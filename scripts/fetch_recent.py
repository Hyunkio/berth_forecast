"""실시간 데이터 갱신 파이프라인.

최근 vessel 데이터 + 기상 데이터를 API에서 수집하고
30-day 입력 윈도우(X_latest.npy / X_ev_latest.npy)를 재생성한다.
이 파일이 존재하면 Flask 대시보드가 테스트셋 말단 대신 이 데이터를 사용한다.

사용법:
    python scripts/fetch_recent.py              # 오늘까지 갱신
    python scripts/fetch_recent.py --dry-run    # API 미사용, 더미 데이터로 갱신 시뮬레이션

필요 환경변수 (.env):
    VESSEL_API_KEY=<data.go.kr 선박운항정보 서비스키>
    WEATHER_API_KEY=<apihub.kma.go.kr 기상청 API 인증키>
"""
import argparse
import json
import os
import pickle
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── 상수 ──────────────────────────────────────────────────────────────────────
PROCESSED    = Path("data/processed")
EXTERNAL     = Path("data/external")
TARGET_PORTS = ["부산", "울산", "인천", "광양"]
SHIP_GROUPS  = ["컨테이너", "유조선", "벌크", "일반화물"]
INPUT_WINDOW = 30
PRED_HORIZON = 7

# 해양수산부 선박운항정보 API
VESSEL_API_URL = "http://apis.data.go.kr/1192000/VsslEtrynd5/Info5"
PORT_CODES = {
    "부산": "020",
    "인천": "030",
    "울산": "032",
    "광양": "055",
}

# 기상청 ASOS API
WEATHER_API_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php"
PORT_STN = {"부산": "159", "울산": "152", "인천": "112", "광양": "168"}
WEATHER_COLS = {"기온": 10, "강수량": 38, "풍속": 2, "최대풍속": 5, "습도": 18}
MISSING_VALS = {"-9", "-9.0", "-9.00", "-99", "-99.0", "-999", "-999.0"}


# ── 1. Vessel API 수집 ────────────────────────────────────────────────────────
def _fetch_vessel_day(api_key: str, port_code: str, date_str: str) -> list[dict]:
    """단일 항만·단일 일자 입항 데이터 조회."""
    params = {
        "serviceKey": api_key,
        "prtAgCd":    port_code,
        "sde":        date_str,
        "ede":        date_str,
        "deGb":       "I",      # I=입항기준
        "numOfRows":  1000,
        "pageNo":     1,
        "_type":      "json",
    }
    try:
        r = requests.get(VESSEL_API_URL, params=params, timeout=30)
        r.raise_for_status()
        body = r.json().get("response", {}).get("body", {})
        items = body.get("items", {})
        if not items:
            return []
        raw = items.get("item", [])
        return raw if isinstance(raw, list) else [raw]
    except Exception as exc:
        print(f"    [vessel API 오류] {date_str}: {exc}")
        return []


def fetch_vessel_range(api_key: str, start: date, end: date) -> pd.DataFrame:
    """start ~ end 기간 전체 항만 입항 데이터 수집 → 일별 집계 DataFrame."""
    rows = []
    cur = start
    while cur <= end:
        date_str = cur.strftime("%Y%m%d")
        for port, code in PORT_CODES.items():
            items = _fetch_vessel_day(api_key, code, date_str)
            for it in items:
                rows.append({
                    "항명":     port,
                    "date":     cur,
                    "입항일시": it.get("arvDt", ""),
                    "출항일시": it.get("dprtDt", ""),
                    "선박용도": it.get("vslKndNm", "기타"),
                    "총톤수":   it.get("grsTn", 0),
                })
            time.sleep(0.1)
        print(f"  {date_str} 수집 완료")
        cur += timedelta(days=1)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["입항일시"] = pd.to_datetime(df["입항일시"], errors="coerce")
    df["출항일시"] = pd.to_datetime(df["출항일시"], errors="coerce")
    df["체류시간_시간"] = (df["출항일시"] - df["입항일시"]).dt.total_seconds() / 3600
    df = df[df["체류시간_시간"] > 0].copy()
    return df


def _aggregate_vessel(df: pd.DataFrame, threshold_map: dict) -> pd.DataFrame:
    """raw vessel records → daily_aggregated 형식."""
    # 선박 유형 → 그룹 매핑
    type_map = {"컨테이너선": "컨테이너", "유조선": "유조선", "산적화물선": "벌크",
                "일반화물선": "일반화물"}
    df["ship_group"] = df["선박용도"].map(type_map).fillna("기타")

    records = []
    for (port, dt), grp in df.groupby(["항명", "date"]):
        thr = threshold_map.get(port, grp["체류시간_시간"].quantile(0.9))
        율 = float((grp["체류시간_시간"] > thr).sum()) / len(grp)
        row = {
            "항명":     port,
            "date":     dt,
            "입항수":   len(grp),
            "평균체류시간": float(grp["체류시간_시간"].mean()),
            "체선율":   round(율, 6),
        }
        for g in SHIP_GROUPS:
            cnt = (grp["ship_group"] == g).sum()
            row[f"{g}_비율"] = round(cnt / len(grp), 6)
        records.append(row)

    return pd.DataFrame(records)


# ── 2. Weather API 수집 ───────────────────────────────────────────────────────
def fetch_weather_range(api_key: str, start: date, end: date) -> pd.DataFrame:
    """기상청 ASOS API → 항만별 기상 피처."""
    tm1 = start.strftime("%Y%m%d")
    tm2 = end.strftime("%Y%m%d")
    frames = {}
    for port, stn in PORT_STN.items():
        try:
            r = requests.get(WEATHER_API_URL,
                             params={"tm1": tm1, "tm2": tm2, "stn": stn, "authKey": api_key},
                             timeout=60)
            r.raise_for_status()
            rows = []
            for line in r.text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 39:
                    continue
                row = {"date": pd.Timestamp(parts[0])}
                for col, idx in WEATHER_COLS.items():
                    val = parts[idx]
                    row[f"{port}_{col}"] = float("nan") if val in MISSING_VALS else float(val)
                rows.append(row)
            if rows:
                tmp = pd.DataFrame(rows).set_index("date")
                frames[port] = tmp
                print(f"  날씨 {port}: {len(tmp)}일")
        except Exception as exc:
            print(f"  [날씨 오류] {port}: {exc}")
        time.sleep(0.5)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(list(frames.values()), axis=1)
    for port in TARGET_PORTS:
        ws_col = f"{port}_최대풍속"
        rn_col = f"{port}_강수량"
        storm = pd.Series(False, index=merged.index)
        if ws_col in merged.columns:
            storm |= merged[ws_col] >= 14.0
        if rn_col in merged.columns:
            storm |= merged[rn_col] >= 50.0
        merged[f"{port}_기상이벤트"] = storm.astype(int)
    return merged.ffill().fillna(0)


# ── 3. 피처 행렬 재생성 ───────────────────────────────────────────────────────
def rebuild_latest_window(daily: pd.DataFrame, weather: pd.DataFrame,
                           meta: dict, scaler_X, scaler_X_ev,
                           event_flags: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    """최신 30일 윈도우 X_latest (79 features) 와 X_ev_latest (87 features) 반환."""
    feature_cols    = meta["feature_cols"]
    ev_feature_cols = meta["ev_feature_cols"]

    # daily → pivot
    date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")

    def pv(col, fill=0):
        p = daily.pivot_table(index="date", columns="항명", values=col)
        return p.reindex(date_range).ffill().fillna(fill)

    pv_rate  = pv("체선율")
    pv_count = pv("입항수")
    pv_stay  = pv("평균체류시간")
    pv_ship  = {g: pv(f"{g}_비율") for g in SHIP_GROUPS if f"{g}_비율" in daily.columns}

    df_feat = pd.DataFrame(index=date_range)
    df_feat.index.name = "date"
    df_feat["dayofweek"]  = df_feat.index.dayofweek
    df_feat["month"]      = df_feat.index.month
    df_feat["is_weekend"] = (df_feat.index.dayofweek >= 5).astype(int)

    for port in TARGET_PORTS:
        if port in pv_rate.columns:
            df_feat[f"{port}_체선율"]   = pv_rate[port]
            df_feat[f"{port}_입항수"]   = pv_count[port]
            df_feat[f"{port}_평균체류"] = pv_stay[port]
        for g, pv_g in pv_ship.items():
            if port in pv_g.columns:
                df_feat[f"{port}_{g}비율"] = pv_g[port]

    for port in TARGET_PORTS:
        col = f"{port}_체선율"
        if col not in df_feat.columns:
            continue
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            df_feat[f"{port}_lag{lag}"] = df_feat[col].shift(lag)
        df_feat[f"{port}_ma7"]  = df_feat[col].shift(1).rolling(7).mean()
        df_feat[f"{port}_ma14"] = df_feat[col].shift(1).rolling(14).mean()
        df_feat[f"{port}_ma30"] = df_feat[col].shift(1).rolling(30).mean()

    # cross-port spread (CCF 기반 연쇄 혼잡 피처)
    CASCADE_LAGS_LOCAL = [("부산","울산",2), ("부산","광양",1), ("부산","인천",1), ("울산","광양",1)]
    for src, tgt, lag in CASCADE_LAGS_LOCAL:
        cs, ct = f"{src}_체선율", f"{tgt}_체선율"
        if cs in df_feat.columns and ct in df_feat.columns:
            df_feat[f"{src}_{tgt}_spread"] = df_feat[cs].shift(lag) - df_feat[ct].shift(lag)

    # 기상 병합
    if not weather.empty:
        for port in TARGET_PORTS:
            for wc in ["기온", "강수량", "풍속", "최대풍속", "습도"]:
                c = f"{port}_{wc}"
                if c in weather.columns:
                    df_feat[c] = weather[c].reindex(df_feat.index).ffill().fillna(0)
                else:
                    df_feat[c] = 0.0

    df_feat = df_feat.dropna().reset_index()
    df_feat_cols = [c for c in df_feat.columns if c != "date"]

    # 마지막 INPUT_WINDOW 행만 사용
    if len(df_feat) < INPUT_WINDOW:
        raise ValueError(f"데이터 부족: {len(df_feat)}일 < {INPUT_WINDOW}일 필요")

    window = df_feat[df_feat_cols].iloc[-INPUT_WINDOW:].values.astype("float32")
    window_cols = df_feat_cols

    # 피처 수 정렬 (meta 기준 컬럼만)
    col_idx = {c: i for i, c in enumerate(window_cols)}
    X_base = np.array([[window[r, col_idx[c]] if c in col_idx else 0.0
                        for c in feature_cols] for r in range(INPUT_WINDOW)],
                      dtype="float32")
    n_base = len(feature_cols)
    X_base_scaled = scaler_X.transform(X_base.reshape(-1, n_base)).reshape(1, INPUT_WINDOW, n_base)

    # 이벤트 버전
    ev_extra_cols = [c for c in ev_feature_cols if c not in feature_cols]
    ev_vals = {}
    # storm 플래그
    for port in TARGET_PORTS:
        c = f"{port}_기상이벤트"
        if not weather.empty and c in weather.columns:
            ev_vals[c] = weather[c].reindex(df_feat.set_index("date").index).fillna(0).values[-INPUT_WINDOW:]
        else:
            ev_vals[c] = np.zeros(INPUT_WINDOW, dtype="float32")
    # LLM 이벤트 플래그
    ev_llm_cols = ["strike", "weather", "surge", "any_event"]
    if event_flags is not None:
        for ec in ev_llm_cols:
            if ec in event_flags.columns:
                ev_vals[ec] = event_flags[ec].reindex(df_feat.set_index("date").index).fillna(0).values[-INPUT_WINDOW:]
            else:
                ev_vals[ec] = np.zeros(INPUT_WINDOW, dtype="float32")
    else:
        for ec in ev_llm_cols:
            ev_vals[ec] = np.zeros(INPUT_WINDOW, dtype="float32")

    X_ev = np.concatenate([
        X_base,
        np.stack([ev_vals.get(c, np.zeros(INPUT_WINDOW)) for c in ev_extra_cols], axis=1)
    ], axis=1)
    n_ev = len(ev_feature_cols)
    X_ev_scaled = scaler_X_ev.transform(X_ev.reshape(-1, n_ev)).reshape(1, INPUT_WINDOW, n_ev)

    return X_base_scaled.astype("float32"), X_ev_scaled.astype("float32")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main(dry_run: bool = False) -> None:
    vessel_key  = os.environ.get("VESSEL_API_KEY", "")
    weather_key = os.environ.get("WEATHER_API_KEY", "")

    # 현재 daily_aggregated 로드
    daily = pd.read_csv(PROCESSED / "daily_aggregated.csv",
                        parse_dates=["date"], encoding="utf-8-sig")
    last_date = daily["date"].max().date()
    today     = date.today()
    print(f"현재 데이터 최신일: {last_date}, 오늘: {today}, 갱신 필요: {(today - last_date).days}일")

    if (today - last_date).days <= 0:
        print("데이터가 이미 최신입니다.")
    elif dry_run:
        print("[dry-run] API 수집 생략 — 더미 행 추가로 시뮬레이션")
        new_rows = []
        for d in range(1, (today - last_date).days + 1):
            dt = last_date + timedelta(days=d)
            for port in TARGET_PORTS:
                sample = daily[daily["항명"] == port].tail(7)
                new_rows.append({
                    "항명":       port,
                    "date":       pd.Timestamp(dt),
                    "입항수":     int(sample["입항수"].mean()),
                    "평균체류시간": float(sample["평균체류시간"].mean()),
                    "체선율":     float(sample["체선율"].mean()),
                    **{f"{g}_비율": float(sample[f"{g}_비율"].mean()) for g in SHIP_GROUPS
                       if f"{g}_비율" in sample.columns},
                })
        daily = pd.concat([daily, pd.DataFrame(new_rows)], ignore_index=True)
        daily.to_csv(PROCESSED / "daily_aggregated.csv", index=False, encoding="utf-8-sig")
        print(f"  → 더미 {len(new_rows)}행 추가, 최신일: {daily['date'].max().date()}")
    elif vessel_key:
        print("vessel API 수집 중...")
        raw_df = fetch_vessel_range(vessel_key,
                                    last_date + timedelta(days=1), today)
        if not raw_df.empty:
            # 90th pct threshold (기존 vessel_clean.csv 기준)
            vc = pd.read_csv(PROCESSED / "vessel_clean.csv")
            thr_map = vc.groupby("항명")["threshold_90p"].first().to_dict()
            new_agg = _aggregate_vessel(raw_df, thr_map)
            daily = pd.concat([daily, new_agg], ignore_index=True)
            daily.to_csv(PROCESSED / "daily_aggregated.csv", index=False, encoding="utf-8-sig")
            print(f"  → vessel {len(raw_df)}건 추가 완료")
        else:
            print("  vessel 수집 결과 없음 — 기존 데이터 유지")
    else:
        print("[경고] VESSEL_API_KEY 없음. vessel 갱신 생략.")

    # 날씨 갱신
    weather_path = EXTERNAL / "weather_daily.csv"
    weather_df = pd.read_csv(weather_path, parse_dates=["date"], encoding="utf-8-sig").set_index("date")
    weather_last = weather_df.index.max().date()
    print(f"날씨 최신일: {weather_last}")

    if (today - weather_last).days > 0 and not dry_run:
        if weather_key:
            print("날씨 API 수집 중...")
            new_weather = fetch_weather_range(weather_key,
                                              weather_last + timedelta(days=1), today)
            if not new_weather.empty:
                weather_df = pd.concat([weather_df, new_weather]).sort_index()
                weather_df.reset_index().to_csv(weather_path, index=False, encoding="utf-8-sig")
                print(f"  → 날씨 {len(new_weather)}일 추가 완료")
        else:
            print("  [경고] WEATHER_API_KEY 없음. 날씨 갱신 생략 — 마지막 값 forward-fill로 사용.")
    elif dry_run:
        print("[dry-run] 날씨 갱신 생략")

    # X_latest.npy 재생성
    print("X_latest.npy 재생성 중...")
    with open(PROCESSED / "dataset_meta.json") as f:
        meta = json.load(f)
    with open(PROCESSED / "scaler_X.pkl",    "rb") as f:
        scaler_X = pickle.load(f)
    with open(PROCESSED / "scaler_X_ev.pkl", "rb") as f:
        scaler_X_ev = pickle.load(f)

    event_flags = None
    ev_path = PROCESSED / "event_flags.csv"
    if ev_path.exists():
        event_flags = pd.read_csv(ev_path, parse_dates=["date"]).set_index("date")

    try:
        X_lat, X_ev_lat = rebuild_latest_window(
            daily, weather_df.reset_index().set_index("date"),
            meta, scaler_X, scaler_X_ev, event_flags
        )
        np.save(PROCESSED / "X_latest.npy",    X_lat)
        np.save(PROCESSED / "X_ev_latest.npy", X_ev_lat)
        # X_tf_latest: X_latest에서 79 기본 피처만 추출 (Transformer용)
        tf_feature_cols = [c for c in meta["feature_cols"]
                           if not any(c.endswith(f"_lag{l}") for l in [21,28])
                           and not c.endswith("_ma30")
                           and not c.endswith("_spread")]
        tf_idx = [meta["feature_cols"].index(c) for c in tf_feature_cols]
        X_tf_lat = X_lat[:, :, tf_idx]
        np.save(PROCESSED / "X_tf_latest.npy", X_tf_lat)
        # 예측 기준일 저장 (flask_app.py가 읽음)
        with open(PROCESSED / "latest_date.txt", "w") as f:
            f.write(str(today))
        print(f"  → X_latest {X_lat.shape}, X_tf_latest {X_tf_lat.shape}, X_ev_latest {X_ev_lat.shape} 저장 완료")
        print(f"  → 예측 기준: {today} → D+1~D+{PRED_HORIZON}")
    except Exception as exc:
        print(f"  [오류] X_latest 생성 실패: {exc}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="API 미사용, 더미 데이터로 갱신 시뮬레이션")
    args = parser.parse_args()
    main(dry_run=args.dry_run)

"""기상청 API Hub ASOS 일자료 수집 → 항만별 기상 피처 생성.

사용법:
    python scripts/collect_weather.py
    python scripts/collect_weather.py --out data/external/weather_daily.csv

필요 설정:
    .env 파일에 WEATHER_API_KEY=<apihub.kma.go.kr 인증키> 추가
    발급: https://apihub.kma.go.kr → 지상관측 > ASOS > kma_sfcdd3.php 신청

항만-관측소 매핑:
    부산 ← STN 159 (부산)
    울산 ← STN 152 (울산)
    인천 ← STN 112 (인천)
    광양 ← STN 168 (여수, 광양 인근 최근접 관측소)

컬럼 위치 (kma_sfcdd3 고정 포맷, 0-based):
    0: 날짜(YYYYMMDD)  1: STN
    2: WS_AVG(평균풍속)  5: WS_MAX(최대풍속)  10: TA_AVG(평균기온)
    18: HM_AVG(평균습도)  38: RN_DAY(일강수량)
"""
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

STUDY_START = "20240620"
STUDY_END   = "20260519"

PORT_STN = {
    "부산": "159",
    "울산": "152",
    "인천": "112",
    "광양": "168",
}

API_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php"

# kma_sfcdd3 응답 컬럼 인덱스 (공백 분리 후 0-based)
COL_IDX = {
    "기온":     10,   # TA_AVG  평균기온 (°C)
    "강수량":   38,   # RN_DAY  일강수량 (mm)
    "풍속":      2,   # WS_AVG  평균풍속 (m/s)
    "최대풍속":  5,   # WS_MAX  최대풍속 (m/s)
    "습도":     18,   # HM_AVG  평균습도 (%)
}

MISSING = {"-9", "-9.0", "-9.00", "-99", "-99.0", "-999", "-999.0"}


def _parse_response(text: str) -> list[dict]:
    """kma_sfcdd3 텍스트 응답 파싱 → 행 리스트."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 39:
            continue
        row = {"date": parts[0]}
        for name, idx in COL_IDX.items():
            val = parts[idx]
            row[name] = float("nan") if val in MISSING else float(val)
        rows.append(row)
    return rows


def fetch_station(auth_key: str, stn_id: str, tm1: str, tm2: str) -> list[dict]:
    """한 관측소 한 기간 요청."""
    params = {"tm1": tm1, "tm2": tm2, "stn": stn_id, "authKey": auth_key}
    resp = requests.get(API_URL, params=params, timeout=60)
    resp.raise_for_status()
    return _parse_response(resp.text)


def fetch_port_weather(auth_key: str, port: str, stn_id: str) -> pd.DataFrame:
    """항만별 전체 기간 수집 (3개월 단위 분할)."""
    # 3개월 단위로 분할 — API 응답 크기 안전 마진
    periods = pd.date_range(STUDY_START, STUDY_END, freq="QS")
    if periods[-1] < pd.Timestamp(STUDY_END):
        periods = periods.append(pd.DatetimeIndex([pd.Timestamp(STUDY_END)]))

    all_rows = []
    starts = [pd.Timestamp(STUDY_START)] + list(periods[1:])
    ends   = list(periods[1:]) + [pd.Timestamp(STUDY_END)]

    for s, e in zip(starts, ends):
        tm1 = s.strftime("%Y%m%d")
        tm2 = e.strftime("%Y%m%d")
        try:
            rows = fetch_station(auth_key, stn_id, tm1, tm2)
            all_rows.extend(rows)
            print(f"    {tm1}~{tm2}: {len(rows)}일")
        except Exception as exc:
            print(f"    [{port}] {tm1}~{tm2} 실패: {exc}")
        time.sleep(0.5)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates("date").set_index("date")
    df.columns = [f"{port}_{c}" for c in df.columns]
    return df


def make_storm_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for port in PORT_STN:
        ws_col = f"{port}_최대풍속"
        rn_col = f"{port}_강수량"
        storm  = pd.Series(False, index=df.index)
        if ws_col in df.columns:
            storm |= df[ws_col] >= 14.0   # 강풍 경보 기준
        if rn_col in df.columns:
            storm |= df[rn_col] >= 50.0   # 호우 주의보 기준
        df[f"{port}_기상이벤트"] = storm.astype(int)
    return df


def main(out_path: Path) -> None:
    auth_key = os.environ.get("WEATHER_API_KEY")
    if not auth_key:
        print("[오류] WEATHER_API_KEY가 설정되지 않았습니다.")
        _make_dummy(out_path)
        return

    print(f"기상 데이터 수집 시작 ({STUDY_START} ~ {STUDY_END})")
    frames = {}
    for port, stn_id in PORT_STN.items():
        print(f"  {port} (관측소 {stn_id}) 수집 중...")
        df_port = fetch_port_weather(auth_key, port, stn_id)
        if not df_port.empty:
            frames[port] = df_port
            print(f"    → 총 {len(df_port)}일 수집 완료")
        else:
            print(f"    → 수집 실패")

    if not frames:
        print("전체 수집 실패. dummy 데이터로 대체합니다.")
        _make_dummy(out_path)
        return

    date_range = pd.date_range(STUDY_START, STUDY_END, freq="D")
    merged = pd.DataFrame(index=date_range)
    merged.index.name = "date"
    for df_port in frames.values():
        merged = merged.join(df_port, how="left")

    merged = merged.ffill().fillna(0)
    merged = make_storm_flag(merged)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.reset_index().to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {out_path}  ({merged.shape[0]}일 × {merged.shape[1]}컬럼)")

    for port in PORT_STN:
        col = f"{port}_기상이벤트"
        if col in merged.columns:
            cnt = int(merged[col].sum())
            print(f"  {port} 기상이벤트: {cnt}일")


def _make_dummy(out_path: Path) -> None:
    date_range = pd.date_range(STUDY_START, STUDY_END, freq="D")
    df = pd.DataFrame(index=date_range)
    df.index.name = "date"
    for port in PORT_STN:
        for col in COL_IDX:
            df[f"{port}_{col}"] = 0.0
        df[f"{port}_기상이벤트"] = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[placeholder] 저장: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/external/weather_daily.csv")
    args = parser.parse_args()
    main(Path(args.out))

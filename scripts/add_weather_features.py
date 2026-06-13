"""기상 피처를 기존 피처에 추가해 X/y 배열을 재생성한다.

실행:
    python scripts/add_weather_features.py

결과:
    data/processed/X_train.npy  (N, 30, 94)  ← 기존 79 + lag21/28(8) + ma30(4) + spread(3)
    data/processed/X_ev_train.npy  (N, 30, 102)  ← 기본 94 + LLM이벤트 4 + 기상이벤트 4
    (val/test도 동일 구조)
    scaler_X.pkl, scaler_X_ev.pkl 갱신
    dataset_meta.json 갱신
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

PROCESSED = Path("data/processed")
EXTERNAL  = Path("data/external")

TARGET_PORTS = ["부산", "울산", "인천", "광양"]
SHIP_GROUPS  = ["컨테이너", "유조선", "벌크", "일반화물"]
CASCADE_LAGS = [("부산","울산",2), ("부산","광양",1), ("부산","인천",1), ("울산","광양",1)]

INPUT_WINDOW = 30
PRED_HORIZON = 7

WEATHER_COLS = ["기온", "강수량", "풍속", "최대풍속", "습도"]
EVENT_COLS   = ["strike", "weather", "surge", "any_event"]


# ── 1. 기본 일별 집계 로드 ────────────────────────────────────────────────────
daily = pd.read_csv(PROCESSED / "daily_aggregated.csv",
                    parse_dates=["date"], encoding="utf-8-sig")
date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")

def pivot_metric(col, fill=0):
    pv = daily.pivot_table(index="date", columns="항명", values=col)
    return pv.reindex(date_range).ffill().fillna(fill)

pv_rate  = pivot_metric("체선율")
pv_count = pivot_metric("입항수")
pv_stay  = pivot_metric("평균체류시간")
pv_ship  = {}
for g in SHIP_GROUPS:
    col = f"{g}_비율"
    if col in daily.columns:
        pv_ship[g] = pivot_metric(col)

# 항만별 체선율 99th percentile cap (이상치 처리)
cap_99 = {}
for port in TARGET_PORTS:
    port_rates = daily[daily["항명"] == port]["체선율"]
    cap_99[port] = port_rates.quantile(0.99)
    print(f"  {port} 체선율 cap: {cap_99[port]:.4f} (원본 max={port_rates.max():.4f})")
pv_rate = pv_rate.copy()
for port in TARGET_PORTS:
    if port in pv_rate.columns:
        pv_rate[port] = pv_rate[port].clip(upper=cap_99[port])

# ── 2. df_feat 구성 (시간 + 항만 + 래그 + 연쇄혼잡 — 기존 57개) ──────────────
try:
    import holidays as hol_pkg
    kr_holidays = hol_pkg.KR(years=range(2024, 2027))
    holiday_dates = list(kr_holidays.keys())
except ImportError:
    holiday_dates = []

df_feat = pd.DataFrame(index=date_range)
df_feat.index.name = "date"
df_feat["dayofweek"]  = df_feat.index.dayofweek
df_feat["month"]      = df_feat.index.month
# quarter 제거: month와 r=0.97 중복
df_feat["is_weekend"] = (df_feat.index.dayofweek >= 5).astype(int)
# is_holiday 제거: 분산≈0 (무의미 피처)

for port in TARGET_PORTS:
    if port in pv_rate.columns:
        df_feat[f"{port}_체선율"]   = pv_rate[port]
        df_feat[f"{port}_입항수"]   = pv_count[port]
        df_feat[f"{port}_평균체류"] = pv_stay[port]
    for g, pv in pv_ship.items():
        if port in pv.columns:
            df_feat[f"{port}_{g}비율"] = pv[port]

for port in TARGET_PORTS:
    col = f"{port}_체선율"
    if col not in df_feat.columns:
        continue
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df_feat[f"{port}_lag{lag}"] = df_feat[col].shift(lag)
    df_feat[f"{port}_ma7"]  = df_feat[col].shift(1).rolling(7).mean()
    df_feat[f"{port}_ma14"] = df_feat[col].shift(1).rolling(14).mean()
    df_feat[f"{port}_ma30"] = df_feat[col].shift(1).rolling(30).mean()

# cross-port spread: 항만 간 체선율 격차 (CCF 최대 상관 시차 기준)
# spread(부산-울산) lag2 → 울산 예측: r=-0.579 (실증)
# spread(부산-광양) lag1 → 광양 예측: r=-0.283
# spread(울산-광양) lag1 → 광양 예측: r=-0.351
cross_port_feat_cols = []
for src, tgt, lag in CASCADE_LAGS:
    col_src = f"{src}_체선율"
    col_tgt = f"{tgt}_체선율"
    if col_src in df_feat.columns and col_tgt in df_feat.columns:
        feat_name = f"{src}_{tgt}_spread"
        df_feat[feat_name] = df_feat[col_src].shift(lag) - df_feat[col_tgt].shift(lag)
        cross_port_feat_cols.append(feat_name)
print(f"Cross-port spread 피처: {cross_port_feat_cols}")

print(f"기존 피처 수 (날씨 전): {df_feat.shape[1]}")

# ── 3. 기상 피처 병합 ─────────────────────────────────────────────────────────
weather_path = EXTERNAL / "weather_daily.csv"
if not weather_path.exists():
    print("[오류] data/external/weather_daily.csv 없음. collect_weather.py 먼저 실행하세요.")
    sys.exit(1)

df_w = pd.read_csv(weather_path, parse_dates=["date"], encoding="utf-8-sig")
df_w = df_w.set_index("date")

# 연속형 기상 피처만 (storm 플래그는 이벤트 버전에서만 사용)
weather_feat_cols = []
for port in TARGET_PORTS:
    for col in WEATHER_COLS:
        c = f"{port}_{col}"
        if c in df_w.columns:
            df_feat[c] = df_w[c].reindex(df_feat.index).ffill().fillna(0)
            weather_feat_cols.append(c)

print(f"기상 피처 추가 후: {df_feat.shape[1]} (추가 {len(weather_feat_cols)}개)")

# ── 4. NaN 제거 & 슬라이딩 윈도우 ────────────────────────────────────────────
df_feat = df_feat.dropna().reset_index()
print(f"NaN 제거 후: {len(df_feat)}일")

feature_cols = [c for c in df_feat.columns if c != "date"]
target_cols  = [f"{p}_체선율" for p in TARGET_PORTS if f"{p}_체선율" in df_feat.columns]

X_list, y_list, date_list = [], [], []
for i in range(INPUT_WINDOW, len(df_feat) - PRED_HORIZON + 1):
    X_list.append(df_feat[feature_cols].iloc[i - INPUT_WINDOW:i].values)
    y_list.append(df_feat[target_cols].iloc[i:i + PRED_HORIZON].values)
    date_list.append(df_feat["date"].iloc[i])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
print(f"X: {X.shape}, y: {y.shape}")

# ── 5. Train/Val/Test 분할 ────────────────────────────────────────────────────
N = len(X)
n_train = int(N * 0.7)
n_val   = int(N * 0.15)
n_test  = N - n_train - n_val

X_train_raw, X_val_raw, X_test_raw = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
y_train_raw, y_val_raw, y_test_raw = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]

n_feats = X.shape[-1]
n_ports = y.shape[-1]

scaler_X = MinMaxScaler()
scaler_X.fit(X_train_raw.reshape(-1, n_feats))
X_train = scaler_X.transform(X_train_raw.reshape(-1, n_feats)).reshape(X_train_raw.shape).astype("float32")
X_val   = scaler_X.transform(X_val_raw.reshape(-1, n_feats)).reshape(X_val_raw.shape).astype("float32")
X_test  = scaler_X.transform(X_test_raw.reshape(-1, n_feats)).reshape(X_test_raw.shape).astype("float32")

scaler_y = MinMaxScaler()
scaler_y.fit(y_train_raw.reshape(-1, n_ports))
y_train = scaler_y.transform(y_train_raw.reshape(-1, n_ports)).reshape(y_train_raw.shape).astype("float32")
y_val   = scaler_y.transform(y_val_raw.reshape(-1, n_ports)).reshape(y_val_raw.shape).astype("float32")
y_test  = scaler_y.transform(y_test_raw.reshape(-1, n_ports)).reshape(y_test_raw.shape).astype("float32")

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ── 6. 기상 저장 ──────────────────────────────────────────────────────────────
np.save(PROCESSED / "X_train.npy", X_train)
np.save(PROCESSED / "X_val.npy",   X_val)
np.save(PROCESSED / "X_test.npy",  X_test)
np.save(PROCESSED / "y_train.npy", y_train)
np.save(PROCESSED / "y_val.npy",   y_val)
np.save(PROCESSED / "y_test.npy",  y_test)

with open(PROCESSED / "scaler_X.pkl", "wb") as f:
    pickle.dump(scaler_X, f)
with open(PROCESSED / "scaler_y.pkl", "wb") as f:
    pickle.dump(scaler_y, f)

# ── 7. 이벤트 버전: 기상(연속) + LLM이벤트 + 기상이벤트 플래그 ──────────────
# storm 플래그 (4개)
storm_feat_cols = []
for port in TARGET_PORTS:
    c = f"{port}_기상이벤트"
    if c in df_w.columns:
        df_feat[c] = df_w[c].reindex(pd.to_datetime(df_feat["date"])).fillna(0).values
        storm_feat_cols.append(c)

# LLM 이벤트 플래그 (4개)
event_path = PROCESSED / "event_flags.csv"
if event_path.exists():
    df_ev = pd.read_csv(event_path, parse_dates=["date"])
    df_ev = df_ev.set_index("date")
    for ec in EVENT_COLS:
        if ec in df_ev.columns:
            df_feat[ec] = df_ev[ec].reindex(pd.to_datetime(df_feat["date"])).fillna(0).values
        else:
            df_feat[ec] = 0
    llm_cols = EVENT_COLS
else:
    for ec in EVENT_COLS:
        df_feat[ec] = 0
    llm_cols = EVENT_COLS

ev_extra = storm_feat_cols + llm_cols
feature_cols_ev = feature_cols + ev_extra
print(f"\n이벤트 버전 피처 수: {len(feature_cols_ev)} "
      f"(기상연속 {len(feature_cols)}, storm {len(storm_feat_cols)}, LLM {len(llm_cols)})")

X_ev_list = []
for i in range(INPUT_WINDOW, len(df_feat) - PRED_HORIZON + 1):
    X_ev_list.append(df_feat[feature_cols_ev].iloc[i - INPUT_WINDOW:i].values)

X_ev = np.array(X_ev_list, dtype=np.float32)
X_ev_train_raw = X_ev[:n_train]
X_ev_val_raw   = X_ev[n_train:n_train+n_val]
X_ev_test_raw  = X_ev[n_train+n_val:]

n_ev_feats = X_ev.shape[-1]
scaler_X_ev = MinMaxScaler()
scaler_X_ev.fit(X_ev_train_raw.reshape(-1, n_ev_feats))

X_ev_train = scaler_X_ev.transform(X_ev_train_raw.reshape(-1, n_ev_feats)).reshape(X_ev_train_raw.shape).astype("float32")
X_ev_val   = scaler_X_ev.transform(X_ev_val_raw.reshape(-1, n_ev_feats)).reshape(X_ev_val_raw.shape).astype("float32")
X_ev_test  = scaler_X_ev.transform(X_ev_test_raw.reshape(-1, n_ev_feats)).reshape(X_ev_test_raw.shape).astype("float32")

scaler_y_ev = MinMaxScaler()
scaler_y_ev.fit(y_train_raw.reshape(-1, n_ports))
y_ev_train = scaler_y_ev.transform(y_train_raw.reshape(-1, n_ports)).reshape(y_train_raw.shape).astype("float32")
y_ev_val   = scaler_y_ev.transform(y_val_raw.reshape(-1, n_ports)).reshape(y_val_raw.shape).astype("float32")
y_ev_test  = scaler_y_ev.transform(y_test_raw.reshape(-1, n_ports)).reshape(y_test_raw.shape).astype("float32")

np.save(PROCESSED / "X_ev_train.npy", X_ev_train)
np.save(PROCESSED / "X_ev_val.npy",   X_ev_val)
np.save(PROCESSED / "X_ev_test.npy",  X_ev_test)
np.save(PROCESSED / "y_ev_train.npy", y_ev_train)
np.save(PROCESSED / "y_ev_val.npy",   y_ev_val)
np.save(PROCESSED / "y_ev_test.npy",  y_ev_test)

with open(PROCESSED / "scaler_X_ev.pkl", "wb") as f:
    pickle.dump(scaler_X_ev, f)
with open(PROCESSED / "scaler_y_ev.pkl", "wb") as f:
    pickle.dump(scaler_y_ev, f)

# ── 8. 메타데이터 저장 ────────────────────────────────────────────────────────
meta = {
    "feature_cols":    feature_cols,
    "ev_feature_cols": feature_cols_ev,
    "target_cols":     target_cols,
    "input_window":    INPUT_WINDOW,
    "pred_horizon":    PRED_HORIZON,
    "n_train":         int(n_train),
    "n_val":           int(n_val),
    "n_test":          int(n_test),
    "cascade_lags":    CASCADE_LAGS,
    "scaler_fit":      "train_only",
    "weather_cols":    weather_feat_cols,
}
with open(PROCESSED / "dataset_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n완료:")
print(f"  기본 피처: {len(feature_cols)}개 (기존+기상+lag21/28+ma30+spread)")
print(f"  이벤트 피처: {len(feature_cols_ev)}개 (+storm {len(storm_feat_cols)} +LLM {len(llm_cols)})")
print(f"  X_train:    {X_train.shape}, X_ev_train: {X_ev_train.shape}")

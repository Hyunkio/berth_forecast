"""피처 엔지니어링: 시간 피처, 래그 피처, 연쇄 혼잡 lag 피처, 슬라이딩 윈도우."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

TARGET_PORTS = ["부산", "울산", "인천", "광양"]

SHIP_GROUPS = ["컨테이너", "유조선", "벌크", "일반화물"]

INPUT_WINDOW = 30
PRED_HORIZON = 7

# EDA Cross-correlation 결과 기반 lag
CASCADE_LAGS = [
    ("부산", "울산", 2),
    ("부산", "광양", 1),
    ("부산", "인천", 1),
    ("울산", "광양", 1),
]

HOLIDAYS = [
    "2024-09-14", "2024-09-15", "2024-09-16", "2024-09-17", "2024-09-18",
    "2025-01-28", "2025-01-29", "2025-01-30",
    "2025-10-02", "2025-10-03", "2025-10-04", "2025-10-05", "2025-10-06",
    "2026-02-16", "2026-02-17", "2026-02-18",
]


def make_pivot(daily: pd.DataFrame) -> dict:
    """daily DataFrame → 항만별 피벗 (metric → DataFrame)."""
    date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")

    def pivot(col, fill=0):
        pv = daily.pivot_table(index="date", columns="항명", values=col)
        return pv.reindex(date_range).ffill().fillna(fill)

    result = {
        "rate":  pivot("체선율"),
        "count": pivot("입항수"),
        "stay":  pivot("평균체류시간"),
    }

    # 선박유형 비율 피벗
    for g in SHIP_GROUPS:
        col = f"{g}_비율"
        if col in daily.columns:
            result[f"ship_{g}"] = pivot(col)

    return result


def build_feature_df(daily: pd.DataFrame) -> tuple:
    """daily → 날짜 인덱스 피처 DataFrame 반환."""
    pivots = make_pivot(daily)
    date_range = pivots["rate"].index

    df = pd.DataFrame(index=date_range)
    df.index.name = "date"

    # 시간 피처
    df["dayofweek"]  = df.index.dayofweek
    df["month"]      = df.index.month
    df["quarter"]    = df.index.quarter
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["is_holiday"] = df.index.isin(pd.to_datetime(HOLIDAYS)).astype(int)

    # 항만 피처
    for port in TARGET_PORTS:
        if port in pivots["rate"].columns:
            df[f"{port}_체선율"]   = pivots["rate"][port]
            df[f"{port}_입항수"]   = pivots["count"][port]
            df[f"{port}_평균체류"] = pivots["stay"][port]

    # 선박유형 비율 피처
    for g in SHIP_GROUPS:
        key = f"ship_{g}"
        if key in pivots:
            for port in TARGET_PORTS:
                if port in pivots[key].columns:
                    df[f"{port}_{g}비율"] = pivots[key][port]

    # 래그 피처
    for port in TARGET_PORTS:
        col = f"{port}_체선율"
        if col not in df.columns:
            continue
        for lag in [7, 14, 30]:
            df[f"{port}_lag{lag}"] = df[col].shift(lag)
        df[f"{port}_ma7"]  = df[col].shift(1).rolling(7).mean()
        df[f"{port}_ma14"] = df[col].shift(1).rolling(14).mean()

    # 연쇄 혼잡 lag 피처
    for src, dst, lag in CASCADE_LAGS:
        src_col = f"{src}_체선율"
        if src_col in df.columns:
            df[f"{src}_to_{dst}_lag{lag}"] = df[src_col].shift(lag)

    feature_cols = list(df.columns)
    target_cols  = [f"{p}_체선율" for p in TARGET_PORTS if f"{p}_체선율" in df.columns]

    return df, feature_cols, target_cols


def make_windows(
    df: pd.DataFrame,
    feature_cols: list,
    target_cols: list,
    input_window: int = INPUT_WINDOW,
    pred_horizon: int = PRED_HORIZON,
) -> tuple:
    """슬라이딩 윈도우로 X, y 생성."""
    df_clean = df.dropna().reset_index()
    X_list, y_list = [], []

    for i in range(input_window, len(df_clean) - pred_horizon + 1):
        X_list.append(df_clean[feature_cols].iloc[i - input_window:i].values)
        y_list.append(df_clean[target_cols].iloc[i:i + pred_horizon].values)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
    )


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict:
    """학습/검증/테스트 분할 + MinMax 정규화."""
    n = len(X)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        "X_train": X[:n_train],               "y_train": y[:n_train],
        "X_val":   X[n_train:n_train+n_val],  "y_val":   y[n_train:n_train+n_val],
        "X_test":  X[n_train+n_val:],          "y_test":  y[n_train+n_val:],
    }

    # 피처 스케일링
    n_s, n_steps, n_f = splits["X_train"].shape
    scaler_X = MinMaxScaler()
    scaler_X.fit(splits["X_train"].reshape(-1, n_f))

    for key in ("X_train", "X_val", "X_test"):
        sh = splits[key].shape
        splits[key] = scaler_X.transform(
            splits[key].reshape(-1, sh[2])
        ).reshape(sh).astype(np.float32)

    # 타겟 스케일링
    n_s2, n_h, n_p = splits["y_train"].shape
    scaler_y = MinMaxScaler()
    scaler_y.fit(splits["y_train"].reshape(-1, n_p))

    for key in ("y_train", "y_val", "y_test"):
        sh = splits[key].shape
        splits[key] = scaler_y.transform(
            splits[key].reshape(-1, sh[2])
        ).reshape(sh).astype(np.float32)

    splits["scaler_X"] = scaler_X
    splits["scaler_y"] = scaler_y
    return splits


def save_dataset(splits: dict, feature_cols: list, target_cols: list, out_dir: Path) -> None:
    """학습 데이터 및 메타 정보 저장."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        np.save(out_dir / f"{key}.npy", splits[key])

    with open(out_dir / "scaler_X.pkl", "wb") as f:
        pickle.dump(splits["scaler_X"], f)
    with open(out_dir / "scaler_y.pkl", "wb") as f:
        pickle.dump(splits["scaler_y"], f)

    meta = {
        "feature_cols": feature_cols,
        "target_cols":  target_cols,
        "input_window": INPUT_WINDOW,
        "pred_horizon": PRED_HORIZON,
        "cascade_lags": [{"src": s, "dst": d, "lag": l} for s, d, l in CASCADE_LAGS],
        "n_train": int(len(splits["X_train"])),
        "n_val":   int(len(splits["X_val"])),
        "n_test":  int(len(splits["X_test"])),
    }
    with open(out_dir / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

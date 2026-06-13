"""전처리 함수: 이상치 제거, 체선 타겟 생성, 일별 집계."""
import pandas as pd

from .loader import COL, TARGET_PORTS


def remove_outliers(df: pd.DataFrame, max_stay_hours: float = 999.0) -> pd.DataFrame:
    """음수 체류시간·극단값 제거."""
    df = df.copy()
    df["체류시간_시간"] = (
        df[COL["depart"]] - df[COL["arrive"]]
    ).dt.total_seconds() / 3600

    mask = (df["체류시간_시간"] >= 0) & (df["체류시간_시간"] <= max_stay_hours)
    return df[mask].copy()


def add_congestion_target(df: pd.DataFrame, percentile: float = 0.90) -> pd.DataFrame:
    """Plan B: 항만별 체류시간 percentile 초과를 체선(1)으로 레이블링."""
    df = df.copy()
    thresholds = (
        df.groupby(COL["port"])["체류시간_시간"]
        .quantile(percentile)
        .rename("threshold")
        .reset_index()
    )
    df = df.merge(thresholds, on=COL["port"], how="left")
    df["체선여부"] = (df["체류시간_시간"] > df["threshold"]).astype(int)
    return df


SHIP_GROUPS = {
    "컨테이너": ["풀컨테이너선", "반컨테이너선"],
    "유조선":   ["석유제품 운반선", "원유운반선", "케미칼 운반선", "기타 유조선"],
    "벌크":     ["산물선(벌크선)", "시멘트운반선", "광석운반선"],
    "일반화물": ["일반화물선", "다목적화물선"],
}


def make_daily(df: pd.DataFrame) -> pd.DataFrame:
    """입항일시 기준 일별 집계 (항만 × 날짜, 선박유형 비율 포함)."""
    df = df.copy()
    df["date"] = df[COL["arrive"]].dt.normalize()

    daily = df.groupby([COL["port"], "date"]).agg(
        입항수=("date", "count"),
        평균체류시간=("체류시간_시간", "mean"),
        체선율=("체선여부", "mean"),
    ).reset_index()

    # 선박유형 비율 추가
    for group, types in SHIP_GROUPS.items():
        ratio = (
            df.groupby([COL["port"], "date"])
            .apply(lambda g: g[COL["ship_type"]].isin(types).mean(), include_groups=False)
            .reset_index(name=f"{group}_비율")
        )
        daily = daily.merge(ratio, on=[COL["port"], "date"], how="left")

    return daily


def preprocess_pipeline(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """원시 데이터 → (vessel_clean, daily_aggregated) 반환."""
    df = remove_outliers(df_raw)
    df = add_congestion_target(df)
    daily = make_daily(df)
    return df, daily

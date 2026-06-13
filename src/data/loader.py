"""선박입출항현황 Excel 파일 로더."""
from pathlib import Path

import pandas as pd


RAW_DIR = Path(__file__).parents[2] / "data" / "raw"

# Excel 파일 구조: 행 0~10은 메타정보, 11번째 행이 헤더
EXCEL_HEADER_ROW = 11

COL = {
    "port":      "항명",
    "arrive":    "입항일시",
    "depart":    "출항일시",
    "berth":     "계선장소",
    "ship_type": "선박용도",
    "tonnage":   "총톤수",
    "inout":     "외내",
}

TARGET_PORTS = ["부산", "울산", "인천", "광양"]

STUDY_START = pd.Timestamp("2024-06-20")
STUDY_END   = pd.Timestamp("2026-05-19 23:59")


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """data/raw/data_*.xlsx 파일을 병합해 최종·연구기간 레코드만 반환."""
    files = sorted(raw_dir.glob("data_*.xlsx"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise FileNotFoundError(f"No data_*.xlsx files found in {raw_dir}")

    dfs = []
    for f in files:
        tmp = pd.read_excel(f, dtype=str, header=EXCEL_HEADER_ROW)
        tmp["source_file"] = f.name
        dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True)

    # 최종 확정 레코드만 사용 (최초/변경 제외)
    df = df[df["구분"] == "최종"].copy()

    # 날짜 변환
    for key in ("arrive", "depart"):
        col = COL[key]
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if COL["tonnage"] in df.columns:
        df[COL["tonnage"]] = pd.to_numeric(
            df[COL["tonnage"]].str.replace(",", "", regex=False), errors="coerce"
        )

    # 연구 기간 필터
    df = df[
        (df[COL["arrive"]] >= STUDY_START) &
        (df[COL["arrive"]] <= STUDY_END)
    ].copy()

    return df


def load_target_ports(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """부산·울산·인천·광양만 필터링해 반환."""
    df = load_raw(raw_dir)
    return df[df[COL["port"]].isin(TARGET_PORTS)].copy()


def load_processed(processed_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """EDA에서 저장한 vessel_clean.csv, daily_aggregated.csv 로드."""
    if processed_dir is None:
        processed_dir = Path(__file__).parents[2] / "data" / "processed"

    vessel = pd.read_csv(
        processed_dir / "vessel_clean.csv",
        parse_dates=["입항일시", "출항일시", "date"],
        encoding="utf-8-sig",
    )
    daily = pd.read_csv(
        processed_dir / "daily_aggregated.csv",
        parse_dates=["date"],
        encoding="utf-8-sig",
    )
    return vessel, daily

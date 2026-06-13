"""항만 관련 뉴스 수집: Google News RSS → JSON 저장.

사용법:
    python scripts/collect_news.py
    python scripts/collect_news.py --out data/raw/news_raw/articles.json
"""
import argparse
import email.utils
import json
import time
from pathlib import Path

import feedparser
import pandas as pd

STUDY_START = pd.Timestamp("2024-06-20")
STUDY_END   = pd.Timestamp("2026-05-19")

# 이벤트 유형별 검색 쿼리
QUERIES = {
    "strike":  [
        "항만 파업",
        "화물연대 파업",
        "항만 노조 파업",
        "부두 운영중단",
        "항만 점거 농성",
    ],
    "weather": [
        "태풍 항만",
        "폭풍 선박 입항 통제",
        "항만 안개 결항",
        "해상 강풍 경보",
        "파고 항만",
    ],
    "surge": [
        "부산항 물동량 급증",
        "항만 체선 컨테이너",
        "홍해 사태 항만",
        "명절 항만 물동량",
        "컨테이너 부족 항만",
        "연말 수출입 급증",
    ],
    "general": [
        "부산항 입항",
        "울산항 선박",
        "인천항 물동량",
        "광양항 컨테이너",
        "항만 혼잡",
    ],
}

RSS_BASE = "https://news.google.com/rss/search?q={}&hl=ko&gl=KR&ceid=KR:ko"


def parse_date(entry: dict) -> pd.Timestamp | None:
    raw = entry.get("published") or entry.get("updated")
    if not raw:
        return None
    try:
        dt = email.utils.parsedate_to_datetime(raw)
        return pd.Timestamp(dt).tz_localize(None)
    except Exception:
        return None


def collect_query(query: str, delay: float = 1.0) -> list[dict]:
    url = RSS_BASE.format(query.replace(" ", "+"))
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries:
        dt = parse_date(entry)
        if dt is None:
            continue
        if not (STUDY_START <= dt <= STUDY_END):
            continue
        articles.append({
            "date":     dt.strftime("%Y-%m-%d"),
            "headline": entry.get("title", "").strip(),
            "source":   entry.get("source", {}).get("title", ""),
            "link":     entry.get("link", ""),
            "body":     "",   # RSS에는 본문 없음
        })
    time.sleep(delay)
    return articles


def deduplicate(articles: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for a in articles:
        key = (a["date"], a["headline"])
        if key not in seen:
            seen.add(key)
            result.append(a)
    return result


def main(out_path: Path) -> None:
    all_articles: list[dict] = []

    for event_type, queries in QUERIES.items():
        for q in queries:
            fetched = collect_query(q)
            print(f"  [{event_type}] '{q}' → {len(fetched)}건")
            all_articles.extend(fetched)

    articles = deduplicate(all_articles)
    articles.sort(key=lambda x: x["date"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"\n수집 완료: {len(articles)}건 → {out_path}")

    # 월별 분포 출력
    df = pd.DataFrame(articles)
    if not df.empty:
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
        print(df.groupby("month").size().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="data/raw/news_raw/articles.json",
        help="출력 JSON 경로",
    )
    args = parser.parse_args()
    main(Path(args.out))

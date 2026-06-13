"""뉴스 기사 → Claude claude-haiku-4-5 → 이벤트 플래그 생성.

사용법:
    python scripts/classify_events.py
    python scripts/classify_events.py --articles data/raw/news_raw/articles.json \\
                                      --out data/processed/event_flags.csv
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 ANTHROPIC_API_KEY 로드

from src.agent.event_classifier import classify_batch, make_event_flags, save_event_flags


def main(articles_path: Path, out_path: Path, dry_run: bool = False) -> None:
    with open(articles_path, encoding="utf-8") as f:
        articles = json.load(f)

    print(f"분류 대상 기사: {len(articles)}건")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("[오류] ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        print("  방법 1: .env 파일에 ANTHROPIC_API_KEY=sk-ant-... 입력")
        print("  방법 2: export ANTHROPIC_API_KEY=sk-ant-...")
        return

    if dry_run:
        articles = articles[:10]
        print("[dry-run] 처음 10건만 분류합니다.")

    results = classify_batch(articles, api_key=api_key, rate_limit_delay=0.3)

    # 이벤트 분포 출력
    from collections import Counter
    counts = Counter(r.event_type for r in results)
    print("\n이벤트 분류 결과:")
    for k, v in counts.most_common():
        print(f"  {k}: {v}건 ({v/len(results)*100:.1f}%)")

    # 연구 기간 date_range
    date_range = pd.date_range("2024-06-20", "2026-05-19", freq="D")
    df_flags = make_event_flags(results, date_range)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_event_flags(df_flags, out_path)

    # 분류 상세 결과 저장 (검증용)
    detail_path = out_path.parent / "event_classifications.json"
    detail = [
        {
            "date":       r.article_date,
            "headline":   r.headline,
            "event_type": r.event_type,
            "reason":     r.reason,
        }
        for r in results
    ]
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)
    print(f"상세 분류 결과 저장: {detail_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--articles",
        default="data/raw/news_raw/articles.json",
        help="수집된 뉴스 JSON 경로",
    )
    parser.add_argument(
        "--out",
        default="data/processed/event_flags.csv",
        help="이벤트 플래그 CSV 출력 경로",
    )
    parser.add_argument("--dry-run", action="store_true", help="처음 10건만 테스트 분류")
    args = parser.parse_args()
    main(Path(args.articles), Path(args.out), dry_run=args.dry_run)

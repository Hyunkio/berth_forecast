"""LLM Agent 이벤트 분류 파이프라인.

뉴스 기사 텍스트 → Claude claude-haiku-4-5 → 이벤트 유형 분류 (4 클래스)
→ 일별 이벤트 플래그 시계열 생성
"""
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic
import pandas as pd

# 이벤트 분류 클래스
EVENT_CLASSES = {
    "strike":   "파업/운영중단 (화물연대, 항만 노조, 운영 중단)",
    "weather":  "기상악화 (태풍, 폭풍, 안개, 해상 경보)",
    "surge":    "물동량급증 (명절 수요, 연말 수요, 글로벌 공급망 충격)",
    "normal":   "정상 (위 3가지에 해당 없음)",
}

SYSTEM_PROMPT = """당신은 항만 물류 전문가입니다. 뉴스 헤드라인이 국내 주요 항만(부산·울산·인천·광양)의
체선(선박 대기·혼잡)에 직접·간접적으로 영향을 미칠 수 있는 이벤트인지 분류합니다.

분류 기준 (넓게 해석하세요):

strike — 파업·운영중단이 항만·공급망에 영향
  - 화물연대·항만 노조 파업 (예고·진행·타결 모두 포함)
  - 항만·터미널 운영 중단, 점거 농성
  - 미국·유럽 등 해외 주요 항만 파업 (국내 수출입 차질 유발)
  - "파업 우려", "파업 가능성", "협상 결렬" 포함

weather — 기상악화로 입항 통제·해운 지연 발생 또는 우려
  - 태풍·열대성 저기압 접근 또는 상륙
  - 강풍·높은 파도·폭풍으로 입출항 통제·지연
  - 안개·황사로 항만 가시거리 저하, 선박 대피
  - "기상 특보", "태풍 대비", "해상 경보" 포함

surge — 물동량 급증·항만 혼잡·운임 급등이 국내 항만에 영향
  - 홍해 사태·수에즈 운하 우회 등 글로벌 공급망 차질
  - 국내·외 항만 적체·혼잡·대기 선박 증가
  - 컨테이너 부족·해상 운임 급등 (SCFI·CCFI 급상승)
  - 명절·연말·특수 시즌 물동량 급증
  - "물류 대란", "공급망 충격", "항만 혼잡", "운임 상승" 포함

normal — 위 3가지와 직접 연관 없는 기사
  - 항만 인프라 투자·개발, 일반 해운 정책, 환경 규제
  - 항만 통계 발표, 인사 이동, 홍보성 기사

핵심 판단 기준:
"이 기사가 국내 항만 선박 대기 시간 증가로 이어질 가능성이 조금이라도 있는가?"
→ 가능성이 있으면 해당 이벤트로 분류 (normal 대신)

반드시 다음 JSON 형식으로만 응답하세요:
{"event_type": "<strike|weather|surge|normal>", "reason": "<한 문장 근거>"}"""


@dataclass
class ClassificationResult:
    article_date: str
    headline: str
    event_type: str
    reason: str
    raw_response: str


def classify_article(
    client: anthropic.Anthropic,
    headline: str,
    body: str = "",
    model: str = "claude-haiku-4-5-20251001",
) -> ClassificationResult:
    """단일 뉴스 기사 분류."""
    text = headline if not body else f"제목: {headline}\n\n본문: {body[:500]}"

    message = client.messages.create(
        model=model,
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    raw = message.content[0].text.strip()
    # 마크다운 코드 펜스 제거 (```json ... ``` 형태)
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:])
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
        event_type = parsed.get("event_type", "normal")
        reason     = parsed.get("reason", "")
    except json.JSONDecodeError:
        event_type = "normal"
        reason     = f"파싱 실패: {raw[:80]}"

    if event_type not in EVENT_CLASSES:
        event_type = "normal"

    return ClassificationResult(
        article_date="",
        headline=headline,
        event_type=event_type,
        reason=reason,
        raw_response=raw,
    )


def classify_batch(
    articles: list[dict],
    api_key: str | None = None,
    rate_limit_delay: float = 0.5,
    model: str = "claude-haiku-4-5-20251001",
) -> list[ClassificationResult]:
    """뉴스 기사 리스트 일괄 분류.

    articles: [{"date": "2024-07-01", "headline": "...", "body": "..."}]
    """
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    results = []

    for i, article in enumerate(articles):
        result = classify_article(
            client,
            headline=article.get("headline", ""),
            body=article.get("body", ""),
            model=model,
        )
        result.article_date = article.get("date", "")
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  분류 중: {i+1}/{len(articles)}")

        time.sleep(rate_limit_delay)

    return results


def make_event_flags(
    results: list[ClassificationResult],
    date_range: pd.DatetimeIndex,
) -> pd.DataFrame:
    """분류 결과 → 일별 이벤트 플래그 DataFrame.

    반환: date 인덱스, 컬럼 = [strike, weather, surge, any_event]
    """
    df = pd.DataFrame(index=date_range)
    df.index.name = "date"
    df["strike"]   = 0
    df["weather"]  = 0
    df["surge"]    = 0

    for r in results:
        try:
            dt = pd.Timestamp(r.article_date)
        except Exception:
            continue
        if dt not in df.index:
            continue
        if r.event_type in ("strike", "weather", "surge"):
            df.loc[dt, r.event_type] = 1

    df["any_event"] = (df[["strike", "weather", "surge"]].sum(axis=1) > 0).astype(int)
    return df


def save_event_flags(df_flags: pd.DataFrame, out_path: Path) -> None:
    df_flags.reset_index().to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"이벤트 플래그 저장: {out_path}")
    print(f"  기간: {df_flags.index.min()} ~ {df_flags.index.max()}")
    for col in ("strike", "weather", "surge", "any_event"):
        print(f"  {col}: {df_flags[col].sum()}일")

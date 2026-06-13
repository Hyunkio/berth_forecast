# 국내 주요 항만 체선·혼잡 예측 시스템

해양수산부 공공데이터를 기반으로 부산·울산·인천·광양 4개 항만의 체선율을 7일 앞까지 예측하고, 선박 최적 입항 시기를 추천하는 딥러닝 시스템입니다.

---

## 주요 특징

- **타겟 변수 직접 생성** — 외부 체선 통계 없이 입출항 데이터의 체류시간·지연시간으로 체선율을 직접 산출
- **LSTM + LSTM 앙상블** — LSTM-A(hidden=128) × 0.20 + LSTM-B(hidden=64) × 0.80, Test RMSE **0.0462**
- **연쇄 혼잡 분석** — CCF(교차상관함수)로 항만 간 혼잡 전이 시간 정량화, 피처로 반영
- **LLM Agent 파이프라인** — Claude Haiku로 항만 뉴스 이벤트 자동 분류(파업/기상/물동량급증/정상) → Transformer 입력 피처
- **Flask 실시간 대시보드** — 7일 예측, SHAP 피처 중요도, 모델 비교, 케이스 스터디

---

## 모델 성능 비교

| 모델 | 피처 수 | Test RMSE | Test MAE |
|------|--------|-----------|----------|
| Transformer | 79 | 0.0531 | 0.0440 |
| Transformer + LLM Event | 103 | 0.0589 | 0.0478 |
| **LSTM + LSTM 앙상블** | **84** | **0.0462** | **0.0389** |

### 항만별 RMSE (LSTM+LSTM 앙상블)

| 부산 | 울산 | 인천 | 광양 |
|------|------|------|------|
| 0.0312 | 0.0503 | 0.0434 | 0.0561 |

---

## 프로젝트 구조

```
berth_forecast/
├── src/
│   ├── data/           # 데이터 로드·전처리
│   ├── features/       # 피처 엔지니어링 (연쇄 혼잡 lag 포함)
│   ├── models/         # LSTM, Transformer 모델 정의
│   ├── agent/          # LLM Agent 이벤트 분류 파이프라인
│   └── utils/          # 평가 지표, 시각화
├── dashboard/
│   ├── flask_app.py    # Flask 대시보드 서버
│   ├── templates/      # HTML 템플릿
│   └── static/         # CSS, JS, 폰트
├── scripts/
│   ├── train.py        # 모델 학습
│   ├── shap_analysis.py
│   ├── classify_events.py
│   ├── collect_news.py
│   ├── collect_weather.py
│   └── fetch_recent.py # 최신 데이터 갱신
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
└── data/
    ├── raw/            # 원본 xlsx (gitignore)
    ├── processed/      # 집계 CSV, 메타 JSON
    └── external/       # 기상 데이터
```

---

## 데이터

- **해양수산부 선박입출항현황** (공공데이터포털, 파일번호 15083024)
  - 기간: 2024.06 ~ 2026.05 (약 75만 건)
  - 항만: 부산·울산·인천·광양
- **기상청 해상 기상 API**: 강수량·풍속·습도 등 일별 기상 피처
- **항만 뉴스**: 네이버 뉴스 크롤링 → LLM Agent 이벤트 분류

---

## 피처 엔지니어링 (84개)

| 카테고리 | 피처 예시 | 수 |
|---------|---------|---|
| 항만 기본 | 체선율, 입항수, 평균체류시간, 선박유형비율 | 28 |
| 래그·이동평균 | lag1~28, MA7/14/30 | 32 |
| 연쇄 혼잡 | 부산→울산 lag2, 부산→광양 lag1 등 | 4 |
| 기상 | 강수량, 최대풍속, 습도 (항만별) | 20 |

---

## 설치 및 실행

### 환경 설정

```bash
conda create -n berth_forecast python=3.10
conda activate berth_forecast
pip install -r requirements.txt

cp .env.example .env
# .env에 API 키 입력
```

### 데이터 준비

원본 xlsx 파일을 `data/raw/`에 배치 후:

```bash
python src/data/loader.py
python src/data/preprocess.py
python src/features/engineer.py
```

### 모델 학습

```bash
python scripts/train.py
```

### 대시보드 실행

```bash
python dashboard/flask_app.py
# http://localhost:5001
```

---

## 환경 변수

`.env.example`을 복사해 `.env` 생성 후 입력:

```
ANTHROPIC_API_KEY=your_key_here   # LLM Agent 이벤트 분류용
WEATHER_API_KEY=your_key_here     # 기상청 API
```

---

## 기술 스택

`Python 3.10` `PyTorch` `Claude Haiku (LLM Agent)` `Flask` `Chart.js` `pandas` `scikit-learn` `SHAP` `statsmodels`

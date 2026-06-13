"""항만 체선 예측 Flask 대시보드.

실행:
    python dashboard/flask_app.py
    http://localhost:5000
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.lstm import build_lstm
from src.models.transformer import build_transformer
import torch.nn as nn


class _LSTMFc(nn.Module):
    """LSTM-A / LSTM-B: fc head (run_reduced_lstm.py 학습 포맷)."""
    def __init__(self, input_size, hidden=128, layers=2, horizon=7, n_ports=4, dropout=0.2):
        super().__init__()
        self.horizon = horizon
        self.n_ports = n_ports
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, horizon * n_ports)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).view(-1, self.horizon, self.n_ports)
from src.utils.metrics import port_metrics, regression_metrics, smape

app = Flask(__name__)

PROCESSED  = Path("data/processed")
PORT_NAMES = ["부산", "울산", "인천", "광양"]
PRED_HORIZON = 7

EVENT_DATES_BY_TYPE = {
    "surge":  ["2026-03-06", "2026-03-15", "2026-03-17"],
    "strike": ["2024-10-01", "2026-04-30"],
}


# ── 모델 로드 (앱 시작 시 1회) ─────────────────────────────────────────────────
def _load_transformer(n_features: int, pt_name: str):
    m = build_transformer(
        n_features=n_features, n_ports=4, pred_horizon=PRED_HORIZON,
        d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128, dropout=0.1,
    )
    m.load_state_dict(torch.load(PROCESSED / "models" / pt_name, map_location="cpu"))
    m.eval()
    return m


def _load_lstm(n_features: int, pt_name: str):
    m = build_lstm(n_features=n_features, n_ports=4, pred_horizon=PRED_HORIZON)
    m.load_state_dict(torch.load(PROCESSED / "models" / pt_name, map_location="cpu"))
    m.eval()
    return m


with open(PROCESSED / "scaler_y.pkl", "rb") as f:
    SCALER_Y = pickle.load(f)
with open(PROCESSED / "scaler_y_ev.pkl", "rb") as f:
    SCALER_Y_EV = pickle.load(f)

# Transformer: 79 피처
# LSTM+LSTM 앙상블: LSTM-A(84f, hidden=128, 2layer) * 0.20 + LSTM-B(84f, hidden=64, 1layer) * 0.80
# Test RMSE 0.0462 (기존 LSTM+TF 0.0491 대비 6.7% 개선)
_TF     = _load_transformer(79, "transformer_best.pt")

def _load_lstm_fc(input_size: int, hidden: int, layers: int, pt_name: str) -> _LSTMFc:
    m = _LSTMFc(input_size=input_size, hidden=hidden, layers=layers)
    m.load_state_dict(torch.load(PROCESSED / "models" / pt_name, map_location="cpu"))
    m.eval()
    return m

_LSTM_A = _load_lstm_fc(84, 128, 2, "lstm_A84_best.pt")
_LSTM_B = _load_lstm_fc(84, 64,  1, "lstm_B84_best.pt")
ENSEMBLE_W = {"lstm_a": 0.20, "lstm_b": 0.80}

MODELS = {
    "transformer":       (_TF,   SCALER_Y),
    "transformer_event": (_load_transformer(103, "transformer_event_best.pt"), SCALER_Y_EV),
    "ensemble":          (None,  SCALER_Y),   # LSTM-A(84f)*0.20 + LSTM-B(84f)*0.80
}

# TF는 X_tf_test (79 피처), 앙상블은 X84 (84 피처)
X_TF_TEST = np.load(PROCESSED / "X_tf_test.npy")    # 79 features (TF용)
X_EV_TEST = np.load(PROCESSED / "X_ev_test.npy")    # 103 features (TF+event용)
X84_TEST  = np.load(PROCESSED / "X84_test.npy")     # 84 features (LSTM+LSTM 앙상블용)
Y_TEST    = np.load(PROCESSED / "y_test.npy")
Y_EV_TEST = np.load(PROCESSED / "y_ev_test.npy")

# X_latest.npy 가 있으면 실시간 예측에 사용
_X84_LATEST_PATH  = PROCESSED / "X84_latest.npy"
_X_EV_LATEST_PATH = PROCESSED / "X_ev_latest.npy"
_X_TF_LATEST_PATH = PROCESSED / "X_tf_latest.npy"
X84_LATEST  = np.load(_X84_LATEST_PATH)   if _X84_LATEST_PATH.exists()  else X84_TEST
X_TF_LATEST = np.load(_X_TF_LATEST_PATH)  if _X_TF_LATEST_PATH.exists() else X_TF_TEST
X_EV_LATEST = np.load(_X_EV_LATEST_PATH)  if _X_EV_LATEST_PATH.exists() else X_EV_TEST
IS_LIVE     = _X84_LATEST_PATH.exists()

with open(PROCESSED / "dataset_meta.json") as f:
    META = json.load(f)

import pandas as _pd
_daily_tmp = _pd.read_csv(PROCESSED / "daily_aggregated.csv", parse_dates=["date"], encoding="utf-8-sig")
_real_last_date = _daily_tmp["date"].max()
del _pd, _daily_tmp
# X_latest.npy 기준일이 있으면 그것을 우선 사용
_latest_date_path = PROCESSED / "latest_date.txt"
if IS_LIVE and _latest_date_path.exists():
    import pandas as _pd2
    DATA_LAST_DATE = _pd2.Timestamp(_latest_date_path.read_text().strip())
    del _pd2
else:
    DATA_LAST_DATE = _real_last_date


def _predict(model_key: str) -> np.ndarray:
    """최신 입력 윈도우로 7일 예측 (원래 스케일). X_latest.npy 있으면 우선 사용."""
    if model_key == "ensemble":
        x84 = torch.from_numpy(X84_LATEST[-1:]).float()   # 84 피처
        with torch.no_grad():
            pred_a = _LSTM_A(x84).numpy()[0]
            pred_b = _LSTM_B(x84).numpy()[0]
        w_a, w_b = ENSEMBLE_W["lstm_a"], ENSEMBLE_W["lstm_b"]
        pred_avg = w_a * pred_a + w_b * pred_b            # (7, 4)
        sh = pred_avg.shape
        return np.clip(SCALER_Y.inverse_transform(pred_avg.reshape(-1, 4)).reshape(sh), 0, 1)

    model, scaler_y = MODELS[model_key]
    if model_key == "transformer_event":
        X = X_EV_LATEST
    elif model_key == "transformer":
        X = X_TF_LATEST
    else:
        X = X_LATEST
    x = torch.from_numpy(X[-1:]).float()
    with torch.no_grad():
        pred = model(x).numpy()[0]   # (7, 4)
    sh = pred.shape
    return np.clip(scaler_y.inverse_transform(pred.reshape(-1, 4)).reshape(sh), 0, 1)


def _eval_metrics() -> dict:
    results = {}
    for key, (model, scaler_y) in MODELS.items():
        if key == "ensemble":
            with torch.no_grad():
                x84 = torch.from_numpy(X84_TEST).float()
                p_a = _LSTM_A(x84).numpy()
                p_b = _LSTM_B(x84).numpy()
            w_a, w_b = ENSEMBLE_W["lstm_a"], ENSEMBLE_W["lstm_b"]
            preds = w_a * p_a + w_b * p_b
            Y, sy = Y_TEST, SCALER_Y
        elif key == "transformer":
            Y, sy = Y_TEST, scaler_y
            with torch.no_grad():
                preds = model(torch.from_numpy(X_TF_TEST).float()).numpy()
        elif key == "transformer_event":
            Y, sy = Y_EV_TEST, scaler_y
            with torch.no_grad():
                preds = model(torch.from_numpy(X_EV_TEST).float()).numpy()
        else:
            Y, sy = Y_TEST, scaler_y
            with torch.no_grad():
                preds = model(torch.from_numpy(X_TEST).float()).numpy()

        sh = preds.shape
        y_pred = np.clip(sy.inverse_transform(preds.reshape(-1, 4)).reshape(sh), 0, 1)
        sh_y   = Y.shape
        y_true = np.clip(sy.inverse_transform(Y.reshape(-1, 4)).reshape(sh_y), 0, 1)
        m = regression_metrics(y_true, y_pred)
        m["port"] = port_metrics(y_true, y_pred, PORT_NAMES)
        results[key] = m
    return results


METRICS = _eval_metrics()


def _precompute_ccf() -> dict:
    """항만 간 CCF 계산 (startup 1회). daily_aggregated에서 실제 데이터만 사용."""
    import pandas as _pd3
    from statsmodels.tsa.stattools import ccf as _ccf
    daily = _pd3.read_csv(PROCESSED / "daily_aggregated.csv",
                          parse_dates=["date"], encoding="utf-8-sig")
    pv = daily.pivot_table(index="date", columns="항명", values="체선율")
    pv = pv.sort_index().ffill().fillna(0)
    pairs = [("부산", "울산"), ("부산", "광양"), ("부산", "인천"), ("울산", "광양")]
    result = {}
    for a, b in pairs:
        if a not in pv.columns or b not in pv.columns:
            continue
        corr = _ccf(pv[a].values, pv[b].values, nlags=14, unbiased=False)
        lag_corrs = corr[1:15].tolist()
        best_lag  = int(np.argmax(np.abs(lag_corrs))) + 1
        best_corr = float(lag_corrs[best_lag - 1])
        result[f"{a}→{b}"] = {
            "lag":   best_lag,
            "corr":  round(best_corr, 4),
            "lags":  list(range(1, 15)),
            "corrs": [round(v, 4) for v in lag_corrs],
        }
    del _pd3
    return result


CCF_CACHE = _precompute_ccf()


def _precompute_backtest() -> dict:
    """테스트셋 윈도우별 월간 RMSE 계산 (startup 1회)."""
    import pandas as pd4

    daily = pd4.read_csv(PROCESSED / "daily_aggregated.csv",
                         parse_dates=["date"], encoding="utf-8-sig")
    w_in           = META.get("input_window", 30)
    n_test         = META.get("n_test", 0)
    test_start_idx = META.get("n_train", 0) + META.get("n_val", 0)

    # df_feat starts ~14 days after the first daily date (lag14 dropna)
    feat_start = daily["date"].min() + pd4.Timedelta(days=14)
    test_dates = [
        feat_start + pd4.Timedelta(days=w_in + test_start_idx + i)
        for i in range(n_test)
    ]

    # 각 모델별 window-by-window 예측
    results = {}
    for key in ["transformer", "transformer_event", "ensemble"]:
        model, scaler_y = MODELS[key] if MODELS[key][0] is not None else (None, SCALER_Y)
        if key == "ensemble":
            w_a, w_b = ENSEMBLE_W["lstm_a"], ENSEMBLE_W["lstm_b"]
            with torch.no_grad():
                x84 = torch.from_numpy(X84_TEST).float()
                preds = w_a * _LSTM_A(x84).numpy() + w_b * _LSTM_B(x84).numpy()
            y_raw = Y_TEST
            sy    = SCALER_Y
        elif key == "transformer":
            Y = Y_TEST; sy = MODELS[key][1]
            with torch.no_grad():
                preds = MODELS[key][0](torch.from_numpy(X_TF_TEST).float()).numpy()
            y_raw = Y
        else:
            X = X_EV_TEST if key == "transformer_event" else X_TEST
            Y = Y_EV_TEST if key == "transformer_event" else Y_TEST
            sy = MODELS[key][1]
            with torch.no_grad():
                preds = MODELS[key][0](torch.from_numpy(X).float()).numpy()
            y_raw = Y

        sh = preds.shape
        y_pred = np.clip(sy.inverse_transform(preds.reshape(-1, 4)).reshape(sh), 0, 1)
        sh_y   = y_raw.shape
        y_true = np.clip(sy.inverse_transform(y_raw.reshape(-1, 4)).reshape(sh_y), 0, 1)

        n = min(len(test_dates), len(y_pred))
        monthly: dict = {}
        for i in range(n):
            dt = test_dates[i]
            month_key = dt.strftime("%Y-%m")
            err = float(np.sqrt(np.mean((y_true[i] - y_pred[i]) ** 2)))
            monthly.setdefault(month_key, []).append(err)

        results[key] = {
            m: round(float(np.mean(errs)), 5) for m, errs in sorted(monthly.items())
        }

    return results


BACKTEST_CACHE = _precompute_backtest()


def _precompute_event_analysis() -> dict:
    """이벤트 기간 vs 평시 조건부 RMSE + 파업 케이스 스터디 (startup 1회)."""
    import pandas as pd6

    daily = pd6.read_csv(PROCESSED / "daily_aggregated.csv",
                         parse_dates=["date"], encoding="utf-8-sig")
    w_in           = META.get("input_window", 30)
    n_test         = META.get("n_test", 0)
    test_start_idx = META.get("n_train", 0) + META.get("n_val", 0)
    feat_start     = daily["date"].min() + pd6.Timedelta(days=14)

    test_window_dates = [
        feat_start + pd6.Timedelta(days=w_in + test_start_idx + i)
        for i in range(n_test)
    ]

    all_event_dts: dict = {}
    for etype, dates in EVENT_DATES_BY_TYPE.items():
        for d in dates:
            all_event_dts[pd6.Timestamp(d)] = etype

    # 예측 범위 내에 이벤트 날짜가 포함되는 윈도우 마스크
    event_mask = np.zeros(n_test, dtype=bool)
    for i, win_date in enumerate(test_window_dates):
        for day_offset in range(1, PRED_HORIZON + 1):
            if win_date + pd6.Timedelta(days=day_offset) in all_event_dts:
                event_mask[i] = True
                break

    n_event  = int(event_mask.sum())
    n_normal = n_test - n_event

    # 모든 forward pass를 한 번에
    with torch.no_grad():
        p_tf   = _TF(torch.from_numpy(X_TF_TEST).float()).numpy()
        x84    = torch.from_numpy(X84_TEST).float()
        p_lstm_a = _LSTM_A(x84).numpy()
        p_lstm_b = _LSTM_B(x84).numpy()
        p_tf_ev = MODELS["transformer_event"][0](torch.from_numpy(X_EV_TEST).float()).numpy()
    w_a, w_b = ENSEMBLE_W["lstm_a"], ENSEMBLE_W["lstm_b"]
    ens_raw = w_a * p_lstm_a + w_b * p_lstm_b

    preds_map = {
        "transformer":       (p_tf,    Y_TEST,    SCALER_Y),
        "transformer_event": (p_tf_ev, Y_EV_TEST, MODELS["transformer_event"][1]),
        "ensemble":          (ens_raw, Y_TEST,    SCALER_Y),
    }

    def _inv(preds, y_raw, sy):
        sh   = preds.shape
        yp   = np.clip(sy.inverse_transform(preds.reshape(-1, 4)).reshape(sh), 0, 1)
        sh_y = y_raw.shape
        yt   = np.clip(sy.inverse_transform(y_raw.reshape(-1, 4)).reshape(sh_y), 0, 1)
        return yp, yt

    cond_rmse: dict = {}
    ens_pred_inv = ens_true_inv = None
    for key, (preds, y_raw, sy) in preds_map.items():
        yp, yt = _inv(preds, y_raw, sy)
        if key == "ensemble":
            ens_pred_inv, ens_true_inv = yp, yt

        n      = min(n_test, len(yp))
        mask_n = event_mask[:n]
        rmse_ev = float(np.sqrt(np.mean((yt[:n][mask_n] - yp[:n][mask_n]) ** 2))) if mask_n.any() else 0.0
        rmse_nm = float(np.sqrt(np.mean((yt[:n][~mask_n] - yp[:n][~mask_n]) ** 2))) if (~mask_n).any() else 0.0
        cond_rmse[key] = {
            "event":    round(rmse_ev, 5),
            "normal":   round(rmse_nm, 5),
            "diff_pct": round((rmse_ev - rmse_nm) / rmse_nm * 100, 1) if rmse_nm > 0 else 0.0,
        }

    # 케이스 스터디: 파업 주변 -7 ~ +3일 앙상블 예측 (부산항)
    strike_dt = pd6.Timestamp("2026-04-30")
    cs_by_date: dict = {}
    for i, win_date in enumerate(test_window_dates[:n_test]):
        if i >= len(ens_pred_inv):
            break
        for day_idx in range(PRED_HORIZON):
            pred_dt = win_date + pd6.Timedelta(days=day_idx + 1)
            delta   = (pred_dt - strike_dt).days
            if -7 <= delta <= 3:
                key_dt    = pred_dt.strftime("%Y-%m-%d")
                day_ahead = day_idx + 1
                if key_dt not in cs_by_date or day_ahead < cs_by_date[key_dt]["day_ahead"]:
                    cs_by_date[key_dt] = {
                        "pred_date":     key_dt,
                        "day_ahead":     day_ahead,
                        "event_type":    all_event_dts.get(pred_dt),
                        "actual_부산":   round(float(ens_true_inv[i, day_idx, 0]), 4),
                        "pred_ensemble": round(float(ens_pred_inv[i, day_idx, 0]), 4),
                    }

    case_study = sorted(cs_by_date.values(), key=lambda x: x["pred_date"])
    for c in case_study:
        c["error"] = round(abs(c["actual_부산"] - c["pred_ensemble"]), 4)
        c.pop("day_ahead")

    event_markers = sorted(
        [{"date": d, "type": et,
          "label": "물동량급증" if et == "surge" else "파업"}
         for et, dates in EVENT_DATES_BY_TYPE.items() for d in dates],
        key=lambda x: x["date"],
    )

    del pd6
    return {
        "conditional_rmse": cond_rmse,
        "n_event":          n_event,
        "n_normal":         n_normal,
        "event_markers":    event_markers,
        "case_study":       case_study,
    }


EVENT_ANALYSIS_CACHE = _precompute_event_analysis()

# 추천 임계값 (실제 데이터 범위 0~0.3 기준)
RISK_HIGH   = 0.12
RISK_MEDIUM = 0.06


def _recommend(pred: np.ndarray) -> dict:
    """7일 예측 기반 항만별 입항 추천. pred: (7, 4)"""
    result: dict = {"ports": {}, "best": None}
    global_min = float("inf")
    best_port, best_day = None, None

    for i, port in enumerate(PORT_NAMES):
        series   = pred[:, i]
        min_day  = int(series.argmin())
        min_val  = float(series[min_day])
        max_val  = float(series.max())
        avg_val  = float(series.mean())

        week_risk = "혼잡" if max_val >= RISK_HIGH else ("보통" if max_val >= RISK_MEDIUM else "원활")

        if min_val >= RISK_HIGH:
            recommendation = "이번 주 입항 자제"
            rec_level      = "avoid"
        elif min_val >= RISK_MEDIUM:
            recommendation = f"D+{min_day + 1}일 입항 가능 (주의)"
            rec_level      = "caution"
        else:
            recommendation = f"D+{min_day + 1}일 입항 권장"
            rec_level      = "recommended"

        result["ports"][port] = {
            "week_risk":      week_risk,
            "best_day":       min_day + 1,
            "best_rate":      round(min_val, 4),
            "avg_rate":       round(avg_val, 4),
            "recommendation": recommendation,
            "rec_level":      rec_level,
        }

        if min_val < global_min:
            global_min = min_val
            best_port  = port
            best_day   = min_day + 1

    result["best"] = {
        "port":    best_port,
        "day":     best_day,
        "rate":    round(global_min, 4),
        "message": f"{best_port}항 D+{best_day}일 최적 입항 (예측 체선율 {global_min*100:.1f}%)",
    }
    return result


def _generate_summary(pred: np.ndarray) -> str:
    """Claude claude-haiku-4-5로 7일 예측 자연어 브리핑 생성."""
    import anthropic
    from datetime import datetime, timedelta

    rows = []
    for d in range(PRED_HORIZON):
        date_str = (DATA_LAST_DATE + timedelta(days=d + 1)).strftime("%m/%d")
        vals = ", ".join(f"{PORT_NAMES[i]}={pred[d, i]*100:.1f}%" for i in range(4))
        rows.append(f"D+{d+1}({date_str}): {vals}")

    rec  = _recommend(pred)
    best = rec["best"]

    prompt = f"""국내 주요 항만 체선율 7일 예측 데이터입니다.
체선율이 높을수록 선박 대기 위험이 크며, 12% 이상이면 혼잡으로 판단합니다.

[예측 데이터]
{chr(10).join(rows)}

[자동 추천 결과]
최저 혼잡: {best['port']}항 D+{best['day']}일 ({best['rate']*100:.1f}%)

위 데이터를 바탕으로 선박 운항사·화물주를 위한 입항 전략 브리핑을 3~4문장으로 작성하세요.
- 이번 주 주요 혼잡 항만·시기 경고 (해당 없으면 "전반적으로 원활" 명시)
- 최적 입항 추천 이유
- 실무 주의사항 한 줄
반드시 한국어, 전문가 어조로 작성. 마크다운(#, **, *, - 등) 사용 금지."""

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


# ── 라우트 ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict/<model_key>")
def api_predict(model_key: str):
    from datetime import timedelta
    if model_key not in MODELS:
        return jsonify({"error": "unknown model"}), 400
    pred = _predict(model_key)   # (7, 4)
    base = DATA_LAST_DATE
    days = [(base + timedelta(days=i+1)).strftime("%m/%d") for i in range(PRED_HORIZON)]
    result = {
        "days":  days,
        "ports": {
            p: {
                "values":   [round(float(pred[d, i]), 4) for d in range(PRED_HORIZON)],
                "max":      round(float(pred[:, i].max()), 4),
                "peak_day": int(pred[:, i].argmax()) + 1,
                "risk":     "high" if pred[:, i].max() >= 0.5 else
                            "medium" if pred[:, i].max() >= 0.3 else "low",
            }
            for i, p in enumerate(PORT_NAMES)
        },
    }
    return jsonify(result)


@app.route("/api/recommend/<model_key>")
def api_recommend(model_key: str):
    from datetime import timedelta
    if model_key not in MODELS:
        return jsonify({"error": "unknown model"}), 400
    pred = _predict(model_key)   # (7, 4)
    rec  = _recommend(pred)
    days = [(DATA_LAST_DATE + timedelta(days=i+1)).strftime("%m/%d(%a)")
            .replace("Mon","월").replace("Tue","화").replace("Wed","수")
            .replace("Thu","목").replace("Fri","금").replace("Sat","토").replace("Sun","일")
            for i in range(PRED_HORIZON)]
    rec["days"] = days
    for idx, port in enumerate(PORT_NAMES):
        rec["ports"][port]["daily_rates"] = [round(float(pred[d, idx]), 4) for d in range(PRED_HORIZON)]
    return jsonify(rec)


@app.route("/api/summary/<model_key>")
def api_summary(model_key: str):
    if model_key not in MODELS:
        return jsonify({"error": "unknown model"}), 400
    try:
        summary = _generate_summary(_predict(model_key))
    except Exception as e:
        summary = f"브리핑 생성 실패: {e}"
    return jsonify({"summary": summary})


@app.route("/api/daily/<port>")
def api_daily(port: str):
    
    import pandas as pd
    path = PROCESSED / "daily_aggregated.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["항명"] == port].sort_values("date").tail(180)
    return jsonify({
        "dates":  df["date"].dt.strftime("%Y-%m-%d").tolist(),
        "rate":   df["체선율"].round(4).tolist(),
        "count":  df["입항수"].tolist(),
    })


@app.route("/api/metrics")
def api_metrics():
    out = {}
    for key, m in METRICS.items():
        out[key] = {
            "MAE":   round(m["MAE"], 4),
            "RMSE":  round(m["RMSE"], 4),
            "MAPE":  round(m["MAPE"], 2),
            "sMAPE": round(m["sMAPE"], 2),
            "port_rmse": {p: round(v, 4) for p, v in m["port"].items()},
        }
    return jsonify(out)


@app.route("/api/ccf")
def api_ccf():
    """항만 간 Cross-Correlation Function 결과 반환 (startup 캐시)."""
    return jsonify(CCF_CACHE)


@app.route("/api/attention")
def api_attention():
    """Transformer 평균 어텐션 가중치 (입력 30일 × 입력 30일) 반환."""
    import torch.nn.functional as F

    x = torch.from_numpy(X_TF_LATEST[-1:]).float()   # (1, 30, 79)

    # 첫 번째 인코더 레이어의 self-attention 출력 후크
    attn_weights = []

    def _hook(module, inp, out):
        # out: (attn_output, attn_weights_avg) if need_weights=True
        if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
            attn_weights.append(out[1].detach())

    hooks = []
    for layer in _TF.encoder.layers:
        hooks.append(layer.self_attn.register_forward_hook(_hook))

    _TF.eval()
    with torch.no_grad():
        _TF(x)

    for h in hooks:
        h.remove()

    if not attn_weights:
        # PyTorch MHA는 need_weights=False가 기본 — softmax 직접 계산
        # 인코더 레이어에 접근하여 Q·K를 추출
        enc_out = _TF.input_proj(x)   # (1, 30, d_model)
        layer = _TF.encoder.layers[0]
        d_model = layer.self_attn.embed_dim
        nhead   = layer.self_attn.num_heads
        head_dim = d_model // nhead

        # in_proj_weight 로 Q, K 투영
        W = layer.self_attn.in_proj_weight   # (3*d_model, d_model)
        b = layer.self_attn.in_proj_bias     # (3*d_model,)
        qkv = F.linear(enc_out, W, b)         # (1, 30, 3*d_model)
        Q, K, _ = qkv.chunk(3, dim=-1)
        Q = Q.view(1, 30, nhead, head_dim).transpose(1, 2)   # (1, nhead, 30, head_dim)
        K = K.view(1, 30, nhead, head_dim).transpose(1, 2)
        scale = head_dim ** -0.5
        scores = (Q @ K.transpose(-2, -1)) * scale            # (1, nhead, 30, 30)
        attn_map = F.softmax(scores, dim=-1).mean(dim=1)      # (1, 30, 30)
        attn_avg = attn_map[0].detach().numpy()
    else:
        attn_avg = torch.stack(attn_weights).mean(dim=0)[0].detach().numpy()  # (30, 30)

    # 각 입력 시점별 평균 어텐션 (쿼리 차원으로 합산 후 평균)
    input_importance = float(attn_avg.mean(axis=0).tolist()[0]) if attn_avg.ndim > 1 else 0
    input_attn = attn_avg.mean(axis=0).tolist() if attn_avg.ndim > 1 else [0.0] * 30

    # 레이블: D-29 … D-0
    labels = [f"D-{29-i}" for i in range(30)]
    return jsonify({"labels": labels, "weights": [round(v, 5) for v in input_attn]})


@app.route("/api/backtest")
def api_backtest():
    """테스트셋 월별 RMSE 반환 (startup 캐시)."""
    return jsonify(BACKTEST_CACHE)


@app.route("/api/shap")
def api_shap():
    """SHAP 피처 중요도 반환 (shap_values.json)."""
    shap_path = PROCESSED / "shap_values.json"
    if not shap_path.exists():
        return jsonify({"error": "shap_values.json 없음 — scripts/shap_analysis.py 실행 필요"}), 404
    with open(shap_path, encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/api/event_period_analysis")
def api_event_period_analysis():
    """이벤트 기간 조건부 RMSE + 케이스 스터디 반환 (startup 캐시)."""
    return jsonify(EVENT_ANALYSIS_CACHE)


@app.route("/api/event_samples")
def api_event_samples():
    """이벤트 분류 샘플 반환 (event_classifications.json)."""
    detail_path = PROCESSED / "event_classifications.json"
    if not detail_path.exists():
        return jsonify({"samples": [], "stats": {}})
    with open(detail_path, encoding="utf-8") as f:
        data = json.load(f)
    from collections import Counter
    counts = Counter(d["event_type"] for d in data)
    # 이벤트 유형별 대표 샘플 반환
    samples = []
    seen_types: set = set()
    for d in reversed(data):   # 최신부터
        if d["event_type"] != "normal" and d["event_type"] not in seen_types:
            samples.append(d)
            seen_types.add(d["event_type"])
    # normal 샘플 2개 추가
    normal_cnt = 0
    for d in data:
        if d["event_type"] == "normal" and normal_cnt < 2:
            samples.append(d)
            normal_cnt += 1
    return jsonify({
        "samples": sorted(samples, key=lambda x: x["date"]),
        "stats": {k: v for k, v in counts.most_common()},
        "total": len(data),
    })


@app.route("/api/classify_headline", methods=["GET"])
def api_classify_headline():
    """단일 헤드라인 → Claude 이벤트 분류 (실시간 데모)."""
    headline = request.args.get("q", "").strip()
    if not headline:
        return jsonify({"error": "headline 파라미터 필요"}), 400
    try:
        import anthropic
        prompt = f"""항만 관련 뉴스 헤드라인을 분석하여 이벤트 유형을 분류하세요.

헤드라인: "{headline}"

다음 중 하나로 분류하고 JSON으로 응답하세요:
- "strike": 항만 파업·운영중단
- "weather": 태풍·기상악화로 인한 항만 운영 차질
- "surge": 물동량 급증·항만 혼잡 심화
- "normal": 해당 없음

반드시 아래 형식으로만 응답 (다른 텍스트 없이):
{{"event_type": "...", "reason": "한 문장 근거"}}"""

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        # strip markdown code fences if present
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()
        parsed = json.loads(raw)
        return jsonify({"headline": headline, **parsed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/data_status")
def api_data_status():
    from datetime import datetime, timedelta
    last_date = DATA_LAST_DATE.date()
    pred_start = last_date + timedelta(days=1)
    pred_end   = last_date + timedelta(days=PRED_HORIZON)
    lag_days   = (datetime.today().date() - last_date).days
    return jsonify({
        "last_data_date": str(last_date),
        "pred_start":     str(pred_start),
        "pred_end":       str(pred_end),
        "lag_days":       lag_days,
        "is_live":        IS_LIVE,
        "source":         "live (X_latest.npy)" if IS_LIVE else "test set tail",
    })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    app.run(debug=False, port=args.port)

"""SHAP 피처 중요도 분석 (LSTM GradientExplainer).

SHAP 실패 시 Permutation Importance로 자동 fallback.

실행:
    python scripts/shap_analysis.py
    python scripts/shap_analysis.py --method permutation   # 강제 permutation

결과:
    data/processed/shap_values.json
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.lstm import build_lstm
from src.utils.metrics import regression_metrics

PROCESSED = Path("data/processed")
PORT_NAMES = ["부산", "울산", "인천", "광양"]


def load_model_and_data():
    with open(PROCESSED / "dataset_meta.json") as f:
        meta = json.load(f)

    feature_cols = meta["feature_cols"]   # 95개
    n_features = len(feature_cols)

    model = build_lstm(n_features=n_features, n_ports=4, pred_horizon=7)
    model.load_state_dict(
        torch.load(PROCESSED / "models" / "lstm_best.pt", map_location="cpu")
    )
    model.eval()

    X_train = np.load(PROCESSED / "X_train.npy")   # (N, 30, 95) LSTM용
    X_test  = np.load(PROCESSED / "X_test.npy")
    y_test  = np.load(PROCESSED / "y_test.npy")

    with open(PROCESSED / "scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)

    return model, X_train, X_test, y_test, scaler_y, feature_cols


# ── SHAP GradientExplainer ─────────────────────────────────────────────────────
def _shap_gradient_for_port(model, bg_data, test_t, port_idx: int):
    """단일 항만(port_idx 0~3)에 대한 SHAP GradientExplainer 실행."""
    import shap

    class PortOutputWrapper(torch.nn.Module):
        def __init__(self, m, pi):
            super().__init__()
            self.m  = m
            self.pi = pi
        def forward(self, x):
            out = self.m(x)                        # (N, 7, 4)
            return out[:, :, self.pi].mean(dim=1, keepdim=True)  # (N, 1)

    wrapper = PortOutputWrapper(model, port_idx)
    wrapper.eval()

    explainer = shap.GradientExplainer(wrapper, bg_data)
    shap_vals = explainer.shap_values(test_t)

    if isinstance(shap_vals, list):
        sv = np.abs(np.stack(shap_vals, axis=0)).mean(axis=0)
    else:
        sv = np.abs(shap_vals)

    return sv.mean(axis=(0, 1))   # (n_features,)


def shap_gradient(model, X_train, X_test, feature_cols, n_bg=80):
    import shap

    print(f"  Background samples: {n_bg} | Test samples: {len(X_test)}")

    rng     = np.random.default_rng(42)
    bg_idx  = rng.choice(len(X_train), size=n_bg, replace=False)
    bg_data = torch.from_numpy(X_train[bg_idx]).float()
    test_t  = torch.from_numpy(X_test).float()

    # 전체 평균 SHAP (기존 동작 유지)
    class MeanOutputWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            out = self.m(x)
            return out.mean(dim=(1, 2), keepdim=False).unsqueeze(1)

    wrapper = MeanOutputWrapper(model)
    wrapper.eval()
    explainer = shap.GradientExplainer(wrapper, bg_data)
    shap_vals = explainer.shap_values(test_t)
    if isinstance(shap_vals, list):
        sv = np.abs(np.stack(shap_vals, axis=0)).mean(axis=0)
    else:
        sv = np.abs(shap_vals)
    importance_all = sv.mean(axis=(0, 1))

    # 항만별 SHAP
    port_importance = {}
    for pi, pname in enumerate(PORT_NAMES):
        print(f"  항만별 SHAP: {pname}...")
        port_importance[pname] = _shap_gradient_for_port(model, bg_data, test_t, pi)

    return importance_all, port_importance


# ── Permutation Importance (fallback) ─────────────────────────────────────────
def permutation_importance(model, X_test, y_test, scaler_y, feature_cols, n_repeats=3):
    print(f"  Permutation importance (repeats={n_repeats})")

    def predict_rmse(X):
        with torch.no_grad():
            preds = model(torch.from_numpy(X).float()).numpy()
        sh = preds.shape
        y_pred = np.clip(scaler_y.inverse_transform(preds.reshape(-1, 4)).reshape(sh), 0, 1)
        sh_y = y_test.shape
        y_true = np.clip(scaler_y.inverse_transform(y_test.reshape(-1, 4)).reshape(sh_y), 0, 1)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    baseline = predict_rmse(X_test)
    print(f"  Baseline RMSE: {baseline:.5f}")

    rng = np.random.default_rng(42)
    importance = np.zeros(len(feature_cols))

    for fi, fname in enumerate(feature_cols):
        deltas = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            # feature fi의 모든 time step을 샘플 간 shuffle
            perm = rng.permutation(len(X_perm))
            X_perm[:, :, fi] = X_perm[perm, :, fi]
            deltas.append(predict_rmse(X_perm) - baseline)
        importance[fi] = float(np.mean(deltas))

        if (fi + 1) % 10 == 0:
            print(f"  진행: {fi+1}/{len(feature_cols)}")

    # 음수(≈ 노이즈) → 0으로 클리핑, 이후 정규화
    importance = np.maximum(importance, 0)
    return importance


# ── 피처 그룹 분류 ─────────────────────────────────────────────────────────────
def classify_group(col: str) -> str:
    ports = ["부산", "울산", "인천", "광양"]
    if col in ("dayofweek", "month", "is_weekend"):
        return "시간"
    for p in ports:
        if col == f"{p}_체선율":
            return "체선율"
        if col in (f"{p}_입항수", f"{p}_평균체류"):
            return "입항통계"
        if any(col == f"{p}_{g}비율" for g in ["컨테이너","유조선","벌크","일반화물"]):
            return "선박유형"
        if col.startswith(f"{p}_lag"):
            return "래그"
        if col.startswith(f"{p}_ma"):
            return "이동평균"
        if any(col == f"{p}_{w}" for w in ["기온","강수량","풍속","최대풍속","습도"]):
            return "기상"
    return "기타"


def _build_records(importance: np.ndarray, feature_cols: list) -> list:
    """importance 배열 → 정렬된 record 리스트."""
    imp_max = importance.max() or 1.0
    records = [
        {
            "feature":        col,
            "importance":     float(importance[i]),
            "importance_norm": float(importance[i] / imp_max),
            "group":          classify_group(col),
            "rank":           0,
        }
        for i, col in enumerate(feature_cols)
    ]
    records.sort(key=lambda x: x["importance"], reverse=True)
    for rank, r in enumerate(records, 1):
        r["rank"] = rank
    return records


def _build_group_summary(records: list) -> list:
    from collections import defaultdict
    group_imp: dict = defaultdict(float)
    for r in records:
        group_imp[r["group"]] += r["importance"]
    total = sum(group_imp.values()) or 1.0
    return sorted(
        [{"group": g, "importance": v, "pct": round(v / total * 100, 1)}
         for g, v in group_imp.items()],
        key=lambda x: x["importance"], reverse=True,
    )


def main(method: str = "auto"):
    model, X_train, X_test, y_test, scaler_y, feature_cols = load_model_and_data()

    print(f"\n=== SHAP 피처 중요도 분석 ===")
    print(f"  피처: {len(feature_cols)}개 | 테스트 샘플: {len(X_test)}개")

    importance_all = None
    port_importance: dict = {}
    used_method = method

    if method in ("auto", "shap"):
        try:
            print("\n[1/2] SHAP GradientExplainer 시도...")
            importance_all, port_importance = shap_gradient(model, X_train, X_test, feature_cols)
            used_method = "shap_gradient"
            print("  SHAP 완료")
        except Exception as e:
            print(f"  SHAP 실패: {e}")
            if method == "shap":
                raise
            print("  → Permutation Importance로 fallback")

    if importance_all is None:
        print("\n[1/2] Permutation Importance...")
        importance_all = permutation_importance(model, X_test, y_test, scaler_y, feature_cols)
        used_method = "permutation"

    # 전체 집계
    records       = _build_records(importance_all, feature_cols)
    group_summary = _build_group_summary(records)

    # 항만별 집계
    port_shap = {}
    for pname, imp in port_importance.items():
        prec = _build_records(imp, feature_cols)
        port_shap[pname] = {
            "top_features":  prec[:15],
            "group_summary": _build_group_summary(prec),
        }

    result = {
        "method":        used_method,
        "top_features":  records[:20],
        "all_features":  records,
        "group_summary": group_summary,
        "port_shap":     port_shap,
    }

    out_path = PROCESSED / "shap_values.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")

    print(f"\nTop 15 피처 (method={used_method}):")
    for r in records[:15]:
        bar = "█" * int(r["importance_norm"] * 20)
        print(f"  {r['rank']:2d}. {r['feature']:<28s} [{r['group']:6s}] {bar}")

    print("\n그룹별 기여도:")
    for g in group_summary:
        print(f"  {g['group']:8s}: {g['pct']:5.1f}%")

    if port_shap:
        print("\n항만별 Top 5 피처:")
        for pname, ps in port_shap.items():
            top5 = [(r["feature"], round(r["importance_norm"], 3)) for r in ps["top_features"][:5]]
            print(f"  {pname}: {top5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["auto", "shap", "permutation"], default="auto")
    args = parser.parse_args()
    main(args.method)

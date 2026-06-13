"""데이터 증강 + 경량화 Transformer 재학습.

증강 전략:
  1. Gaussian jitter  — 각 윈도우에 N(0, σ) 노이즈 추가 (σ=0.01)
  2. Window shift     — ±1~2 스텝 이동으로 새 윈도우 생성
  → 454 → 약 1,800 샘플

Transformer 변경:
  - d_model: 64 → 32
  - num_encoder_layers: 2 → 1
  - dim_feedforward: 128 → 64
  - optimizer: Adam + weight_decay=1e-4
  - loss: MAE (MSE 대신)

실행:
    python scripts/augment_and_train.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, ".")
from src.models.lstm import build_lstm
from src.models.transformer import build_transformer
from src.utils.metrics import port_metrics, regression_metrics

PROCESSED  = Path("data/processed")
MODELS_DIR = PROCESSED / "models"
PORT_NAMES = ["부산", "울산", "인천", "광양"]


# ── 증강 ──────────────────────────────────────────────────────────────────────
def augment(X: np.ndarray, y: np.ndarray, jitter_sigma: float = 0.008,
            n_copies: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """X에만 Gaussian jitter 적용 — y 관계 유지, X-y 관계 보존."""
    X_aug, y_aug = [X], [y]
    for _ in range(n_copies):
        noise = np.random.normal(0, jitter_sigma, X.shape).astype(np.float32)
        X_aug.append(np.clip(X + noise, 0, 1))
        y_aug.append(y)   # y는 원본 그대로
    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)


# ── 학습 루프 ──────────────────────────────────────────────────────────────────
def make_loader(X, y, batch=32, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def train(model, train_loader, val_loader, epochs=200, lr=5e-4,
          weight_decay=1e-4, patience=20, save_path=None, use_mae=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.L1Loss() if use_mae else nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val, no_improve = float("inf"), 0
    for epoch in range(epochs):
        model.train()
        tloss = sum(
            (optimizer.zero_grad() or True) and
            (loss := criterion(model(xb), yb)) and
            loss.backward() or
            (torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)) or
            optimizer.step() or loss.item()
            for xb, yb in train_loader
        ) / len(train_loader)

        model.eval()
        with torch.no_grad():
            vloss = sum(criterion(model(xb), yb).item() for xb, yb in val_loader) / len(val_loader)

        scheduler.step(vloss)
        if vloss < best_val:
            best_val = vloss
            no_improve = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"    Early stop @ epoch {epoch+1}")
            break
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}: train={tloss:.4f}, val={vloss:.4f}")


def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb).numpy())
            trues.append(yb.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def inverse(scaler, y, n_ports=4):
    sh = y.shape
    return np.clip(scaler.inverse_transform(y.reshape(-1, n_ports)).reshape(sh), 0, 1)


def load_scaler(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)

    X_tr_raw = np.load(PROCESSED / "X_train.npy")
    y_tr_raw = np.load(PROCESSED / "y_train.npy")
    X_vl     = np.load(PROCESSED / "X_val.npy")
    y_vl     = np.load(PROCESSED / "y_val.npy")
    X_te     = np.load(PROCESSED / "X_test.npy")
    y_te     = np.load(PROCESSED / "y_test.npy")

    scaler_y = load_scaler(PROCESSED / "scaler_y.pkl")
    n_feats  = X_tr_raw.shape[2]
    n_ports  = y_tr_raw.shape[2]

    print(f"원본 train: {X_tr_raw.shape}")
    X_tr, y_tr = augment(X_tr_raw, y_tr_raw)
    print(f"증강 후:    {X_tr.shape}  ({len(X_tr)/len(X_tr_raw):.1f}x)")

    val_loader  = make_loader(X_vl, y_vl)
    test_loader = make_loader(X_te, y_te)

    # ── 1. LSTM (증강 데이터로 재학습) ──
    print("\n=== LSTM (증강) ===")
    lstm = build_lstm(n_features=n_feats, n_ports=n_ports, pred_horizon=7)
    print(f"  파라미터: {sum(p.numel() for p in lstm.parameters()):,}")
    train(lstm, make_loader(X_tr, y_tr, shuffle=True), val_loader,
          epochs=200, lr=1e-3, weight_decay=1e-4, patience=20,
          save_path=MODELS_DIR / "lstm_best.pt", use_mae=False)
    lstm.load_state_dict(torch.load(MODELS_DIR / "lstm_best.pt", map_location="cpu"))
    yt, yp = evaluate(lstm, test_loader)
    m = regression_metrics(inverse(scaler_y, yt), inverse(scaler_y, yp))
    pm = port_metrics(inverse(scaler_y, yt), inverse(scaler_y, yp), PORT_NAMES)
    print(f"  MAE:{m['MAE']:.4f} RMSE:{m['RMSE']:.4f} sMAPE:{m['sMAPE']:.1f}%")
    print("  항만별 RMSE:", {p: f"{r:.4f}" for p, r in pm.items()})

    # ── 2. Transformer 경량화 (증강 데이터로 재학습) ──
    print("\n=== Transformer 경량화 (d=32, 1 layer, MAE loss) ===")
    tf = build_transformer(n_features=n_feats, n_ports=n_ports, pred_horizon=7,
                           d_model=32, nhead=4, num_encoder_layers=1,
                           dim_feedforward=64, dropout=0.1)
    print(f"  파라미터: {sum(p.numel() for p in tf.parameters()):,}")
    train(tf, make_loader(X_tr, y_tr, shuffle=True), val_loader,
          epochs=200, lr=5e-4, weight_decay=1e-4, patience=20,
          save_path=MODELS_DIR / "transformer_best.pt", use_mae=True)
    tf.load_state_dict(torch.load(MODELS_DIR / "transformer_best.pt", map_location="cpu"))
    yt, yp = evaluate(tf, test_loader)
    m = regression_metrics(inverse(scaler_y, yt), inverse(scaler_y, yp))
    pm = port_metrics(inverse(scaler_y, yt), inverse(scaler_y, yp), PORT_NAMES)
    print(f"  MAE:{m['MAE']:.4f} RMSE:{m['RMSE']:.4f} sMAPE:{m['sMAPE']:.1f}%")
    print("  항만별 RMSE:", {p: f"{r:.4f}" for p, r in pm.items()})

    # ── 3. Transformer+Event 경량화 ──
    print("\n=== Transformer+Event 경량화 ===")
    X_ev_tr_raw = np.load(PROCESSED / "X_ev_train.npy")
    y_ev_tr_raw = np.load(PROCESSED / "y_ev_train.npy")
    X_ev_vl     = np.load(PROCESSED / "X_ev_val.npy")
    y_ev_vl     = np.load(PROCESSED / "y_ev_val.npy")
    X_ev_te     = np.load(PROCESSED / "X_ev_test.npy")
    y_ev_te     = np.load(PROCESSED / "y_ev_test.npy")
    scaler_y_ev = load_scaler(PROCESSED / "scaler_y_ev.pkl")

    X_ev_tr, y_ev_tr = augment(X_ev_tr_raw, y_ev_tr_raw)
    n_ev_feats = X_ev_tr.shape[2]

    tf_ev = build_transformer(n_features=n_ev_feats, n_ports=n_ports, pred_horizon=7,
                              d_model=32, nhead=4, num_encoder_layers=1,
                              dim_feedforward=64, dropout=0.1)
    print(f"  피처:{n_ev_feats} 파라미터:{sum(p.numel() for p in tf_ev.parameters()):,}")
    train(tf_ev, make_loader(X_ev_tr, y_ev_tr, shuffle=True),
          make_loader(X_ev_vl, y_ev_vl),
          epochs=200, lr=5e-4, weight_decay=1e-4, patience=20,
          save_path=MODELS_DIR / "transformer_event_best.pt", use_mae=True)
    tf_ev.load_state_dict(torch.load(MODELS_DIR / "transformer_event_best.pt", map_location="cpu"))
    yt, yp = evaluate(tf_ev, make_loader(X_ev_te, y_ev_te))
    m = regression_metrics(inverse(scaler_y_ev, yt), inverse(scaler_y_ev, yp))
    pm = port_metrics(inverse(scaler_y_ev, yt), inverse(scaler_y_ev, yp), PORT_NAMES)
    print(f"  MAE:{m['MAE']:.4f} RMSE:{m['RMSE']:.4f} sMAPE:{m['sMAPE']:.1f}%")
    print("  항만별 RMSE:", {p: f"{r:.4f}" for p, r in pm.items()})

    print("\n완료. 모델 저장 위치:", MODELS_DIR)


if __name__ == "__main__":
    main()

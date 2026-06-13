"""모델 학습 스크립트 (LSTM, Transformer, Transformer+Event).

사용법:
    python scripts/train.py --model all
    python scripts/train.py --model transformer_event --epochs 200
"""
import argparse
import sys
from pathlib import Path

import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, ".")
from src.models.lstm import build_lstm
from src.models.transformer import build_transformer
from src.utils.metrics import port_metrics, regression_metrics, smape


def _load_scaler(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _inverse(scaler, y_scaled: np.ndarray) -> np.ndarray:
    """(N, 7, 4) scaled → original scale, clipped to [0, 1]."""
    sh = y_scaled.shape
    return np.clip(
        scaler.inverse_transform(y_scaled.reshape(-1, sh[-1])).reshape(sh), 0, 1
    )

PROCESSED = Path("data/processed")
MODELS_DIR = PROCESSED / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PORT_NAMES = ["부산", "울산", "인천", "광양"]


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = False) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 150,
    lr: float = 5e-4,
    patience: int = 15,
    save_path: Path | None = None,
) -> tuple[list, list]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    best_val = float("inf")
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            if save_path:
                torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Early stop @ epoch {epoch + 1}")
            break

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train={train_loss:.4f}, val={val_loss:.4f}")

    return train_losses, val_losses


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb).numpy())
            trues.append(yb.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def run_lstm(epochs: int) -> dict:
    print("\n=== LSTM (대조군) ===")
    X_train = np.load(PROCESSED / "X_train.npy")
    y_train = np.load(PROCESSED / "y_train.npy")
    X_val   = np.load(PROCESSED / "X_val.npy")
    y_val   = np.load(PROCESSED / "y_val.npy")
    X_test  = np.load(PROCESSED / "X_test.npy")
    y_test  = np.load(PROCESSED / "y_test.npy")

    model = build_lstm(n_features=X_train.shape[2], n_ports=y_train.shape[2], pred_horizon=y_train.shape[1])
    print(f"  파라미터: {sum(p.numel() for p in model.parameters()):,}")

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val)
    test_loader  = make_loader(X_test,  y_test)

    train_model(model, train_loader, val_loader,
                epochs=epochs, lr=1e-3, patience=15,
                save_path=MODELS_DIR / "lstm_best.pt")

    model.load_state_dict(torch.load(MODELS_DIR / "lstm_best.pt", map_location="cpu"))
    y_scaled_true, y_scaled_pred = evaluate(model, test_loader)
    scaler_y = _load_scaler(PROCESSED / "scaler_y.pkl")
    y_true = _inverse(scaler_y, y_scaled_true)
    y_pred = _inverse(scaler_y, y_scaled_pred)
    metrics = regression_metrics(y_true, y_pred)
    pm = port_metrics(y_true, y_pred, PORT_NAMES)
    print(f"  Test — MAE:{metrics['MAE']:.4f}, RMSE:{metrics['RMSE']:.4f}, MAPE:{metrics['MAPE']:.1f}%, sMAPE:{metrics['sMAPE']:.1f}%")
    print("  항만별 RMSE:", {p: f"{r:.4f}" for p, r in pm.items()})
    metrics["port"] = pm
    return metrics


def run_transformer(epochs: int) -> dict:
    print("\n=== Transformer (수치 전용) ===")
    X_train = np.load(PROCESSED / "X_train.npy")
    y_train = np.load(PROCESSED / "y_train.npy")
    X_val   = np.load(PROCESSED / "X_val.npy")
    y_val   = np.load(PROCESSED / "y_val.npy")
    X_test  = np.load(PROCESSED / "X_test.npy")
    y_test  = np.load(PROCESSED / "y_test.npy")

    model = build_transformer(n_features=X_train.shape[2], n_ports=y_train.shape[2],
                               pred_horizon=y_train.shape[1], d_model=64, nhead=4,
                               num_encoder_layers=2, dim_feedforward=128, dropout=0.1)
    print(f"  파라미터: {sum(p.numel() for p in model.parameters()):,}")

    train_model(model, make_loader(X_train, y_train, shuffle=True),
                make_loader(X_val, y_val),
                epochs=epochs, lr=5e-4, patience=15,
                save_path=MODELS_DIR / "transformer_best.pt")

    model.load_state_dict(torch.load(MODELS_DIR / "transformer_best.pt", map_location="cpu"))
    y_scaled_true, y_scaled_pred = evaluate(model, make_loader(X_test, y_test))
    scaler_y = _load_scaler(PROCESSED / "scaler_y.pkl")
    y_true = _inverse(scaler_y, y_scaled_true)
    y_pred = _inverse(scaler_y, y_scaled_pred)
    metrics = regression_metrics(y_true, y_pred)
    pm = port_metrics(y_true, y_pred, PORT_NAMES)
    print(f"  Test — MAE:{metrics['MAE']:.4f}, RMSE:{metrics['RMSE']:.4f}, MAPE:{metrics['MAPE']:.1f}%, sMAPE:{metrics['sMAPE']:.1f}%")
    print("  항만별 RMSE:", {p: f"{r:.4f}" for p, r in pm.items()})
    metrics["port"] = pm
    return metrics


def run_transformer_event(epochs: int) -> dict:
    print("\n=== Transformer + 이벤트 피처 ===")
    ev_path = PROCESSED / "X_ev_train.npy"
    if not ev_path.exists():
        print("  [경고] 이벤트 데이터셋 없음 — 02_feature_engineering.ipynb 셀 19~24 먼저 실행")
        return {}

    X_train = np.load(PROCESSED / "X_ev_train.npy")
    y_train = np.load(PROCESSED / "y_ev_train.npy")
    X_val   = np.load(PROCESSED / "X_ev_val.npy")
    y_val   = np.load(PROCESSED / "y_ev_val.npy")
    X_test  = np.load(PROCESSED / "X_ev_test.npy")
    y_test  = np.load(PROCESSED / "y_ev_test.npy")

    model = build_transformer(n_features=X_train.shape[2], n_ports=y_train.shape[2],
                               pred_horizon=y_train.shape[1], d_model=64, nhead=4,
                               num_encoder_layers=2, dim_feedforward=128, dropout=0.1)
    print(f"  피처: {X_train.shape[2]}개 | 파라미터: {sum(p.numel() for p in model.parameters()):,}")

    train_model(model, make_loader(X_train, y_train, shuffle=True),
                make_loader(X_val, y_val),
                epochs=epochs, lr=5e-4, patience=15,
                save_path=MODELS_DIR / "transformer_event_best.pt")

    model.load_state_dict(torch.load(MODELS_DIR / "transformer_event_best.pt", map_location="cpu"))
    y_scaled_true, y_scaled_pred = evaluate(model, make_loader(X_test, y_test))
    scaler_y_ev = _load_scaler(PROCESSED / "scaler_y_ev.pkl")
    y_true = _inverse(scaler_y_ev, y_scaled_true)
    y_pred = _inverse(scaler_y_ev, y_scaled_pred)
    metrics = regression_metrics(y_true, y_pred)
    pm = port_metrics(y_true, y_pred, PORT_NAMES)
    print(f"  Test — MAE:{metrics['MAE']:.4f}, RMSE:{metrics['RMSE']:.4f}, MAPE:{metrics['MAPE']:.1f}%, sMAPE:{metrics['sMAPE']:.1f}%")
    print("  항만별 RMSE:", {p: f"{r:.4f}" for p, r in pm.items()})
    metrics["port"] = pm
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer", "transformer_event", "all"],
                        default="all")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    results = {}
    if args.model in ("lstm", "all"):
        results["LSTM (대조군)"] = run_lstm(args.epochs)
    if args.model in ("transformer", "all"):
        results["Transformer"] = run_transformer(args.epochs)
    if args.model in ("transformer_event", "all"):
        results["Transformer+Event"] = run_transformer_event(args.epochs)

    if len(results) > 1:
        summary = {k: {m: v for m, v in r.items() if m != "port"} for k, r in results.items() if r}
        df = pd.DataFrame(summary).T
        print("\n=== 최종 비교 (원래 스케일 체선율 기준) ===")
        print(df.to_string())
        df.to_csv(PROCESSED / "model_comparison_3way.csv", encoding="utf-8-sig")
        print(f"\n저장: {PROCESSED}/model_comparison_3way.csv")


if __name__ == "__main__":
    main()

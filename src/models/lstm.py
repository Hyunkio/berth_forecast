"""LSTM 대조군 모델."""
import torch
import torch.nn as nn


class CongestionLSTM(nn.Module):
    """
    입력: (batch, seq_len, n_features)
    출력: (batch, pred_horizon, n_ports)
    """

    def __init__(
        self,
        n_features: int,
        n_ports: int,
        pred_horizon: int = 7,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.n_ports      = n_ports

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, pred_horizon * n_ports)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # (batch, seq, hidden)
        out = out[:, -1, :]            # 마지막 시각 hidden
        out = self.head(out)           # (batch, horizon * n_ports)
        return out.view(-1, self.pred_horizon, self.n_ports)


def build_lstm(n_features: int, n_ports: int, pred_horizon: int = 7, **kwargs) -> CongestionLSTM:
    return CongestionLSTM(n_features, n_ports, pred_horizon, **kwargs)

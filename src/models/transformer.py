"""Transformer 메인 모델 — 단일/멀티 항만 통합 지원."""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class CongestionTransformer(nn.Module):
    """
    입력: (batch, seq_len, n_features)
    출력: (batch, pred_horizon, n_ports)

    Multi-head Self-Attention으로 시간 간·항만 간 상관관계를 자동 학습.
    """

    def __init__(
        self,
        n_features: int,
        n_ports: int,
        pred_horizon: int = 7,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.n_ports      = n_ports
        self.d_model      = d_model

        # 입력 투영
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: 학습 안정성 향상
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # 출력 헤드
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_horizon * n_ports),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)          # (batch, seq, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)             # (batch, seq, d_model)
        x = x[:, -1, :]                 # 마지막 시각 표현
        x = self.head(x)                # (batch, horizon * n_ports)
        return x.view(-1, self.pred_horizon, self.n_ports)


def build_transformer(
    n_features: int,
    n_ports: int,
    pred_horizon: int = 7,
    **kwargs,
) -> CongestionTransformer:
    return CongestionTransformer(n_features, n_ports, pred_horizon, **kwargs)

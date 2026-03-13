# models/predictor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PredictorConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_ff: int = 256
    dropout: float = 0.1
    G: int = 64


class TinyCausalPredictor(nn.Module):
    """
    A lightweight causal sequence model.
    Input:
        x: [B, L, d_model]
    Output:
        support_logits: [B, G]
        energy_pred:    [B, G]
    """

    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        self.cfg = cfg

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)

        self.support_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.G),
        )

        self.energy_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.G),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, L, d_model]
        """
        B, L, D = x.shape
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        h = self.encoder(x, mask=causal_mask)
        h_last = h[:, -1, :]

        support_logits = self.support_head(h_last)
        energy_pred = self.energy_head(h_last)
        return support_logits, energy_pred


def predictor_loss(
    support_logits: torch.Tensor,
    energy_pred: torch.Tensor,
    support_true: torch.Tensor,
    energy_true: torch.Tensor,
    lambda_b: float = 1.0,
    lambda_e: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    loss_b = F.binary_cross_entropy_with_logits(support_logits, support_true)
    loss_e = F.huber_loss(energy_pred, energy_true)
    loss = lambda_b * loss_b + lambda_e * loss_e
    return loss, {
        "loss": float(loss.item()),
        "loss_b": float(loss_b.item()),
        "loss_e": float(loss_e.item()),
    }


def pool_frame_tokens(Z_t: torch.Tensor) -> torch.Tensor:
    """
    First version: mean-pool frame tokens into one frame embedding.
    Z_t: [Ntok, d]
    return: [d]
    """
    return Z_t.mean(dim=0)
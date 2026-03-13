# models/tokenizer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn


@dataclass
class TokenizerConfig:
    G: int
    top_k: int
    d_model: int
    eps: float = 1e-8
    support_threshold: float = 1e-6


def group_energy_from_hvec(
    hvec: torch.Tensor,
    group_id: torch.Tensor,
    G: int,
) -> torch.Tensor:
    """
    hvec: [N] complex
    group_id: [N] long
    return: [G] real
    """
    abs2 = (torch.abs(hvec) ** 2).real.to(torch.float32)
    Eg = torch.zeros(G, device=hvec.device, dtype=torch.float32)
    Eg.scatter_add_(0, group_id, abs2)
    return Eg


def build_support_and_logenergy(
    hvec: torch.Tensor,
    group_id: torch.Tensor,
    G: int,
    tau: float,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    Eg = group_energy_from_hvec(hvec, group_id, G)
    bg = (Eg >= tau).to(torch.float32)
    eg = torch.log(Eg + eps)
    return bg, eg


class WavenumberTokenizer(nn.Module):
    """
    Physics-aware tokenizer for the predictive support learning stage.

    The token design follows your note:
      1. Coarse estimate via matched-filter snapshot
      2. Compute coarse group energies
      3. Keep top-K groups
      4. Build token = group-id embedding + energy embedding + position embedding
      5. Add a global token
    """

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg

        self.group_embed = nn.Embedding(cfg.G, cfg.d_model)

        self.energy_mlp = nn.Sequential(
            nn.Linear(1, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        self.pos_mlp = nn.Sequential(
            nn.Linear(2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(5, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    @torch.no_grad()
    def coarse_estimate(
        self,
        Aop,
        y_t: torch.Tensor,
        Pdim: float,
        LR: int,
        LS: int,
    ) -> torch.Tensor:
        """
        Matched-filter snapshot:
            H_a^(0) = sqrt(Pdim) * A^*(y_t)
        Returns vec(H_a^(0)) in the same column-major convention as the core code.
        """
        h0 = (Pdim ** 0.5) * Aop.adjoint(y_t, LR, LS)
        return h0

    @torch.no_grad()
    def topk_groups(
        self,
        h0: torch.Tensor,
        group_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Eg = group_energy_from_hvec(h0, group_id, self.cfg.G)
        K = min(self.cfg.top_k, self.cfg.G)
        vals, idx = torch.topk(Eg, k=K, largest=True, sorted=True)
        return idx, vals

    def build_frame_tokens(
        self,
        *,
        h0: torch.Tensor,
        snr_db: float,
        P: int,
        NrRF: int,
        NtRF: int,
        group_id: torch.Tensor,
        group_uv: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        group_uv: [G,2] float tensor, e.g. normalized coarse group coordinates.
        Returns:
            Z_t: [1+K, d_model]
            aux: dict with coarse group statistics
        """
        topk_gid, topk_E = self.topk_groups(h0, group_id)

        gid_emb = self.group_embed(topk_gid)  # [K,d]
        energy_emb = self.energy_mlp(torch.log(topk_E + self.cfg.eps).unsqueeze(-1))
        pos_emb = self.pos_mlp(group_uv[topk_gid].to(torch.float32))

        group_tokens = gid_emb + energy_emb + pos_emb

        global_feat = torch.tensor(
            [
                float(snr_db),
                float(P),
                float(NrRF),
                float(NtRF),
                float((torch.norm(h0) ** 2).real.item()),
            ],
            device=h0.device,
            dtype=torch.float32,
        ).unsqueeze(0)

        global_token = self.global_mlp(global_feat)  # [1,d]

        Z_t = torch.cat([global_token, group_tokens], dim=0)

        aux = {
            "topk_gid": topk_gid,
            "topk_energy": topk_E,
            "coarse_group_energy": group_energy_from_hvec(h0, group_id, self.cfg.G),
        }
        return Z_t, aux

    @torch.no_grad()
    def build_labels_from_true_h(
        self,
        h_true: torch.Tensor,
        group_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build next-step labels:
            support b_g
            log-energy e_g
        """
        bg, eg = build_support_and_logenergy(
            h_true=h_true if False else h_true,
            hvec=h_true,
            group_id=group_id,
            G=self.cfg.G,
            tau=self.cfg.support_threshold,
            eps=self.cfg.eps,
        )
        return bg, eg


def build_support_and_logenergy(
    hvec: torch.Tensor,
    group_id: torch.Tensor,
    G: int,
    tau: float,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    Eg = group_energy_from_hvec(hvec, group_id, G)
    bg = (Eg >= tau).to(torch.float32)
    eg = torch.log(Eg + eps)
    return bg, eg
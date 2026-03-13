# experiments/build_temporal_dataset.py
from __future__ import annotations

import os
import math
import argparse
from typing import Dict, List

import numpy as np
import torch

from core import demo_sbl_block_operator_fixed_angles_report as core
from models.tokenizer import WavenumberTokenizer, TokenizerConfig, build_support_and_logenergy


def evolve_profiles(profile: torch.Tensor, drift_std: float = 0.03) -> torch.Tensor:
    logp = torch.log(torch.clamp(profile, min=1e-8)) + drift_std * torch.randn_like(profile)
    out = torch.exp(logp)
    out = out / torch.clamp(out.sum(), min=1e-8)
    return out


def evolve_W(W_prev: torch.Tensor, rho_w: float, ctype: torch.dtype) -> torch.Tensor:
    Wf = (torch.randn_like(W_prev.real) + 1j * torch.randn_like(W_prev.real)).to(ctype) / math.sqrt(2.0)
    return rho_w * W_prev + math.sqrt(max(1.0 - rho_w ** 2, 0.0)) * Wf


def build_group_uv_stub(G: int, device: str) -> torch.Tensor:
    """
    First version:
    Use a simple normalized line/grid-like placeholder for group positions.
    Later you can replace it with a true coarse group-grid coordinate map.
    """
    g = torch.arange(G, device=device, dtype=torch.float32)
    u = (g / max(G - 1, 1)) * 2.0 - 1.0
    v = torch.zeros_like(u)
    return torch.stack([u, v], dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/processed/temporal_dataset.pt")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", type=str, default="c64", choices=["c64", "c128"])
    parser.add_argument("--num_sequences", type=int, default=80)
    parser.add_argument("--frames", type=int, default=18)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--snr_db", type=float, default=10.0)
    parser.add_argument("--rho_w", type=float, default=0.95)
    parser.add_argument("--profile_drift", type=float, default=0.03)
    parser.add_argument("--support_tau", type=float, default=1e-5)

    parser.add_argument("--P", type=int, default=24)
    parser.add_argument("--NrRF", type=int, default=32)
    parser.add_argument("--NtRF", type=int, default=4)

    parser.add_argument("--NRx", type=int, default=33)
    parser.add_argument("--NRy", type=int, default=33)
    parser.add_argument("--NSx", type=int, default=5)
    parser.add_argument("--NSy", type=int, default=5)

    args = parser.parse_args()

    core.seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctype = torch.complex64 if args.dtype == "c64" else torch.complex128

    fc = 30e9
    c0 = 3e8
    lamb = c0 / fc
    delta = lamb / 4

    NRx, NRy = args.NRx, args.NRy
    NSx, NSy = args.NSx, args.NSy

    LRx = (NRx - 1) * delta
    LRy = (NRy - 1) * delta
    LSx = (NSx - 1) * delta
    LSy = (NSy - 1) * delta

    Nc = 2
    alpha0 = 540.0
    w_mix = np.ones(Nc, dtype=np.float64) / Nc
    Nu, Nv = 9, 9
    alphaR = (alpha0 * np.ones(Nc)).astype(np.float64)
    alphaS = (alpha0 * np.ones(Nc)).astype(np.float64)

    xiR = core.build_xi_ellipse(LRx, LRy, lamb)
    xiS = core.build_xi_ellipse(LSx, LSy, lamb)
    LR = xiR.shape[0]
    LS = xiS.shape[0]

    PsiR = core.build_wavenumber_basis_vec(NRx, NRy, delta, LRx, LRy, xiR, device=device, ctype=ctype)
    PsiS = core.build_wavenumber_basis_vec(NSx, NSy, delta, LSx, LSy, xiS, device=device, ctype=ctype)

    idxR_np, idxS_np, _ = core.design_selection_physical(
        xiS, xiR, LS, LR,
        args.NtRF, args.NrRF, args.P,
        seed=args.seed + 7,
        BxS=1, ByS=1, BxR=3, ByR=3,
        avoid_repeat=True,
        reset_policy="when_exhausted",
    )
    idxR = torch.tensor(idxR_np, device=device, dtype=torch.long)

    Xmat = torch.zeros((LS, args.P), device=device, dtype=ctype)
    signs = torch.sign(torch.randn(args.P, device=device, dtype=torch.float32))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    for p in range(args.P):
        for t in range(args.NtRF):
            Xmat.real[idxS_np[p, t], p] += signs[p]
    Xmat = Xmat / torch.clamp(torch.norm(Xmat, dim=0, keepdim=True), min=1e-12)

    Aop = core.IndexSensing(idxR=idxR, Xmat=Xmat, Pdim=1.0)

    group_id_np, GR, GS, G = core.build_group_ids_from_xi(xiR, xiS, Brx=3, Bry=3, Bsx=1, Bsy=1)
    group_id = torch.tensor(group_id_np, device=device, dtype=torch.long)

    tokenizer = WavenumberTokenizer(
        TokenizerConfig(
            G=G,
            top_k=args.top_k,
            d_model=128,
            eps=1e-8,
            support_threshold=args.support_tau,
        )
    ).to(device)

    group_uv = build_group_uv_stub(G, device=device)

    thetaR_fix, phiR_fix, thetaS_fix, phiS_fix = core.generate_fixed_angles(Nc, args.seed)

    sigma2R0 = core.sigma2_from_vmf_uvcells_torch(
        xiR, LRx, LRy, lamb, thetaR_fix, phiR_fix, alphaR, w_mix, Nu, Nv, device="cpu"
    ).to(torch.float32).to(device)
    sigma2S0 = core.sigma2_from_vmf_uvcells_torch(
        xiS, LSx, LSy, lamb, thetaS_fix, phiS_fix, alphaS, w_mix, Nu, Nv, device="cpu"
    ).to(torch.float32).to(device)

    dataset = []
    for seq_id in range(args.num_sequences):
        core.seed_all(args.seed + seq_id)

        sigma2R = sigma2R0.clone()
        sigma2S = sigma2S0.clone()
        W = (torch.randn(LR, LS, device=device) + 1j * torch.randn(LR, LS, device=device)).to(ctype) / math.sqrt(2.0)

        seq_tokens = []
        seq_support = []
        seq_energy = []
        seq_h = []
        seq_y = []

        for t in range(args.frames):
            if t > 0:
                sigma2R = evolve_profiles(sigma2R, drift_std=args.profile_drift)
                sigma2S = evolve_profiles(sigma2S, drift_std=args.profile_drift)
                W = evolve_W(W, rho_w=args.rho_w, ctype=ctype)

            Ha_true = (torch.sqrt(torch.clamp(sigma2R, min=0.0)).to(ctype).unsqueeze(1) * W) * \
                      (torch.sqrt(torch.clamp(sigma2S, min=0.0)).to(ctype).unsqueeze(0))
            h_true = Ha_true.transpose(0, 1).contiguous().view(-1)

            y_clean = Aop.forward(h_true, LR, LS)
            M = y_clean.numel()
            sigPow = (torch.norm(y_clean) ** 2).real.item() / M
            sigma2n = sigPow / (10 ** (args.snr_db / 10.0))
            noise = math.sqrt(sigma2n / 2) * (torch.randn(M, device=device) + 1j * torch.randn(M, device=device)).to(ctype)
            y_t = y_clean + noise

            h0 = tokenizer.coarse_estimate(Aop=Aop, y_t=y_t, Pdim=1.0, LR=LR, LS=LS)
            Z_t, _ = tokenizer.build_frame_tokens(
                h0=h0,
                snr_db=args.snr_db,
                P=args.P,
                NrRF=args.NrRF,
                NtRF=args.NtRF,
                group_id=group_id,
                group_uv=group_uv,
            )

            b_t, e_t = build_support_and_logenergy(
                hvec=h_true,
                group_id=group_id,
                G=G,
                tau=args.support_tau,
                eps=1e-8,
            )

            seq_tokens.append(Z_t.detach().cpu())
            seq_support.append(b_t.detach().cpu())
            seq_energy.append(e_t.detach().cpu())
            seq_h.append(h_true.detach().cpu())
            seq_y.append(y_t.detach().cpu())

        dataset.append({
            "tokens": seq_tokens,
            "support": seq_support,
            "energy": seq_energy,
            "h": seq_h,
            "y": seq_y,
        })

        print(f"[Seq {seq_id+1}/{args.num_sequences}] done")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "dataset": dataset,
        "meta": {
            "LR": LR,
            "LS": LS,
            "G": G,
            "history": args.history,
            "top_k": args.top_k,
            "snr_db": args.snr_db,
            "P": args.P,
            "NrRF": args.NrRF,
            "NtRF": args.NtRF,
            "group_id": group_id.detach().cpu(),
            "group_uv": group_uv.detach().cpu(),
        }
    }, args.out)

    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()
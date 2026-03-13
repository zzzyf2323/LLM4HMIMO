# experiments/run_predictive_warmstart.py
from __future__ import annotations

import math
import argparse
from typing import List

import numpy as np
import torch

from core import demo_sbl_block_operator_fixed_angles_report as core
from models.tokenizer import WavenumberTokenizer, TokenizerConfig
from models.predictor import TinyCausalPredictor, PredictorConfig, pool_frame_tokens


def build_predictive_gamma_init(
    pred_support_prob: torch.Tensor,
    pred_log_energy: torch.Tensor,
    gamma_min: float,
    gamma_max: float,
    rho: float,
) -> torch.Tensor:
    gamma0 = torch.exp(pred_log_energy).clamp(min=gamma_min, max=gamma_max)
    gamma0 = torch.where(pred_support_prob >= rho, gamma0, torch.full_like(gamma0, gamma_min))
    return gamma0


def build_screened_mask(
    pred_support_prob: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    return pred_support_prob >= rho


def nmse_from_vec(mu_vec, h_true, PsiR, PsiS, LR, LS):
    Ha_hat = mu_vec.view(LS, LR).transpose(0, 1)
    Ha_true = h_true.view(LS, LR).transpose(0, 1)
    H_hat = PsiR @ Ha_hat @ PsiS.conj().transpose(0, 1)
    H_true = PsiR @ Ha_true @ PsiS.conj().transpose(0, 1)
    err = torch.norm(H_hat - H_true) ** 2
    den = torch.clamp(torch.norm(H_true) ** 2, min=1e-12)
    return float((err / den).real.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/temporal_dataset.pt")
    parser.add_argument("--ckpt", type=str, default="data/checkpoints/wavenumber_predictor.pt")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_eval", type=int, default=20)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--gamma_min", type=float, default=1e-8)
    parser.add_argument("--gamma_max", type=float, default=1e1)
    parser.add_argument("--support_tau", type=float, default=1e-5)

    parser.add_argument("--sbl_iter", type=int, default=20)
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--cg_iter", type=int, default=100)
    parser.add_argument("--cg_tol", type=float, default=1e-4)
    parser.add_argument("--trace_probes", type=int, default=4)
    parser.add_argument("--stop_rel", type=float, default=1e-3)
    parser.add_argument("--damping", type=float, default=0.4)

    args = parser.parse_args()

    core.seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bundle = torch.load(args.data, map_location="cpu")
    dataset = bundle["dataset"]
    meta = bundle["meta"]

    group_id = meta["group_id"].to(device)
    G = int(meta["G"])
    history = int(meta["history"])
    top_k = int(meta["top_k"])
    snr_db = float(meta["snr_db"])
    P = int(meta["P"])
    NrRF = int(meta["NrRF"])
    NtRF = int(meta["NtRF"])

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]

    predictor = TinyCausalPredictor(
        PredictorConfig(
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_ff=256,
            dropout=0.1,
            G=cfg["G"],
        )
    ).to(device)
    predictor.load_state_dict(ckpt["state_dict"])
    predictor.eval()

    tokenizer = WavenumberTokenizer(
        TokenizerConfig(
            G=G,
            top_k=top_k,
            d_model=cfg["d_model"],
            eps=1e-8,
            support_threshold=args.support_tau,
        )
    ).to(device)

    fc = 30e9
    c0 = 3e8
    lamb = c0 / fc
    delta = lamb / 4

    NRx, NRy = 33, 33
    NSx, NSy = 5, 5

    LRx = (NRx - 1) * delta
    LRy = (NRy - 1) * delta
    LSx = (NSx - 1) * delta
    LSy = (NSy - 1) * delta

    xiR = core.build_xi_ellipse(LRx, LRy, lamb)
    xiS = core.build_xi_ellipse(LSx, LSy, lamb)
    LR = xiR.shape[0]
    LS = xiS.shape[0]

    ctype = torch.complex64
    PsiR = core.build_wavenumber_basis_vec(NRx, NRy, delta, LRx, LRy, xiR, device=device, ctype=ctype)
    PsiS = core.build_wavenumber_basis_vec(NSx, NSy, delta, LSx, LSy, xiS, device=device, ctype=ctype)

    idxR_np, idxS_np, _ = core.design_selection_physical(
        xiS, xiR, LS, LR,
        NtRF, NrRF, P,
        seed=args.seed + 7,
        BxS=1, ByS=1, BxR=3, ByR=3,
        avoid_repeat=True,
        reset_policy="when_exhausted",
    )
    idxR = torch.tensor(idxR_np, device=device, dtype=torch.long)

    Xmat = torch.zeros((LS, P), device=device, dtype=ctype)
    signs = torch.sign(torch.randn(P, device=device, dtype=torch.float32))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    for p in range(P):
        for t in range(NtRF):
            Xmat.real[idxS_np[p, t], p] += signs[p]
    Xmat = Xmat / torch.clamp(torch.norm(Xmat, dim=0, keepdim=True), min=1e-12)

    Aop = core.IndexSensing(idxR=idxR, Xmat=Xmat, Pdim=1.0)

    ones = torch.ones(LR * LS, device=device, dtype=torch.float32)
    cnt = torch.zeros(G, device=device, dtype=torch.float32).scatter_add_(0, group_id, ones).clamp_min(1.0)

    group_uv = meta["group_uv"].to(device)

    uniform_nmse = []
    pred_nmse = []

    for seq_id, seq in enumerate(dataset[:args.num_eval]):
        tokens = [z.to(device) for z in seq["tokens"]]
        hs = [h.to(device) for h in seq["h"]]
        ys = [y.to(device) for y in seq["y"]]

        pooled = [pool_frame_tokens(z) for z in tokens]

        for t in range(history, len(tokens)):
            x_hist = torch.stack(pooled[t-history:t], dim=0).unsqueeze(0)  # [1,L,d]
            with torch.no_grad():
                support_logits, energy_pred = predictor(x_hist)
                support_prob = torch.sigmoid(support_logits[0])
                pred_log_energy = energy_pred[0]

            gamma0_pred = build_predictive_gamma_init(
                pred_support_prob=support_prob,
                pred_log_energy=pred_log_energy,
                gamma_min=args.gamma_min,
                gamma_max=args.gamma_max,
                rho=args.rho,
            )
            screened_mask = build_screened_mask(support_prob, rho=args.rho)

            y_t = ys[t]
            h_true = hs[t]

            M = y_t.numel()
            sigma2n = 1e-3

            mu_u, _, _ = core.sbl_block_em(
                Aop=Aop, y=y_t, sigma2=sigma2n,
                group_id=group_id, cnt=cnt, G=G,
                LR=LR, LS=LS,
                max_iter=args.sbl_iter,
                cg_tol=args.cg_tol, cg_maxiter=args.cg_iter,
                probes=args.probes,
                stop_rel=args.stop_rel,
                damping=args.damping,
                learn_sigma2=False,
                trace_probes=args.trace_probes,
                sigma2_damping=0.3,
                diag_eps=0.0,
                use_jacobi=False,
                diag_print_every=0,
                collect_info=False
            )

            mu_p, _, _ = core.sbl_block_em(
                Aop=Aop, y=y_t, sigma2=sigma2n,
                group_id=group_id, cnt=cnt, G=G,
                LR=LR, LS=LS,
                max_iter=args.sbl_iter,
                cg_tol=args.cg_tol, cg_maxiter=args.cg_iter,
                probes=args.probes,
                stop_rel=args.stop_rel,
                damping=args.damping,
                learn_sigma2=False,
                trace_probes=args.trace_probes,
                sigma2_damping=0.3,
                diag_eps=0.0,
                use_jacobi=False,
                diag_print_every=0,
                collect_info=False,
                gamma0=gamma0_pred,
                screened_mask=screened_mask,
            )

            uniform_nmse.append(nmse_from_vec(mu_u, h_true, PsiR, PsiS, LR, LS))
            pred_nmse.append(nmse_from_vec(mu_p, h_true, PsiR, PsiS, LR, LS))

        print(f"[Eval Seq {seq_id+1}/{min(args.num_eval, len(dataset))}] done")

    print(f"[Uniform-init SBL] mean NMSE = {10*np.log10(np.mean(uniform_nmse)+1e-300):.2f} dB")
    print(f"[Predictive-init SBL] mean NMSE = {10*np.log10(np.mean(pred_nmse)+1e-300):.2f} dB")


if __name__ == "__main__":
    main()
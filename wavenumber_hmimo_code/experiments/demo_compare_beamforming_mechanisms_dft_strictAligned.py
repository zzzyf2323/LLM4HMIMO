# experiments/demo_compare_beamforming_mechanisms_dft_strictAligned.py
from __future__ import annotations

import math
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from core import demo_sbl_block_operator_fixed_angles_report as core


# -------------------------- Traditional hybrid sensing operator --------------------------
class DenseHybridSensing:
    def __init__(self, A_list: torch.Tensor, x_list: torch.Tensor, Pdim: float):
        self.A_list = A_list
        self.x_list = x_list
        self.Pdim = float(Pdim)

        self.P, self.NrRF, self.LR = A_list.shape
        P2, self.LS = x_list.shape
        assert P2 == self.P

        self._Gbuf = None

    @staticmethod
    def vec_to_Ha(hvec, LR, LS):
        return hvec.view(LS, LR).transpose(0, 1)

    @staticmethod
    def Ha_to_vec(Ha):
        return Ha.transpose(0, 1).contiguous().view(-1)

    @torch.inference_mode()
    def forward(self, hvec, LR, LS):
        Ha = self.vec_to_Ha(hvec, LR, LS)
        out = []
        scale = math.sqrt(self.Pdim)
        for p in range(self.P):
            Ap = self.A_list[p]
            xp = self.x_list[p]
            yp = Ap @ (Ha @ xp)
            out.append(yp)
        return scale * torch.cat(out, dim=0)

    @torch.inference_mode()
    def adjoint(self, v, LR, LS):
        V = v.view(self.P, self.NrRF)
        scale = math.sqrt(self.Pdim)

        if (self._Gbuf is None) or (self._Gbuf.shape != (LR, LS)) or (self._Gbuf.device != v.device) or (self._Gbuf.dtype != v.dtype):
            self._Gbuf = torch.zeros((LR, LS), device=v.device, dtype=v.dtype)
        else:
            self._Gbuf.zero_()

        for p in range(self.P):
            rp = V[p]
            Ap = self.A_list[p]
            xp = self.x_list[p]
            tmp = Ap.conj().transpose(0, 1) @ rp
            self._Gbuf += tmp[:, None] * xp.conj()[None, :]

        self._Gbuf *= scale
        return self.Ha_to_vec(self._Gbuf)


@torch.inference_mode()
def check_adjoint_generic(Aop, LR, LS, device, ctype, trials=2):
    for t in range(trials):
        h = (torch.randn(LR * LS, device=device) + 1j * torch.randn(LR * LS, device=device)).to(ctype)
        v = (torch.randn(Aop.P * Aop.NrRF, device=device) + 1j * torch.randn(Aop.P * Aop.NrRF, device=device)).to(ctype)
        Ah = Aop.forward(h, LR, LS)
        Atv = Aop.adjoint(v, LR, LS)
        lhs = torch.vdot(Ah, v)
        rhs = torch.vdot(h, Atv)
        rel = (lhs - rhs).abs() / (lhs.abs() + rhs.abs() + 1e-12)
        print(f"[AdjointTest-Dense] trial {t + 1}: rel_err={rel.item():.2e}")


# -------------------------- FISTA Group-LASSO --------------------------
@torch.inference_mode()
def prox_group_l2(x: torch.Tensor, tau: float, group_id: torch.Tensor, G: int):
    abs2 = (torch.abs(x) ** 2).real.to(torch.float32)
    gn2 = torch.zeros(G, device=x.device, dtype=torch.float32).scatter_add_(0, group_id, abs2).clamp_min(1e-30)
    gn = torch.sqrt(gn2)
    scale_g = torch.clamp(1.0 - (tau / gn), min=0.0)
    return x * scale_g[group_id].to(x.dtype)


@torch.inference_mode()
def estimate_lipschitz(Aop, LR, LS, device, ctype, iters=12):
    N = LR * LS
    x = (torch.randn(N, device=device) + 1j * torch.randn(N, device=device)).to(ctype)
    x = x / torch.clamp(torch.norm(x), min=1e-30)
    L = 1.0
    for _ in range(iters):
        y = Aop.forward(x, LR, LS)
        z = Aop.adjoint(y, LR, LS)
        nz = torch.clamp(torch.norm(z), min=1e-30)
        x = z / nz
        L = float((torch.real(torch.vdot(x, z))).item())
    return max(L, 1e-12)


@torch.inference_mode()
def fista_group_lasso(
    Aop, y, group_id, G, LR, LS,
    lam: float,
    max_iter: int = 120,
    stop_rel: float = 1e-4,
    L: float | None = None,
):
    device = y.device
    ctype = y.dtype
    N = LR * LS

    if L is None:
        L = estimate_lipschitz(Aop, LR, LS, device, ctype, iters=12)
    step = 1.0 / L

    x = torch.zeros(N, device=device, dtype=ctype)
    z = x.clone()
    t = 1.0

    for it in range(max_iter):
        Az = Aop.forward(z, LR, LS)
        grad = Aop.adjoint(Az - y, LR, LS)
        x_new = prox_group_l2(z - step * grad, tau=lam * step, group_id=group_id, G=G)

        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)

        rel = torch.norm(x_new - x) / torch.clamp(torch.norm(x), min=1e-30)
        x = x_new
        t = t_new

        if it >= 5 and float(rel.item()) < stop_rel:
            break

    return x, {"L": L, "iters": it + 1}


# -------------------------- DFT helpers --------------------------
def dft_matrix_1d(N: int) -> np.ndarray:
    n = np.arange(N, dtype=np.float64)
    k = n[:, None]
    W = np.exp(-1j * 2.0 * np.pi * k * n[None, :] / N) / np.sqrt(N)
    return W.astype(np.complex128)


def dft_codebook_upa(Nx: int, Ny: int) -> np.ndarray:
    Fx = dft_matrix_1d(Nx)
    Fy = dft_matrix_1d(Ny)
    return np.kron(Fy, Fx).astype(np.complex128)


def _k_eff(k: int, N: int) -> int:
    return k if k <= (N // 2) else (k - N)


def valid_dft_cols_propagating(Nx: int, Ny: int, delta: float, lamb: float):
    k0 = 2.0 * math.pi / lamb
    cols = []
    for ky in range(Ny):
        ky_eff = _k_eff(ky, Ny)
        ky_val = 2.0 * math.pi * ky_eff / (Ny * delta)
        for kx in range(Nx):
            kx_eff = _k_eff(kx, Nx)
            kx_val = 2.0 * math.pi * kx_eff / (Nx * delta)
            if (kx_val * kx_val + ky_val * ky_val) <= (k0 * k0 + 1e-12):
                cols.append(ky * Nx + kx)
    return cols


def build_traditional_sensing_dft_propagating(
    PsiR, PsiS,
    NRx, NRy,
    NSx, NSy,
    LR, LS,
    P, NrRF, NtRF,
    pilot_mode,
    seed,
    delta,
    lamb,
    device, ctype
):
    rng = np.random.RandomState(seed)

    F_rx = dft_codebook_upa(NRx, NRy)
    F_tx = dft_codebook_upa(NSx, NSy)

    valid_r = valid_dft_cols_propagating(NRx, NRy, delta, lamb)
    valid_t = valid_dft_cols_propagating(NSx, NSy, delta, lamb)

    A_list = []
    x_list = []

    for p in range(P):
        cols_r = [valid_r[(p * NrRF + i) % len(valid_r)] for i in range(NrRF)]
        cols_t = [valid_t[(p * NtRF + i) % len(valid_t)] for i in range(NtRF)]

        Wr_np = F_rx[:, cols_r]
        Fr_np = F_tx[:, cols_t]

        if pilot_mode == "selection":
            sp_np = np.ones((NtRF,), dtype=np.complex128) / math.sqrt(NtRF)
        else:
            sgn = rng.choice([-1.0, 1.0], size=(NtRF,))
            sp_np = sgn.astype(np.complex128) / math.sqrt(NtRF)

        Wr = torch.tensor(Wr_np, device=device, dtype=ctype)
        Fr = torch.tensor(Fr_np, device=device, dtype=ctype)
        sp = torch.tensor(sp_np, device=device, dtype=ctype)

        Up = Wr.conj().transpose(0, 1).contiguous()
        Vp = Fr

        Ap = Up @ PsiR
        xp = PsiS.conj().transpose(0, 1) @ (Vp @ sp)

        A_list.append(Ap)
        x_list.append(xp)

    A_list = torch.stack(A_list, dim=0)
    x_list = torch.stack(x_list, dim=0)
    return A_list, x_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", type=str, default="c128", choices=["c64", "c128"])
    parser.add_argument("--pilot", type=str, default="selection", choices=["selection", "rademacher"])
    parser.add_argument("--snr_list", type=int, nargs="+", default=[-5, 0, 5, 10, 15])
    parser.add_argument("--Nmc", type=int, default=50)

    parser.add_argument("--P", type=int, default=32)
    parser.add_argument("--NrRF", type=int, default=64)
    parser.add_argument("--NtRF", type=int, default=4)
    parser.add_argument("--Pdim", type=float, default=1.0)

    parser.add_argument("--BxR", type=int, default=3)
    parser.add_argument("--ByR", type=int, default=3)
    parser.add_argument("--BxS", type=int, default=1)
    parser.add_argument("--ByS", type=int, default=1)
    parser.add_argument("--avoid_repeat", dest="avoid_repeat", action="store_true")
    parser.add_argument("--no_avoid_repeat", dest="avoid_repeat", action="store_false")
    parser.set_defaults(avoid_repeat=True)
    parser.add_argument("--reset_policy", type=str, default="when_exhausted",
                        choices=["never", "when_exhausted", "per_slot"])

    parser.add_argument("--Brx", type=int, default=3)
    parser.add_argument("--Bry", type=int, default=3)
    parser.add_argument("--Bsx", type=int, default=1)
    parser.add_argument("--Bsy", type=int, default=1)

    parser.add_argument("--sbl_iter", type=int, default=80)
    parser.add_argument("--probes", type=int, default=16)
    parser.add_argument("--cg_iter", type=int, default=200)
    parser.add_argument("--cg_tol", type=float, default=1e-5)
    parser.add_argument("--stop_rel", type=float, default=1e-4)
    parser.add_argument("--damping", type=float, default=0.40)
    parser.add_argument("--learn_sigma2", action="store_true")
    parser.add_argument("--trace_probes", type=int, default=8)
    parser.add_argument("--sigma2_damping", type=float, default=0.3)
    parser.add_argument("--diag_eps", type=float, default=0.0)
    parser.add_argument("--use_jacobi", action="store_true")
    parser.add_argument("--diag_print_every", type=int, default=0)

    parser.add_argument("--angles_mode", type=str, default="fixed", choices=["fixed", "random_each_mc"])
    parser.add_argument("--fixed_angles_seed", type=int, default=2026)

    parser.add_argument("--glasso_iter", type=int, default=120)
    parser.add_argument("--glasso_stop", type=float, default=1e-4)
    parser.add_argument("--lam_mult", type=float, default=0.15)

    parser.add_argument("--out_csv", type=str, default="outputs/csv/nmse_compare_mechanisms.csv")
    parser.add_argument("--out_png", type=str, default="outputs/figures/nmse_compare_mechanisms.png")
    args = parser.parse_args()

    core.seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctype = torch.complex64 if args.dtype == "c64" else torch.complex128
    print(f"[Device] {device} | dtype={ctype} | seed={args.seed}")

    fc = 30e9
    c0 = 3e8
    lamb = c0 / fc
    delta = lamb / 4

    NRx, NRy = 65, 65
    NSx, NSy = 5, 5

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
    print(f"[Modes] LR={LR}, LS={LS}, Nvar={LR * LS}")

    PsiR = core.build_wavenumber_basis_vec(NRx, NRy, delta, LRx, LRy, xiR, device=device, ctype=ctype)
    PsiS = core.build_wavenumber_basis_vec(NSx, NSy, delta, LSx, LSy, xiS, device=device, ctype=ctype)

    idxR_np, idxS_np, _ = core.design_selection_physical(
        xiS, xiR, LS, LR,
        args.NtRF, args.NrRF, args.P,
        seed=args.seed + 7,
        BxS=args.BxS, ByS=args.ByS, BxR=args.BxR, ByR=args.ByR,
        avoid_repeat=args.avoid_repeat,
        reset_policy=args.reset_policy,
    )
    idxR = torch.tensor(idxR_np, device=device, dtype=torch.long)

    if args.pilot == "selection":
        Xmat = torch.zeros((LS, args.P), device=device, dtype=ctype)
        signs = torch.sign(torch.randn(args.P, device=device, dtype=torch.float32))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        for p in range(args.P):
            for t in range(args.NtRF):
                Xmat.real[idxS_np[p, t], p] += signs[p]
        Xmat = Xmat / torch.clamp(torch.norm(Xmat, dim=0, keepdim=True), min=1e-12)
    else:
        Xmat = (2 * torch.randint(0, 2, (LS, args.P), device=device) - 1).to(torch.float32)
        Xmat = (Xmat / math.sqrt(LS)).to(ctype)

    Aop_ours = core.IndexSensing(idxR=idxR, Xmat=Xmat, Pdim=args.Pdim)
    core.check_adjoint(Aop_ours, LR, LS, device, ctype, trials=2)

    A_list, x_list = build_traditional_sensing_dft_propagating(
        PsiR=PsiR, PsiS=PsiS,
        NRx=NRx, NRy=NRy,
        NSx=NSx, NSy=NSy,
        LR=LR, LS=LS,
        P=args.P, NrRF=args.NrRF, NtRF=args.NtRF,
        pilot_mode=args.pilot,
        seed=args.seed + 99,
        delta=delta, lamb=lamb,
        device=device, ctype=ctype
    )
    Aop_trad = DenseHybridSensing(A_list=A_list, x_list=x_list, Pdim=args.Pdim)
    check_adjoint_generic(Aop_trad, LR, LS, device, ctype, trials=2)

    group_id_np, GR, GS, G = core.build_group_ids_from_xi(xiR, xiS, args.Brx, args.Bry, args.Bsx, args.Bsy)
    group_id = torch.tensor(group_id_np, device=device, dtype=torch.long)
    ones = torch.ones(LR * LS, device=device, dtype=torch.float32)
    cnt = torch.zeros(G, device=device, dtype=torch.float32).scatter_add_(0, group_id, ones).clamp_min(1.0)

    if args.angles_mode == "fixed":
        thetaR_fix, phiR_fix, thetaS_fix, phiS_fix = core.generate_fixed_angles(Nc, args.fixed_angles_seed)
    else:
        thetaR_fix = phiR_fix = thetaS_fix = phiS_fix = None

    nmse = np.zeros((2, 2, args.Nmc, len(args.snr_list)), dtype=np.float64)
    t0 = time.time()

    for mc in range(args.Nmc):
        core.seed_all(args.seed + mc)

        if args.angles_mode == "fixed":
            thetaR, phiR, thetaS, phiS = thetaR_fix, phiR_fix, thetaS_fix, phiS_fix
        else:
            thetaR = (np.random.rand(Nc) * (math.pi / 2)).astype(np.float64)
            phiR = (np.random.rand(Nc) * (2 * math.pi)).astype(np.float64)
            thetaS = (np.random.rand(Nc) * (math.pi / 2)).astype(np.float64)
            phiS = (np.random.rand(Nc) * (2 * math.pi)).astype(np.float64)

        sigma2R = core.sigma2_from_vmf_uvcells_torch(
            xiR, LRx, LRy, lamb, thetaR, phiR, alphaR, w_mix, Nu, Nv, device="cpu"
        ).to(torch.float32).to(device)
        sigma2S = core.sigma2_from_vmf_uvcells_torch(
            xiS, LSx, LSy, lamb, thetaS, phiS, alphaS, w_mix, Nu, Nv, device="cpu"
        ).to(torch.float32).to(device)

        W = (torch.randn(LR, LS, device=device) + 1j * torch.randn(LR, LS, device=device)).to(ctype) / math.sqrt(2.0)
        Ha_true = (torch.sqrt(torch.clamp(sigma2R, min=0.0)).to(ctype).unsqueeze(1) * W) * \
                  (torch.sqrt(torch.clamp(sigma2S, min=0.0)).to(ctype).unsqueeze(0))
        H_true = PsiR @ Ha_true @ PsiS.conj().transpose(0, 1)
        h_true = Ha_true.transpose(0, 1).contiguous().view(-1)

        yclean_ours = Aop_ours.forward(h_true, LR, LS)
        yclean_trad = Aop_trad.forward(h_true, LR, LS)

        M = yclean_ours.numel()
        sigPow_ours = (torch.norm(yclean_ours) ** 2).real.item() / M
        sigPow_trad = (torch.norm(yclean_trad) ** 2).real.item() / M

        def nmse_from_vec(mu_vec):
            Ha_hat = mu_vec.view(LS, LR).transpose(0, 1)
            H_hat = PsiR @ Ha_hat @ PsiS.conj().transpose(0, 1)
            err = torch.norm(H_hat - H_true) ** 2
            den = torch.clamp(torch.norm(H_true) ** 2, min=1e-12)
            return float((err / den).real.item())

        for is_, SNRdB in enumerate(args.snr_list):
            sigma2n_ours = sigPow_ours / (10 ** (SNRdB / 10.0))
            n_ours = math.sqrt(sigma2n_ours / 2) * (torch.randn(M, device=device) + 1j * torch.randn(M, device=device)).to(ctype)
            y_ours = yclean_ours + n_ours

            mu_ours, _, _ = core.sbl_block_em(
                Aop=Aop_ours, y=y_ours, sigma2=sigma2n_ours,
                group_id=group_id, cnt=cnt, G=G,
                LR=LR, LS=LS,
                max_iter=args.sbl_iter,
                cg_tol=args.cg_tol, cg_maxiter=args.cg_iter,
                probes=args.probes,
                stop_rel=args.stop_rel,
                damping=args.damping,
                learn_sigma2=args.learn_sigma2,
                trace_probes=args.trace_probes,
                sigma2_damping=args.sigma2_damping,
                diag_eps=args.diag_eps,
                use_jacobi=args.use_jacobi,
                diag_print_every=args.diag_print_every,
                collect_info=False
            )
            nmse[0, 0, mc, is_] = nmse_from_vec(mu_ours)

            sigma2n_trad = sigPow_trad / (10 ** (SNRdB / 10.0))
            n_trad = math.sqrt(sigma2n_trad / 2) * (torch.randn(M, device=device) + 1j * torch.randn(M, device=device)).to(ctype)
            y_trad = yclean_trad + n_trad

            mu_trad, _, _ = core.sbl_block_em(
                Aop=Aop_trad, y=y_trad, sigma2=sigma2n_trad,
                group_id=group_id, cnt=cnt, G=G,
                LR=LR, LS=LS,
                max_iter=args.sbl_iter,
                cg_tol=args.cg_tol, cg_maxiter=args.cg_iter,
                probes=args.probes,
                stop_rel=args.stop_rel,
                damping=args.damping,
                learn_sigma2=args.learn_sigma2,
                trace_probes=args.trace_probes,
                sigma2_damping=args.sigma2_damping,
                diag_eps=args.diag_eps,
                use_jacobi=args.use_jacobi,
                diag_print_every=args.diag_print_every,
                collect_info=False
            )
            nmse[1, 0, mc, is_] = nmse_from_vec(mu_trad)

            Nvar = LR * LS
            lam_ours = args.lam_mult * math.sqrt(sigma2n_ours) * math.sqrt(2.0 * math.log(max(Nvar, 2)))
            x_ours, _ = fista_group_lasso(
                Aop=Aop_ours, y=y_ours, group_id=group_id, G=G, LR=LR, LS=LS,
                lam=lam_ours, max_iter=args.glasso_iter, stop_rel=args.glasso_stop, L=None
            )
            nmse[0, 1, mc, is_] = nmse_from_vec(x_ours)

            lam_trad = args.lam_mult * math.sqrt(sigma2n_trad) * math.sqrt(2.0 * math.log(max(Nvar, 2)))
            x_trad, _ = fista_group_lasso(
                Aop=Aop_trad, y=y_trad, group_id=group_id, G=G, LR=LR, LS=LS,
                lam=lam_trad, max_iter=args.glasso_iter, stop_rel=args.glasso_stop, L=None
            )
            nmse[1, 1, mc, is_] = nmse_from_vec(x_trad)

            print(
                f"[MC {mc+1}/{args.Nmc}] SNR={SNRdB:>3} dB | "
                f"OURS: SBL {10*np.log10(nmse[0,0,mc,is_]+1e-300):6.2f} dB, "
                f"GLASSO {10*np.log10(nmse[0,1,mc,is_]+1e-300):6.2f} dB | "
                f"TRAD: SBL {10*np.log10(nmse[1,0,mc,is_]+1e-300):6.2f} dB, "
                f"GLASSO {10*np.log10(nmse[1,1,mc,is_]+1e-300):6.2f} dB"
            )

    nmse_mean = nmse.mean(axis=2)
    nmse_mean_db = 10 * np.log10(nmse_mean + 1e-300)

    out = np.column_stack([
        np.array(args.snr_list, dtype=np.float64),
        nmse_mean_db[0, 0, :],
        nmse_mean_db[0, 1, :],
        nmse_mean_db[1, 0, :],
        nmse_mean_db[1, 1, :],
    ])
    header = "SNR_dB,OURS_SBL_dB,OURS_FISTA_GroupLASSO_dB,TRAD_DFT_SBL_dB,TRAD_DFT_FISTA_GroupLASSO_dB"
    np.savetxt(args.out_csv, out, delimiter=",", header=header, comments="")

    plt.figure()
    plt.plot(args.snr_list, nmse_mean_db[0, 0, :], "-o", label="OURS + SBL")
    plt.plot(args.snr_list, nmse_mean_db[0, 1, :], "-o", label="OURS + GroupLASSO")
    plt.plot(args.snr_list, nmse_mean_db[1, 0, :], "-o", label="TRAD + SBL")
    plt.plot(args.snr_list, nmse_mean_db[1, 1, :], "-o", label="TRAD + GroupLASSO")
    plt.grid(True)
    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE (dB)")
    plt.title(f"Beamforming Mechanisms vs Algorithms | Nmc={args.Nmc}")
    plt.legend()
    plt.savefig(args.out_png, dpi=200)
    plt.show()

    print(f"[Done] total {time.time()-t0:.1f}s")
    print(f"[Saved] {args.out_csv}")
    print(f"[Saved] {args.out_png}")


if __name__ == "__main__":
    main()
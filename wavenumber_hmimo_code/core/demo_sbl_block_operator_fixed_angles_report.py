# demo_sbl_block_operator_fixed_angles_report.py
# ------------------------------------------------------------
# Block-SBL operator-form for your sensing:
#   y_p = sqrt(Pdim) * Qp * Ha * x_p + n_p
# Stack: y = A(Ha) + n  (operator A and adjoint A*)
#
# FIXED-ANGLES PATCH (for effG stability):
#   - Fix (thetaR,phiR,thetaS,phiS) across all Monte Carlo trials
#   - Only randomize W per MC (and noise per SNR)
#
# EXTRA DIAGNOSTICS:
#   - Plot sigma_R^2, sigma_S^2 (1D sorted + 2D grid heatmap if possible)
#   - Plot group energy Eg sorted curve for Ha_true
#
# OUTPUTS:
#   - nmse_sbl_vs_snr.csv / .mat (if scipy) / diag_dump.npz
#   - badpoint_report.txt / badpoint_table.csv / badpoint_summary.png
#   - sigma2_debug.png / group_energy_debug.png
#
# Usage example:
#   python demo_sbl_block_operator_fixed_angles_report.py --Nmc 20 --probes 16 --cg_tol 1e-5
#   python demo_sbl_block_operator_fixed_angles_report.py --fixed_angles_seed 2026 --angles_mode fixed
# ------------------------------------------------------------

import math, time, argparse, random, os
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- save results for MATLAB ---
try:
    import scipy.io as sio  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# -------------------------- Reproducibility --------------------------
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------- Mode-set (ellipse) --------------------------
def build_xi_ellipse(Lx, Ly, lamb):
    Mx = int(math.ceil(Lx / lamb))
    My = int(math.ceil(Ly / lamb))
    xi = []
    for lx in range(-Mx, Mx + 1):
        for ly in range(-My, My + 1):
            if (lx * lamb / Lx) ** 2 + (ly * lamb / Ly) ** 2 <= 1.0 + 1e-12:
                xi.append((lx, ly))
    return np.array(xi, dtype=np.int32)  # [L,2]


def build_wavenumber_basis_vec(Nx, Ny, delta, Lx, Ly, xi, device, ctype):
    """
    Vectorized Psi build:
      Psi[:,k] = kron(ay_k, ax_k),  ax_k in C^Nx, ay_k in C^Ny
    Output: (Ny*Nx) x L
    """
    xi_t = torch.tensor(xi, device=device, dtype=torch.float32)  # [L,2]
    lx = xi_t[:, 0]
    ly = xi_t[:, 1]

    nx = torch.arange(-(Nx - 1) / 2, (Nx - 1) / 2 + 1, device=device, dtype=torch.float32)
    ny = torch.arange(-(Ny - 1) / 2, (Ny - 1) / 2 + 1, device=device, dtype=torch.float32)

    two_pi = 2.0 * math.pi
    ax = torch.exp(1j * two_pi * (nx[:, None] * lx[None, :]) * delta / Lx).to(ctype) / math.sqrt(Nx)
    ay = torch.exp(1j * two_pi * (ny[:, None] * ly[None, :]) * delta / Ly).to(ctype) / math.sqrt(Ny)

    Psi3 = ay[:, None, :] * ax[None, :, :]
    Psi = Psi3.reshape(Ny * Nx, -1).contiguous()
    return Psi


# -------------------------- vMF PAS + uv-cell sigma^2 --------------------------
def vmf_pdf_stable_s2(sdot, kappa):
    sdot = torch.clamp(sdot, -1.0, 1.0)
    denom = 2.0 * math.pi * (-torch.expm1(-2.0 * kappa))
    c = kappa / torch.clamp(denom, min=1e-30)
    c = torch.where(kappa < 1e-6, torch.full_like(c, 1.0 / (4.0 * math.pi)), c)
    return c * torch.exp(kappa * (sdot - 1.0))


@torch.inference_mode()
def sigma2_from_vmf_uvcells_torch(
    xi, Lx, Ly, lamb,
    theta_mu, phi_mu, alpha, w_mix,
    Nu=9, Nv=9, device="cpu", dtype=torch.float64
):
    """
    xi: [L,2] ints. output sigma2: [L] float (real)
    """
    eps0 = 1e-12
    xi_t = torch.tensor(xi, device=device, dtype=dtype)
    L = xi_t.shape[0]
    sigma2 = torch.zeros(L, device=device, dtype=dtype)

    theta_mu = torch.tensor(theta_mu, device=device, dtype=dtype)
    phi_mu = torch.tensor(phi_mu, device=device, dtype=dtype)
    alpha = torch.tensor(alpha, device=device, dtype=dtype)
    w_mix = torch.tensor(w_mix, device=device, dtype=dtype)

    for k in range(L):
        lx = xi_t[k, 0]
        ly = xi_t[k, 1]

        u0 = (lx - 0.5) * lamb / Lx
        u1 = (lx + 0.5) * lamb / Lx
        v0 = (ly - 0.5) * lamb / Ly
        v1 = (ly + 0.5) * lamb / Ly

        u0c, u1c = max(u0, -1.0), min(u1, 1.0)
        v0c, v1c = max(v0, -1.0), min(v1, 1.0)
        if u0c >= u1c or v0c >= v1c:
            sigma2[k] = 0.0
            continue

        u = torch.linspace(u0c, u1c, Nu, device=device, dtype=dtype)
        v = torch.linspace(v0c, v1c, Nv, device=device, dtype=dtype)
        U, V = torch.meshgrid(u, v, indexing="xy")

        R2 = U * U + V * V
        mask = (R2 < 1.0 - eps0)

        W0 = torch.sqrt(torch.clamp(1.0 - R2, min=0.0))
        theta = torch.acos(torch.clamp(W0, 0.0, 1.0))
        phi = torch.atan2(V, U)
        phi = torch.where(phi < 0, phi + 2.0 * math.pi, phi)

        A2 = torch.zeros_like(U, dtype=dtype)
        for c in range(w_mix.numel()):
            sdot = (
                torch.sin(theta) * torch.sin(theta_mu[c]) * torch.cos(phi - phi_mu[c])
                + torch.cos(theta) * torch.cos(theta_mu[c])
            )
            A2 = A2 + w_mix[c] * vmf_pdf_stable_s2(sdot, alpha[c])

        integrand = torch.zeros_like(U, dtype=dtype)
        integrand[mask] = A2[mask] / torch.clamp(W0[mask], min=eps0)

        du = (u1c - u0c) / max(Nu - 1, 1)
        dv = (v1c - v0c) / max(Nv - 1, 1)
        sigma2[k] = torch.sum(integrand) * du * dv

    return sigma2


# -------------------------- Physical-neighborhood selection --------------------------
def build_blocks_from_xi_disjoint(xi, Bx, By):
    lx = xi[:, 0]
    ly = xi[:, 1]
    lx_min, lx_max = int(lx.min()), int(lx.max())
    ly_min, ly_max = int(ly.min()), int(ly.max())

    nBx = (lx_max - lx_min) // Bx + 1
    nBy = (ly_max - ly_min) // By + 1

    bx_id = (lx - lx_min) // Bx
    by_id = (ly - ly_min) // By
    gid = bx_id + by_id * nBx

    G = nBx * nBy
    tmp = [[] for _ in range(G)]
    for i, g in enumerate(gid):
        tmp[int(g)].append(i)

    return [grp for grp in tmp if len(grp) > 0]


def pick_k_from_group(group_idx, k, used_mask, avoid_repeat=True, rng=None):
    if rng is None:
        rng = np.random

    group_idx = np.asarray(list(group_idx), dtype=np.int64)
    N = int(used_mask.shape[0])

    if k <= 0:
        return []

    if group_idx.size == 0:
        if N >= k:
            return rng.choice(np.arange(N), size=k, replace=False).tolist()
        return rng.choice(np.arange(N), size=k, replace=True).tolist()

    chosen = []
    pool1 = group_idx[~used_mask[group_idx]] if avoid_repeat else group_idx

    if pool1.size >= k:
        return rng.choice(pool1, size=k, replace=False).tolist()

    if pool1.size > 0:
        chosen.extend(pool1.tolist())

    remain = k - len(chosen)
    if remain > 0:
        if avoid_repeat:
            pool2 = np.where(~used_mask)[0]
            if pool2.size == 0:
                pool2 = np.arange(N)
        else:
            pool2 = np.arange(N)

        if len(chosen) > 0:
            pool2 = np.setdiff1d(pool2, np.asarray(chosen, dtype=np.int64), assume_unique=False)

        if pool2.size >= remain:
            chosen.extend(rng.choice(pool2, size=remain, replace=False).tolist())
            remain = 0
        else:
            chosen.extend(pool2.tolist())
            remain = k - len(chosen)

    if remain > 0:
        src = group_idx if group_idx.size > 0 else np.arange(N)
        chosen.extend(rng.choice(src, size=remain, replace=True).tolist())

    chosen = chosen[:k]
    if len(set(chosen)) < k and N >= k:
        uniq = list(dict.fromkeys(chosen))
        need = k - len(uniq)
        pool = np.setdiff1d(np.arange(N), np.asarray(uniq, dtype=np.int64), assume_unique=False)
        if pool.size >= need:
            uniq.extend(rng.choice(pool, size=need, replace=False).tolist())
        else:
            uniq.extend(rng.choice(np.arange(N), size=need, replace=True).tolist())
        chosen = uniq[:k]

    return chosen


def design_selection_physical(
    xiS, xiR, LS, LR,
    NtRF, NrRF, P, seed,
    BxS, ByS, BxR, ByR,
    avoid_repeat=True,
    reset_policy="when_exhausted",
):
    rng = np.random.RandomState(seed)

    groupsR = build_blocks_from_xi_disjoint(xiR, BxR, ByR)
    groupsS = build_blocks_from_xi_disjoint(xiS, BxS, ByS)
    GR, GS = len(groupsR), len(groupsS)

    pairs = [(gr, gs) for gs in range(GS) for gr in range(GR)]
    rng.shuffle(pairs)

    usedR = np.zeros(LR, dtype=bool)
    usedS = np.zeros(LS, dtype=bool)

    idxR = np.zeros((P, NrRF), dtype=np.int64)
    idxS = np.zeros((P, NtRF), dtype=np.int64)

    for p in range(P):
        if reset_policy == "per_slot":
            usedR[:] = False
            usedS[:] = False
        elif reset_policy == "when_exhausted":
            if avoid_repeat and (LR - int(usedR.sum()) < NrRF):
                usedR[:] = False
            if avoid_repeat and (LS - int(usedS.sum()) < NtRF):
                usedS[:] = False

        gr, gs = pairs[p % len(pairs)]
        rsel = pick_k_from_group(groupsR[gr], NrRF, usedR, avoid_repeat=avoid_repeat, rng=rng)
        ssel = pick_k_from_group(groupsS[gs], NtRF, usedS, avoid_repeat=avoid_repeat, rng=rng)

        if len(rsel) != NrRF or len(ssel) != NtRF:
            raise RuntimeError(
                f"[SelectionError] p={p} got len(rsel)={len(rsel)} (need {NrRF}), "
                f"len(ssel)={len(ssel)} (need {NtRF})."
            )

        idxR[p, :] = np.asarray(rsel, dtype=np.int64)
        idxS[p, :] = np.asarray(ssel, dtype=np.int64)
        usedR[idxR[p, :]] = True
        usedS[idxS[p, :]] = True

    meta = {"GR": GR, "GS": GS, "pairs": pairs, "reset_policy": reset_policy, "avoid_repeat": avoid_repeat}
    return idxR, idxS, meta


def coverage_report(idxR_np, idxS_np, LR, LS):
    P, _ = idxR_np.shape
    per_slot_R = np.mean([len(np.unique(idxR_np[p, :])) / LR for p in range(P)])
    overall_R = len(np.unique(idxR_np.reshape(-1))) / LR
    per_slot_S = np.mean([len(np.unique(idxS_np[p, :])) / LS for p in range(P)])
    overall_S = len(np.unique(idxS_np.reshape(-1))) / LS
    return per_slot_R, overall_R, per_slot_S, overall_S


# -------------------------- Group IDs on vec(Ha) (column-major) --------------------------
def build_group_ids_from_xi(xiR, xiS, Brx, Bry, Bsx, Bsy):
    lxR, lyR = xiR[:, 0], xiR[:, 1]
    lxS, lyS = xiS[:, 0], xiS[:, 1]

    bxR = (lxR - lxR.min()) // Brx
    byR = (lyR - lyR.min()) // Bry
    _, gR_id = np.unique(np.stack([bxR, byR], axis=1), axis=0, return_inverse=True)
    GR = int(gR_id.max()) + 1

    bxS = (lxS - lxS.min()) // Bsx
    byS = (lyS - lyS.min()) // Bsy
    _, gS_id = np.unique(np.stack([bxS, byS], axis=1), axis=0, return_inverse=True)
    GS = int(gS_id.max()) + 1

    LR = xiR.shape[0]
    LS = xiS.shape[0]
    G = GR * GS

    group_id = np.zeros(LR * LS, dtype=np.int64)
    for j in range(LS):
        gid_col = gR_id + (gS_id[j] * GR)
        group_id[j * LR:(j + 1) * LR] = gid_col

    return group_id, GR, GS, G


# -------------------------- Index-based operator A, A* (FAST) --------------------------
class IndexSensing:
    """
    y = A(h) where:
      Ha = unvec(h) col-major
      Z = Ha @ Xmat   [LR,P]
      y contains Z[idxR[p], p] for each p, flattened length P*NrRF
    """
    def __init__(self, idxR: torch.Tensor, Xmat: torch.Tensor, Pdim: float = 1.0):
        self.idxR = idxR
        self.Xmat = Xmat
        self.Pdim = float(Pdim)
        self.P, self.NrRF = idxR.shape
        self.LS, P2 = Xmat.shape
        assert P2 == self.P

        self.BconjT = Xmat.conj().transpose(0, 1).contiguous()  # [P,LS]

        # linear indices for Z.reshape(-1) (row-major): index = row*P + col
        p = torch.arange(self.P, device=idxR.device, dtype=torch.long).view(self.P, 1).expand(self.P, self.NrRF)
        self.lin_idx = (idxR * self.P + p).reshape(-1)
        self.rows_flat = idxR.reshape(-1)

        self._Gbuf = None

    @staticmethod
    def vec_to_Ha(hvec, LR, LS):
        return hvec.view(LS, LR).transpose(0, 1)  # [LR,LS]

    @staticmethod
    def Ha_to_vec(Ha):
        return Ha.transpose(0, 1).contiguous().view(-1)

    @torch.inference_mode()
    def forward(self, hvec, LR, LS):
        Ha = self.vec_to_Ha(hvec, LR, LS)
        Z = Ha @ self.Xmat  # [LR,P]
        zflat = Z.reshape(-1)
        gathered = zflat[self.lin_idx]
        return math.sqrt(self.Pdim) * gathered

    @torch.inference_mode()
    def adjoint(self, v, LR, LS):
        V = v.view(self.P, self.NrRF)  # [P,NrRF]
        scale = math.sqrt(self.Pdim)

        outer = (V[:, :, None] * self.BconjT[:, None, :]) * scale  # [P,NrRF,LS]
        outer_flat = outer.reshape(-1, LS)  # [(P*NrRF),LS]

        if (self._Gbuf is None) or (self._Gbuf.shape != (LR, LS)) or (self._Gbuf.dtype != v.dtype) or (self._Gbuf.device != v.device):
            self._Gbuf = torch.zeros((LR, LS), device=v.device, dtype=v.dtype)
        else:
            self._Gbuf.zero_()

        self._Gbuf.index_add_(0, self.rows_flat, outer_flat)
        return self.Ha_to_vec(self._Gbuf)


# -------------------------- CG (Hermitian PD) --------------------------
@torch.inference_mode()
def cg(matvec, b, x0=None, tol=1e-6, maxiter=200, M_inv=None, return_info=False):
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    bnorm = torch.sqrt(torch.real(torch.vdot(b, b)).clamp_min(1e-30))

    if M_inv is not None:
        z = M_inv(r)
        p = z.clone()
        rz = torch.real(torch.vdot(r, z))
    else:
        p = r.clone()
        rz = torch.real(torch.vdot(r, r))

    rnorm0 = torch.sqrt(torch.real(torch.vdot(r, r)).clamp_min(0.0))
    relres = float((rnorm0 / bnorm).item())
    if relres < tol:
        if return_info:
            return x, {"iters": 0, "relres": relres}
        return x

    it = 0
    for it in range(1, maxiter + 1):
        Ap = matvec(p)
        pAp = torch.real(torch.vdot(p, Ap)).clamp_min(1e-30)
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        rnorm = torch.sqrt(torch.real(torch.vdot(r, r)).clamp_min(0.0))
        relres = float((rnorm / bnorm).item())
        if relres < tol:
            break

        if M_inv is not None:
            z_new = M_inv(r)
            rz_new = torch.real(torch.vdot(r, z_new))
            beta = rz_new / rz.clamp_min(1e-30)
            p = z_new + beta * p
            z = z_new
            rz = rz_new
        else:
            rz_new = torch.real(torch.vdot(r, r))
            beta = rz_new / rz.clamp_min(1e-30)
            p = r + beta * p
            rz = rz_new

    if return_info:
        return x, {"iters": it, "relres": float(relres)}
    return x


@torch.inference_mode()
def complex_rademacher(n, device, ctype):
    a = torch.sign(torch.randn(n, device=device))
    b = torch.sign(torch.randn(n, device=device))
    return ((a + 1j * b).to(ctype)) / math.sqrt(2.0)


# -------------------------- Diagnostics helpers --------------------------
@torch.inference_mode()
def check_adjoint(Aop: IndexSensing, LR, LS, device, ctype, trials=2):
    for t in range(trials):
        h = (torch.randn(LR * LS, device=device) + 1j * torch.randn(LR * LS, device=device)).to(ctype)
        v = (torch.randn(Aop.P * Aop.NrRF, device=device) + 1j * torch.randn(Aop.P * Aop.NrRF, device=device)).to(ctype)
        Ah = Aop.forward(h, LR, LS)
        Atv = Aop.adjoint(v, LR, LS)
        lhs = torch.vdot(Ah, v)
        rhs = torch.vdot(h, Atv)
        rel = (lhs - rhs).abs() / (lhs.abs() + rhs.abs() + 1e-12)
        print(f"[AdjointTest] trial {t + 1}: rel_err={rel.item():.2e}")


@torch.inference_mode()
def group_energy_report_and_curve(Ha_true: torch.Tensor, group_id: torch.Tensor, G: int, topk=(5, 10, 20)):
    """
    Return:
      - dict metrics: eff_groups, topK_ratio...
      - Eg_sorted (cpu numpy)
    """
    h = Ha_true.transpose(0, 1).contiguous().view(-1)  # vec col-major
    Ei = (torch.abs(h) ** 2).float()
    Eg = torch.zeros(G, device=Ei.device, dtype=torch.float32).scatter_add_(0, group_id, Ei)
    Eg_sum = Eg.sum().clamp_min(1e-30)
    p = (Eg / Eg_sum).clamp_min(0.0)
    eff = (1.0 / torch.sum(p * p).clamp_min(1e-30)).item()
    Eg_sorted = torch.sort(Eg, descending=True).values
    out = {"eff_groups": eff}
    for K in topk:
        K = min(K, G)
        out[f"top{K}_ratio"] = (Eg_sorted[:K].sum() / Eg_sum).item()
    return out, Eg_sorted.detach().cpu().numpy(), Eg_sum.detach().cpu().item()


def plot_sigma2_and_groups(
    xiR, xiS,
    sigma2R: torch.Tensor,
    sigma2S: torch.Tensor,
    Eg_sorted: np.ndarray,
    Eg_sum: float,
    out_sigma_png="sigma2_debug.png",
    out_group_png="group_energy_debug.png",
    title_prefix=""
):
    """
    Make 2 pngs:
      - sigma2_debug.png: 1D sorted curves + (optional) 2D heatmap by (lx,ly)
      - group_energy_debug.png: Eg_sorted cumulative energy curve
    """
    # ----- sigma2 1D sorted -----
    sR = sigma2R.detach().cpu().numpy().astype(np.float64)
    sS = sigma2S.detach().cpu().numpy().astype(np.float64)

    sR_sort = np.sort(sR)[::-1]
    sS_sort = np.sort(sS)[::-1]

    plt.figure()
    plt.semilogy(np.maximum(sR_sort, 1e-30), label=r"$\sigma_R^2$ sorted")
    plt.semilogy(np.maximum(sS_sort, 1e-30), label=r"$\sigma_S^2$ sorted")
    plt.grid(True, which="both")
    plt.xlabel("index (sorted)")
    plt.ylabel("value (log scale)")
    plt.title(f"{title_prefix} sigma^2 sorted")
    plt.legend()
    plt.savefig(out_sigma_png, dpi=200)
    plt.close()

    # ----- group energy cumulative -----
    Eg_sorted = np.maximum(Eg_sorted.astype(np.float64), 0.0)
    cdf = np.cumsum(Eg_sorted) / max(Eg_sum, 1e-30)

    plt.figure()
    plt.plot(cdf, "-")
    plt.grid(True)
    plt.xlabel("group rank (sorted)")
    plt.ylabel("cumulative energy ratio")
    plt.title(f"{title_prefix} group energy cumulative")
    plt.savefig(out_group_png, dpi=200)
    plt.close()


# -------------------------- Block-SBL (group-ARD) EM --------------------------
@torch.inference_mode()
def sbl_block_em(
    Aop: IndexSensing,
    y: torch.Tensor,
    sigma2: float,
    group_id: torch.Tensor,
    cnt: torch.Tensor,
    G: int,
    LR: int, LS: int,
    max_iter=80,
    cg_tol=1e-6, cg_maxiter=200,
    probes=64,
    gamma_floor=1e-12,
    stop_rel=1e-4,
    damping=0.4,
    learn_sigma2=False,
    trace_probes=8,
    sigma2_damping=0.3,
    diag_eps=0.0,
    use_jacobi=False,
    diag_print_every=0,
    collect_info=True,
    gamma0: torch.Tensor | None = None,
    screened_mask: torch.Tensor | None = None,
):
    device = y.device
    ctype = y.dtype
    N = LR * LS
    M = y.numel()

    # # init gamma
    # x0 = Aop.adjoint(y, LR, LS)
    # v0 = (torch.abs(x0) ** 2).float()
    # sumv = torch.zeros(G, device=device, dtype=torch.float32).scatter_add_(0, group_id, v0)
    # gamma_g = (sumv / cnt).clamp_min(gamma_floor)
    # gamma_i = gamma_g[group_id]  # float32

    # init gamma
    if gamma0 is None:
        x0 = Aop.adjoint(y, LR, LS)
        v0 = (torch.abs(x0) ** 2).float()
        sumv = torch.zeros(G, device=device, dtype=torch.float32).scatter_add_(0, group_id, v0)
        gamma_g = (sumv / cnt).clamp_min(gamma_floor)
    else:
        gamma_g = gamma0.to(device=device, dtype=torch.float32).clone().clamp_min(gamma_floor)

    if screened_mask is not None:
        screened_mask = screened_mask.to(device=device, dtype=torch.bool)
        gamma_g = torch.where(
            screened_mask,
            gamma_g,
            torch.full_like(gamma_g, gamma_floor)
        )

    gamma_i = gamma_g[group_id]

    # probes
    Z = torch.stack([complex_rademacher(N, device, ctype) for _ in range(probes)], dim=1)
    u0 = torch.zeros(M, device=device, dtype=ctype)
    u0_probe = torch.zeros((M, probes), device=device, dtype=ctype)
    u0_trace = torch.zeros((M, trace_probes), device=device, dtype=ctype)

    Minv = None
    if use_jacobi:
        z = complex_rademacher(N, device, ctype)
        Az = Aop.forward(torch.sqrt(gamma_i).to(ctype) * z, LR, LS)
        diagC = (sigma2 + (torch.abs(Az) ** 2).real).clamp_min(1e-12)
        Minv = lambda r: r / diagC.to(r.dtype)

    relchg = []
    neg_ratio_list = []
    cg_u_iters = []
    cg_u_relres = []
    cg_probe_avg_iters = []
    cg_probe_avg_relres = []
    gamma_minmax = []
    active_groups = []
    sigma2_hist = []

    for it in range(max_iter):
        gamma_old = gamma_g.clone()
        gamma_i_c = gamma_i.to(ctype)

        def C_mv(v):
            tmp = Aop.adjoint(v, LR, LS)
            tmp = gamma_i_c * tmp
            return sigma2 * v + Aop.forward(tmp, LR, LS)

        # u = C^{-1} y
        u, info_u = cg(C_mv, y, x0=u0, tol=cg_tol, maxiter=cg_maxiter, M_inv=Minv, return_info=True)
        u0 = u
        cg_u_iters.append(info_u["iters"])  # type: ignore
        cg_u_relres.append(info_u["relres"])

        mu = gamma_i_c * Aop.adjoint(u, LR, LS)

        # Hutchinson diag(K)
        diagK = torch.zeros(N, device=device, dtype=ctype)
        probe_iters = []
        probe_relres = []
        for k in range(probes):
            z = Z[:, k]
            w = Aop.forward(gamma_i_c * z, LR, LS)
            xk, info_k = cg(C_mv, w, x0=u0_probe[:, k], tol=cg_tol, maxiter=cg_maxiter, M_inv=Minv, return_info=True)
            u0_probe[:, k] = xk
            probe_iters.append(info_k["iters"])
            probe_relres.append(info_k["relres"])
            Kz = gamma_i_c * Aop.adjoint(xk, LR, LS)
            diagK += torch.conj(z) * Kz
        diagK = diagK / probes

        diagK_real = torch.real(diagK).to(gamma_i.dtype)  # float32
        neg_ratio = (diagK_real > gamma_i).float().mean().item()
        neg_ratio_list.append(neg_ratio)
        cg_probe_avg_iters.append(float(np.mean(probe_iters)))
        cg_probe_avg_relres.append(float(np.mean(probe_relres)))

        # diagSigma is REAL (fix warning + correct)
        diagSigma = gamma_i - diagK_real
        if diag_eps > 0:
            diagSigma = torch.maximum(diagSigma, diag_eps * gamma_i)
        else:
            diagSigma = diagSigma.clamp_min(0.0)

        Emu = (torch.abs(mu) ** 2).to(diagSigma.dtype)  # real
        Ei = (Emu + diagSigma).to(torch.float32)

        sumEi = torch.zeros(G, device=device, dtype=torch.float32).scatter_add_(0, group_id, Ei)
        gamma_new = (sumEi / cnt).clamp_min(gamma_floor)

        gamma_g = (1.0 - damping) * gamma_g + damping * gamma_new
        gamma_g = gamma_g.clamp_min(gamma_floor)

        if screened_mask is not None:
            gamma_g = torch.where(
                screened_mask,
                gamma_g,
                torch.full_like(gamma_g, gamma_floor)
        )

        gamma_i = gamma_g[group_id]

        gamma_i = gamma_g[group_id]

        gmin = gamma_g.min().item()
        gmax = gamma_g.max().item()
        gamma_minmax.append((gmin, gmax))
        active = int((gamma_g > (10.0 * gamma_floor)).sum().item())
        active_groups.append(active)

        if learn_sigma2:
            yhat = Aop.forward(mu, LR, LS)
            res2 = torch.real(torch.vdot(y - yhat, y - yhat)).item()

            tr_inv = 0.0
            for k in range(trace_probes):
                q = complex_rademacher(M, device, ctype)
                r, _ = cg(C_mv, q, x0=u0_trace[:, k], tol=cg_tol, maxiter=cg_maxiter, M_inv=Minv, return_info=True)
                u0_trace[:, k] = r
                tr_inv += torch.real(torch.vdot(q, r)).item()
            tr_inv /= max(trace_probes, 1)

            sigma2_new = max(1e-18, (res2 + sigma2 * (M - sigma2 * tr_inv)) / M)
            sigma2 = (1.0 - sigma2_damping) * sigma2 + sigma2_damping * sigma2_new

        sigma2_hist.append(float(sigma2))

        rel = torch.norm(gamma_g - gamma_old) / torch.clamp(torch.norm(gamma_old), min=1e-12)
        relchg.append(rel.item())

        if diag_print_every and ((it + 1) % diag_print_every == 0):
            print(
                f"[SBL] it={it+1:03d} relchg={rel.item():.2e} sigma2={sigma2:.2e} "
                f"neg={neg_ratio:.3f} cg_u(it={info_u['iters']},rr={info_u['relres']:.1e}) "
                f"cg_probe(it~{np.mean(probe_iters):.1f},rr~{np.mean(probe_relres):.1e}) "
                f"gamma[min,max]=[{gmin:.1e},{gmax:.1e}] activeG={active}"
            )

        if it >= 4 and rel.item() < stop_rel:
            break

    info = {}
    if collect_info:
        info = {
            "relchg": relchg,
            "sigma2": float(sigma2),
            "sigma2_hist": sigma2_hist,
            "neg_ratio": neg_ratio_list,
            "cg_u_iters": cg_u_iters,
            "cg_u_relres": cg_u_relres,
            "cg_probe_avg_iters": cg_probe_avg_iters,
            "cg_probe_avg_relres": cg_probe_avg_relres,
            "gamma_minmax": gamma_minmax,
            "active_groups": active_groups,
        }
    return mu, gamma_g, info


# -------------------------- Bad-point Report --------------------------
def _safe_last(x, default=np.nan):
    try:
        return float(x[-1])
    except Exception:
        return float(default)


def _safe_mean(x, default=np.nan):
    try:
        return float(np.mean(x))
    except Exception:
        return float(default)


def _safe_max(x, default=np.nan):
    try:
        return float(np.max(x))
    except Exception:
        return float(default)


def summarize_info_dict(info: dict):
    relchg = np.array(info.get("relchg", []), dtype=float)
    neg = np.array(info.get("neg_ratio", []), dtype=float)
    cg_u_rr = np.array(info.get("cg_u_relres", []), dtype=float)
    cg_p_rr = np.array(info.get("cg_probe_avg_relres", []), dtype=float)
    cg_u_it = np.array(info.get("cg_u_iters", []), dtype=float)
    cg_p_it = np.array(info.get("cg_probe_avg_iters", []), dtype=float)
    active = np.array(info.get("active_groups", []), dtype=float)
    gminmax = info.get("gamma_minmax", [])
    if isinstance(gminmax, list) and len(gminmax) > 0:
        gmin = np.array([x[0] for x in gminmax], dtype=float)
        gmax = np.array([x[1] for x in gminmax], dtype=float)
    else:
        gmin = np.array([], dtype=float)
        gmax = np.array([], dtype=float)

    out = {
        "iters": int(len(relchg)),
        "relchg_last": _safe_last(relchg),
        "neg_max": _safe_max(neg),
        "neg_last": _safe_last(neg),
        "cg_u_relres_last": _safe_last(cg_u_rr),
        "cg_probe_relres_mean": _safe_mean(cg_p_rr),
        "cg_u_iters_mean": _safe_mean(cg_u_it),
        "cg_probe_iters_mean": _safe_mean(cg_p_it),
        "active_last": _safe_last(active),
        "gamma_min_last": _safe_last(gmin),
        "gamma_max_last": _safe_last(gmax),
    }
    return out


def infer_root_causes(sp_effG, sp_top10, sbl_sum, nmse_db, snr_db, cfg):
    causes = []
    notes = []

    # NOTE: with fixed angles, "sparsity mismatch" should be stable;
    # set thresholds conservatively; you can tune later.
    if (sp_effG > 80.0) and (sp_top10 < 0.55):
        causes.append("Sparsity mismatch (group not sparse)")
        notes.append(f"effG={sp_effG:.1f} high & top10={sp_top10:.3f} low")

    if (sbl_sum["cg_u_relres_last"] > 5e-4) or (sbl_sum["cg_probe_relres_mean"] > 5e-4):
        causes.append("CG not converged enough")
        notes.append(f"cg_u_rr={sbl_sum['cg_u_relres_last']:.1e}, cg_probe_rr={sbl_sum['cg_probe_relres_mean']:.1e}")

    if sbl_sum["neg_max"] > 0.15:
        causes.append("Hutchinson diag(K) overshoot")
        notes.append(f"neg_max={sbl_sum['neg_max']:.3f} (try probes↑ / precond / diag_eps)")

    if (sbl_sum["active_last"] < 0.05 * cfg["G"]) and (nmse_db > -10.0):
        causes.append("Gamma collapse / over-pruning")
        notes.append(f"active_last={sbl_sum['active_last']:.0f} too small vs G={cfg['G']}")

    if len(causes) == 0:
        causes.append("Unclear (needs deeper check)")
        notes.append("No obvious diagnostic threshold triggered")

    headline = " | ".join(causes)
    return headline, notes


def write_badpoint_report(
    path_txt,
    path_csv,
    path_png,
    snr_list,
    nmse_mc_db,
    diag_dump,
    top_bad,
    bad_nmse_db,
    cfg
):
    Nmc, Ns = nmse_mc_db.shape
    snr_list = list(snr_list)

    flat = []
    for mc in range(Nmc):
        for is_ in range(Ns):
            flat.append((nmse_mc_db[mc, is_], mc, is_))
    flat.sort(reverse=True, key=lambda x: x[0])
    worst = flat[:max(1, top_bad)]

    max_snr = max(snr_list)
    is_max = snr_list.index(max_snr)
    bad_at_max = []
    for mc in range(Nmc):
        if nmse_mc_db[mc, is_max] > bad_nmse_db:
            bad_at_max.append((nmse_mc_db[mc, is_max], mc))
    bad_at_max.sort(reverse=True, key=lambda x: x[0])

    rows = []
    for rank, (nmse_db, mc, is_) in enumerate(worst, start=1):
        sp_effG = float(diag_dump["sparsity_eff_groups"][mc])
        sp_top10 = float(diag_dump["top10_ratio"][mc])
        info = diag_dump["per_snr_info"][mc][is_]
        sbl_sum = summarize_info_dict(info)
        headline, notes = infer_root_causes(sp_effG, sp_top10, sbl_sum, nmse_db, snr_list[is_], cfg)

        rows.append({
            "rank": rank,
            "mc": mc + 1,
            "snr_db": snr_list[is_],
            "nmse_db": float(nmse_db),
            "effG": sp_effG,
            "top10_ratio": sp_top10,
            "iters": sbl_sum["iters"],
            "relchg_last": sbl_sum["relchg_last"],
            "neg_max": sbl_sum["neg_max"],
            "cg_u_relres_last": sbl_sum["cg_u_relres_last"],
            "cg_probe_relres_mean": sbl_sum["cg_probe_relres_mean"],
            "active_last": sbl_sum["active_last"],
            "cause": headline,
            "notes": "; ".join(notes)
        })

    import csv
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("=== Bad-point Attribution Report (Block-SBL operator) ===\n")
        f.write(f"Nmc={Nmc}, Ns={Ns}, top_bad={top_bad}, bad_nmse_db(threshold@maxSNR)={bad_nmse_db}\n")
        f.write(f"Config: G={cfg['G']}, probes={cfg['probes']}, cg_tol={cfg['cg_tol']}, cg_iter={cfg['cg_iter']}, diag_eps={cfg['diag_eps']}, use_jacobi={cfg['use_jacobi']}\n\n")

        f.write("--- Worst points (highest NMSE dB) ---\n")
        for r in rows:
            f.write(f"[#{r['rank']}] MC={r['mc']:>3}  SNR={r['snr_db']:>4} dB  NMSE={r['nmse_db']:.2f} dB\n")
            f.write(f"     sparsity: effG={r['effG']:.1f}, top10={r['top10_ratio']:.3f}\n")
            f.write(f"     solve: iters={r['iters']}, relchg_last={r['relchg_last']:.2e}, cg_u_rr={r['cg_u_relres_last']:.1e}, cg_probe_rr~={r['cg_probe_relres_mean']:.1e}\n")
            f.write(f"     hutch: neg_max={r['neg_max']:.3f}, active_last={r['active_last']:.0f}\n")
            f.write(f"     CAUSE: {r['cause']}\n")
            f.write(f"     notes: {r['notes']}\n\n")

        f.write("--- Bad at max SNR ---\n")
        f.write(f"maxSNR={max_snr} dB, condition: NMSE(maxSNR) > {bad_nmse_db} dB\n")
        if len(bad_at_max) == 0:
            f.write("None.\n")
        else:
            for nmse_db, mc in bad_at_max[:min(50, len(bad_at_max))]:
                sp_effG = float(diag_dump["sparsity_eff_groups"][mc])
                sp_top10 = float(diag_dump["top10_ratio"][mc])
                info = diag_dump["per_snr_info"][mc][is_max]
                sbl_sum = summarize_info_dict(info)
                headline, notes = infer_root_causes(sp_effG, sp_top10, sbl_sum, nmse_db, max_snr, cfg)

                f.write(f"MC={mc+1:>3} NMSE@{max_snr}dB={nmse_db:.2f} dB | {headline}\n")
                f.write(f"   effG={sp_effG:.1f}, top10={sp_top10:.3f}, neg_max={sbl_sum['neg_max']:.3f}, cg_u_rr={sbl_sum['cg_u_relres_last']:.1e}\n")
                f.write(f"   notes: {'; '.join(notes)}\n")

    try:
        nmse_all = nmse_mc_db.reshape(-1)
        plt.figure()
        plt.hist(nmse_all, bins=50)
        plt.grid(True)
        plt.xlabel("NMSE (dB)")
        plt.ylabel("count")
        plt.title("Distribution of NMSE over (MC,SNR)")
        plt.savefig(path_png, dpi=200)
        plt.close()
    except Exception:
        pass


# -------------------------- Fixed angles helper --------------------------
def generate_fixed_angles(Nc: int, seed: int):
    """
    Generate deterministic (theta,phi) for R/S side.
    theta in [0, pi/2], phi in [0, 2pi)
    """
    rng = np.random.RandomState(seed)
    thetaR = (rng.rand(Nc) * (math.pi / 2)).astype(np.float64)
    phiR = (rng.rand(Nc) * (2 * math.pi)).astype(np.float64)
    thetaS = (rng.rand(Nc) * (math.pi / 2)).astype(np.float64)
    phiS = (rng.rand(Nc) * (2 * math.pi)).astype(np.float64)
    return thetaR, phiR, thetaS, phiS


# -------------------------- Main --------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", type=str, default="c128", choices=["c64", "c128"])
    parser.add_argument("--pilot", type=str, default="rademacher", choices=["selection", "rademacher"])

    parser.add_argument("--snr_list", type=int, nargs="+", default=[-5, 0, 5, 10, 15])
    parser.add_argument("--Nmc", type=int, default=200)

    # sensing
    parser.add_argument("--P", type=int, default=64)
    parser.add_argument("--NrRF", type=int, default=64)
    parser.add_argument("--NtRF", type=int, default=4)
    parser.add_argument("--Pdim", type=float, default=1.0)

    # selection design
    parser.add_argument("--BxR", type=int, default=3)
    parser.add_argument("--ByR", type=int, default=3)
    parser.add_argument("--BxS", type=int, default=1)
    parser.add_argument("--ByS", type=int, default=1)
    parser.add_argument("--avoid_repeat", dest="avoid_repeat", action="store_true")
    parser.add_argument("--no_avoid_repeat", dest="avoid_repeat", action="store_false")
    parser.set_defaults(avoid_repeat=True)
    parser.add_argument("--reset_policy", type=str, default="when_exhausted",
                        choices=["never", "when_exhausted", "per_slot"])

    # group-ARD blocks
    parser.add_argument("--Brx", type=int, default=3)
    parser.add_argument("--Bry", type=int, default=3)
    parser.add_argument("--Bsx", type=int, default=1)
    parser.add_argument("--Bsy", type=int, default=1)

    # SBL params
    parser.add_argument("--sbl_iter", type=int, default=200)
    parser.add_argument("--probes", type=int, default=16)
    parser.add_argument("--cg_iter", type=int, default=200)
    parser.add_argument("--cg_tol", type=float, default=1e-5)
    parser.add_argument("--stop_rel", type=float, default=1e-4)
    parser.add_argument("--damping", type=float, default=0.40)
    parser.add_argument("--learn_sigma2", action="store_true")
    parser.add_argument("--trace_probes", type=int, default=16)
    parser.add_argument("--sigma2_damping", type=float, default=0.3)

    # robustness
    parser.add_argument("--diag_eps", type=float, default=0.0)
    parser.add_argument("--use_jacobi", action="store_true")
    parser.add_argument("--diag_print_every", type=int, default=0)

    # reporting knobs
    parser.add_argument("--bad_nmse_db", type=float, default=-5.0)
    parser.add_argument("--top_bad", type=int, default=10)

    # FIXED ANGLES controls
    parser.add_argument("--angles_mode", type=str, default="fixed", choices=["fixed", "random_each_mc"],
                        help="fixed: fix theta/phi across all MC; random_each_mc: old behavior")
    parser.add_argument("--fixed_angles_seed", type=int, default=2026,
                        help="seed used to generate fixed theta/phi when angles_mode=fixed")

    # diagnostics plot control
    parser.add_argument("--plot_debug", action="store_true",
                        help="If set, will plot sigma^2 and group energy for MC=1 only")

    # outputs
    parser.add_argument("--out_mat", type=str, default="nmse_sbl_vs_snr.mat")
    parser.add_argument("--out_csv", type=str, default="nmse_sbl_vs_snr.csv")
    parser.add_argument("--out_npz", type=str, default="diag_dump.npz")

    parser.add_argument("--report_txt", type=str, default="badpoint_report.txt")
    parser.add_argument("--report_csv", type=str, default="badpoint_table.csv")
    parser.add_argument("--report_png", type=str, default="badpoint_summary.png")

    args = parser.parse_args()
    seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctype = torch.complex64 if args.dtype == "c64" else torch.complex128
    print(f"[Device] {device} | dtype={ctype} | seed={args.seed}")

    # ---------------- System params ----------------
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

    # vMF channel params
    Nc = 2
    alpha0 = 540.0
    w_mix = np.ones(Nc, dtype=np.float64) / Nc
    Nu, Nv = 9, 9

    # ---------------- Build mode sets ----------------
    xiR = build_xi_ellipse(LRx, LRy, lamb)
    xiS = build_xi_ellipse(LSx, LSy, lamb)
    LR = xiR.shape[0]
    LS = xiS.shape[0]
    print(f"[Modes] LR={LR}, LS={LS}, Nvar={LR*LS}")

    # ---------------- Build bases for NMSE on H ----------------
    print("[Build] PsiR/PsiS ...")
    PsiR = build_wavenumber_basis_vec(NRx, NRy, delta, LRx, LRy, xiR, device=device, ctype=ctype)
    PsiS = build_wavenumber_basis_vec(NSx, NSy, delta, LSx, LSy, xiS, device=device, ctype=ctype)

    # ---------------- Design selection ----------------
    idxR_np, idxS_np, _ = design_selection_physical(
        xiS, xiR, LS, LR,
        args.NtRF, args.NrRF, args.P,
        seed=args.seed + 7,
        BxS=args.BxS, ByS=args.ByS, BxR=args.BxR, ByR=args.ByR,
        avoid_repeat=args.avoid_repeat,
        reset_policy=args.reset_policy,
    )
    idxR = torch.tensor(idxR_np, device=device, dtype=torch.long)

    perR, allR, perS, allS = coverage_report(idxR_np, idxS_np, LR, LS)
    print(f"[Selection] avoid_repeat={args.avoid_repeat} reset_policy={args.reset_policy}")
    print(f"[Coverage-R] per-slot avg={perR:.3f} | overall={allR:.3f}")
    print(f"[Coverage-S] per-slot avg={perS:.3f} | overall={allS:.3f}")

    # Tx probing matrix Xmat: [LS,P]
    if args.pilot == "selection":
        Xmat = torch.zeros((LS, args.P), device=device, dtype=ctype)
        signs = torch.sign(torch.randn(args.P, device=device, dtype=torch.float32))
        for p in range(args.P):
            for t in range(args.NtRF):
                Xmat.real[idxS_np[p, t], p] += signs[p]
        Xmat = Xmat / torch.clamp(torch.norm(Xmat, dim=0, keepdim=True), min=1e-12)
    else:
        Xmat = (2 * torch.randint(0, 2, (LS, args.P), device=device) - 1).to(torch.float32)
        Xmat = (Xmat / math.sqrt(LS)).to(ctype)

    Aop = IndexSensing(idxR=idxR, Xmat=Xmat, Pdim=args.Pdim)
    check_adjoint(Aop, LR, LS, device, ctype, trials=2)

    # ---------------- group ids + counts ----------------
    group_id_np, GR, GS, G = build_group_ids_from_xi(xiR, xiS, args.Brx, args.Bry, args.Bsx, args.Bsy)
    group_id = torch.tensor(group_id_np, device=device, dtype=torch.long)
    ones = torch.ones(LR * LS, device=device, dtype=torch.float32)
    cnt = torch.zeros(G, device=device, dtype=torch.float32).scatter_add_(0, group_id, ones).clamp_min(1.0)
    print(f"[Groups] GR={GR}, GS={GS}, G={G}")

    # ---------------- Fixed angles (global) ----------------
    if args.angles_mode == "fixed":
        thetaR_fix, phiR_fix, thetaS_fix, phiS_fix = generate_fixed_angles(Nc, args.fixed_angles_seed)
        print(f"[Angles] mode=fixed | fixed_angles_seed={args.fixed_angles_seed}")
    else:
        thetaR_fix = phiR_fix = thetaS_fix = phiS_fix = None
        print("[Angles] mode=random_each_mc (old behavior)")

    # ---------------- Monte Carlo ----------------
    Ns = len(args.snr_list)
    nmse = np.zeros((args.Nmc, Ns), dtype=np.float64)

    diag_dump = {
        "mc_idx": [],
        "sigPow": [],
        "sparsity_eff_groups": [],
        "top10_ratio": [],
        "per_snr_info": [],
    }

    t0 = time.time()

    # precompute alpha arrays (constant)
    alphaR = (alpha0 * np.ones(Nc)).astype(np.float64)
    alphaS = (alpha0 * np.ones(Nc)).astype(np.float64)

    for mc in range(args.Nmc):
        seed_all(args.seed + mc)

        # ---- angles: fixed vs random per MC ----
        if args.angles_mode == "fixed":
            thetaR = thetaR_fix
            phiR = phiR_fix
            thetaS = thetaS_fix
            phiS = phiS_fix
        else:
            thetaR = (np.random.rand(Nc) * (math.pi / 2)).astype(np.float64)
            phiR = (np.random.rand(Nc) * (2 * math.pi)).astype(np.float64)
            thetaS = (np.random.rand(Nc) * (math.pi / 2)).astype(np.float64)
            phiS = (np.random.rand(Nc) * (2 * math.pi)).astype(np.float64)

        # ---- sigma^2 depends only on angles (fixed => constant across MC) ----
        sigma2R = sigma2_from_vmf_uvcells_torch(
            xiR, LRx, LRy, lamb, thetaR, phiR, alphaR, w_mix, Nu, Nv, device="cpu"
        ).to(torch.float32).to(device)
        sigma2S = sigma2_from_vmf_uvcells_torch(
            xiS, LSx, LSy, lamb, thetaS, phiS, alphaS, w_mix, Nu, Nv, device="cpu"
        ).to(torch.float32).to(device)

        # ---- ONLY randomize W across MC ----
        W = (torch.randn(LR, LS, device=device) + 1j * torch.randn(LR, LS, device=device)).to(ctype) / math.sqrt(2.0)
        Ha_true = (torch.sqrt(torch.clamp(sigma2R, min=0.0)).to(ctype).unsqueeze(1) * W) * \
                  (torch.sqrt(torch.clamp(sigma2S, min=0.0)).to(ctype).unsqueeze(0))
        H_true = PsiR @ Ha_true @ PsiS.conj().transpose(0, 1)

        sp, Eg_sorted, Eg_sum = group_energy_report_and_curve(Ha_true, group_id, G, topk=(10,))
        h_true = Ha_true.transpose(0, 1).contiguous().view(-1)
        y_clean = Aop.forward(h_true, LR, LS)
        M = y_clean.numel()
        sigPow = (torch.norm(y_clean) ** 2).real.item() / M

        print(f"[MC {mc+1}/{args.Nmc}] sigPow={sigPow:.3e} | effG={sp['eff_groups']:.1f} top10={sp['top10_ratio']:.3f}")

        # optional plots for MC=1
        if args.plot_debug and (mc == 0):
            plot_sigma2_and_groups(
                xiR, xiS,
                sigma2R, sigma2S,
                Eg_sorted, Eg_sum,
                out_sigma_png="sigma2_debug.png",
                out_group_png="group_energy_debug.png",
                title_prefix=f"[MC{mc+1}]"
            )
            print("[DebugPlot] saved sigma2_debug.png and group_energy_debug.png")

        per_snr_infos = []
        for is_, SNRdB in enumerate(args.snr_list):
            sigma2n = sigPow / (10 ** (SNRdB / 10.0))
            noise = math.sqrt(sigma2n / 2) * (torch.randn(M, device=device) + 1j * torch.randn(M, device=device)).to(ctype)
            y = y_clean + noise

            mu, _, info = sbl_block_em(
                Aop=Aop, y=y, sigma2=sigma2n,
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
                collect_info=True
            )

            Ha_hat = mu.view(LS, LR).transpose(0, 1)
            H_hat = PsiR @ Ha_hat @ PsiS.conj().transpose(0, 1)
            err = torch.norm(H_hat - H_true) ** 2
            den = torch.clamp(torch.norm(H_true) ** 2, min=1e-12)
            nmse_lin = (err / den).real.item()
            nmse[mc, is_] = nmse_lin

            nmse_db = 10 * np.log10(nmse_lin + 1e-300)
            print(f"  SNR={SNRdB:>3} dB | NMSE={nmse_db:.2f} dB")
            per_snr_infos.append(info)

        diag_dump["mc_idx"].append(mc + 1)
        diag_dump["sigPow"].append(sigPow)
        diag_dump["sparsity_eff_groups"].append(sp["eff_groups"])
        diag_dump["top10_ratio"].append(sp["top10_ratio"])
        diag_dump["per_snr_info"].append(per_snr_infos)

    # ---------------- finalize averages ----------------
    snr_db = np.array(args.snr_list, dtype=np.float64)
    nmse_mean = nmse.mean(axis=0)
    nmse_mean_db = 10 * np.log10(nmse_mean + 1e-300)

    # save csv
    csv_mat = np.column_stack([snr_db, nmse_mean_db])
    np.savetxt(args.out_csv, csv_mat, delimiter=",", header="SNR_dB,NMSE_mean_dB", comments="")

    # save mat
    if _HAVE_SCIPY:
        mdict = {
            "snr_db": snr_db.reshape(1, -1),
            "nmse_mc_lin": nmse,
            "nmse_mean_lin": nmse_mean.reshape(1, -1),
            "nmse_mean_db": nmse_mean_db.reshape(1, -1),
            "angles_mode": np.array(args.angles_mode),
            "fixed_angles_seed": np.array([args.fixed_angles_seed], dtype=np.int32),
        }
        sio.savemat(args.out_mat, mdict, do_compression=True, oned_as="row")
    else:
        print("[Warn] scipy not found, .mat not saved. Install with: pip install scipy")

    # save diag dump
    np.savez_compressed(
        args.out_npz,
        mc_idx=np.array(diag_dump["mc_idx"]),
        sigPow=np.array(diag_dump["sigPow"]),
        sparsity_eff_groups=np.array(diag_dump["sparsity_eff_groups"]),
        top10_ratio=np.array(diag_dump["top10_ratio"]),
        per_snr_info=np.array(diag_dump["per_snr_info"], dtype=object),
    )

    # plot mean curve
    plt.figure()
    plt.plot(args.snr_list, 10 * np.log10(nmse_mean + 1e-300), "-o")
    plt.grid(True)
    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE (dB)")
    plt.title(f"Block-SBL (fixed angles) | Nmc={args.Nmc} | pilot={args.pilot}")
    plt.savefig("nmse_sbl.png", dpi=200)
    plt.show()

    # ---------------- auto badpoint report ----------------
    nmse_mc_db = 10 * np.log10(nmse + 1e-300)
    cfg = {
        "G": G,
        "probes": args.probes,
        "cg_tol": args.cg_tol,
        "cg_iter": args.cg_iter,
        "diag_eps": args.diag_eps,
        "use_jacobi": int(args.use_jacobi),
    }
    write_badpoint_report(
        args.report_txt,
        args.report_csv,
        args.report_png,
        args.snr_list,
        nmse_mc_db,
        diag_dump,
        top_bad=args.top_bad,
        bad_nmse_db=args.bad_nmse_db,
        cfg=cfg
    )

    print(f"[Done] total {time.time()-t0:.1f}s")
    print(f"[Saved] nmse_sbl.png, {args.out_csv}, {args.out_npz}, {args.report_txt}, {args.report_csv}, {args.report_png}")
    if args.plot_debug:
        print("[Saved] sigma2_debug.png, group_energy_debug.png (MC1)")
    if _HAVE_SCIPY:
        print(f"[Saved] {args.out_mat}")


if __name__ == "__main__":
    main()

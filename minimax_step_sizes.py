"""Torch helper utilities for estimating MINIMAX step sizes."""

from __future__ import annotations

import math

import torch


# -----------------------------
# 1. Build Phi and K0 in torch
# -----------------------------
def build_Phi_and_K0_torch(phi, X):
    """
    Build feature matrix Phi and K0 = Phi Phi^T from a feature map phi.

    Parameters
    ----------
    phi : callable
        phi(x) -> 1D torch tensor of shape (d,)

    X : array-like or torch.Tensor, shape (n, d_x)

    Returns
    -------
    Phi : torch.Tensor, shape (n, d)
    K0  : torch.Tensor, shape (n, n)
    """

    X = torch.as_tensor(X, dtype=torch.get_default_dtype())

    n = X.shape[0]
    feats = []
    for i in range(n):
        f = phi(X[i])
        f = torch.as_tensor(f, dtype=X.dtype).reshape(-1)
        feats.append(f)

    Phi = torch.stack(feats, dim=0)  # (n, d)
    K0 = Phi @ Phi.T
    K0 = 0.5 * (K0 + K0.T)  # symmetrize for safety

    return Phi, K0


# -----------------------------------------
# 2. Grad + Hessian in raw parameters
#    (sigma_f, sigma_eps)
# -----------------------------------------
def grad_hess_L_rawparams_torch(y, K0, sigma_f, sigma_eps):
    """
    Gradient and Hessian of
        L(sigma_f, sigma_eps) = y^T C^{-1} y + log det C
    where C = sigma_f^2 K0 + sigma_eps^2 I,
    w.r.t. raw parameters (sigma_f, sigma_eps).

    Returns
    -------
    g : torch.Tensor, shape (2,)
        [dL/d sigma_f, dL/d sigma_eps]
    H : torch.Tensor, shape (2, 2)
        Hessian w.r.t. (sigma_f, sigma_eps)
    """

    y = torch.as_tensor(y, dtype=K0.dtype)

    n = y.shape[0]
    sigma_f = torch.as_tensor(sigma_f, dtype=K0.dtype)
    sigma_eps = torch.as_tensor(sigma_eps, dtype=K0.dtype)

    C = sigma_f**2 * K0 + sigma_eps**2 * torch.eye(n, dtype=K0.dtype, device=K0.device)
    C = 0.5 * (C + C.T)  # enforce symmetry

    Q = torch.inverse(C)
    alpha = Q @ y

    # scalar pieces
    aK0a = alpha @ (K0 @ alpha)
    aQa = alpha @ (Q @ alpha)
    aK0QK0a = alpha @ (K0 @ (Q @ (K0 @ alpha)))
    trQK0 = torch.trace(Q @ K0)
    trQK0QK0 = torch.trace(Q @ K0 @ Q @ K0)
    trQ = torch.trace(Q)
    trQ2 = torch.trace(Q @ Q)
    aKa = alpha @ alpha
    aK0Qa = alpha @ (K0 @ (Q @ alpha))
    trQ2K0 = torch.trace(Q @ Q @ K0)

    # gradient
    g_f = 2 * sigma_f * (-aK0a + trQK0)
    g_e = 2 * sigma_eps * (-aKa + trQ)
    g = torch.stack([g_f, g_e])

    # Hessian (analytic, verified vs finite differences)
    H_ff = 2 * (-aK0a + trQK0) + 8 * sigma_f**2 * aK0QK0a - 4 * sigma_f**2 * trQK0QK0
    H_ee = 2 * (-aKa + trQ) + 8 * sigma_eps**2 * aQa - 4 * sigma_eps**2 * trQ2
    H_fe = 8 * sigma_f * sigma_eps * aK0Qa - 4 * sigma_f * sigma_eps * trQ2K0

    H = torch.stack([torch.stack([H_ff, H_fe]), torch.stack([H_fe, H_ee])])

    return g, H


# -----------------------------------------
# 3. Grad + Hessian in log parameters
#    u = log sigma_f, v = log sigma_eps
# -----------------------------------------
def grad_hess_L_logparams_torch(y, K0, u, v):
    """
    Gradient and Hessian of L in log-parameters:
      u = log sigma_f, v = log sigma_eps.

    Returns
    -------
    g_log : torch.Tensor, shape (2,)
        [dL/du, dL/dv]
    H_log : torch.Tensor, shape (2, 2)
        Hessian w.r.t. (u, v)
    """

    u = torch.as_tensor(u, dtype=K0.dtype)
    v = torch.as_tensor(v, dtype=K0.dtype)

    sigma_f = torch.exp(u)
    sigma_eps = torch.exp(v)

    g_raw, H_raw = grad_hess_L_rawparams_torch(y, K0, sigma_f, sigma_eps)
    g_f, g_e = g_raw
    H_ff, H_fe = H_raw[0, 0], H_raw[0, 1]
    H_ee = H_raw[1, 1]

    # gradient in log-space
    g_u = sigma_f * g_f
    g_v = sigma_eps * g_e

    # Hessian in log-space (chain rule)
    H_uu = sigma_f**2 * H_ff + sigma_f * g_f
    H_vv = sigma_eps**2 * H_ee + sigma_eps * g_e
    H_uv = sigma_f * sigma_eps * H_fe

    g_log = torch.stack([g_u, g_v])
    H_log = torch.stack([torch.stack([H_uu, H_uv]), torch.stack([H_uv, H_vv])])

    return g_log, H_log


# -----------------------------------------
# 4. Step-size chooser (a, b)
#    with flags:
#      - use_log_sigma
#      - use_normalization
# -----------------------------------------
def choose_step_sizes_torch(
    phi,
    X,
    y,
    sigma_f_init=1.0,
    sigma_eps_init=None,
    eta=0.25,
    use_log_sigma=True,
    use_normalization=True,
    ratio_norm=10.0,
    ratio_no_norm=3.0,
):
    """
    Choose primal (a) and dual (b) step sizes for your minimax GP training.

    Parameters
    ----------
    phi : callable
        Feature map phi(x) -> 1D torch tensor.

    X : array-like or torch.Tensor, shape (n, d_x)

    y : array-like or torch.Tensor, shape (n,)

    sigma_f_init : float
        Initial sigma_f.

    sigma_eps_init : float or None
        Initial sigma_eps; if None, set to 0.1 * std(y) (or 1.0 if std==0).

    eta : float in (0, 1)
        Safety factor: a ≈ eta / L, where L is local curvature.

    use_log_sigma : bool
        If True, we compute curvature in (log sigma_f, log sigma_eps).
        If False, in (sigma_f, sigma_eps) directly.

    use_normalization : bool
        If True, you are using penalty ||A-F|| / ||A||, and we push b a bit higher.
        If False, you are using ||A-F||, and we use a more conservative b.

    ratio_norm : float > 1
        Base dual/primal ratio when use_normalization=True.

    ratio_no_norm : float > 1
        Base dual/primal ratio when use_normalization=False.

    Returns
    -------
    a : float
        Primal step size.
    b : float
        Dual step size.
    """

    y = torch.as_tensor(y, dtype=torch.get_default_dtype())

    # Default noise init
    if sigma_eps_init is None:
        std_y = y.std()
        sigma_eps_init = 0.1 * std_y.item() if std_y > 0 else 1.0

    sigma_f_init = float(sigma_f_init)
    sigma_eps_init = float(sigma_eps_init)

    _, K0 = build_Phi_and_K0_torch(phi, X)

    if use_log_sigma:
        if sigma_f_init <= 0 or sigma_eps_init <= 0:
            raise ValueError("sigma_f_init and sigma_eps_init must be > 0 when use_log_sigma=True")

        u0 = math.log(sigma_f_init)
        v0 = math.log(sigma_eps_init)
        _, H = grad_hess_L_logparams_torch(y, K0, u0, v0)
    else:
        _, H = grad_hess_L_rawparams_torch(
            y,
            K0,
            sigma_f=torch.tensor(sigma_f_init, dtype=K0.dtype),
            sigma_eps=torch.tensor(sigma_eps_init, dtype=K0.dtype),
        )

    # Local Lipschitz estimate: spectral norm of 2x2 Hessian
    eigvals = torch.linalg.eigvalsh(H)
    L_est = eigvals.abs().max().item()
    if not math.isfinite(L_est) or L_est <= 0:
        L_est = 1.0

    # Primal step
    a = eta / L_est

    # Dual step: different ratios depending on normalization flag
    ratio = ratio_norm if use_normalization else ratio_no_norm
    b = ratio * a

    return float(a), float(b)


__all__ = [
    "build_Phi_and_K0_torch",
    "grad_hess_L_rawparams_torch",
    "grad_hess_L_logparams_torch",
    "choose_step_sizes_torch",
]



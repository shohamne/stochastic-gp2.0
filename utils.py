import math

import torch


torch.set_default_dtype(torch.float64)


# -------------------------------------------------------------------
# 1. RBF kernel and synthetic GP data (NeurIPS setup)
# -------------------------------------------------------------------

def rbf_kernel(X1: torch.Tensor, X2: torch.Tensor, lengthscale: float) -> torch.Tensor:
    X1_sq = (X1 ** 2).sum(-1, keepdim=True)
    X2_sq = (X2 ** 2).sum(-1, keepdim=True)
    sqdist = X1_sq - 2.0 * X1 @ X2.T + X2_sq.T
    return torch.exp(-0.5 * sqdist / (lengthscale ** 2))


def generate_gp_data(
    n: int = 1024,
    lengthscale: float = 0.5,
    sigma_f2: float = 4.0,
    sigma_eps2: float = 1.0,
    device: str | None = None,
    seed: int = 0,
    cluster_strength: float = 1.0,
    heterogeneity: float = 0.0,
):
    """
    Synthetic GP data with optional input heterogeneity and a knob that
    globally concentrates the input cloud.

    Base case (cluster_strength = 1.0, heterogeneity = 0.0):
      x_i ~ N(0, 5^2), kernel: k(x,x') = σ_f² k_f(x,x') + σ_ε² δ.
      This reproduces the original NeurIPS setup.

    Cluster-strength effect (cluster_strength > 1.0):
      Shrinks the entire input cloud relative to the RBF lengthscale,
      yielding a more globally correlated / near low-rank kernel. This
      regime favors ORF-based SCGD / MINIMAX over submatrix BSGD.

    Heterogeneous case (heterogeneity > 0):
      Mixture of two 1D Gaussian clusters with different scales and centers.
      As `heterogeneity` increases:
        - Cluster centers move further apart,
        - One cluster becomes denser/narrower, the other wider,
      which induces a more realistic, block / multi-scale kernel structure.
      This structure is harder for submatrix-based BSGD, but still well
      handled by global ORF approximations (SCGD / MINIMAX), and cannot
      be "fixed" simply by changing the RBF lengthscale.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    cluster_strength = float(cluster_strength)
    if cluster_strength <= 0.0:
        raise ValueError("cluster_strength must be positive.")
    base_std = 5.0 / cluster_strength

    # --- Input distribution ---
    if heterogeneity <= 0.0:
        # Original setup: single wide Gaussian cloud
        X = base_std * torch.randn(n, 1, device=device)
    else:
        heterogeneity = float(heterogeneity)
        scale = 1.0 / cluster_strength

        # Fraction of points assigned to the second (wider) cluster.
        # Grows with heterogeneity but capped at 40%.
        frac2 = min(0.4, 0.1 + 0.3 * heterogeneity)
        n2 = max(1, int(frac2 * n))
        n1 = max(1, n - n2)

        # Cluster separation grows with heterogeneity
        base_sep = 10.0
        sep = base_sep * (1.0 + heterogeneity)

        # Cluster centers
        c1 = -0.5 * sep * scale
        c2 = +0.5 * sep * scale

        # Within-cluster stds: one dense, one wide
        std1 = (2.5 / (1.0 + 0.5 * heterogeneity)) * scale  # denser as heterogeneity grows
        std2 = (5.0 * (1.0 + 0.5 * heterogeneity)) * scale  # wider as heterogeneity grows

        X1 = c1 + std1 * torch.randn(n1, 1, device=device)
        X2 = c2 + std2 * torch.randn(n2, 1, device=device)
        X = torch.cat([X1, X2], dim=0)

        # Shuffle so indices are not ordered by cluster
        perm = torch.randperm(X.shape[0], device=device)
        X = X[perm]

        # If we overshot n due to max(1, ·) safeguards, trim back to n
        if X.shape[0] > n:
            X = X[:n]

    # --- GP kernel & outputs (unchanged logic) ---
    K_f = rbf_kernel(X, X, lengthscale)
    eye_n = torch.eye(X.shape[0], device=device)
    K = sigma_f2 * K_f + sigma_eps2 * eye_n

    jitter = 1e-6
    L = torch.linalg.cholesky(K + jitter * eye_n)
    y = L @ torch.randn(X.shape[0], 1, device=device)
    y = y.squeeze(-1)
    return X, y, K_f


# -------------------------------------------------------------------
# 2. Exact and minibatch NLML
# -------------------------------------------------------------------

def exact_nlml_from_precomputed(
    K_f: torch.Tensor,
    y: torch.Tensor,
    sigma_f2: float,
    sigma_eps2: float,
    jitter: float = 1e-6,
) -> float:
    """
    ℓ(θ) = (1/(2n)) [ y^T K^{-1} y + log|K| + n log(2π) ]
    where K = σ_f² K_f + σ_ε² I.
    """
    n = y.shape[0]
    device = K_f.device
    eye_n = torch.eye(n, device=device, dtype=K_f.dtype)

    K = sigma_f2 * K_f + sigma_eps2 * eye_n
    K = K + jitter * eye_n

    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(y.view(-1, 1), L)
    logdetK = 2.0 * torch.log(torch.diagonal(L)).sum()
    quad = (y.view(1, -1) @ alpha).squeeze()
    nll = 0.5 * (quad + logdetK + n * math.log(2.0 * math.pi))
    return (nll / n).item()


def minibatch_nlml_from_precomputed(
    K_f: torch.Tensor,
    y: torch.Tensor,
    sigma_f2: float,
    sigma_eps2: float,
    batch_idx: torch.Tensor,
    jitter: float = 1e-6,
) -> float:
    idx = batch_idx
    m = idx.numel()
    device = K_f.device
    eye_m = torch.eye(m, device=device, dtype=K_f.dtype)

    y_b = y[idx]
    K_f_b = K_f[idx][:, idx]
    K = sigma_f2 * K_f_b + sigma_eps2 * eye_m
    K = K + jitter * eye_m

    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(y_b.view(-1, 1), L)
    logdetK = 2.0 * torch.log(torch.diagonal(L)).sum()
    quad = (y_b.view(1, -1) @ alpha).squeeze()
    nll = 0.5 * (quad + logdetK + m * math.log(2.0 * math.pi))
    return (nll / m).item()


def exact_nlml_gradients(
    K_f: torch.Tensor,
    y: torch.Tensor,
    rho: float | torch.Tensor,
    sigma_eps2: float | torch.Tensor,
    jitter: float = 1e-6,
) -> tuple[float, float, float]:
    """
    Compute the exact NLML (per data point) and its gradients w.r.t. ρ and σ².
    """
    n = y.shape[0]
    device = K_f.device
    dtype = K_f.dtype
    eye_n = torch.eye(n, device=device, dtype=dtype)

    rho_t = torch.as_tensor(rho, device=device, dtype=dtype)
    sigma_eps2_t = torch.as_tensor(sigma_eps2, device=device, dtype=dtype)
    sigma_f2 = torch.exp(rho_t)

    y_vec = y.view(-1, 1).to(device=device, dtype=dtype)
    K = sigma_f2 * K_f + sigma_eps2_t * eye_n
    K = K + jitter * eye_n

    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(y_vec, L)
    logdetK = 2.0 * torch.log(torch.diagonal(L)).sum()
    quad = (y_vec.view(1, -1) @ alpha).squeeze()
    nll = 0.5 * (quad + logdetK + n * math.log(2.0 * math.pi))
    nlml = nll / n

    K_inv = torch.cholesky_inverse(L)
    alpha_vec = alpha.view(-1)

    alpha_term_f = torch.dot(alpha_vec, K_f @ alpha_vec)
    alpha_term_eps = torch.dot(alpha_vec, alpha_vec)

    trace_Kinv_Kf = (K_inv * K_f).sum()
    trace_Kinv = torch.trace(K_inv)

    coeff = 0.5 / n
    d_nlml_dsigma_f2 = coeff * (-alpha_term_f + trace_Kinv_Kf)
    d_nlml_dsigma_eps2 = coeff * (-alpha_term_eps + trace_Kinv)
    d_nlml_drho = d_nlml_dsigma_f2 * sigma_f2

    return (
        nlml.item(),
        d_nlml_drho.item(),
        d_nlml_dsigma_eps2.item(),
    )


# -------------------------------------------------------------------
# 3. Orthogonal Random Features (ORF) for RBF kernel
# -------------------------------------------------------------------

def orf_features(
    X: torch.Tensor,
    num_features: int,
    lengthscale: float,
    seed: int = 0,
):
    """
    ORF features φ(x) ≈ k_f(x,·) for Gaussian RBF (Yu et al., 2016).
    We approximate k_f(x,x') with amplitude 1, then introduce σ_f²
    via a separate amplitude parameter.
    """
    torch.manual_seed(seed)
    n, d_in = X.shape
    device = X.device
    D = num_features

    blocks = D // d_in
    remainder = D % d_in
    W_blocks = []

    def sample_orf_block():
        G = torch.randn(d_in, d_in, device=device)
        Q, R = torch.linalg.qr(G)
        sign = torch.sign(torch.diagonal(R))
        Q = Q * sign

        G2 = torch.randn(d_in, d_in, device=device)
        r = torch.linalg.norm(G2, dim=1)
        W_block = (r.unsqueeze(1) * Q) / lengthscale
        return W_block

    for _ in range(blocks):
        W_blocks.append(sample_orf_block())
    if remainder > 0:
        W_block = sample_orf_block()
        W_blocks.append(W_block[:remainder])

    W = torch.vstack(W_blocks)
    b = 2.0 * math.pi * torch.rand(D, device=device)

    Z_base = math.sqrt(2.0 / D) * torch.cos(X @ W.T + b)
    return Z_base, W, b



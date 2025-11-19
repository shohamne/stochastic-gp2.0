import math
import time

import torch

from utils import exact_nlml_from_precomputed, minibatch_nlml_from_precomputed


# -------------------------------------------------------------------
# 4. NeurIPS BSGD (biased SGD on exact kernel)
# -------------------------------------------------------------------

def bsgd_gradient(
    K_f: torch.Tensor,
    y: torch.Tensor,
    theta: torch.Tensor,
    batch_idx: torch.Tensor,
    s1: float,
    s2: float,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """
    Biased gradient estimator g(θ; X_ξ, y_ξ) from NeurIPS:
      g_l = 1/(2 s_l(m)) tr(K_ξ^{-1}(I - y_ξ y_ξ^T K_ξ^{-1}) ∂K_ξ/∂θ_l)
    with θ = (σ_f², σ_ε²), s1(m)=3 log m, s2(m)=m.
    """
    sigma_f2 = theta[0]
    sigma_eps2 = theta[1]

    idx = batch_idx
    m = idx.numel()
    device = K_f.device
    eye_m = torch.eye(m, device=device, dtype=K_f.dtype)

    y_b = y[idx]
    K_f_b = K_f[idx][:, idx]

    K = sigma_f2 * K_f_b + sigma_eps2 * eye_m
    K = K + jitter * eye_m
    L = torch.linalg.cholesky(K)
    K_inv = torch.cholesky_inverse(L)

    y_col = y_b.view(-1, 1)
    alpha = K_inv @ y_col
    A = K_inv - alpha @ alpha.T

    dK1 = K_f_b
    dK2 = eye_m

    g1 = 0.5 * (A * dK1).sum() / s1
    g2 = 0.5 * (A * dK2).sum() / s2

    return torch.stack([g1, g2])


def train_bsgd_neurips(
    K_f: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int = 25,
    batch_size: int = 128,
    sigma_f2_init: float = 1.0,
    sigma_eps2_init: float = 1.0,
    alpha1: float = 9.0,    # NeurIPS Fig.1 uses α1 ~ 9
    theta_min: float = 1e-4,
    theta_max: float = 20.0,
    print_every: int = 1,
    sigma_f2_true: float | None = None,
    sigma_eps2_true: float | None = None,
):
    device = K_f.device
    n = y.shape[0]
    theta = torch.tensor([sigma_f2_init, sigma_eps2_init],
                         device=device, dtype=K_f.dtype)

    global_step = 0
    jitter = 1e-6

    print("\n=== BSGD (biased SGD on exact kernel) ===")
    print(
        "iter\tepoch\tσ_f²\tσ_ε²\t"
        "real_nlml\treal_nlml_true\tmini_nlml\t"
        "||grad||\tduration_s\twall_time_s"
    )

    run_start = time.perf_counter()

    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start+batch_size]
            if batch_idx.numel() == 0:
                continue

            iter_start = time.perf_counter()
            global_step += 1
            m = batch_idx.numel()
            s1 = 3.0 * math.log(m)
            s2 = float(m)

            grad = bsgd_gradient(K_f, y, theta, batch_idx, s1, s2, jitter=jitter)
            lr = alpha1 / float(global_step)  # α_k = α1 / k
            theta = theta - lr * grad
            theta = theta.clamp(min=theta_min, max=theta_max)

            if global_step % print_every == 0:
                now = time.perf_counter()
                iter_duration = now - iter_start
                wall_elapsed = now - run_start
                sigma_f2 = theta[0].item()
                sigma_eps2 = theta[1].item()
                real_nlml = exact_nlml_from_precomputed(K_f, y, sigma_f2, sigma_eps2, jitter=jitter)
                if (sigma_f2_true is not None) and (sigma_eps2_true is not None):
                    real_nlml_true = exact_nlml_from_precomputed(
                        K_f, y, float(sigma_f2_true), float(sigma_eps2_true), jitter=jitter
                    )
                else:
                    real_nlml_true = float("nan")
                mini_nlml = minibatch_nlml_from_precomputed(
                    K_f, y, sigma_f2, sigma_eps2, batch_idx, jitter=jitter
                )
                full_grad = full_gradient_exact(K_f, y, theta, jitter=jitter)
                grad_norm = full_grad.norm().item()
                print(
                    f"{global_step:4d}\t{epoch+1:2d}\t"
                    f"{sigma_f2:7.4f}\t{sigma_eps2:7.4f}\t"
                    f"{real_nlml:9.4f}\t{real_nlml_true:14.4f}\t{mini_nlml:9.4f}\t"
                    f"{grad_norm:9.3e}\t{iter_duration:8.3f}\t{wall_elapsed:10.3f}"
                )

    return theta


# -------------------------------------------------------------------
# 5. NeurIPS Figure 2: full gradient vs minibatch size
# -------------------------------------------------------------------

def full_gradient_exact(
    K_f: torch.Tensor,
    y: torch.Tensor,
    theta: torch.Tensor,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """
    Exact gradient ∇_θ ℓ(θ) for the full GP NLML, with θ = (σ_f², σ_ε²).
    """
    sigma_f2 = theta[0]
    sigma_eps2 = theta[1]

    n = y.shape[0]
    device = K_f.device
    dtype = K_f.dtype
    eye_n = torch.eye(n, device=device, dtype=dtype)

    K = sigma_f2 * K_f + sigma_eps2 * eye_n
    K = K + jitter * eye_n

    L = torch.linalg.cholesky(K)
    K_inv = torch.cholesky_inverse(L)

    y_col = y.view(-1, 1)
    alpha = K_inv @ y_col
    A = K_inv - alpha @ alpha.T

    dK1 = K_f
    dK2 = eye_n

    g1 = 0.5 * (A * dK1).sum() / n
    g2 = 0.5 * (A * dK2).sum() / n

    return torch.stack([g1, g2])


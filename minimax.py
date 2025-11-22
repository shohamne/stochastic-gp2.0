import math
import time

import torch

from bsgd import full_gradient_exact
from utils import exact_nlml_from_precomputed


# -------------------------------------------------------------------
# 6. TMLR MINIMAX + ORF (Algorithm 1)
# -------------------------------------------------------------------

def minimax_train_orf(
    Z_base: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    K_f_exact: torch.Tensor,
    K_f_orf: torch.Tensor,
    n_epochs: int = 25,
    batch_size: int = 128,
    mu: float = 1.0,
    mu_increase_factor: float = 1.0,
    mu_increase_epochs: int = 0,
    a: float = 1e-3,
    b: float = 1e-3,
    lr_decay: float = 1.0,
    lr_decay_start_epoch: int = 1,
    sigma_f2_init: float = 1.0,
    sigma_eps2_init: float = 1.0,
    w_init_scale: float = 0.1,
    sigma_f2_true: float | None = None,
    sigma_eps2_true: float | None = None,
    print_every: int = 1,
):
    """
    Minimax algorithm from TMLR (Algorithm 1), with ORF approximation.
    Parameters:
      θ = (w, ρ, σ²), ϕ(x) = exp(ρ/2) Z_base(x),
      F(θ) = Σ_i [ϕ(x_i)ϕ(x_i)^T + (σ²/n) I_d].
    Learning rates a (MIN) and b (MAX) can be exponentially decayed each
    iteration via lr_decay once epoch lr_decay_start_epoch is reached;
    lr_decay=1.0 (or lr_decay_start_epoch <= 1) reproduces the original schedule.
    μ can optionally be multiplied by `mu_increase_factor` every
    `mu_increase_epochs` epochs to gradually tighten the penalty.
    """
    device = Z_base.device
    n, d = Z_base.shape
    jitter = 1e-6
    eye_d = torch.eye(d, device=device, dtype=Z_base.dtype)

    if not (0.0 < lr_decay <= 1.0):
        raise ValueError("lr_decay must lie in (0, 1].")

    lr_decay_start_epoch = max(1, int(lr_decay_start_epoch))

    w = (w_init_scale * torch.randn(d, device=device)).clone().detach().requires_grad_(True)
    rho = torch.tensor(math.log(sigma_f2_init), device=device, requires_grad=True)
    sigma2 = torch.tensor(sigma_eps2_init, device=device, requires_grad=True)

    A = (sigma_eps2_init * eye_d).clone().detach().requires_grad_(True)
    B = torch.zeros(d, d, device=device, requires_grad=True)

    total_steps = 0
    lr_scale = 1.0
    last_lr_scale = lr_scale
    decay_active = lr_decay_start_epoch <= 1
    mu_current = float(mu)
    mu_increase_epochs = max(0, int(mu_increase_epochs))
    if mu_increase_factor <= 0:
        raise ValueError("mu_increase_factor must be positive.")
    mu_increase_factor = float(mu_increase_factor)

    print("\n=== MINIMAX + ORF (TMLR) ===")
    print(
        "iter\tepoch\tσ_f²\tσ_ε²\t"
        "real_nlml\treal_nlml_true\treal_nlml_abs_err\t"
        "orf_nlml\torf_nlml_true\t"
        "ℓ_μ\t|A-F|/|A|\t|B|\tcosBΔ\tpen/pen*\t"
        "lr_scale\t||grad||\tduration_s\twall_time_s"
    )

    run_start = time.perf_counter()

    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)

        for start in range(0, n, batch_size):
            S = perm[start:start+batch_size]
            if S.numel() == 0:
                continue

            iter_start = time.perf_counter()
            S_bar = torch.randperm(n, device=device)[:batch_size]
            total_steps += 1
            if (not decay_active) and ((epoch + 1) >= lr_decay_start_epoch):
                decay_active = True
            current_lr_scale = lr_scale
            a_eff = a * current_lr_scale

            # ---------- MIN step ----------
            for p in (w, rho, sigma2, A, B):
                if p.grad is not None:
                    p.grad.zero_()

            amp = torch.exp(0.5 * rho)
            A_norm = A.norm()
            logdetA = torch.logdet(A + jitter * eye_d)

            psi_min = 0.0
            for i in S:
                phi_i = amp * Z_base[i]
                pred_i = phi_i @ w
                res_i = pred_i - y[i]

                gi = (
                    (res_i ** 2) / sigma2
                    + (w.pow(2).sum() / n)
                    + ((n - d) / n) * torch.log(sigma2)
                )
                Fi = torch.ger(phi_i, phi_i) + (sigma2 / n) * eye_d
                penalty_i = mu_current * torch.sum(B * ((A / n) - Fi)) / (A_norm + 1e-12)
                psi_i = gi + (logdetA / n) + penalty_i
                psi_min = psi_min + psi_i

            psi_min = (n / S.numel()) * psi_min
            psi_min.backward()

            with torch.no_grad():
                w -= a_eff * w.grad
                rho -= a_eff * rho.grad
                sigma2 -= a_eff * sigma2.grad
                A -= a_eff * A.grad

                sigma2.clamp_(min=1e-4, max=20.0)
                w.clamp_(-10.0, 10.0)

                A.copy_(0.5 * (A + A.T))
                eigvals, eigvecs = torch.linalg.eigh(A)
                min_eig = sigma2.item()
                eigvals_clipped = torch.clamp(eigvals, min=min_eig)
                A.copy_(eigvecs @ torch.diag(eigvals_clipped) @ eigvecs.T)

            # ---------- MAX step ----------
            for p in (w, rho, sigma2, A, B):
                if p.grad is not None:
                    p.grad.zero_()

            amp = torch.exp(0.5 * rho)
            A_norm = A.norm()
            logdetA = torch.logdet(A + jitter * eye_d)

            psi_max = 0.0
            for i in S_bar:
                phi_i = amp * Z_base[i]
                pred_i = phi_i @ w
                res_i = pred_i - y[i]

                gi = (
                    (res_i ** 2) / sigma2
                    + (w.pow(2).sum() / n)
                    + ((n - d) / n) * torch.log(sigma2)
                )
                Fi = torch.ger(phi_i, phi_i) + (sigma2 / n) * eye_d
                penalty_i = mu_current * torch.sum(B * ((A / n) - Fi)) / (A_norm + 1e-12)
                psi_i = gi + (logdetA / n) + penalty_i
                psi_max = psi_max + psi_i

            psi_max = (n / S_bar.numel()) * psi_max
            psi_max.backward()

            with torch.no_grad():
                b_eff = b * current_lr_scale
                B += b_eff * B.grad
                B_norm = B.norm()
                if B_norm > 1.0:
                    B /= B_norm

            last_lr_scale = current_lr_scale
            if decay_active:
                lr_scale = current_lr_scale * lr_decay
            else:
                lr_scale = current_lr_scale

            # ---------- Logging ----------
            if total_steps % print_every == 0:
                with torch.no_grad():
                    sigma_f2 = torch.exp(rho)
                    sigma_eps2 = sigma2.clamp(min=1e-4)

                    real_nlml = exact_nlml_from_precomputed(
                        K_f_exact, y, sigma_f2.item(), sigma_eps2.item(), jitter=jitter
                    )
                    orf_nlml = exact_nlml_from_precomputed(
                        K_f_orf, y, sigma_f2.item(), sigma_eps2.item(), jitter=jitter
                    )

                    if (sigma_f2_true is not None) and (sigma_eps2_true is not None):
                        real_nlml_true = exact_nlml_from_precomputed(
                            K_f_exact, y, float(sigma_f2_true), float(sigma_eps2_true), jitter=jitter
                        )
                        orf_nlml_true = exact_nlml_from_precomputed(
                            K_f_orf, y, float(sigma_f2_true), float(sigma_eps2_true), jitter=jitter
                        )
                    else:
                        real_nlml_true = float("nan")
                        orf_nlml_true = float("nan")

                    real_nlml_abs_err = abs(real_nlml - real_nlml_true)
                    amp2 = torch.exp(0.5 * rho)
                    Z_scaled = amp2 * Z_base
                    F_full = Z_scaled.T @ Z_scaled + sigma_eps2 * eye_d

                    L_F = torch.linalg.cholesky(F_full + jitter * eye_d)
                    logdetF = 2.0 * torch.log(torch.diagonal(L_F)).sum()

                    resid_full = Z_scaled @ w - y
                    g_theta = (
                        resid_full.pow(2).sum() / sigma_eps2
                        + w.pow(2).sum()
                        + (n - d) * torch.log(sigma_eps2)
                    )
                    l_theta = g_theta + logdetF
                    approx_nlml = (l_theta / n).item()

                    logdetA_now = torch.logdet(A + jitter * eye_d)
                    delta_AF = A - F_full
                    norm_delta = delta_AF.norm()
                    norm_A = A.norm()
                    norm_B = B.norm()

                    penalty_norm = norm_delta / (norm_A + 1e-12)
                    l_mu_total = g_theta + logdetA_now + mu_current * penalty_norm
                    l_mu = (l_mu_total / n).item()

                    if norm_delta > 1e-12 and norm_B > 1e-12:
                        cos_B_delta = (B * delta_AF).sum() / (norm_B * norm_delta + 1e-12)
                        penalty_curr = mu_current * (B * delta_AF).sum() / (norm_A + 1e-12)
                        penalty_max = mu_current * norm_delta / (norm_A + 1e-12)
                        pen_ratio = (penalty_curr / (penalty_max + 1e-12)).item()
                        cos_val = cos_B_delta.item()
                    else:
                        cos_val = 0.0
                        pen_ratio = 0.0

                    theta_curr = torch.stack((sigma_f2, sigma_eps2))
                    grad_exact = full_gradient_exact(
                        K_f_exact, y, theta_curr, jitter=jitter
                    )
                    grad_norm = grad_exact.norm().item()

                    now = time.perf_counter()
                    iter_duration = now - iter_start
                    wall_elapsed = now - run_start
                    print(
                        f"{total_steps:4d}\t{epoch+1:2d}\t"
                        f"{sigma_f2.item():7.4f}\t{sigma_eps2.item():7.4f}\t"
                        f"{real_nlml:9.4f}\t"
                        f"{real_nlml_true:14.4f}\t"
                        f"{real_nlml_abs_err:16.4f}\t"
                        f"{orf_nlml:9.4f}\t"
                        f"{orf_nlml_true:14.4f}\t"
                        f"{l_mu:7.4f}\t"
                        f"{mu_current:7.4f}\t"
                        f"{penalty_norm.item():9.3e}\t"
                        f"{norm_B.item():7.4f}\t"
                        f"{cos_val:7.4f}\t"
                        f"{pen_ratio:7.4f}\t"
                        f"{last_lr_scale:8.5f}\t"
                        f"{grad_norm:9.3e}\t"
                        f"{iter_duration:8.3f}\t"
                        f"{wall_elapsed:10.3f}"
                    )

        if (
            mu_increase_epochs > 0
            and ((epoch + 1) % mu_increase_epochs == 0)
            and mu_increase_factor != 1.0
        ):
            mu_current *= mu_increase_factor

    return w, rho, sigma2, A, B


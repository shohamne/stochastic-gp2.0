import math

import torch

from bsgd import full_gradient_exact
from utils import exact_nlml_from_precomputed


# -------------------------------------------------------------------
# 7. TMLR SCGD + ORF (Algorithm 2)
# -------------------------------------------------------------------

def scgd_train_orf(
    Z_base: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    K_f_exact: torch.Tensor,
    K_f_orf: torch.Tensor,
    n_epochs: int = 25,
    batch_size: int = 128,
    a0: float = 1.0,
    b0: float = 1.0,
    a_decay: float = 0.75,
    b_decay: float = 0.5,
    decay_start_epoch: int = 1,
    sigma_f2_init: float = 1.0,
    sigma_eps2_init: float = 1.0,
    w_init_scale: float = 0.1,
    sigma_f2_true: float | None = None,
    sigma_eps2_true: float | None = None,
    print_every: int = 1,
):
    """
    SCGD Algorithm 2 from TMLR, specialized to ORF GP:
      gi(θ) = 1/σ² (ϕ(x_i)^T w - y_i)^2
               + 1/n ||w||² + 1/n (n-d) log σ²
      Fi(θ) = ϕ(x_i)ϕ(x_i)^T + (σ²/n) I_d
    with ϕ(x) = exp(ρ/2) Z_base(x) and θ = (w, ρ, σ²).

    Step sizes:
      a_t = a0 / t^a_decay,  b_t = b0 / t^b_decay (after epoch decay_start_epoch).
      Before decay_start_epoch, a_t = a0 and b_t = b0.
    """
    device = Z_base.device
    n, d = Z_base.shape
    jitter = 1e-6
    eye_d = torch.eye(d, device=device, dtype=Z_base.dtype)

    decay_start_epoch = max(1, int(decay_start_epoch))
    decay_start_step: int | None = None

    w = (w_init_scale * torch.randn(d, device=device)).clone().detach().requires_grad_(True)
    rho = torch.tensor(math.log(sigma_f2_init), device=device, requires_grad=True)
    sigma2 = torch.tensor(sigma_eps2_init, device=device, requires_grad=True)

    # SPD tracker F~_0
    F_tilde = (sigma_eps2_init * eye_d).clone()

    total_steps = 0

    print("\n=== SCGD + ORF (TMLR Algorithm 2) ===")
    print(
        "iter\tepoch\tσ_f²\tσ_ε²\t"
        "real_nlml\torf_nlml\t"
        "real_nlml_true\torf_nlml_true\t"
        "approx_nlml\t|F~ - F|/|F|\t||grad||"
    )

    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)

        for start in range(0, n, batch_size):
            S = perm[start:start+batch_size]
            if S.numel() == 0:
                continue

            total_steps += 1
            m = S.numel()
            if (decay_start_step is None) and ((epoch + 1) >= decay_start_epoch):
                decay_start_step = total_steps

            if decay_start_step is None:
                a_t = a0
                b_t = b0
            else:
                effective_t = max(1, total_steps - decay_start_step + 1)
                a_t = a0 / (effective_t ** a_decay)
                b_t = b0 / (effective_t ** b_decay)

            for p in (w, rho, sigma2):
                if p.grad is not None:
                    p.grad.zero_()

            amp = torch.exp(0.5 * rho)
            sigma_eps2 = sigma2.clamp(min=1e-4)

            phi_batch = amp * Z_base[S]   # (m,d)
            y_batch = y[S]

            # Σ_i gi(θ)
            shared_w = w.pow(2).sum() / n
            shared_log = (n - d) * torch.log(sigma_eps2) / n
            res = phi_batch @ w - y_batch
            loss_g = (res.pow(2) / sigma_eps2).sum() + m * shared_w + m * shared_log

            # Σ_i Fi(θ) over minibatch
            F_i_sum = phi_batch.T @ phi_batch + (sigma_eps2 * m / n) * eye_d

            # F~_t^{-1}
            F_tilde_spd = F_tilde + jitter * eye_d
            L_Ft = torch.linalg.cholesky(F_tilde_spd)
            F_tilde_inv = torch.cholesky_inverse(L_Ft)

            penalty_sum = (F_tilde_inv * F_i_sum).sum()

            psi_batch = loss_g + penalty_sum
            psi_scaled = (n / m) * psi_batch
            psi_scaled.backward()

            with torch.no_grad():
                w -= a_t * w.grad
                rho -= a_t * rho.grad
                sigma2 -= a_t * sigma2.grad

                sigma2.clamp_(min=1e-4, max=20.0)
                w.clamp_(-10.0, 10.0)

                # Exponential tracking of F(θ_t): F~_{t+1}
                F_tilde.mul_(1.0 - b_t).add_((n / m) * F_i_sum.detach())

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

                    F_error = (F_tilde - F_full).norm() / (F_full.norm() + 1e-12)

                    theta_curr = torch.stack((sigma_f2, sigma_eps2))
                    grad_exact = full_gradient_exact(
                        K_f_exact, y, theta_curr, jitter=jitter
                    )
                    grad_norm = grad_exact.norm().item()

                    print(
                        f"{total_steps:4d}\t{epoch+1:2d}\t"
                        f"{sigma_f2.item():7.4f}\t{sigma_eps2.item():7.4f}\t"
                        f"{real_nlml:9.4f}\t"
                        f"{orf_nlml:9.4f}\t"
                        f"{real_nlml_true:14.4f}\t"
                        f"{orf_nlml_true:14.4f}\t"
                        f"{approx_nlml:11.4f}\t"
                        f"{F_error.item():9.3e}\t"
                        f"{grad_norm:9.3e}"
                    )

    return w, rho, sigma2, F_tilde


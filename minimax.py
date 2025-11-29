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
    lr_decay_every_epochs: int = 0,
    sigma_f2_init: float = 1.0,
    sigma_eps2_init: float = 1.0,
    w_init_scale: float = 0.1,
    warm_start_w: bool = True,
    sigma_f2_true: float | None = None,
    sigma_eps2_true: float | None = None,
    print_every: int = 1,
    real_psi_if_full_batch: bool = False,
    penalty_normalization: str = "relative",
    skip_eigh: bool = False,
    log_sigma_grads: bool = False,
):
    """
    Minimax algorithm from TMLR (Algorithm 1), with ORF approximation.
    Parameters:
      θ = (w, ρ, σ²), ϕ(x) = exp(ρ/2) Z_base(x),
      F(θ) = Σ_i [ϕ(x_i)ϕ(x_i)^T + (σ²/n) I_d].
    Learning rates a (MIN) and b (MAX) can be exponentially decayed by
    multiplying with `lr_decay` every `lr_decay_every_epochs` epochs.
    Setting lr_decay_every_epochs <= 0 disables the schedule. lr_decay=1.0
    reproduces the original constant step-size regime.
    μ can optionally be multiplied by `mu_increase_factor` every
    `mu_increase_epochs` epochs to gradually tighten the penalty.
    The penalty term μ‖A-F‖ can be normalized by ‖A‖ when
    `penalty_normalization="relative"` (default) or left unnormalized with
    `penalty_normalization="absolute"`.
    When `real_psi_if_full_batch` is True and `batch_size == n`, the updates
    use the exact ψ (with ‖A - F‖) and skip the auxiliary B ascent step.
    When `skip_eigh` is True the expensive torch.linalg.eigh projection on A
    is skipped; this can speed up runs at the cost of not explicitly enforcing
    A ≽ σ²I. Setting `log_sigma_grads=True` switches the σ_ε² updates to operate
    in log-variance space (ρ already parameterizes log σ_f²) while keeping all
    printed diagnostics in variance space.
    """
    device = Z_base.device
    n, d = Z_base.shape
    jitter = 1e-6
    eye_d = torch.eye(d, device=device, dtype=Z_base.dtype)

    if not (0.0 < lr_decay <= 1.0):
        raise ValueError("lr_decay must lie in (0, 1].")
    lr_decay_every_epochs = int(lr_decay_every_epochs)
    if lr_decay_every_epochs < 0:
        raise ValueError("lr_decay_every_epochs must be >= 0.")

    sigma2_min = 1e-4
    sigma2_max = 20.0

    w = (w_init_scale * torch.randn(d, device=device)).clone().detach().requires_grad_(True)
    rho = torch.tensor(math.log(sigma_f2_init), device=device, requires_grad=True)
    sigma2 = torch.tensor(sigma_eps2_init, device=device, requires_grad=True)

    A = (sigma_eps2_init * eye_d).clone().detach().requires_grad_(True)
    B = torch.zeros(d, d, device=device, requires_grad=True)

    if warm_start_w:
        with torch.no_grad():
            sigma_f2_0 = torch.tensor(
                float(sigma_f2_init), device=device, dtype=Z_base.dtype
            ).clamp(min=1e-8, max=20.0)
            sigma_eps2_0 = torch.tensor(
                float(sigma_eps2_init), device=device, dtype=Z_base.dtype
            ).clamp(min=sigma2_min, max=sigma2_max)
            phi_full0 = torch.sqrt(sigma_f2_0) * Z_base
            F0 = phi_full0.T @ phi_full0 + sigma_eps2_0 * eye_d
            try:
                rhs0 = phi_full0.T @ y
                L0 = torch.linalg.cholesky(F0 + jitter * eye_d)
                w_ls = torch.cholesky_solve(rhs0.unsqueeze(1), L0).squeeze(1)
                w.copy_(w_ls)
                A.copy_(F0)
            except RuntimeError:
                # Fall back to random initialization if the system is ill-conditioned
                pass

    total_steps = 0
    lr_scale = 1.0
    last_lr_scale = lr_scale
    mu_current = float(mu)
    mu_increase_epochs = max(0, int(mu_increase_epochs))
    if mu_increase_factor <= 0:
        raise ValueError("mu_increase_factor must be positive.")
    mu_increase_factor = float(mu_increase_factor)
    params = (w, rho, sigma2, A, B)
    real_psi_if_full_batch = bool(real_psi_if_full_batch)
    penalty_normalization = str(penalty_normalization).strip().lower()
    if penalty_normalization not in {"relative", "absolute"}:
        raise ValueError("penalty_normalization must be 'relative' or 'absolute'.")
    use_relative_penalty = penalty_normalization == "relative"
    skip_eigh = bool(skip_eigh)
    use_log_sigma_grads = bool(log_sigma_grads)

    def _penalty_denom(norm_tensor: torch.Tensor) -> torch.Tensor:
        """Return the denominator used in the penalty term."""
        return norm_tensor + 1e-12 if use_relative_penalty else norm_tensor.new_tensor(1.0)

    use_real_psi_training = real_psi_if_full_batch and (batch_size == n)
    if real_psi_if_full_batch and not use_real_psi_training:
        print(
            "[MINIMAX] --real-psi-if-full-batch ignored because batch_size != n "
            "(falling back to stochastic ψ)."
        )

    def compute_psi_sim(batch_idx: torch.Tensor) -> torch.Tensor:
        if batch_idx.numel() == 0:
            raise ValueError("Batch for psi computation must be non-empty.")

        amp = torch.exp(0.5 * rho)
        A_norm = A.norm()
        penalty_denom = _penalty_denom(A_norm)
        logdetA = torch.logdet(A + jitter * eye_d)

        phi_batch = amp * Z_base[batch_idx]  # (m, d)
        y_batch = y[batch_idx]
        preds = phi_batch @ w
        residuals = preds - y_batch

        sigma2_inv = 1.0 / sigma2
        w_sq_term = w.pow(2).sum() / n
        log_sigma_term = ((n - d) / n) * torch.log(sigma2)
        logdet_term = logdetA / n

        g_per_sample = (
            residuals.pow(2) * sigma2_inv
            + w_sq_term
            + log_sigma_term
            + logdet_term
        )

        sigma2_over_n = sigma2 / n
        const_penalty_term = torch.sum(B * ((A / n) - sigma2_over_n * eye_d))
        phi_B_phi = torch.einsum("bi,ij,bj->b", phi_batch, B, phi_batch)
        penalty_per_sample = (mu_current / penalty_denom) * (const_penalty_term - phi_B_phi)

        psi_val = g_per_sample.sum() + penalty_per_sample.sum()
        return (n / batch_idx.numel()) * psi_val

    def compute_psi_real_full() -> torch.Tensor:
        amp = torch.exp(0.5 * rho)
        Z_scaled = amp * Z_base
        sigma_eps2 = sigma2.clamp(min=sigma2_min, max=sigma2_max)
        F_full = Z_scaled.T @ Z_scaled + sigma_eps2 * eye_d
        resid_full = Z_scaled @ w - y

        g_theta = (
            (resid_full.pow(2).sum() / sigma_eps2)
            + w.pow(2).sum()
            + (n - d) * torch.log(sigma_eps2)
        )
        logdetA_now = torch.logdet(A + jitter * eye_d)
        penalty_norm = (A - F_full).norm() / _penalty_denom(A.norm())
        return (g_theta + logdetA_now + mu_current * penalty_norm) / n

    def value_and_grad_norm(obj_fn) -> tuple[float, float]:
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        value = obj_fn()
        value.backward()
        grad_sq = 0.0
        for p in params:
            if p.grad is not None:
                grad_sq += p.grad.detach().pow(2).sum().item()
        grad_norm = math.sqrt(grad_sq) if grad_sq > 0.0 else 0.0
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        return value.detach().item(), grad_norm

    penalty_formula = "μ‖A-F‖/‖A‖" if use_relative_penalty else "μ‖A-F‖"
    penalty_column_label = "|A-F|/|A|" if use_relative_penalty else "|A-F|"
    psi_penalty_label = f"{penalty_formula}/n"

    print(f"\n=== MINIMAX + ORF (TMLR) ===  ψ = (g(θ) + log|A| + {penalty_formula})/n")
    print(
        "iter\tepoch\tσ_f²\tσ_ε²\t"
        "real_nlml\treal_nlml_true\treal_nlml_abs_err\t"
        "orf_nlml\torf_nlml_true\t"
        f"μ\t{penalty_column_label}\t|A-F|\t\t|A|\t\t|F|\t\tcosAF\t|B|\tcosBΔ\tpen/pen*\t"
        f"lr_scale\tψ_real\tg(θ)/n\t\t‖Φw-y‖²/(nσ_ε²)\t‖w‖²/n\t((n-d)logσ_ε²)/n\t(log|A|)/n\t{psi_penalty_label}\t||grad_psi_real||\t"
        "||grad||\tduration_s\twall_time_s"
    )

    run_start = time.perf_counter()

    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)

        for start in range(0, n, batch_size):
            S = perm[start:start+batch_size]
            if S.numel() == 0:
                continue

            iter_start = time.perf_counter()
            S_bar = None if use_real_psi_training else torch.randperm(n, device=device)[:batch_size]
            total_steps += 1
            current_lr_scale = lr_scale
            a_eff = a * current_lr_scale

            # ---------- MIN step ----------
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()

            psi_min = compute_psi_real_full() if use_real_psi_training else compute_psi_sim(S)
            psi_min.backward()

            with torch.no_grad():
                w -= a_eff * w.grad
                rho -= a_eff * rho.grad
                if use_log_sigma_grads:
                    if sigma2.grad is not None:
                        safe_sigma2 = sigma2.clamp(min=1e-8)
                        log_sigma2 = torch.log(safe_sigma2)
                        log_sigma2 = log_sigma2 - a_eff * (sigma2 * sigma2.grad)
                        sigma2.copy_(torch.exp(log_sigma2))
                else:
                    sigma2 -= a_eff * sigma2.grad
                A -= a_eff * A.grad

                sigma2.clamp_(min=sigma2_min, max=sigma2_max)
                w.clamp_(-10.0, 10.0)

                A.copy_(0.5 * (A + A.T))
                if not skip_eigh:
                    eigvals, eigvecs = torch.linalg.eigh(A)
                    min_eig = sigma2_min
                    eigvals_clipped = torch.clamp(eigvals, min=min_eig)
                    A.copy_(eigvecs @ torch.diag(eigvals_clipped) @ eigvecs.T)

            # ---------- MAX step ----------
            if not use_real_psi_training:
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()

                psi_max = compute_psi_sim(S_bar)
                psi_max.backward()

                with torch.no_grad():
                    b_eff = b * current_lr_scale
                    B += b_eff * B.grad
                    B_norm = B.norm()
                    if B_norm > 1.0:
                        B /= B_norm

            last_lr_scale = current_lr_scale
            grad_psi_real = float("nan")
            psi_real_all = float("nan")
            if total_steps % print_every == 0:
                psi_real_all, grad_psi_real = value_and_grad_norm(compute_psi_real_full)

            # ---------- Logging ----------
            if total_steps % print_every == 0:
                with torch.no_grad():
                    sigma_f2 = torch.exp(rho)
                    sigma_eps2 = sigma2.clamp(min=sigma2_min)

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
                    g_term_resid = resid_full.pow(2).sum() / sigma_eps2
                    g_term_w = w.pow(2).sum()
                    g_term_logsigma = (n - d) * torch.log(sigma_eps2)
                    g_theta = g_term_resid + g_term_w + g_term_logsigma
                    l_theta = g_theta + logdetF
                    approx_nlml = (l_theta / n).item()

                    logdetA_now = torch.logdet(A + jitter * eye_d)
                    delta_AF = A - F_full
                    norm_delta = delta_AF.norm()
                    norm_A = A.norm()
                    norm_F = F_full.norm()
                    norm_B = B.norm()

                    penalty_norm_denom = _penalty_denom(norm_A)
                    penalty_norm = norm_delta / penalty_norm_denom
                    delta_abs_val = norm_delta.item()
                    penalty_norm_val = penalty_norm.item()
                    norm_A_val = norm_A.item()
                    norm_F_val = norm_F.item()
                    if norm_A > 1e-12 and norm_F > 1e-12:
                        cos_AF = (A * F_full).sum() / (norm_A * norm_F + 1e-12)
                        cos_AF_val = cos_AF.item()
                    else:
                        cos_AF_val = 0.0
                    psi_g = (g_theta / n).item()
                    psi_g_resid = (g_term_resid / n).item()
                    psi_g_w = (g_term_w / n).item()
                    psi_g_logsigma = (g_term_logsigma / n).item()
                    psi_logdetA = (logdetA_now / n).item()
                    psi_penalty = (mu_current * penalty_norm / n).item()

                    if norm_delta > 1e-12 and norm_B > 1e-12:
                        cos_B_delta = (B * delta_AF).sum() / (norm_B * norm_delta + 1e-12)
                        penalty_curr = mu_current * (B * delta_AF).sum() / penalty_norm_denom
                        penalty_max = mu_current * norm_delta / penalty_norm_denom
                        pen_ratio = (penalty_curr / (penalty_max + 1e-12)).item()
                        cos_val = cos_B_delta.item()
                    else:
                        cos_val = 0.0
                        pen_ratio = 0.0

                    theta_curr = torch.stack((sigma_f2, sigma_eps2))
                    grad_exact = full_gradient_exact(
                        K_f_exact, y, theta_curr, jitter=jitter
                    )
                    if use_log_sigma_grads:
                        grad_exact = grad_exact * theta_curr
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
                        f"{mu_current:7.4f}\t"
                        f"{penalty_norm_val:9.3e}\t"
                        f"{delta_abs_val:9.3e}\t"
                        f"{norm_A_val:9.3e}\t"
                        f"{norm_F_val:9.3e}\t"
                        f"{cos_AF_val:7.4f}\t"
                        f"{norm_B.item():7.4f}\t"
                        f"{cos_val:7.4f}\t"
                        f"{pen_ratio:7.4f}\t"
                        f"{last_lr_scale:8.5f}\t"
                        f"{psi_real_all:9.4f}\t"
                        f"{psi_g:9.4f}\t"
                        f"{psi_g_resid:9.4f}\t"
                        f"{psi_g_w:9.4f}\t"
                        f"{psi_g_logsigma:9.4f}\t"
                        f"{psi_logdetA:9.4f}\t"
                        f"{psi_penalty:9.4f}\t"
                        f"{grad_psi_real:9.3e}\t"
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

        if lr_decay_every_epochs > 0 and ((epoch + 1) % lr_decay_every_epochs == 0):
            lr_scale *= lr_decay

    return w, rho, sigma2, A, B


import argparse
import math
from typing import Sequence

import torch

from bsgd import train_bsgd_neurips
from minimax import minimax_train_orf
from scgd import scgd_train_orf
from utils import exact_nlml_gradients, generate_gp_data, orf_features

_PHI_MODE_ALIASES = {"phi", "finite", "feature"}


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return requested


def _normalize_kernel_mode(mode: str) -> str:
    return (mode or "rbf").lower()


def _is_phi_mode(mode: str) -> bool:
    return _normalize_kernel_mode(mode) in _PHI_MODE_ALIASES


def _fork_rng_for_device(device: torch.device):
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return torch.random.fork_rng(devices=[idx])
    return torch.random.fork_rng()


def _build_phi_feature_map(args):
    num_features = args.phi_num_features
    if num_features <= 0:
        raise ValueError("--phi-num-features must be positive when kernel_mode='phi'.")

    phi_seed = args.phi_seed if args.phi_seed is not None else args.seed
    phi_lengthscale = (
        args.phi_lengthscale if args.phi_lengthscale is not None else args.lengthscale
    )

    def phi_fn(X: torch.Tensor) -> torch.Tensor:
        device = X.device
        with _fork_rng_for_device(device):
            torch.manual_seed(phi_seed)
            Z_base, _, _ = orf_features(
                X,
                num_features=num_features,
                lengthscale=phi_lengthscale,
                seed=phi_seed,
            )
        return Z_base

    return phi_fn


def _prepare_data(args):
    device = _resolve_device(args.device)
    kernel_mode = args.kernel_mode
    phi_fn = None
    return_phi = False
    if _is_phi_mode(kernel_mode):
        phi_fn = _build_phi_feature_map(args)
        return_phi = True

    data = generate_gp_data(
        n=args.n,
        lengthscale=args.lengthscale,
        sigma_f2=args.sigma_f2_true,
        sigma_eps2=args.sigma_eps2_true,
        device=device,
        seed=args.seed,
        cluster_strength=args.cluster_strength,
        heterogeneity=args.heterogeneity,
        kernel_mode=kernel_mode,
        phi=phi_fn,
        return_phi=return_phi,
    )
    if return_phi:
        X, y, K_f, Z = data
    else:
        X, y, K_f = data
        Z = None
    return device, X, y, K_f, Z


def _prepare_orf_features(X, lengthscale: float, num_features: int, seed: int):
    Z_base, _, _ = orf_features(
        X,
        num_features=num_features,
        lengthscale=lengthscale,
        seed=seed,
    )
    K_f_orf = Z_base @ Z_base.T
    return Z_base, K_f_orf


def run_bsgd(args):
    device, _, y, K_f, _ = _prepare_data(args)
    theta = train_bsgd_neurips(
        K_f,
        y,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        sigma_f2_init=args.sigma_f2_init,
        sigma_eps2_init=args.sigma_eps2_init,
        alpha1=args.alpha1,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        print_every=args.print_every,
        sigma_f2_true=args.sigma_f2_true,
        sigma_eps2_true=args.sigma_eps2_true,
    )
    print(
        "\n[BSGD] Finished on device {} -> σ_f² ≈ {:.4f}, σ_ε² ≈ {:.4f}".format(
            device, theta[0].item(), theta[1].item()
        )
    )


def run_minimax(args):
    device, X, y, K_f, Z_phi = _prepare_data(args)
    if Z_phi is not None:
        Z_base = Z_phi
        K_f_orf = K_f
    else:
        Z_base, K_f_orf = _prepare_orf_features(
            X,
            lengthscale=args.lengthscale,
            num_features=args.num_features,
            seed=args.seed,
        )
    w, rho, sigma2, A, B = minimax_train_orf(
        Z_base,
        X,
        y,
        K_f_exact=K_f,
        K_f_orf=K_f_orf,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        mu=args.mu,
        mu_increase_factor=args.mu_increase_factor,
        mu_increase_epochs=args.mu_increase_epochs,
        a=args.a,
        b=args.b,
        lr_decay=args.lr_decay,
        lr_decay_start_epoch=args.lr_decay_start_epoch,
        sigma_f2_init=args.sigma_f2_init,
        sigma_eps2_init=args.sigma_eps2_init,
        w_init_scale=args.w_init_scale,
        sigma_f2_true=args.sigma_f2_true,
        sigma_eps2_true=args.sigma_eps2_true,
        print_every=args.print_every,
    )
    sigma_f2 = torch.exp(rho).item()
    sigma_eps2 = sigma2.item()
    print(
        "\n[MINIMAX] Finished on device {} -> σ_f² ≈ {:.4f}, σ_ε² ≈ {:.4f}".format(
            device, sigma_f2, sigma_eps2
        )
    )
    return w, rho, sigma2, A, B


def run_scgd(args):
    device, X, y, K_f, Z_phi = _prepare_data(args)
    if Z_phi is not None:
        Z_base = Z_phi
        K_f_orf = K_f
    else:
        Z_base, K_f_orf = _prepare_orf_features(
            X,
            lengthscale=args.lengthscale,
            num_features=args.num_features,
            seed=args.seed,
        )
    w, rho, sigma2, F_tilde = scgd_train_orf(
        Z_base,
        X,
        y,
        K_f_exact=K_f,
        K_f_orf=K_f_orf,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        a0=args.a0,
        b0=args.b0,
        a_decay=args.a_decay,
        b_decay=args.b_decay,
        decay_start_epoch=args.decay_start_epoch,
        sigma_f2_init=args.sigma_f2_init,
        sigma_eps2_init=args.sigma_eps2_init,
        w_init_scale=args.w_init_scale,
        sigma_f2_true=args.sigma_f2_true,
        sigma_eps2_true=args.sigma_eps2_true,
        print_every=args.print_every,
        warm_start_w=args.warm_start_w,
    )
    sigma_f2 = torch.exp(rho).item()
    sigma_eps2 = sigma2.item()
    print(
        "\n[SCGD] Finished on device {} -> σ_f² ≈ {:.4f}, σ_ε² ≈ {:.4f}".format(
            device, sigma_f2, sigma_eps2
        )
    )
    return w, rho, sigma2, F_tilde


def run_nlml_grad(args):
    device, _, y, K_f, _ = _prepare_data(args)
    nlml, grad_rho, grad_sigma2 = exact_nlml_gradients(
        K_f,
        y,
        rho=args.rho,
        sigma_eps2=args.sigma2,
    )
    sigma_f2 = math.exp(args.rho)
    print(
        "\n[NLML-GRAD] Finished on device {} (seed {})".format(device, args.seed)
    )
    print(
        "Inputs:\tσ_f² = {:.4f}\tσ_ε² = {:.4f}".format(sigma_f2, args.sigma2)
    )
    print(
        "NLML = {:.6f}\t∂ℓ/∂ρ = {:.6f}\t∂ℓ/∂σ_ε² = {:.6f}".format(
            nlml, grad_rho, grad_sigma2
        )
    )


def _add_common_data_args(parser: argparse.ArgumentParser):
    parser.add_argument("--n", type=int, default=1024, help="Number of data points.")
    parser.add_argument("--lengthscale", type=float, default=0.5, help="RBF lengthscale.")
    parser.add_argument("--sigma-f2-true", type=float, default=4.0, help="Ground-truth σ_f² used to sample data.")
    parser.add_argument("--sigma-eps2-true", type=float, default=1.0, help="Ground-truth σ_ε² used to sample data.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data and ORF features.")
    parser.add_argument(
        "--cluster-strength",
        type=float,
        default=1.0,
        help=(
            "Controls how concentrated the inputs x_i are. 1.0 reproduces the "
            "original GP setup; values > 1 shrink the input cloud, making the "
            "RBF kernel increasingly globally correlated (near low-rank). "
            "In that regime, ORF-based SCGD/MINIMAX tend to outperform BSGD."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Computation device. 'auto' selects CUDA when available.",
    )
    parser.add_argument(
        "--heterogeneity",
        type=float,
        default=0.0,
        help=(
            "Controls multi-cluster / multi-scale structure of X. "
            "0.0 = original single Gaussian cloud. Larger values create "
            "two clusters with different densities and separation, producing "
            "a heterogeneous kernel that favors ORF-based methods (SCGD / "
            "MINIMAX) over submatrix-based BSGD, in a way that changing the "
            "RBF lengthscale alone cannot fix."
        ),
    )
    parser.add_argument(
        "--kernel-mode",
        default="rbf",
        choices=tuple(sorted(_PHI_MODE_ALIASES | {"rbf"})),
        help=(
            "Kernel used by generate_gp_data. 'rbf' reproduces the original "
            "exact RBF GP. 'phi'/'finite'/'feature' switch to a finite feature "
            "map so that K_f = φ(X) φ(X)^T."
        ),
    )
    parser.add_argument(
        "--phi-num-features",
        type=int,
        default=128,
        help=(
            "Feature dimension d for kernel_mode='phi'. Ignored for kernel_mode='rbf'."
        ),
    )
    parser.add_argument(
        "--phi-lengthscale",
        type=float,
        default=None,
        help=(
            "Lengthscale used when constructing φ(X) (defaults to --lengthscale "
            "when omitted)."
        ),
    )
    parser.add_argument(
        "--phi-seed",
        type=int,
        default=None,
        help=(
            "Random seed for the φ feature map; defaults to --seed when not provided."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Command line entry point for stochastic GP optimization algorithms.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    _add_common_data_args(common)

    orf_common = argparse.ArgumentParser(add_help=False)
    orf_common.add_argument(
        "--num-features",
        type=int,
        default=128,
        help="Number of ORF features for MINIMAX/SCGD.",
    )

    # BSGD parser
    bsgd_parser = subparsers.add_parser(
        "bsgd",
        parents=[common],
        help="Run the biased SGD algorithm on the exact kernel.",
    )
    bsgd_parser.add_argument("--n-epochs", type=int, default=25)
    bsgd_parser.add_argument("--batch-size", type=int, default=128)
    bsgd_parser.add_argument("--sigma-f2-init", type=float, default=1.0)
    bsgd_parser.add_argument("--sigma-eps2-init", type=float, default=1.0)
    bsgd_parser.add_argument("--alpha1", type=float, default=9.0, help="Initial LR scale, α_k = α1 / k.")
    bsgd_parser.add_argument("--theta-min", type=float, default=1e-4)
    bsgd_parser.add_argument("--theta-max", type=float, default=20.0)
    bsgd_parser.add_argument("--print-every", type=int, default=1)
    bsgd_parser.set_defaults(func=run_bsgd)

    # MINIMAX parser
    minimax_parser = subparsers.add_parser(
        "minimax",
        parents=[common, orf_common],
        help="Run the TMLR MINIMAX + ORF algorithm.",
    )
    minimax_parser.add_argument("--n-epochs", type=int, default=25)
    minimax_parser.add_argument("--batch-size", type=int, default=128)
    minimax_parser.add_argument("--mu", type=float, default=1.0)
    minimax_parser.add_argument(
        "--mu-increase-factor",
        type=float,
        default=1.0,
        help="Multiplicative factor applied to μ when increases trigger.",
    )
    minimax_parser.add_argument(
        "--mu-increase-epochs",
        type=int,
        default=0,
        help="Every this many epochs, multiply μ by the specified factor (0 disables).",
    )
    minimax_parser.add_argument("--a", type=float, default=1e-3, help="Step size for MIN step.")
    minimax_parser.add_argument("--b", type=float, default=1e-3, help="Step size for MAX step.")
    minimax_parser.add_argument(
        "--lr-decay",
        type=float,
        default=1.0,
        help="Exponential decay factor applied each minibatch (<=1).",
    )
    minimax_parser.add_argument(
        "--lr-decay-start-epoch",
        type=int,
        default=1,
        help="Epoch index (1-based) when the lr_decay factor starts applying.",
    )
    minimax_parser.add_argument("--sigma-f2-init", type=float, default=1.0)
    minimax_parser.add_argument("--sigma-eps2-init", type=float, default=1.0)
    minimax_parser.add_argument("--w-init-scale", type=float, default=0.1)
    minimax_parser.add_argument("--print-every", type=int, default=1)
    minimax_parser.set_defaults(func=run_minimax)

    # SCGD parser
    scgd_parser = subparsers.add_parser(
        "scgd",
        parents=[common, orf_common],
        help="Run the TMLR SCGD + ORF algorithm.",
    )
    scgd_parser.add_argument("--n-epochs", type=int, default=25)
    scgd_parser.add_argument("--batch-size", type=int, default=128)
    scgd_parser.add_argument("--a0", type=float, default=1e-3)
    scgd_parser.add_argument("--b0", type=float, default=1e-3)
    scgd_parser.add_argument("--a-decay", type=float, default=0.75, help="Exponent for a_t schedule.")
    scgd_parser.add_argument("--b-decay", type=float, default=0.5, help="Exponent for b_t schedule.")
    scgd_parser.add_argument(
        "--decay-start-epoch",
        type=int,
        default=1,
        help="Epoch index (1-based) after which the a/b schedules start decaying.",
    )
    scgd_parser.add_argument("--sigma-f2-init", type=float, default=1.0)
    scgd_parser.add_argument("--sigma-eps2-init", type=float, default=1.0)
    scgd_parser.add_argument("--w-init-scale", type=float, default=0.1)
    scgd_parser.add_argument(
        "--warm-start-w",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm-start w via a ridge solve (pass --no-warm-start-w to disable).",
    )
    scgd_parser.add_argument("--print-every", type=int, default=1)
    scgd_parser.set_defaults(func=run_scgd)

    # Exact NLML gradient parser
    grad_parser = subparsers.add_parser(
        "nlml-grad",
        parents=[common],
        help="Compute exact NLML value and gradients at (ρ, σ²).",
    )
    grad_parser.add_argument(
        "--rho",
        type=float,
        default=0.0,
        help="Log σ_f² at which to evaluate the NLML gradient.",
    )
    grad_parser.add_argument(
        "--sigma2",
        type=float,
        default=1.0,
        help="Noise variance σ_ε² at which to evaluate the NLML gradient.",
    )
    grad_parser.set_defaults(func=run_nlml_grad)

    return parser


def main(argv: Sequence[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()


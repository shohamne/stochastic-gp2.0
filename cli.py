import argparse
import math
from typing import Sequence

import torch

from bsgd import train_bsgd_neurips
from experiments import plot_neurips_figure2, run_neurips_figure2_experiment
from minimax import minimax_train_orf
from scgd import scgd_train_orf
from utils import exact_nlml_gradients, generate_gp_data, orf_features


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return requested


def _prepare_data(args):
    device = _resolve_device(args.device)
    X, y, K_f = generate_gp_data(
        n=args.n,
        lengthscale=args.lengthscale,
        sigma_f2=args.sigma_f2_true,
        sigma_eps2=args.sigma_eps2_true,
        device=device,
        seed=args.seed,
    )
    return device, X, y, K_f


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
    device, _, y, K_f = _prepare_data(args)
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
    )
    print(
        "\n[BSGD] Finished on device {} -> σ_f² ≈ {:.4f}, σ_ε² ≈ {:.4f}".format(
            device, theta[0].item(), theta[1].item()
        )
    )


def run_minimax(args):
    device, X, y, K_f = _prepare_data(args)
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
    device, X, y, K_f = _prepare_data(args)
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
    device, _, y, K_f = _prepare_data(args)
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


def run_figure2(args):
    device = _resolve_device(args.device)
    results = run_neurips_figure2_experiment(
        m_list=tuple(args.m_list),
        n=args.n,
        lengthscale=args.lengthscale,
        sigma_f2_true=args.sigma_f2_true,
        sigma_eps2_true=args.sigma_eps2_true,
        n_epochs=args.n_epochs,
        n_reps=args.n_reps,
        device=device,
    )
    plot_neurips_figure2(results, m_list=tuple(args.m_list), filename=args.output)
    print(
        "\n[Figure2] Finished on device {} -> results saved to {}".format(
            device, args.output
        )
    )


def _add_common_data_args(parser: argparse.ArgumentParser):
    parser.add_argument("--n", type=int, default=1024, help="Number of data points.")
    parser.add_argument("--lengthscale", type=float, default=0.5, help="RBF lengthscale.")
    parser.add_argument("--sigma-f2-true", type=float, default=4.0, help="Ground-truth σ_f² used to sample data.")
    parser.add_argument("--sigma-eps2-true", type=float, default=1.0, help="Ground-truth σ_ε² used to sample data.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data and ORF features.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Computation device. 'auto' selects CUDA when available.",
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
    scgd_parser.add_argument("--print-every", type=int, default=1)
    scgd_parser.set_defaults(func=run_scgd)

    # Figure 2 parser
    fig2_parser = subparsers.add_parser(
        "figure2",
        parents=[common],
        help="Reproduce NeurIPS Figure 2 experiment.",
    )
    fig2_parser.add_argument(
        "--m-list",
        type=int,
        nargs="+",
        default=(64, 32, 16),
        help="Mini-batch sizes to evaluate.",
    )
    fig2_parser.add_argument("--n-epochs", type=int, default=25)
    fig2_parser.add_argument("--n-reps", type=int, default=10)
    fig2_parser.add_argument("--output", default="neurips_figure2.png")
    fig2_parser.set_defaults(func=run_figure2)

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


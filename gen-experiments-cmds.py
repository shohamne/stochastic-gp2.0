#!/usr/bin/env python
"""
Generate shell commands to reproduce the NeurIPS-2020 Figure 1 and Figure 2
experiments for ALL three optimizers implemented in this codebase:

  - BSGD on the exact kernel          (cli.py bsgd / figure2)
  - MINIMAX + ORF approximation      (cli.py minimax)
  - SCGD   + ORF approximation       (cli.py scgd)

Figure 1 (parameter trajectories):
  * n = 1024, lengthscale = 0.5
  * true (σ_f², σ_ε²) = (4.0, 1.0)
  * minibatch size m = 128
  * 25 epochs  ->  25 * (1024 / 128) = 200 iterations
  * 10 independent data pools (seeds 0..9)
  * 3 initial points:
        θ(0) = (5.0, 3.0), α1 = 9
        θ(0) = (2.5, 3.5), α1 = 9
        θ(0) = (2.5, 0.7), α1 = 6          (for BSGD only)

Figure 2 (full gradient vs minibatch size, BSGD):
  * m ∈ {64, 32, 16}
  * α_k = α1 / k with α1 = 9
  * 25 epochs, 10 repetitions (seeds 0..9)
  * implemented by cli.py "figure2" command.

For MINIMAX and SCGD we generate Figure‑1‑style runs (same data regime,
same initial θ), so you can plot σ_f² and σ_ε² vs iteration from their logs.
Reproducing a *gradient‑norm* style Figure 2 for them would require extra
instrumentation inside minimax.py / scgd.py.

Usage
-----
From the repo root (where cli.py lives):

    python generate_fig_cmds.py > run_fig_experiments.sh
    bash run_fig_experiments.sh

You can also tweak defaults below (device, number of reps, etc.).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate shell commands for NeurIPS Figure 1 & 2 experiments."
    )
    p.add_argument(
        "--python-bin",
        default="python",
        help="Python executable to use in generated commands (default: python).",
    )
    p.add_argument(
        "--cli-path",
        default="cli.py",
        help="Path to cli.py relative to this script (default: cli.py).",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device argument passed to cli.py (default: auto).",
    )
    p.add_argument(
        "--n-reps",
        type=int,
        default=10,
        help="Number of independent data pools / seeds (default: 10, as in the paper).",
    )
    p.add_argument(
        "--logdir",
        default="logs",
        help="Base directory for stdout logs in the generated commands.",
    )
    p.add_argument(
        "--figdir",
        default="figures",
        help="Directory where Figure 2 images will be written.",
    )
    p.add_argument(
        "--kernel-mode",
        default="finite",
        choices=("rbf", "phi", "finite", "feature"),
        help="Kernel mode passed through to cli.py (default: rbf).",
    )
    return p


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def sh_quote(s: str) -> str:
    """Very small helper for single-quoting shell arguments."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


def main() -> None:
    args = build_parser().parse_args()

    logdir = Path(args.logdir)
    figdir = Path(args.figdir)

    # We only *mention* these directories in the commands; bash will create them.
    # Still, create them now so the user sees them on disk after redirecting.
    logdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    py = args.python_bin
    cli = args.cli_path
    device = args.device
    n_reps = args.n_reps
    kernel_mode = args.kernel_mode

    # Common GP setup (matches Section 7.1 / Figures 1-2 in the paper).
    common_gp_flags = (
        "--n 1024 "
        "--lengthscale 0.5 "
        "--sigma-f2-true 4.0 "
        "--sigma-eps2-true 1.0 "
        "--n-epochs 200 "
        f"--device {device} "
        f"--kernel-mode {kernel_mode} "
    )

    algo_common_flags = {
        "bsgd": (
            "--theta-min 1e-4 "
            "--theta-max 20.0 "
            "--print-every 1 "
        ),
        "minimax": (
            "--a 1e-4 "
            "--b 1e-3 "
            "--w-init-scale 0.1 "
            "--num-features 128 "
            "--print-every 1 "
            "--mu-increase-factor 1.0 "
            "--mu-increase-epochs 0 "
        ),
        "scgd": (
            "--a0 2e-3 "
            "--a-decay 0.75 "
            "--w-init-scale 0.1 "
            "--num-features 128 "
            "--print-every 1 "
        ),
    }

    def build_algo_cmd(algo: str, extra_flags: str) -> str:
        algo_flags = algo_common_flags.get(algo, "")
        return (
            f"{py} {cli} {algo} "
            f"{common_gp_flags}"
            f"{algo_flags}"
            f"{extra_flags}"
        )

    def build_algo_extra_flags(
        algo: str,
        *,
        cfg: dict[str, float] | None,
        seed: int,
        batch_size: int,
        sigma_f2_init: float | None = None,
        sigma_eps2_init: float | None = None,
    ) -> str:
        if sigma_f2_init is None:
            if cfg is None or "sigma_f2_init" not in cfg:
                raise ValueError(f"Missing sigma_f2_init for {algo}")
            sigma_f2_init = cfg["sigma_f2_init"]
        if sigma_eps2_init is None:
            if cfg is None or "sigma_eps2_init" not in cfg:
                raise ValueError(f"Missing sigma_eps2_init for {algo}")
            sigma_eps2_init = cfg["sigma_eps2_init"]

        flags = [
            f"--seed {seed}",
            f"--batch-size {batch_size}",
        ]

        if algo == "bsgd":
            if cfg is None or "alpha1" not in cfg:
                raise ValueError("BSGD requires alpha1 in cfg")
            flags.extend(
                [
                    f"--sigma-f2-init {sigma_f2_init}",
                    f"--sigma-eps2-init {sigma_eps2_init}",
                    f"--alpha1 {cfg['alpha1']}",
                ]
            )
        elif algo == "minimax":
            flags.extend(
                [
                    "--mu 2.0",
                    "--lr-decay 1.0",
                    "--lr-decay-start-epoch 0",
                    f"--sigma-f2-init {sigma_f2_init}",
                    f"--sigma-eps2-init {sigma_eps2_init}",
                ]
            )
        elif algo == "scgd":
            flags.extend(
                [
                    "--b0 0.9",
                    "--b-decay 0.25",
                    "--decay-start-epoch 25",
                    f"--sigma-f2-init {sigma_f2_init}",
                    f"--sigma-eps2-init {sigma_eps2_init}",
                ]
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        return " ".join(flags) + " "

    # ------------------------------------------------------------------
    # Figure 1: parameter convergence for ALL algorithms
    # ------------------------------------------------------------------
    #
    # Three initial points from Figure 1 (page 8):
    #   θ(0) = (5.0, 3.0), α1 = 9.0
    #   θ(0) = (2.5, 3.5), α1 = 9.0
    #   θ(0) = (2.5, 0.7), α1 = 6.0
    #
    # (α1 is only relevant for BSGD; MINIMAX/SCGD keep their own step sizes.)
    #

    init_configs = [
        {
            "name": "theta0_5.0_3.0",
            "sigma_f2_init": 5.0,
            "sigma_eps2_init": 3.0,
            "alpha1": 9.0,
        },
        {
            "name": "theta0_2.5_3.5",
            "sigma_f2_init": 2.5,
            "sigma_eps2_init": 3.5,
            "alpha1": 9.0,
        },
        {
            "name": "theta0_2.5_0.7",
            "sigma_f2_init": 2.5,
            "sigma_eps2_init": 0.7,
            "alpha1": 6.0,
        },
    ]

    seeds = list(range(n_reps))

    # -----------------------------
    # Figure 1 – BSGD
    # -----------------------------
    print("# ------------------------------------------------------------------")
    print("# Figure 1: BSGD (biased SGD on exact kernel) – all initial points")
    print("# ------------------------------------------------------------------")
    for cfg in init_configs:
        for seed in seeds:
            log_path = logdir / f"fig1_bsgd_{cfg['name']}_seed{seed}.log"
            extra_flags = build_algo_extra_flags(
                "bsgd", cfg=cfg, seed=seed, batch_size=128
            )
            print(build_algo_cmd("bsgd", extra_flags))
    print()

    # -----------------------------
    # Figure 1 – MINIMAX + ORF
    # -----------------------------
    print("# ------------------------------------------------------------------")
    print("# Figure 1: MINIMAX + ORF – same initial θ, 25 epochs, m = 128")
    print("# (You can reconstruct σ_f² and σ_ε² trajectories from the logs.)")
    print("# ------------------------------------------------------------------")
    for cfg in init_configs:
        for seed in seeds:
            log_path = logdir / f"fig1_minimax_{cfg['name']}_seed{seed}.log"
            extra_flags = build_algo_extra_flags(
                "minimax", cfg=cfg, seed=seed, batch_size=128
            )
            print(build_algo_cmd("minimax", extra_flags))
    print()

    # -----------------------------
    # Figure 1 – SCGD + ORF
    # -----------------------------
    print("# ------------------------------------------------------------------")
    print("# Figure 1: SCGD + ORF – same initial θ, 25 epochs, m = 128")
    print("# (Again, σ_f² and σ_ε² vs iteration can be parsed from logs.)")
    print("# ------------------------------------------------------------------")
    for cfg in init_configs:
        for seed in seeds:
            log_path = logdir / f"fig1_scgd_{cfg['name']}_seed{seed}.log"
            extra_flags = build_algo_extra_flags(
                "scgd", cfg=cfg, seed=seed, batch_size=128
            )
            print(build_algo_cmd("scgd", extra_flags))
    print()

    # ------------------------------------------------------------------
    # Figure 2: full gradient vs minibatch size – BSGD
    # ------------------------------------------------------------------
    #
    # Matches the Figure‑1 setup for every other hyperparameter; only the
    # minibatch size and (σ_f², σ_ε²) initializations are different.
    #
    print("# ------------------------------------------------------------------")
    print("# Figure 2: BSGD gradient-norm experiment (match Figure 1 params)")
    print("# ------------------------------------------------------------------")

    m_list = [256, 512, 1024]
    for cfg in init_configs:
        for m in m_list:
            for seed in seeds:
                log_path = logdir / f"fig2_bsgd_{cfg['name']}_m{m}_seed{seed}.log"
                extra_flags = build_algo_extra_flags(
                    "bsgd",
                    cfg=cfg,
                    seed=seed,
                    batch_size=m,
                    sigma_f2_init=5.0,
                    sigma_eps2_init=3.0,
                )
                print(build_algo_cmd("bsgd", extra_flags))
    print()

    # ------------------------------------------------------------------
    # Optional: Figure‑2‑style regimes for MINIMAX / SCGD
    # ------------------------------------------------------------------
    #
    # We cannot compute the *full GP gradient* for MINIMAX / SCGD without
    # adding new instrumentation. However, we still run them in the exact
    # Figure‑1 configuration and only change the batch size (and optionally
    # the θ initialization) so every other hyperparameter matches Figure 1.
    #
    print("# ------------------------------------------------------------------")
    print("# OPTIONAL: Figure-2-style runs for MINIMAX (varying minibatch size)")
    print("# ------------------------------------------------------------------")
    for m in m_list:
        for seed in seeds:
            log_path = logdir / f"fig2_minimax_m{m}_seed{seed}.log"
            extra_flags = build_algo_extra_flags(
                "minimax",
                cfg=None,
                seed=seed,
                batch_size=m,
                sigma_f2_init=5.0,
                sigma_eps2_init=3.0,
            )
            print(build_algo_cmd("minimax", extra_flags))
    print()

    print("# ------------------------------------------------------------------")
    print("# OPTIONAL: Figure-2-style runs for SCGD (varying minibatch size)")
    print("# ------------------------------------------------------------------")
    for m in m_list:
        for seed in seeds:
            log_path = logdir / f"fig2_scgd_m{m}_seed{seed}.log"
            extra_flags = build_algo_extra_flags(
                "scgd",
                cfg=None,
                seed=seed,
                batch_size=m,
                sigma_f2_init=5.0,
                sigma_eps2_init=3.0,
            )
            print(build_algo_cmd("scgd", extra_flags))
    print()

    print("# End of generated commands.")


if __name__ == "__main__":
    main()

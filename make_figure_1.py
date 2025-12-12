#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from collect_results import collect_stdout_logs


# Initial points used in NeurIPS Figure 1 / gen-experiments-cmds.py
INIT_CONFIGS = [
    {
        "name": r"$\theta^{(0)} = (5.0, 3.0)$",
        "sigma_f2_init": 5.0,
        "sigma_eps2_init": 3.0,
        "alpha1": 9.0,
    },
    {
        "name": r"$\theta^{(0)} = (2.5, 3.5)$",
        "sigma_f2_init": 2.5,
        "sigma_eps2_init": 3.5,
        "alpha1": 9.0,
    },
    {
        "name": r"$\theta^{(0)} = (2.5, 0.7)$",
        "sigma_f2_init": 2.5,
        "sigma_eps2_init": 0.7,
        "alpha1": 6.0,
    },
]

ALGO_ORDER = ("bsgd", "minimax", "scgd")
ALGO_TITLES = {
    "bsgd": "BSGD (exact kernel)",
    "minimax": "MINIMAX + ORF",
    "scgd": "SCGD + ORF",
}


def make_figure1_from_df(
    df,
    output: str | Path = "neurips_figure1_all_algos.png",
    smooth_window: int = 1,
    batch_size: int | None = 128,
) -> None:
    """
    Create a 3x3 panel plot:

        rows    = algorithms (BSGD, MINIMAX, SCGD)
        columns = initial θ (5.0,3.0), (2.5,3.5), (2.5,0.7)

    Each panel shows trajectories of σ_f² and σ_ε² vs iteration for 10 seeds.
    """

    if df.empty:
        raise RuntimeError("DataFrame is empty – no stdout logs found.")

    if batch_size is not None:
        if "batch_size" not in df.columns:
            raise RuntimeError(
                "DataFrame is missing the 'batch_size' column; cannot filter runs."
            )
        df = df[df["batch_size"] == batch_size].copy()
        if df.empty:
            raise RuntimeError(
                f"No rows left after filtering to batch_size == {batch_size}."
            )

    # Some SCGD/MINIMAX directories hit filesystem filename limits, which
    # truncated "--sigma-eps2-init <value>" down to "--sigma-eps2-init 3."/
    # "0.". That in turn makes the parsed metadata show 3.0 / 0.0 even though
    # the intended initializations were 3.5 / 0.7.  Since all ambiguous runs
    # use σ_f²_init = 2.5, we can safely fix them up here before applying
    # per-configuration filters.
    truncated_init_fixups = [
        {
            "algos": ("scgd", "minimax"),
            "sigma_f2_init": 2.5,
            "observed_eps2": 3.0,
            "corrected_eps2": 3.5,
        },
        {
            "algos": ("scgd", "minimax"),
            "sigma_f2_init": 2.5,
            "observed_eps2": 0.0,
            "corrected_eps2": 0.7,
        },
    ]
    for fix in truncated_init_fixups:
        mask_fix = (
            df["algo"].isin(fix["algos"])
            & np.isclose(df["sigma_f2_init"], fix["sigma_f2_init"])
            & np.isclose(df["sigma_eps2_init"], fix["observed_eps2"])
        )
        df.loc[mask_fix, "sigma_eps2_init"] = fix["corrected_eps2"]

    # Keep all available runs per algorithm (each run may have a different
    # number of epochs) while still respecting the batch-size constraint above.
    df_bsgd = df[df["algo"] == "bsgd"]
    df_minimax = df[df["algo"] == "minimax"]
    df_scgd = df[df["algo"] == "scgd"]

    algo_to_df = {
        "bsgd": df_bsgd,
        "minimax": df_minimax,
        "scgd": df_scgd,
    }

    smooth_window = int(max(smooth_window, 1))

    def smooth_series(y: np.ndarray) -> np.ndarray:
        if smooth_window <= 1:
            return y
        kernel = np.ones(smooth_window, dtype=float) / smooth_window
        return np.convolve(y, kernel, mode="same")

    required_init_cols = ("sigma_f2_init", "sigma_eps2_init")
    missing_init_cols = [col for col in required_init_cols if col not in df.columns]
    if missing_init_cols:
        cols = ", ".join(missing_init_cols)
        raise RuntimeError(
            f"DataFrame is missing required initialization metadata: {cols}"
        )

    fig, axes = plt.subplots(
        nrows=len(ALGO_ORDER),
        ncols=len(INIT_CONFIGS),
        figsize=(12, 9),
        sharex=True,
        sharey=True,
    )

    # Colormap for repetitions (seeds)
    all_seeds = sorted(df["seed"].unique())
    num_colors = max(len(all_seeds), 1)
    cmap = plt.get_cmap("tab10", num_colors)
    seed_to_color = {seed: cmap(i % cmap.N) for i, seed in enumerate(all_seeds)}

    y_min, y_max = np.inf, -np.inf

    for row_idx, algo in enumerate(ALGO_ORDER):
        df_algo = algo_to_df[algo]
        ax_row = axes[row_idx]

        if df_algo.empty:
            # Nothing to plot for this algorithm; skip row
            for ax in (ax_row if isinstance(ax_row, np.ndarray) else [ax_row]):
                ax.set_visible(False)
            continue

        for col_idx, cfg in enumerate(INIT_CONFIGS):
            ax = ax_row[col_idx] if isinstance(ax_row, np.ndarray) else ax_row

            mask_cfg = (
                (df_algo["sigma_f2_init"] == cfg["sigma_f2_init"])
                & (df_algo["sigma_eps2_init"] == cfg["sigma_eps2_init"])
            )
            # For BSGD, also require matching α1 to avoid mixing configs
            if algo == "bsgd":
                mask_cfg &= df_algo["alpha1"] == cfg["alpha1"]

            df_cfg = df_algo[mask_cfg]
            if df_cfg.empty:
                ax.set_visible(False)
                continue

            seeds = sorted(df_cfg["seed"].unique())

            # Plot each repetition
            for seed in seeds:
                df_seed = (
                    df_cfg[df_cfg["seed"] == seed]
                    .sort_values("iter")
                    .reset_index(drop=True)
                )
                x = df_seed["iter"].to_numpy()

                sigma_f2 = smooth_series(
                    df_seed["sigma_f2"].to_numpy(dtype=float)
                )
                sigma_eps2 = smooth_series(
                    df_seed["sigma_eps2"].to_numpy(dtype=float)
                )

                trim = min(smooth_window, len(x))
                if trim >= len(x):
                    continue
                if trim > 0:
                    x = x[:-trim]
                    sigma_f2 = sigma_f2[:-trim]
                    sigma_eps2 = sigma_eps2[:-trim]

                color = seed_to_color.get(seed, "C0")
                ax.plot(
                    x,
                    sigma_f2,
                    color=color,
                    linewidth=1.0,
                    alpha=0.8,
                    linestyle="-",
                )
                ax.plot(
                    x,
                    sigma_eps2,
                    color=color,
                    linewidth=1.0,
                    alpha=0.8,
                    linestyle="--",
                )

                y_min = min(y_min, sigma_f2.min(), sigma_eps2.min())
                y_max = max(y_max, sigma_f2.max(), sigma_eps2.max())

            # Ground-truth horizontal lines
            def _unique_scalar(df_section, column: str) -> float | None:
                if column not in df_section.columns:
                    return None
                values = df_section[column].dropna().unique()
                if len(values) != 1:
                    return None
                try:
                    return float(values[0])
                except (TypeError, ValueError):
                    return None

            sigma_f2_true = _unique_scalar(df_cfg, "sigma_f2_true")
            sigma_eps2_true = _unique_scalar(df_cfg, "sigma_eps2_true")

            if sigma_f2_true is not None:
                ax.axhline(
                    sigma_f2_true,
                    color="k",
                    linestyle="-",
                    linewidth=1.0,
                    alpha=0.9,
                )
            if sigma_eps2_true is not None:
                ax.axhline(
                    sigma_eps2_true,
                    color="k",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.9,
                )

            # Titles: top row gets θ⁽⁰⁾ labels
            if row_idx == 0:
                ax.set_title(cfg["name"])

            # Leftmost column: algorithm labels on the left
            if col_idx == 0:
                ax.set_ylabel(
                    ALGO_TITLES[algo] + "\n" + r"$\theta^{(k)}$", fontsize=10
                )

            # Bottom row: x-axis label
            if row_idx == len(ALGO_ORDER) - 1:
                ax.set_xlabel("Iteration $k$")

            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    # Set a reasonable shared y-range
    if np.isfinite(y_min) and np.isfinite(y_max):
        pad = 0.05 * (y_max - y_min)
        y_min = max(0.0, y_min - pad)
        y_max = y_max + pad
        for ax in fig.axes:
            if ax.get_visible():
                ax.set_ylim(y_min, y_max)

    # Global legend: parameter styles + ground truth
    from matplotlib.lines import Line2D

    example_color = seed_to_color[all_seeds[0]] if all_seeds else "C0"
    legend_handles = []
    legend_labels = []

    # Explicitly show which color corresponds to which seed/run.
    if all_seeds:
        for seed in all_seeds:
            legend_handles.append(
                Line2D([0], [0], color=seed_to_color[seed], linestyle="-", linewidth=1.5)
            )
            legend_labels.append(f"Seed {seed}")

    legend_handles.extend(
        [
            Line2D([0], [0], color=example_color, linestyle="-", linewidth=1.5),
            Line2D([0], [0], color=example_color, linestyle="--", linewidth=1.5),
            Line2D([0], [0], color="k", linestyle="-", linewidth=1.0),
            Line2D([0], [0], color="k", linestyle=":", linewidth=1.0),
        ]
    )
    legend_labels.extend(
        [
            r"$\sigma_f^2$ (per repetition)",
            r"$\sigma_\varepsilon^2$ (per repetition)",
            r"True $\sigma_f^2$",
            r"True $\sigma_\varepsilon^2$",
        ]
    )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.98),
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    output = Path(output)
    fig.savefig(output, dpi=150)
    print(f"[Figure 1] Saved plot to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct NeurIPS Figure 1 (parameter trajectories) for "
            "BSGD, MINIMAX and SCGD from stdout logs."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("res-17"),
        help="Base directory containing stdout logs (default: res-11).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("neurips_figure1_all_algos.png"),
        help="Output image filename.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help=(
            "Moving-average window size (in iterations) applied to each "
            "trajectory before plotting (default: 1, i.e., no smoothing)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help=(
            "Minibatch size that runs must match to be included. "
            "Set to another value (or -1) to select different runs; "
            "use -1 to disable batch-size filtering."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    df = collect_stdout_logs(args.root)
    if df.empty:
        print(f"No stdout records found under {args.root}")
        return

    make_figure1_from_df(
        df,
        output=args.output,
        smooth_window=args.smooth_window,
        batch_size=None if args.batch_size == -1 else args.batch_size,
    )


if __name__ == "__main__":
    main()

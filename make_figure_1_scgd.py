#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from collect_results import collect_stdout_logs


INIT_CONFIGS = [
    {
        "name": r"$\theta^{(0)} = (5.0, 3.0)$",
        "sigma_f2_init": 5.0,
        "sigma_eps2_init": 3.0,
    },
    {
        "name": r"$\theta^{(0)} = (2.5, 3.5)$",
        "sigma_f2_init": 2.5,
        "sigma_eps2_init": 3.5,
    },
    {
        "name": r"$\theta^{(0)} = (2.5, 0.7)$",
        "sigma_f2_init": 2.5,
        "sigma_eps2_init": 0.7,
    },
]

TARGET_ALGO = "scgd"
ALGO_TITLE = "SCGD + ORF"


def make_scgd_figure_from_df(
    df,
    output: str | Path = "neurips_figure1_scgd.png",
    smooth_window: int = 1,
    batch_size: int | None = 128,
    max_iter: int | None = None,
    max_epoch: int | None = None,
) -> None:
    """
    Create a 1x3 panel plot for SCGD showing σ_f² and σ_ε² trajectories across seeds.

    Columns correspond to the three initializations from INIT_CONFIGS.
    Optionally clamp trajectories to iterations <= max_iter and/or epochs <= max_epoch.
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

    # Fix truncated "--sigma-eps2-init" flags for SCGD runs (see make_figure_1.py).
    truncated_init_fixups = [
        {
            "sigma_f2_init": 2.5,
            "observed_eps2": 3.0,
            "corrected_eps2": 3.5,
        },
        {
            "sigma_f2_init": 2.5,
            "observed_eps2": 0.0,
            "corrected_eps2": 0.7,
        },
    ]
    for fix in truncated_init_fixups:
        mask_fix = (
            np.isclose(df["sigma_f2_init"], fix["sigma_f2_init"])
            & np.isclose(df["sigma_eps2_init"], fix["observed_eps2"])
            & (df["algo"] == TARGET_ALGO)
        )
        df.loc[mask_fix, "sigma_eps2_init"] = fix["corrected_eps2"]

    df = df[df["algo"] == TARGET_ALGO]
    if df.empty:
        raise RuntimeError("No SCGD runs available after filtering.")

    if max_iter is not None:
        if "iter" not in df.columns:
            raise RuntimeError(
                "DataFrame is missing the 'iter' column; cannot apply --max-iter."
            )
        df = df[df["iter"] <= max_iter].copy()
        if df.empty:
            raise RuntimeError(
                f"No rows left after filtering to iterations <= {max_iter}."
            )

    if max_epoch is not None:
        if "epoch" not in df.columns:
            raise RuntimeError(
                "DataFrame is missing the 'epoch' column; cannot apply --max-epoch."
            )
        df = df[df["epoch"] <= max_epoch].copy()
        if df.empty:
            raise RuntimeError(
                f"No rows left after filtering to epochs <= {max_epoch}."
            )

    smooth_window = int(max(smooth_window, 1))

    def smooth_series(y: np.ndarray) -> np.ndarray:
        if smooth_window <= 1:
            return y
        kernel = np.ones(smooth_window, dtype=float) / smooth_window
        return np.convolve(y, kernel, mode="same")

    required_cols = ("sigma_f2_init", "sigma_eps2_init")
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        cols = ", ".join(missing_cols)
        raise RuntimeError(
            f"DataFrame is missing required initialization metadata: {cols}"
        )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(INIT_CONFIGS),
        figsize=(14, 5.2),
        sharex=True,
        sharey=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax in axes.ravel():
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        ax.tick_params(axis="x", labelrotation=0)

    all_seeds = sorted(df["seed"].unique())
    num_colors = max(len(all_seeds), 1)
    cmap = plt.get_cmap("tab10", num_colors)
    seed_to_color = {seed: cmap(i % cmap.N) for i, seed in enumerate(all_seeds)}

    y_min, y_max = np.inf, -np.inf

    for col_idx, cfg in enumerate(INIT_CONFIGS):
        ax = axes[col_idx]
        mask_cfg = (
            np.isclose(df["sigma_f2_init"], cfg["sigma_f2_init"])
            & np.isclose(df["sigma_eps2_init"], cfg["sigma_eps2_init"])
        )
        df_cfg = df[mask_cfg]
        if df_cfg.empty:
            ax.set_visible(False)
            continue

        seeds = sorted(df_cfg["seed"].unique())
        for seed in seeds:
            df_seed = (
                df_cfg[df_cfg["seed"] == seed]
                .sort_values("iter")
                .reset_index(drop=True)
            )
            x = df_seed["iter"].to_numpy()

            sigma_f2 = smooth_series(df_seed["sigma_f2"].to_numpy(dtype=float))
            sigma_eps2 = smooth_series(df_seed["sigma_eps2"].to_numpy(dtype=float))

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

        ax.set_title(cfg["name"])
        ax.set_xlabel("Iteration $t$")
        if col_idx == 0:
            ax.set_ylabel(r"$\theta^{(t)}$", fontsize=10)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    if np.isfinite(y_min) and np.isfinite(y_max):
        pad = 0.05 * (y_max - y_min)
        y_min = max(0.0, y_min - pad)
        y_max = y_max + pad
        for ax in fig.axes:
            if ax.get_visible():
                ax.set_ylim(y_min, y_max)

    from matplotlib.lines import Line2D

    example_color = seed_to_color[all_seeds[0]] if all_seeds else "C0"
    seed_handles = []
    seed_labels = []
    if all_seeds:
        for seed in all_seeds:
            seed_handles.append(
                Line2D([0], [0], color=seed_to_color[seed], linestyle="-", linewidth=1.8)
            )
            seed_labels.append(f"Seed {seed}")

    style_handles = [
        Line2D([0], [0], color=example_color, linestyle="-", linewidth=2.0),
        Line2D([0], [0], color=example_color, linestyle="--", linewidth=2.0),
    ]
    style_labels = [
        r"$\sigma_f^2$",
        r"$\sigma_\varepsilon^2$",
    ]

    if seed_handles:
        seed_legend = fig.legend(
            seed_handles,
            seed_labels,
            loc="lower center",
            bbox_to_anchor=(0.3, 0.12),
            ncol=min(len(seed_handles), 5),
        )
        fig.add_artist(seed_legend)

    style_legend = fig.legend(
        style_handles,
        style_labels,
        loc="lower center",
        bbox_to_anchor=(0.7, 0.13),
        ncol=2,
    )
    fig.add_artist(style_legend)

    plt.tight_layout(rect=[0.02, 0.24, 0.98, 0.98])
    output = Path(output)
    fig.savefig(output, dpi=150)
    print(f"[SCGD Figure] Saved plot to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot SCGD parameter trajectories for the NeurIPS Figure 1 setup."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("res-17"),
        help="Base directory containing stdout logs (default: res-17).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("neurips_figure1_scgd.png"),
        help="Output image filename.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Moving-average window size applied before plotting (default: 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help=(
            "Minibatch size that runs must match to be included. Set to another "
            "value (or -1) to select different runs; use -1 to disable filtering."
        ),
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=-1,
        help=(
            "Maximum iteration included in the plot. Use -1 to keep all iterations."
        ),
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=50000,
        help="Maximum epoch included in the plot. Use -1 to keep all epochs.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    df = collect_stdout_logs(args.root)
    if df.empty:
        print(f"No stdout records found under {args.root}")
        return

    make_scgd_figure_from_df(
        df,
        output=args.output,
        smooth_window=args.smooth_window,
        batch_size=None if args.batch_size == -1 else args.batch_size,
        max_iter=None if args.max_iter == -1 else args.max_iter,
        max_epoch=None if args.max_epoch == -1 else args.max_epoch,
    )


if __name__ == "__main__":
    main()



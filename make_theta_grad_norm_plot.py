#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from collect_results import collect_stdout_logs

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


def _nan_moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.copy()

    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y)
    if not valid.any():
        return np.full_like(y, np.nan, dtype=float)

    kernel = np.ones(window, dtype=float)
    num = np.convolve(np.where(valid, y, 0.0), kernel, mode="same")
    den = np.convolve(valid.astype(float), kernel, mode="same")

    out = np.full_like(y, np.nan, dtype=float)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def _smooth_columns(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    smoothed = np.full_like(values, np.nan, dtype=float)
    for j in range(values.shape[1]):
        smoothed[:, j] = _nan_moving_average(values[:, j], window)
    return smoothed


def make_theta_grad_norm_plot_from_df(
    df,
    output: str | Path = "theta_grad_norm_mean_std.png",
    grad_column: str = "grad_norm",
    smooth_window: int = 1,
    batch_size: int | None = 128,
) -> None:
    """Plot mean ± std of grad norms in the Figure-1 layout."""

    if df.empty:
        raise RuntimeError("DataFrame is empty – no stdout logs found.")

    required_cols = {
        "algo",
        "iter",
        "seed",
        grad_column,
        "sigma_f2_init",
        "sigma_eps2_init",
        "n_epochs",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(
            "DataFrame is missing required columns: " + ", ".join(sorted(missing))
        )

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

    df_bsgd = df[(df["algo"] == "bsgd") & (df["n_epochs"] == 200)]
    df_minimax = df[(df["algo"] == "minimax") & (df["n_epochs"] == 200)]
    df_scgd = df[(df["algo"] == "scgd") & (df["n_epochs"] == 200)]

    algo_to_df = {
        "bsgd": df_bsgd,
        "minimax": df_minimax,
        "scgd": df_scgd,
    }

    smooth_window = int(max(smooth_window, 1))

    fig, axes = plt.subplots(
        nrows=len(ALGO_ORDER),
        ncols=len(INIT_CONFIGS),
        figsize=(12, 9),
        sharex=True,
        sharey=True,
    )

    y_min, y_max = np.inf, -np.inf

    for row_idx, algo in enumerate(ALGO_ORDER):
        df_algo = algo_to_df[algo]
        ax_row = axes[row_idx]

        if df_algo.empty:
            for ax in (ax_row if isinstance(ax_row, np.ndarray) else [ax_row]):
                ax.set_visible(False)
            continue

        for col_idx, cfg in enumerate(INIT_CONFIGS):
            ax = ax_row[col_idx] if isinstance(ax_row, np.ndarray) else ax_row
            mask_cfg = (
                np.isclose(df_algo["sigma_f2_init"], cfg["sigma_f2_init"])
                & np.isclose(df_algo["sigma_eps2_init"], cfg["sigma_eps2_init"])
            )
            if algo == "bsgd" and "alpha1" in df_algo.columns:
                mask_cfg &= np.isclose(df_algo["alpha1"], cfg["alpha1"])

            df_cfg = df_algo[mask_cfg].copy()
            if df_cfg.empty:
                ax.set_visible(False)
                continue

            pivot = (
                df_cfg.pivot_table(
                    index="iter",
                    columns="seed",
                    values=grad_column,
                    aggfunc="last",
                )
                .sort_index()
            )
            if pivot.empty:
                ax.set_visible(False)
                continue

            values = pivot.to_numpy(dtype=float)
            values = _smooth_columns(values, smooth_window)

            mean_grad = np.nanmean(values, axis=1)
            std_grad = np.nanstd(values, axis=1, ddof=1)
            iters = pivot.index.to_numpy(dtype=float)

            lower = mean_grad - std_grad
            upper = mean_grad + std_grad

            ax.plot(iters, mean_grad, color="C0", linewidth=1.5, label="Mean grad norm")
            ax.fill_between(
                iters,
                lower,
                upper,
                color="C0",
                alpha=0.3,
                linewidth=0.0,
                label="±1 std",
            )

            finite_indices = np.where(np.isfinite(mean_grad))[0]
            if finite_indices.size:
                final_idx = finite_indices[-1]
                final_mean = float(mean_grad[final_idx])
                ax.axhline(
                    final_mean,
                    color="k",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.8,
                )
                ax.text(
                    1.01,
                    final_mean,
                    f"{final_mean:.3g}",
                    transform=ax.get_yaxis_transform(),
                    fontsize=8,
                    ha="left",
                    va="center",
                    color="k",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.7,
                    ),
                    clip_on=False,
                )

            panel_y_min = np.nanmin(lower)
            panel_y_max = np.nanmax(upper)
            y_min = min(y_min, panel_y_min)
            y_max = max(y_max, panel_y_max)

            if row_idx == 0:
                ax.set_title(cfg["name"])

            if col_idx == 0:
                ax.set_ylabel(
                    ALGO_TITLES[algo] + "\n" + r"$\|\nabla \ell(\theta^{(k)})\|_2$",
                    fontsize=10,
                )

            if row_idx == len(ALGO_ORDER) - 1:
                ax.set_xlabel("Iteration $k$")

            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    if np.isfinite(y_min) and np.isfinite(y_max):
        pad = 0.05 * (y_max - y_min)
        lo = max(0.0, y_min - pad)
        hi = y_max + pad
        for ax in fig.axes:
            if ax.get_visible():
                ax.set_ylim(lo, hi)

    handles, labels = [], []
    if any(ax.get_visible() for ax in fig.axes):
        handles = [plt.Line2D([0], [0], color="C0", linewidth=1.5)]
        labels = [r"Mean $\|\nabla \ell(\theta^{(k)})\|_2$"]

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, 0.98),
        )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    output = Path(output)
    fig.savefig(output, dpi=150)
    print(f"[Theta Grad Norm] Saved plot to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean ± std of gradient norms across seeds in the Figure 1 layout."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("res/results/1"),
        help="Base directory containing stdout logs (default: res/results/1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("theta_grad_norm_mean_std.png"),
        help="Output image filename.",
    )
    parser.add_argument(
        "--grad-column",
        type=str,
        default="grad_norm",
        help="Column name that stores gradient norms (default: grad_norm).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help=(
            "Moving-average window size (in iterations) applied per seed before "
            "computing statistics (default: 1)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help=(
            "Minibatch size runs must match to be included. Use -1 to disable "
            "batch-size filtering."
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

    make_theta_grad_norm_plot_from_df(
        df,
        output=args.output,
        grad_column=args.grad_column,
        smooth_window=args.smooth_window,
        batch_size=None if args.batch_size == -1 else args.batch_size,
    )


if __name__ == "__main__":
    main()



#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from collect_results import collect_stdout_logs

ALGO_ORDER = ("bsgd", "minimax", "scgd")
ALGO_TITLES = {
    "bsgd": "BSGD (exact kernel)",
    "minimax": "MINIMAX + ORF",
    "scgd": "SCGD + ORF",
}

# Order chosen to match the NeurIPS Figure 2 layout: m = 64, 32, 16
BATCH_SIZES = (64, 32, 16)


def _compute_theta_error(df) -> np.ndarray:
    """
    Return |theta^(k) - theta*| for each row of the dataframe.

    theta = (sigma_f2, sigma_eps2)
    theta* = (sigma_f2_true, sigma_eps2_true)
    """

    required = {"sigma_f2", "sigma_eps2", "sigma_f2_true", "sigma_eps2_true"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns needed to compute θ error: {missing}")

    delta_f = df["sigma_f2"].astype(float) - df["sigma_f2_true"].astype(float)
    delta_eps = df["sigma_eps2"].astype(float) - df["sigma_eps2_true"].astype(float)
    return np.sqrt(delta_f.to_numpy() ** 2 + delta_eps.to_numpy() ** 2)


def make_theta_gap_figure(
    df,
    output: str | Path = "neurips_figure2_theta_gap.png",
    smooth_window: int = 1,
    use_log10: bool = False,
) -> None:
    """
    Create the same 3×3 panel layout as Figure 2, but plot |θ^(k)−θ*|.

        rows    = algorithms (BSGD, MINIMAX, SCGD)
        columns = minibatch sizes m = 64, 32, 16

    Each panel shows the mean and ±1 standard error of either
        |θ^(k)−θ*|         (linear scale)
        log10 |θ^(k)−θ*|   (if --log10 is enabled)
    aggregated over seeds.
    """

    if df.empty:
        raise RuntimeError("DataFrame is empty – no stdout logs found.")

    # Keep only the GP setup used in Figures 1–2.
    mask_common = (
        (df["n"] == 1024)
        & (df["lengthscale"] == 0.5)
        & (df["sigma_f2_true"] == 4.0)
        & (df["sigma_eps2_true"] == 1.0)
    )
    df = df[mask_common].copy()
    if df.empty:
        raise RuntimeError("No rows left after filtering to the NeurIPS GP setup.")

    # SCGD filename truncation fix copied from make_figure_1.py.
    scgd_fixups = [
        {"sigma_f2_init": 2.5, "observed_eps2": 3.0, "corrected_eps2": 3.5},
        {"sigma_f2_init": 2.5, "observed_eps2": 0.0, "corrected_eps2": 0.7},
    ]
    for fix in scgd_fixups:
        mask_fix = (
            (df["algo"] == "scgd")
            & np.isclose(df["sigma_f2_init"], fix["sigma_f2_init"])
            & np.isclose(df["sigma_eps2_init"], fix["observed_eps2"])
        )
        df.loc[mask_fix, "sigma_eps2_init"] = fix["corrected_eps2"]

    # Infer missing σ_ε² metadata for MINIMAX/SCGD runs whose directory
    # names were truncated before "--sigma-eps2-init 3.0".
    infer_eps2_mask = (
        (df["sigma_f2_init"] == 5.0)
        & df["sigma_eps2_init"].isna()
        & df["algo"].isin(("minimax", "scgd"))
    )
    df.loc[infer_eps2_mask, "sigma_eps2_init"] = 3.0

    # Figure 2 uses θ^(0) = (5.0, 3.0) and (for BSGD) α1 = 9.0
    # and 25-epoch runs with batch sizes m in {16, 32, 64}.  We will
    # further split by batch size per column.
    df = df[
        (df["sigma_f2_init"] == 5.0)
        & (df["sigma_eps2_init"] == 3.0)
        & (df["n_epochs"] == 25)
        & (df["batch_size"].isin(BATCH_SIZES))
    ].copy()
    if df.empty:
        raise RuntimeError(
            "No rows left after filtering to θ^(0)=(5.0,3.0), n_epochs=25, "
            f"and batch_size in {BATCH_SIZES}."
        )

    df["theta_error"] = _compute_theta_error(df)

    smooth_window = int(max(smooth_window, 1))

    def smooth_1d(y: np.ndarray) -> np.ndarray:
        """NaN-aware moving average with window `smooth_window`."""
        if smooth_window <= 1:
            return y

        y = np.asarray(y, dtype=float)
        valid = np.isfinite(y)
        if not valid.any():
            return y

        kernel = np.ones(smooth_window, dtype=float)
        num = np.convolve(np.where(valid, y, 0.0), kernel, mode="same")
        den = np.convolve(valid.astype(float), kernel, mode="same")

        out = np.full_like(num, np.nan, dtype=float)
        mask = den > 0
        out[mask] = num[mask] / den[mask]
        return out

    fig, axes = plt.subplots(
        nrows=len(ALGO_ORDER),
        ncols=len(BATCH_SIZES),
        figsize=(12, 9),
        sharey=True,
    )

    y_min, y_max = np.inf, -np.inf
    y_label = (
        r"$\log_{10}\|\theta^{(k)} - \theta^\star\|_2$"
        if use_log10
        else r"$\|\theta^{(k)} - \theta^\star\|_2$"
    )

    for row_idx, algo in enumerate(ALGO_ORDER):
        for col_idx, m in enumerate(BATCH_SIZES):
            ax = axes[row_idx, col_idx]

            panel_mask = (df["algo"] == algo) & (df["batch_size"] == m)
            if algo == "bsgd" and "alpha1" in df.columns:
                panel_mask &= df["alpha1"] == 9.0

            df_panel = df[panel_mask].copy()
            if df_panel.empty:
                ax.set_visible(False)
                continue

            pivot = df_panel.pivot_table(
                index="iter",
                columns="seed",
                values="theta_error",
                aggfunc="last",
            ).sort_index()

            values = pivot.to_numpy(dtype=float)
            if use_log10:
                with np.errstate(divide="ignore", invalid="ignore"):
                    values = np.where(values > 0.0, np.log10(values), np.nan)

            mean_vals = np.nanmean(values, axis=1)
            n_eff = np.sum(np.isfinite(values), axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                se_vals = np.nanstd(values, axis=1, ddof=1) / np.sqrt(n_eff)

            iters = pivot.index.to_numpy(dtype=float)

            mean_vals = smooth_1d(mean_vals)
            se_vals = smooth_1d(se_vals)

            lower = mean_vals - se_vals
            upper = mean_vals + se_vals

            ax.plot(iters, mean_vals, color="C1", linewidth=1.5)
            ax.fill_between(
                iters,
                lower,
                upper,
                color="C1",
                alpha=0.3,
                linewidth=0.0,
            )

            panel_y_min = np.nanmin(lower)
            panel_y_max = np.nanmax(upper)
            y_min = min(y_min, panel_y_min)
            y_max = max(y_max, panel_y_max)

            if row_idx == 0:
                ax.set_title(f"$m = {m}$")
            if col_idx == 0:
                ax.set_ylabel(ALGO_TITLES[algo] + "\n" + y_label, fontsize=10)
            if row_idx == len(ALGO_ORDER) - 1:
                ax.set_xlabel("Iteration $k$")

            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    if np.isfinite(y_min) and np.isfinite(y_max):
        pad = 0.05 * (y_max - y_min)
        lo = y_min - pad
        hi = y_max + pad
        for ax in fig.axes:
            if ax.get_visible():
                ax.set_ylim(lo, hi)

    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    output = Path(output)
    fig.savefig(output, dpi=150)
    print(f"[θ-gap] Saved plot to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Figure 2 style plot showing |θ^(k) - θ*| (or log10 of that quantity) "
            "for BSGD, MINIMAX, and SCGD."
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
        default=Path("neurips_figure2_theta_gap.png"),
        help="Output image filename.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=200,
        help=(
            "Moving-average window size (in iterations) applied to the "
            "mean and standard-error curves before plotting."
        ),
    )
    parser.add_argument(
        "--log10",
        action="store_true",
        help="Plot log10 |θ^(k) - θ*| instead of |θ^(k) - θ*|.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    df = collect_stdout_logs(args.root)
    if df.empty:
        print(f"No stdout records found under {args.root}")
        return

    make_theta_gap_figure(
        df,
        output=args.output,
        smooth_window=args.smooth_window,
        use_log10=args.log10,
    )


if __name__ == "__main__":
    main()


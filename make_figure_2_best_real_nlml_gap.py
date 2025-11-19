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
BATCH_SIZES = (64, 32, 16)
BEST_REAL_ALGOS = {"minimax", "scgd"}


def _choose_column(df, override: str | None, default: str, flag: str) -> str:
    column = override or default
    if column not in df.columns:
        raise ValueError(
            f"Requested {flag} '{column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    return column


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return values
    kernel = np.ones(window, dtype=float)
    num = np.convolve(np.where(valid, values, 0.0), kernel, mode="same")
    den = np.convolve(valid.astype(float), kernel, mode="same")
    out = np.full_like(num, np.nan, dtype=float)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def _best_real_from_orf(real_vals: np.ndarray, orf_vals: np.ndarray) -> np.ndarray:
    if real_vals.shape != orf_vals.shape:
        raise ValueError(
            "real_vals and orf_vals must share the same shape. "
            f"Got {real_vals.shape} vs {orf_vals.shape}."
        )
    out = np.full_like(real_vals, np.nan, dtype=float)
    if real_vals.size == 0:
        return out
    n_iters, n_cols = real_vals.shape
    for col in range(n_cols):
        real_col = real_vals[:, col]
        orf_col = orf_vals[:, col]
        best_idx = -1
        best_orf = np.inf
        for iter_idx in range(n_iters):
            candidate = orf_col[iter_idx]
            if np.isfinite(candidate) and candidate < best_orf:
                best_orf = candidate
                best_idx = iter_idx
            if best_idx >= 0:
                out[iter_idx, col] = real_col[best_idx]
    return out


def _pivot_panel(df_panel, value_column: str):
    return (
        df_panel.pivot_table(
            index="iter",
            columns="seed",
            values=value_column,
            aggfunc="last",
        ).sort_index()
    )


def make_figure2_best_real_nlml_gap_from_df(
    df,
    output: str | Path = "neurips_figure2_best_real_nlml_gap.png",
    nlml_column: str | None = None,
    orf_column: str | None = None,
    true_column: str | None = None,
    smooth_window: int = 1,
) -> None:
    """
    Plot mean ± SE of the difference between the best-real NLML and a
    corresponding ground-truth NLML:

      * MINIMAX/SCGD: best-real NLML minus their own `real_nlml_true`.
      * BSGD: real NLML minus MINIMAX `real_nlml_true` (matched by seed).
    """

    if df.empty:
        raise RuntimeError("DataFrame is empty – no stdout logs found.")

    mask_common = (
        (df["n"] == 1024)
        & (df["lengthscale"] == 0.5)
        & (df["sigma_f2_true"] == 4.0)
        & (df["sigma_eps2_true"] == 1.0)
    )
    df = df[mask_common].copy()
    if df.empty:
        raise RuntimeError("No rows left after filtering to the NeurIPS GP setup.")

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

    infer_eps2_mask = (
        (df["sigma_f2_init"] == 5.0)
        & df["sigma_eps2_init"].isna()
        & df["algo"].isin(("minimax", "scgd"))
    )
    df.loc[infer_eps2_mask, "sigma_eps2_init"] = 3.0

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

    nlml_column = _choose_column(df, nlml_column, "real_nlml", "--nlml-column")
    orf_column = _choose_column(df, orf_column, "orf_nlml", "--orf-column")
    true_column = _choose_column(df, true_column, "real_nlml_true", "--true-column")
    smooth_window = int(max(smooth_window, 1))

    minimax_truth = df[df["algo"] == "minimax"]

    fig, axes = plt.subplots(
        nrows=len(ALGO_ORDER),
        ncols=len(BATCH_SIZES),
        figsize=(12, 9),
        sharey=True,
    )

    y_min, y_max = np.inf, -np.inf

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

            pivot_real = _pivot_panel(df_panel, nlml_column)
            values = pivot_real.to_numpy(dtype=float)

            if algo in BEST_REAL_ALGOS:
                pivot_orf = (
                    _pivot_panel(df_panel, orf_column)
                    .reindex(index=pivot_real.index, columns=pivot_real.columns)
                )
                orf_vals = pivot_orf.to_numpy(dtype=float)
                values = _best_real_from_orf(values, orf_vals)

            if algo in BEST_REAL_ALGOS:
                pivot_true = (
                    _pivot_panel(df_panel, true_column)
                    .reindex(index=pivot_real.index, columns=pivot_real.columns)
                )
            else:  # BSGD uses MINIMAX truth, matched by seed
                df_true = minimax_truth[minimax_truth["batch_size"] == m]
                if df_true.empty:
                    raise RuntimeError(
                        f"No MINIMAX truth rows found for batch_size={m}; "
                        "needed to compare against BSGD."
                    )
                pivot_true = (
                    _pivot_panel(df_true, true_column)
                    .reindex(index=pivot_real.index, columns=pivot_real.columns)
                )

            truth_vals = pivot_true.to_numpy(dtype=float)
            diff_vals = values - truth_vals

            mean_vals = np.nanmean(diff_vals, axis=1)
            n_eff = np.sum(np.isfinite(diff_vals), axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                se_vals = np.nanstd(diff_vals, axis=1, ddof=1) / np.sqrt(n_eff)

            iters = pivot_real.index.to_numpy(dtype=float)
            mean_vals = _smooth(mean_vals, smooth_window)
            se_vals = _smooth(se_vals, smooth_window)

            lower = mean_vals - se_vals
            upper = mean_vals + se_vals

            ax.plot(iters, mean_vals, color="C0", linewidth=1.5)
            ax.fill_between(
                iters,
                lower,
                upper,
                color="C0",
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
                ax.set_ylabel(
                    ALGO_TITLES[algo] + "\nBest Real – Truth",
                    fontsize=10,
                )
            if row_idx == len(ALGO_ORDER) - 1:
                ax.set_xlabel("Iteration $k$")

            ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
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
    print(f"[Figure 2 - Best Real Gap] Saved plot to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the gap between best-real NLML trajectories and ground-truth NLML "
            "(using MINIMAX truth for BSGD, aligned by seed)."
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
        default=Path("neurips_figure2_best_real_nlml_gap.png"),
        help="Output image filename.",
    )
    parser.add_argument(
        "--nlml-column",
        type=str,
        default="real_nlml",
        help="Column containing the real NLML values (default: 'real_nlml').",
    )
    parser.add_argument(
        "--orf-column",
        type=str,
        default="orf_nlml",
        help=(
            "Column containing the ORF-based NLML proxy "
            "used to pick the best-real iterate (default: 'orf_nlml')."
        ),
    )
    parser.add_argument(
        "--true-column",
        type=str,
        default="real_nlml_true",
        help=(
            "Column containing the ground-truth NLML values "
            "(default: 'real_nlml_true')."
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help=(
            "Moving-average window size (in iterations) applied to the "
            "mean and standard-error curves before plotting (default: 1)."
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

    make_figure2_best_real_nlml_gap_from_df(
        df,
        output=args.output,
        nlml_column=args.nlml_column,
        orf_column=args.orf_column,
        true_column=args.true_column,
        smooth_window=args.smooth_window,
    )


if __name__ == "__main__":
    main()



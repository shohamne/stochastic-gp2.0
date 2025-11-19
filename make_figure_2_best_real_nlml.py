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

# Only the ORF-based methods need the best-real NLML correction.
BEST_REAL_ALGOS = {"minimax", "scgd"}


def _choose_nlml_column(df, override: str | None) -> str:
    """
    Decide which column in the DataFrame contains the real NLML values.

    The column must be explicitly specified via --nlml-column.
    If omitted, we expect the column to be named "real_nlml".
    """
    column = override or "real_nlml"
    if column not in df.columns:
        raise ValueError(
            f"Requested nlml-column '{column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    return column


def _choose_orf_column(df, override: str | None) -> str:
    """
    Decide which column contains the ORF-based NLML proxy used for selection.
    """
    column = override or "orf_nlml"
    if column not in df.columns:
        raise ValueError(
            f"Requested orf-column '{column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    return column


def _best_real_from_orf(real_vals: np.ndarray, orf_vals: np.ndarray) -> np.ndarray:
    """
    Given two (iter × seed) matrices, return best-real NLML per iteration:

        best_real[i] = real_vals[argmin_j<=i orf_vals[j]]

    Both inputs must already be float arrays with identical shapes.
    """
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


def make_figure2_best_real_nlml_from_df(
    df,
    output: str | Path = "neurips_figure2_best_real_nlml.png",
    nlml_column: str | None = None,
    orf_column: str | None = None,
    smooth_window: int = 1,
) -> None:
    """
    Create a 3x3 panel plot mirroring NeurIPS Figure 2 but for the "best-real"
    NLML trajectory:

        rows    = algorithms (BSGD, MINIMAX, SCGD)
        columns = minibatch sizes m = 64, 32, 16

    For MINIMAX and SCGD we track the iteration whose ORF NLML is currently
    minimal and plot the corresponding *real* NLML. BSGD uses its real NLML
    directly. Each panel shows mean and ±1 standard error across seeds, with
    θ^(0)=(5.0,3.0) and the 25-epoch runs used in NeurIPS Figure 2.
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

    # SCGD filename truncation fix copied from make_figure_2_real_nlml.py:
    # some directories truncated "--sigma-eps2-init 3.5"/"0.7" to "3."/"0."
    # so metadata shows 3.0 / 0.0 instead. Fix those before we filter.
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

    # Figure 2 uses θ^(0) = (5.0, 3.0) and (for BSGD) α1 = 9.0 and
    # 25-epoch runs with batch sizes m in {16, 32, 64}.
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

    nlml_column = _choose_nlml_column(df, nlml_column)
    orf_column = _choose_orf_column(df, orf_column)

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

    for row_idx, algo in enumerate(ALGO_ORDER):
        for col_idx, m in enumerate(BATCH_SIZES):
            ax = axes[row_idx, col_idx]

            panel_mask = (df["algo"] == algo) & (df["batch_size"] == m)
            # For BSGD, insist on α1 = 9.0 (the value used in the paper).
            if algo == "bsgd" and "alpha1" in df.columns:
                panel_mask &= df["alpha1"] == 9.0

            df_panel = df[panel_mask].copy()
            if df_panel.empty:
                ax.set_visible(False)
                continue

            pivot = df_panel.pivot_table(
                index="iter",
                columns="seed",
                values=nlml_column,
                aggfunc="last",
            ).sort_index()

            values = pivot.to_numpy(dtype=float)

            if algo in BEST_REAL_ALGOS:
                pivot_orf = (
                    df_panel.pivot_table(
                        index="iter",
                        columns="seed",
                        values=orf_column,
                        aggfunc="last",
                    )
                    .sort_index()
                    .reindex(index=pivot.index, columns=pivot.columns)
                )
                orf_values = pivot_orf.to_numpy(dtype=float)
                values = _best_real_from_orf(values, orf_values)

            mean_vals = np.nanmean(values, axis=1)
            n_eff = np.sum(np.isfinite(values), axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                se_vals = np.nanstd(values, axis=1, ddof=1) / np.sqrt(n_eff)

            iters = pivot.index.to_numpy(dtype=float)

            mean_vals = smooth_1d(mean_vals)
            se_vals = smooth_1d(se_vals)

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
                    ALGO_TITLES[algo] + "\nBest Real NLML",
                    fontsize=10,
                )

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
    print(f"[Figure 2 - Best Real NLML] Saved plot to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct a NeurIPS Figure 2-style plot (full-batch vs minibatch) "
            "using the best-real NLML trajectories (selected via ORF NLML)."
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
        default=Path("neurips_figure2_best_real_nlml.png"),
        help="Output image filename.",
    )
    parser.add_argument(
        "--nlml-column",
        type=str,
        default="real_nlml",
        help=(
            "Name of the column containing the real NLML values "
            "(default: 'real_nlml')."
        ),
    )
    parser.add_argument(
        "--orf-column",
        type=str,
        default="orf_nlml",
        help=(
            "Name of the column containing the ORF-based NLML proxy "
            "used to pick the best-real iterate (default: 'orf_nlml')."
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help=(
            "Moving-average window size (in iterations) applied to the "
            "mean and standard-error curves before plotting (default: 200)."
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

    make_figure2_best_real_nlml_from_df(
        df,
        output=args.output,
        nlml_column=args.nlml_column,
        orf_column=args.orf_column,
        smooth_window=args.smooth_window,
    )


if __name__ == "__main__":
    main()



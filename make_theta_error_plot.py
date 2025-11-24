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


def _print_data_description(df, batch_size: int | None) -> None:
    """Print a short textual description of the dataframe being plotted."""
    batch_label = "any" if batch_size is None else str(batch_size)
    total_rows = len(df)
    has_seed = "seed" in df.columns
    n_seeds = df["seed"].nunique() if has_seed else "n/a"
    iter_min = df["iter"].min() if "iter" in df.columns else None
    iter_max = df["iter"].max() if "iter" in df.columns else None

    print("[Theta Error] Data description:")
    print(f"  Batch-size filter : {batch_label}")
    print(f"  Rows / unique seeds: {total_rows:,} / {n_seeds}")
    if iter_min is not None and iter_max is not None:
        print(f"  Iteration range   : {int(iter_min)} – {int(iter_max)}")

    algo_counts = df["algo"].value_counts(sort=False) if "algo" in df.columns else {}
    for algo in ALGO_ORDER:
        if algo in algo_counts:
            seed_count = (
                df[df["algo"] == algo]["seed"].nunique() if has_seed else "n/a"
            )
            print(
                f"    {ALGO_TITLES[algo]:<20} · rows={int(algo_counts[algo]):,}, "
                f"seeds={seed_count}"
            )

    if {"sigma_f2_init", "sigma_eps2_init"} <= set(df.columns) and has_seed:
        init_counts = (
            df.groupby(["sigma_f2_init", "sigma_eps2_init"])
            .agg(rows=("iter", "size"), seeds=("seed", "nunique"))
            .reset_index()
            .sort_values(["sigma_f2_init", "sigma_eps2_init"])
        )
        for row in init_counts.itertuples(index=False):
            print(
                "    Init σ_f²={:.2g}, σ_ε²={:.2g}: rows={}, seeds={}".format(
                    row.sigma_f2_init, row.sigma_eps2_init, int(row.rows), int(row.seeds)
                )
            )


def _print_result_summary(panel_summaries: list[dict]) -> None:
    """Print final mean ± std for each panel that was rendered."""
    print("[Theta Error] Result summary:")
    if not panel_summaries:
        print("  No valid panels — nothing to summarize.")
        return

    algo_rank = {algo: idx for idx, algo in enumerate(ALGO_ORDER)}
    init_rank = {cfg["name"]: idx for idx, cfg in enumerate(INIT_CONFIGS)}
    panel_summaries.sort(
        key=lambda entry: (
            algo_rank.get(entry["algo"], len(ALGO_ORDER)),
            init_rank.get(entry["init"], len(INIT_CONFIGS)),
        )
    )

    for entry in panel_summaries:
        mean_val = entry["final_mean"]
        std_val = entry["final_std"]
        mean_str = f"{mean_val:.3g}" if np.isfinite(mean_val) else "n/a"
        std_str = f"{std_val:.3g}" if np.isfinite(std_val) else "n/a"
        print(
            f"  - {ALGO_TITLES.get(entry['algo'], entry['algo'])} | {entry['init']}: "
            f"{entry['n_seeds']} seeds, {entry['n_iters']} iters, "
            f"final mean ± std = {mean_str} ± {std_str}"
        )


def _nan_moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware moving average."""
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


def _nan_running_min(values: np.ndarray) -> np.ndarray:
    """Running min that ignores NaNs until a finite value appears."""
    arr = np.asarray(values, dtype=float)
    running = np.full_like(arr, np.nan, dtype=float)
    best = np.inf
    has_value = False
    for idx, val in enumerate(arr):
        if np.isfinite(val):
            best = val if not has_value else min(best, val)
            has_value = True
        if has_value:
            running[idx] = best
    return running


def make_theta_error_plot_from_df(
    df,
    output: str | Path = "theta_error_mean_std.png",
    smooth_window: int = 1,
    batch_size: int | None = 128,
) -> None:
    """
    Build a 3x3 panel plot (algorithms × initializations) showing
    mean ± 1 standard deviation of ||θ^(k) - θ||_2 across seeds.
    """

    if df.empty:
        raise RuntimeError("DataFrame is empty – no stdout logs found.")

    required_cols = {
        "algo",
        "iter",
        "seed",
        "sigma_f2",
        "sigma_eps2",
        "sigma_f2_true",
        "sigma_eps2_true",
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
    _print_data_description(df, batch_size)

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

    # Use the long 200-epoch runs (one iteration per epoch) like Figure 1
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

    panel_summaries: list[dict] = []

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
            if algo == "bsgd":
                mask_cfg &= np.isclose(df_algo["alpha1"], cfg["alpha1"])

            df_cfg = df_algo[mask_cfg].copy()
            if df_cfg.empty:
                ax.set_visible(False)
                continue

            # Compute ||θ^(k) - θ||_2 per row
            theta_error = np.sqrt(
                (df_cfg["sigma_f2"] - df_cfg["sigma_f2_true"]) ** 2
                + (df_cfg["sigma_eps2"] - df_cfg["sigma_eps2_true"]) ** 2
            )
            df_cfg["theta_error"] = theta_error.astype(float)

            pivot = (
                df_cfg.pivot_table(
                    index="iter",
                    columns="seed",
                    values="theta_error",
                    aggfunc="last",
                )
                .sort_index()
            )
            if pivot.empty:
                ax.set_visible(False)
                continue

            values = pivot.to_numpy(dtype=float)
            values = _smooth_columns(values, smooth_window)

            mean_err = np.nanmean(values, axis=1)
            std_err = np.nanstd(values, axis=1, ddof=1)
            running_best = _nan_running_min(mean_err)

            iters = pivot.index.to_numpy(dtype=float)

            lower = mean_err - std_err
            upper = mean_err + std_err

            ax.plot(iters, mean_err, color="C0", linewidth=1.5, label="Mean error")
            ax.fill_between(
                iters,
                lower,
                upper,
                color="C0",
                alpha=0.3,
                linewidth=0.0,
                label="±1 std",
            )
            if np.isfinite(running_best).any():
                ax.plot(
                    iters,
                    running_best,
                    color="C1",
                    linewidth=1.2,
                    linestyle="--",
                    label="Best-so-far mean",
                )

            finite_indices = np.where(np.isfinite(mean_err))[0]
            if finite_indices.size:
                final_idx = finite_indices[-1]
                final_mean = float(mean_err[final_idx])
                final_std = float(std_err[final_idx])
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
            else:
                final_mean = np.nan
                final_std = np.nan

            panel_y_min = np.nanmin(lower)
            panel_y_max = np.nanmax(upper)
            if np.isfinite(running_best).any():
                panel_y_min = min(panel_y_min, np.nanmin(running_best))
                panel_y_max = max(panel_y_max, np.nanmax(running_best))
            y_min = min(y_min, panel_y_min)
            y_max = max(y_max, panel_y_max)

            panel_summaries.append(
                {
                    "algo": algo,
                    "init": cfg["name"],
                    "n_seeds": pivot.shape[1],
                    "n_iters": pivot.shape[0],
                    "final_mean": final_mean,
                    "final_std": final_std,
                }
            )

            if row_idx == 0:
                ax.set_title(cfg["name"])

            if col_idx == 0:
                ax.set_ylabel(
                    ALGO_TITLES[algo] + "\n" + r"$\|\theta^{(k)} - \theta\|_2$",
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
        handles = [
            plt.Line2D([0], [0], color="C0", linewidth=1.5),
            plt.Line2D([0], [0], color="C1", linewidth=1.2, linestyle="--"),
        ]
        labels = [
            r"Mean $\|\theta^{(k)} - \theta\|_2$",
            r"Best-so-far mean $\|\theta^{(k)} - \theta\|_2$",
        ]

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
    print(f"[Theta Error] Saved plot to {output}")
    _print_result_summary(panel_summaries)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean ± std of ||theta^(k) - theta||_2 for BSGD, MINIMAX, "
            "and SCGD across initializations using stdout logs."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("res-11"),
        help="Base directory containing stdout logs (default: res-9).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("theta_error_mean_std.png"),
        help="Output image filename.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help=(
            "Moving-average window size (in iterations) applied to each seed's "
            "error trajectory before computing statistics (default: 1)."
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

    make_theta_error_plot_from_df(
        df,
        output=args.output,
        smooth_window=args.smooth_window,
        batch_size=None if args.batch_size == -1 else args.batch_size,
    )


if __name__ == "__main__":
    main()



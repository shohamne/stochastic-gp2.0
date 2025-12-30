#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from collect_results import collect_stdout_logs


TARGET_ALGO = "minimax"
DEFAULT_SEEDS = (3, 1)


def _smooth_series(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode="same")


def _format_mu_value(mu: float) -> str:
    """Format μ using scientific notation (mantissa·10^exp) for consistency."""
    if np.isclose(mu, 0.0):
        return "0.0e0"
    mantissa, exponent = f"{mu:.1e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent = int(exponent)  # strip leading zeros/plus sign
    return f"{mantissa}e{exponent}"


def _mu_label(df_run) -> str:
    if "mu" not in df_run.columns:
        return "μ (unknown)"
    mu_series = df_run["mu"].dropna().to_numpy(dtype=float)
    if mu_series.size == 0:
        return "μ (unknown)"
    start, end = mu_series[0], mu_series[-1]
    if np.isclose(start, end):
        return rf"$\mu$={_format_mu_value(start)}"
    return rf"$\mu$={_format_mu_value(start)}\rightarrow{_format_mu_value(end)}"


def make_res18_seed_subplots(
    df,
    output: str | Path = "res18_seed_subplot.pdf",
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    mus: tuple[float, ...] | None = None,
    smooth_window: int = 1,
    batch_size: int | None = 1024,
    max_iter: int | None = None,
    max_epoch: int | None = None,
    ref_root: str | Path | None = Path("res-17"),
) -> None:
    """
    Create a 3xN panel plot (N = number of columns):

    - Base columns come from `df` using `seeds`.
    - Row 1: σ_f² (solid) and σ_ε² (dashed) per run of the seed, with reference
      lines pulled from `ref_root`.
    - Row 2: gradient norm per run (if available in dataframe).
    - Row 3: real NLML per run (if available in dataframe).

    Legend entries use μ values (scientific notation). Optionally filter to a
    specific set of μ values. A black reference line is added for each θ
    parameter using the last iteration value from the same seed under
    `ref_root`.
    """

    if df.empty:
        raise RuntimeError("DataFrame is empty – no stdout logs found.")

    def _apply_truncated_fixups(df_target) -> None:
        """Fix truncated '--sigma-eps2-init' flags for MINIMAX runs."""
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
                (df_target["algo"] == TARGET_ALGO)
                & np.isclose(df_target["sigma_f2_init"], fix["sigma_f2_init"])
                & np.isclose(df_target["sigma_eps2_init"], fix["observed_eps2"])
            )
            df_target.loc[mask_fix, "sigma_eps2_init"] = fix["corrected_eps2"]

    def _filter_dataset(df_raw, target_seeds: tuple[int, ...], label: str):
        """Apply the same filtering pipeline to any dataset."""
        df_cur = df_raw.copy()

        if batch_size is not None:
            if "batch_size" not in df_cur.columns:
                raise RuntimeError(
                    f"{label}: DataFrame is missing the 'batch_size' column; cannot filter runs."
                )
            df_cur = df_cur[df_cur["batch_size"] == batch_size].copy()
            if df_cur.empty:
                raise RuntimeError(
                    f"{label}: No rows left after filtering to batch_size == {batch_size}."
                )

        _apply_truncated_fixups(df_cur)

        df_cur = df_cur[df_cur["algo"] == TARGET_ALGO]
        if df_cur.empty:
            raise RuntimeError(f"{label}: No {TARGET_ALGO} runs available after filtering.")

        df_cur = df_cur[df_cur["seed"].isin(target_seeds)]
        if df_cur.empty:
            raise RuntimeError(f"{label}: No runs found for seeds {target_seeds}.")

        if mus is not None:
            if "mu" not in df_cur.columns:
                raise RuntimeError(
                    f"{label}: DataFrame is missing the 'mu' column; cannot filter runs."
                )
            mask_mu = np.zeros(len(df_cur), dtype=bool)
            for target_mu in mus:
                mask_mu |= np.isclose(df_cur["mu"], target_mu, rtol=1e-6, atol=1e-12)
            df_cur = df_cur[mask_mu].copy()
            if df_cur.empty:
                mus_str = ", ".join(_format_mu_value(mu) for mu in mus)
                raise RuntimeError(f"{label}: No runs found for μ in [{mus_str}].")

        if max_iter is not None:
            if "iter" not in df_cur.columns:
                raise RuntimeError(
                    f"{label}: DataFrame is missing the 'iter' column; cannot apply --max-iter."
                )
            df_cur = df_cur[df_cur["iter"] <= max_iter].copy()
            if df_cur.empty:
                raise RuntimeError(
                    f"{label}: No rows left after filtering to iterations <= {max_iter}."
                )

        if max_epoch is not None:
            if "epoch" not in df_cur.columns:
                raise RuntimeError(
                    f"{label}: DataFrame is missing the 'epoch' column; cannot apply --max-epoch."
                )
            df_cur = df_cur[df_cur["epoch"] <= max_epoch].copy()
            if df_cur.empty:
                raise RuntimeError(
                    f"{label}: No rows left after filtering to epochs <= {max_epoch}."
                )

        return df_cur

    df = _filter_dataset(df, seeds, "main")

    datasets: list[dict[str, object]] = [
        {"label": "", "df": df, "seeds": tuple(seeds)}
    ]

    if not datasets:
        raise RuntimeError("No datasets available to plot after filtering.")

    smooth_window = int(max(smooth_window, 1))

    # Collect unique μ values and labels so colors stay consistent across subplots.
    mu_values_set: set[float] = set()
    for dataset in datasets:
        df_dataset = dataset["df"]
        seeds_dataset = dataset["seeds"]
        for seed in seeds_dataset:  # type: ignore[assignment]
            df_seed = df_dataset[df_dataset["seed"] == seed]  # type: ignore[index]
            if df_seed.empty:
                continue
            if "stdout_path" in df_seed.columns:
                grouped = list(df_seed.groupby("stdout_path"))
            else:
                grouped = [(f"seed-{seed}", df_seed)]
            for _, df_run in grouped:
                if "mu" in df_run.columns:
                    mu_series = df_run["mu"].dropna().to_numpy(dtype=float)
                    if mu_series.size > 0:
                        mu_values_set.add(mu_series[0])
    # Sort μ values numerically and generate labels
    sorted_mu_values = sorted(mu_values_set)
    mu_labels: list[str] = [rf"$\mu$={_format_mu_value(mu)}" for mu in sorted_mu_values]

    num_colors = max(len(mu_labels), 1)
    cmap = plt.get_cmap("tab10", num_colors)
    mu_to_color = {mu: cmap(i % cmap.N) for i, mu in enumerate(mu_labels)}

    # Optional reference values drawn as black lines (last iteration per seed).
    ref_values: dict[int, dict[str, float | None]] = {}
    all_seeds = {seed for dataset in datasets for seed in dataset["seeds"]}  # type: ignore[arg-type]
    if ref_root is not None:
        df_ref = collect_stdout_logs(Path(ref_root))
        if not df_ref.empty:
            # Try to match batch size; if that drops everything, fall back to all.
            if batch_size is not None and "batch_size" in df_ref.columns:
                df_ref_filtered = df_ref[df_ref["batch_size"] == batch_size].copy()
                if not df_ref_filtered.empty:
                    df_ref = df_ref_filtered
            for seed in all_seeds:
                df_seed_ref = df_ref[df_ref["seed"] == seed]
                if df_seed_ref.empty:
                    continue
                sort_key = "iter" if "iter" in df_seed_ref.columns else (
                    "epoch" if "epoch" in df_seed_ref.columns else None
                )
                if sort_key:
                    df_seed_ref = df_seed_ref.sort_values(sort_key)
                last = df_seed_ref.iloc[-1]
                ref_values[seed] = {
                    "sigma_f2": float(last["sigma_f2"])
                    if "sigma_f2" in last and last["sigma_f2"] is not None
                    else None,
                    "sigma_eps2": float(last["sigma_eps2"])
                    if "sigma_eps2" in last and last["sigma_eps2"] is not None
                    else None,
                    "grad_norm": float(last["grad_norm"])
                    if "grad_norm" in last and last["grad_norm"] is not None
                    else None,
                    "real_nlml": float(last["real_nlml"])
                    if "real_nlml" in last and last["real_nlml"] is not None
                    else None,
                }

    has_grad = all(
        "grad_norm" in dataset["df"].columns  # type: ignore[operator]
        for dataset in datasets
        if not dataset["df"].empty  # type: ignore[truthy-bool]
    )
    has_nlml = all(
        "real_nlml" in dataset["df"].columns  # type: ignore[operator]
        for dataset in datasets
        if not dataset["df"].empty  # type: ignore[truthy-bool]
    )

    panels: list[dict[str, object]] = []
    for dataset in datasets:
        for seed in dataset["seeds"]:  # type: ignore[assignment]
            panels.append(
                {
                    "label": dataset["label"],
                    "seed": seed,
                    "df": dataset["df"],
                }
            )

    total_cols = len(panels)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=total_cols,
        figsize=(6 * total_cols, 12),
        sharex=True,
    )
    if axes.ndim == 1:  # len(seeds) == 1 gives shape (3,)
        axes = axes[:, np.newaxis]

    y_limits = {
        "sigma": [np.inf, -np.inf],
        "grad": [np.inf, -np.inf],
        "nlml": [np.inf, -np.inf],
    }

    for ax_idx, panel in enumerate(panels):
        seed = panel["seed"]  # type: ignore[assignment]
        df_dataset = panel["df"]  # type: ignore[assignment]
        ax_sigma = axes[0, ax_idx]
        ax_grad = axes[1, ax_idx]
        ax_nlml = axes[2, ax_idx]
        df_seed = df_dataset[df_dataset["seed"] == seed]  # type: ignore[index]
        if df_seed.empty:
            ax_sigma.set_visible(False)
            ax_grad.set_visible(False)
            ax_nlml.set_visible(False)
            continue

        # Group by stdout_path so multiple runs for the same seed are separated.
        if "stdout_path" in df_seed.columns:
            grouped = list(df_seed.groupby("stdout_path"))
        else:
            grouped = [(f"seed-{seed}", df_seed)]

        for _, df_run in grouped:
            df_run = df_run.sort_values("iter").reset_index(drop=True)
            x = df_run["iter"].to_numpy()
            sigma_f2 = _smooth_series(
                df_run["sigma_f2"].to_numpy(dtype=float), smooth_window
            )
            sigma_eps2 = _smooth_series(
                df_run["sigma_eps2"].to_numpy(dtype=float), smooth_window
            )

            trim = min(smooth_window, len(x))
            if trim >= len(x):
                continue
            if trim > 0:
                x = x[:-trim]
                sigma_f2 = sigma_f2[:-trim]
                sigma_eps2 = sigma_eps2[:-trim]

            mu_text = _mu_label(df_run)
            color = mu_to_color.get(mu_text, "C0")

            ax_sigma.plot(
                x,
                sigma_f2,
                color=color,
                linewidth=1.2,
                alpha=0.9,
                linestyle="-",
                label=rf"$\sigma_f^2$ ({mu_text})",
            )
            ax_sigma.plot(
                x,
                sigma_eps2,
                color=color,
                linewidth=1.2,
                alpha=0.9,
                linestyle="--",
                label=rf"$\sigma_\varepsilon^2$ ({mu_text})",
            )

            y_limits["sigma"][0] = min(
                y_limits["sigma"][0], sigma_f2.min(), sigma_eps2.min()
            )
            y_limits["sigma"][1] = max(
                y_limits["sigma"][1], sigma_f2.max(), sigma_eps2.max()
            )

            if has_grad:
                grad_vals = _smooth_series(
                    df_run["grad_norm"].to_numpy(dtype=float), smooth_window
                )
                if trim > 0:
                    grad_vals = grad_vals[:-trim]
                ax_grad.plot(
                    x,
                    grad_vals,
                    color=color,
                    linewidth=1.2,
                    alpha=0.9,
                    linestyle="-",
                )
                y_limits["grad"][0] = min(y_limits["grad"][0], grad_vals.min())
                y_limits["grad"][1] = max(y_limits["grad"][1], grad_vals.max())

            if has_nlml:
                nlml_vals = _smooth_series(
                    df_run["real_nlml"].to_numpy(dtype=float), smooth_window
                )
                if trim > 0:
                    nlml_vals = nlml_vals[:-trim]
                ax_nlml.plot(
                    x,
                    nlml_vals,
                    color=color,
                    linewidth=1.2,
                    alpha=0.9,
                    linestyle="-",
                )
                y_limits["nlml"][0] = min(y_limits["nlml"][0], nlml_vals.min())
                y_limits["nlml"][1] = max(y_limits["nlml"][1], nlml_vals.max())

        # Reference horizontal lines (last iteration per seed)
        ref_vals = ref_values.get(seed)
        if ref_vals:
            if ref_vals.get("sigma_f2") is not None:
                y_ref = ref_vals["sigma_f2"]
                y_limits["sigma"][0] = min(y_limits["sigma"][0], y_ref)
                y_limits["sigma"][1] = max(y_limits["sigma"][1], y_ref)
                ax_sigma.axhline(
                    y_ref,
                    color="k",
                    linestyle="-",
                    linewidth=1.4,
                    alpha=0.85,
                    label=r"$\sigma_f^2$ optimal",
                )
            if ref_vals.get("sigma_eps2") is not None:
                y_ref = ref_vals["sigma_eps2"]
                y_limits["sigma"][0] = min(y_limits["sigma"][0], y_ref)
                y_limits["sigma"][1] = max(y_limits["sigma"][1], y_ref)
                ax_sigma.axhline(
                    y_ref,
                    color="k",
                    linestyle=":",
                    linewidth=1.4,
                    alpha=0.85,
                    label=r"$\sigma_\varepsilon^2$ optimal",
                )
            if has_grad and ref_vals.get("grad_norm") is not None:
                y_ref = ref_vals["grad_norm"]
                y_limits["grad"][0] = min(y_limits["grad"][0], y_ref)
                y_limits["grad"][1] = max(y_limits["grad"][1], y_ref)
                ax_grad.axhline(
                    y_ref,
                    color="k",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.8,
                    label="grad optimal",
                )
            if has_nlml and ref_vals.get("real_nlml") is not None:
                y_ref = ref_vals["real_nlml"]
                y_limits["nlml"][0] = min(y_limits["nlml"][0], y_ref)
                y_limits["nlml"][1] = max(y_limits["nlml"][1], y_ref)
                ax_nlml.axhline(
                    y_ref,
                    color="k",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.8,
                    label="NLML optimal",
                )

        ax_sigma.set_title(f"seed {seed}")  # type: ignore[index]
        if ax_idx == 0:
            ax_sigma.set_ylabel(r"$\theta^{(t)}$")
            if has_grad:
                ax_grad.set_ylabel(r"$\|\nabla \ell(\theta^{(t)})\|_2$")
            if has_nlml:
                ax_nlml.set_ylabel("Negative Log Marginal Likelihood")
        ax_nlml.set_xlabel("Iteration $t$")
        ax_sigma.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        if has_grad:
            ax_grad.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        if has_nlml:
            ax_nlml.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

    def _apply_shared_ylim(row_axes, key: str) -> None:
        y_min, y_max = y_limits[key]
        if np.isfinite(y_min) and np.isfinite(y_max):
            pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
            y_min = max(0.0, y_min - pad)
            y_max = y_max + pad
            visible_axes = [ax for ax in row_axes if ax.get_visible()]
            if visible_axes:
                visible_axes[0].set_ylim(y_min, y_max)
                for ax in visible_axes[1:]:
                    ax.set_ylim(y_min, y_max)

    _apply_shared_ylim(axes[0, :], "sigma")
    if has_grad:
        _apply_shared_ylim(axes[1, :], "grad")
        for ax in axes[1, :]:
            if ax.get_visible():
                ax.set_yscale("log")
    else:
        for ax in axes[1, :]:
            ax.set_visible(False)
    if has_nlml:
        _apply_shared_ylim(axes[2, :], "nlml")
    else:
        for ax in axes[2, :]:
            ax.set_visible(False)

    from matplotlib.lines import Line2D

    example_color = next(iter(mu_to_color.values()), "C0")
    mu_handles = [
        Line2D([0], [0], color=mu_to_color[mu], linestyle="-", linewidth=1.8)
        for mu in mu_labels
    ]
    mu_legend_labels = mu_labels

    style_handles = [
        Line2D([0], [0], color=example_color, linestyle="-", linewidth=2.0),
        Line2D([0], [0], color=example_color, linestyle="--", linewidth=2.0),
        Line2D([0], [0], color="k", linestyle="-", linewidth=1.4),
        Line2D([0], [0], color="k", linestyle=":", linewidth=1.4),
        Line2D([0], [0], color="k", linestyle="--", linewidth=1.2),
        Line2D([0], [0], color="k", linestyle="--", linewidth=1.2),
    ]
    style_labels = [
        r"$\sigma_f^2$",
        r"$\sigma_\varepsilon^2$",
        r"$\sigma_f^2$ optimal",
        r"$\sigma_\varepsilon^2$ optimal",
        "grad optimal",
        "NLML optimal",
    ]

    if mu_handles:
        mu_legend = fig.legend(
            mu_handles,
            mu_legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.3, 0.12),
            ncol=min(len(mu_handles), 5),
        )
        fig.add_artist(mu_legend)

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
    print(f"[minimax] Saved plot to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot minimax trajectories for specified seeds, "
            "optionally adding a comparison column from another root."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("res-25"),
        help="Base directory containing stdout logs (default: res-25).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("res25_seed_subplot.pdf"),
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
            "Minibatch size that runs must match to be included. "
            "Set to another value (or -1) to select different runs; "
            "use -1 to disable filtering."
        ),
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=-1,
        help="Maximum iteration included in the plot. Use -1 to keep all iterations.",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=40000,
        help="Maximum epoch included in the plot. Use -1 to keep all epochs.",
    )
    parser.add_argument(
        "--mus",
        type=float,
        nargs="+",
        default=[0, 1, 10, 100, 1000],
        help=(
            "Optional list of μ values to include (scientific/decimal allowed). "
            "If omitted, include all μ."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help=(
            "Optional list of seeds to include. "
            "Default: [1, 3, 5]."
        ),
    )
    parser.add_argument(
        "--ref-root",
        type=Path,
        default=Path("res-20"),
        help="Reference root used to draw black lines from the last iteration (default: res-17).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    df = collect_stdout_logs(args.root)
    if df.empty:
        print(f"No stdout records found under {args.root}")
        return

    seeds = tuple(args.seeds) if args.seeds is not None else DEFAULT_SEEDS
    make_res18_seed_subplots(
        df,
        output=args.output,
        seeds=seeds,
        mus=None if args.mus is None else tuple(args.mus),
        smooth_window=args.smooth_window,
        batch_size=None if args.batch_size == -1 else args.batch_size,
        max_iter=None if args.max_iter == -1 else args.max_iter,
        max_epoch=None if args.max_epoch == -1 else args.max_epoch,
        ref_root=args.ref_root,
    )


if __name__ == "__main__":
    main()


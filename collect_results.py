#!/usr/bin/env python
from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    pd = None  # type: ignore[assignment]

DEFAULT_ROOT = Path("res/output/1")

COLUMN_ALIASES: dict[str, str] = {
    "σ_f²": "sigma_f2",
    "σ_ε²": "sigma_eps2",
    "ℓ_μ": "ell_mu",
    "|A-F|/|A|": "rel_a_minus_f",
    "|B|": "abs_b",
    "cosBΔ": "cosb_delta",
    "pen/pen*": "pen_ratio",
    "||grad||": "grad_norm",
    "|F~": "rel_ftilde_minus_f",  # handled via startswith check
}


def require_pandas() -> None:
    if pd is None:
        raise RuntimeError(
            "pandas is required to build the aggregated DataFrame. "
            "Install it via `python -m pip install pandas` and re-run the script."
        )


def normalize_header(token: str) -> str:
    if token in COLUMN_ALIASES:
        return COLUMN_ALIASES[token]
    if token.startswith("|F~"):
        return "rel_ftilde_minus_f"
    clean = []
    for ch in token:
        clean.append(ch if ch.isalnum() else "_")
    normalized = "".join(clean).strip("_").lower()
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized or "value"


def coerce_scalar(value: str | None) -> Any:
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_command_metadata(run_dir: Path) -> dict[str, Any]:
    command_str = run_dir.name
    tokens = shlex.split(command_str)
    meta: dict[str, Any] = {
        "command_str": command_str,
        "algo": None,
    }
    if "cli.py" in tokens:
        cli_idx = tokens.index("cli.py")
        algo_idx = cli_idx + 1
        if algo_idx < len(tokens):
            meta["algo"] = tokens[algo_idx]
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            value = None
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                value = tokens[i + 1]
                i += 1
            meta[key] = coerce_scalar(value)
        i += 1
    meta = {k: v for k, v in meta.items() if v is not None}
    meta["algo"] = meta.get("algo", "unknown")
    return meta


def parse_table(path: Path) -> list[dict[str, Any]]:
    def split_header_tokens(line: str) -> list[str]:
        tokens = line.split()
        merged: list[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if (
                token == "|F~"
                and i + 2 < len(tokens)
                and tokens[i + 1] == "-"
                and tokens[i + 2] == "F|/|F|"
            ):
                merged.append("|F~ - F|/|F|")
                i += 3
                continue
            merged.append(token)
            i += 1
        return merged

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("iter"):
            header_idx = idx
            break
    if header_idx is None:
        return rows
    raw_header = split_header_tokens(lines[header_idx])
    headers = [normalize_header(token) for token in raw_header]
    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("["):
            break
        tokens = line.split()
        if len(tokens) < len(headers):
            continue
        row = {}
        for key, value in zip(headers, tokens):
            coerced = coerce_scalar(value)
            if key in {"iter", "epoch"} and coerced is not None:
                coerced = int(float(coerced))
            row[key] = coerced
        rows.append(row)
    return rows


def collect_stdout_logs(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    require_pandas()
    records: list[dict[str, Any]] = []
    for stdout_path in sorted(root.glob("**/stdout")):
        table_rows = parse_table(stdout_path)
        if not table_rows:
            continue
        meta = parse_command_metadata(stdout_path.parent)
        meta["stdout_path"] = str(stdout_path)
        for row in table_rows:
            record = {**meta, **row}
            records.append(record)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect stdout logs under res/output/1 into a pandas DataFrame."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Base directory containing stdout logs (default: res/output/1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the aggregated data (CSV if endswith .csv, parquet if .parquet).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Print the first N rows to stdout for a quick sanity check.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        df = collect_stdout_logs(args.root)
    except RuntimeError as err:
        print(err)
        return
    if df.empty:
        print("No stdout records found.")
        return
    if args.limit:
        print(df.head(args.limit).to_string(index=False))
    else:
        print(df.info())
    if args.output:
        if args.output.suffix.lower() == ".parquet":
            df.to_parquet(args.output, index=False)
        else:
            df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()


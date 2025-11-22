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


def extract_command_from_stdout(stdout_path: Path) -> str | None:
    """Extract the full command from the first line of stdout file if available."""
    try:
        with stdout_path.open("r", encoding="utf-8") as fh:
            first_line = fh.readline()
        # Look for command in quotes, e.g., '[GPU X][JOB Y] 'command''
        if "'" in first_line:
            start = first_line.find("'")
            end = first_line.rfind("'")
            if start < end:
                return first_line[start + 1 : end]
        # Alternative: look for python cli.py pattern
        if "python" in first_line and "cli.py" in first_line:
            # Try to extract from the line
            parts = first_line.split("'")
            for part in parts:
                if "cli.py" in part:
                    return part.strip()
    except Exception:
        pass
    return None


def parse_command_metadata(run_dir: Path, stdout_path: Path | None = None) -> dict[str, Any]:
    stdout_cmd = extract_command_from_stdout(stdout_path) if stdout_path else None
    command_str = stdout_cmd or run_dir.name
    tokens = shlex.split(command_str) if command_str else []
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
    
    # If critical parameters are missing and we have stdout_path, try parsing from stdout
    if stdout_path is not None:
        stdout_cmd = extract_command_from_stdout(stdout_path)
        if stdout_cmd:
            stdout_tokens = shlex.split(stdout_cmd)
            # Fill in missing parameters from stdout command
            i = 0
            while i < len(stdout_tokens):
                token = stdout_tokens[i]
                if token.startswith("--"):
                    key = token[2:].replace("-", "_")
                    # Only fill if missing or None
                    if key not in meta or meta[key] is None:
                        value = None
                        if i + 1 < len(stdout_tokens) and not stdout_tokens[i + 1].startswith("--"):
                            value = stdout_tokens[i + 1]
                            i += 1
                        if value is not None:
                            meta[key] = coerce_scalar(value)
                i += 1
    
    meta = {k: v for k, v in meta.items() if v is not None}
    meta["algo"] = meta.get("algo", "unknown")
    return meta


def discover_log_files(root: Path) -> list[tuple[Path, str]]:
    """Return candidate log files along with their structure type."""
    legacy: list[tuple[Path, str]] = [
        (path, "legacy") for path in root.glob("**/stdout") if path.is_file()
    ]
    results_dirs: set[Path] = set()
    if root.is_dir() and root.name == "results":
        results_dirs.add(root)
    for results_dir in root.glob("**/results"):
        if results_dir.is_dir():
            results_dirs.add(results_dir)
    flat: list[tuple[Path, str]] = []
    for results_dir in results_dirs:
        for candidate in results_dir.iterdir():
            if candidate.is_file() and candidate.name.isdigit():
                flat.append((candidate, "flat"))
    if not legacy and not flat:
        return []
    combined = legacy + flat
    combined.sort(key=lambda item: str(item[0]))
    return combined


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
        if not stripped:
            break
        # Skip informational lines like "[MINIMAX] Epoch X: increased μ to Y"
        if stripped.startswith("[MINIMAX]") and "increased" in stripped:
            continue
        if stripped.startswith("["):
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
    log_files = discover_log_files(root)
    for stdout_path, structure_type in log_files:
        table_rows = parse_table(stdout_path)
        if not table_rows:
            continue
        meta = parse_command_metadata(stdout_path.parent, stdout_path=stdout_path)
        meta["stdout_path"] = str(stdout_path)
        meta["structure_type"] = structure_type
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


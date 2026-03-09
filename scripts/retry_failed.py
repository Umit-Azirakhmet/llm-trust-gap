#!/usr/bin/env python3
"""
Retry failed experiment items and merge results back.

Scans result directories for JSON files, finds entries that failed
(error key present or outputted_answer is null), re-runs only those
question indices via the experiment runner, then merges the new
results back into the original files.

Usage:
  # Default: scan outputs/results_llama, results_mistral, results_gemma, results_qwen
  python scripts/retry_failed.py

  # Custom result dirs and dataset
  python scripts/retry_failed.py --results-dirs outputs/results_llama outputs/results_mistral \\
      --dataset dataset/medmcqa_1000.json

  # Dry run: only print what would be retried, do not call API or merge
  python scripts/retry_failed.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiment import ExperimentRunner
from src.utils import load_json, save_json

PROMPT_GROUPS = ("g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8")
DEFAULT_RESULTS_DIRS = [
    "outputs/results_llama",
    "outputs/results_mistral",
    "outputs/results_gemma",
    "outputs/results_qwen",
]


def is_failed_entry(entry: dict) -> bool:
    """True if this result entry is considered failed (should be retried)."""
    if "error" in entry:
        return True
    if entry.get("outputted_answer") is None:
        return True
    return False


def parse_result_filename(stem: str) -> dict | None:
    """
    Parse result filename stem to get model, prompt_group, dataset.
    Expects format: {model}_{prompt_group}_{dataset}_{timestamp}.
    Timestamp is YYYYMMDD_HHMMSS (last two parts). Prompt group is one of g0..g8.
    """
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    # Last two parts: timestamp (digits)
    if not (parts[-1].isdigit() and parts[-2].isdigit()):
        return None
    rest = parts[:-2]
    # Find prompt_group (g0..g8)
    for i, p in enumerate(rest):
        if p in PROMPT_GROUPS:
            model = "_".join(rest[:i])
            dataset = "_".join(rest[i + 1 :])
            return {
                "model": model,
                "prompt_group": p,
                "dataset": dataset,
            }
    return None


def collect_failed_by_file(
    results_dirs: list[Path],
    dataset_path: Path,
) -> list[tuple[Path, str, str, str, list[int]]]:
    """
    Scan result dirs for JSON files, find failed question_ids per file.
    Returns list of (result_path, model, prompt_group, dataset_stem, failed_indices).
    """
    out = []
    for dir_path in results_dirs:
        if not dir_path.is_dir():
            continue
        for p in dir_path.glob("*.json"):
            if "_retry_" in p.name:
                continue
            stem = p.stem
            parsed = parse_result_filename(stem)
            if not parsed:
                continue
            try:
                data = load_json(str(p))
            except Exception as e:
                print(f"Warning: could not load {p}: {e}", file=sys.stderr)
                continue
            if not isinstance(data, list):
                continue
            failed = [item["question_id"] for item in data if is_failed_entry(item)]
            if not failed:
                continue
            out.append(
                (
                    p,
                    parsed["model"],
                    parsed["prompt_group"],
                    parsed["dataset"],
                    failed,
                )
            )
    return out


def run_retry(
    model: str,
    prompt_group: str,
    dataset_path: Path,
    only_indices: list[int],
    output_dir: Path,
) -> Path:
    """Run experiment for only_indices and return path to retry JSON."""
    runner = ExperimentRunner(
        model=model,
        prompt_group=prompt_group,
        dataset_path=str(dataset_path),
    )
    retry_path = runner.run(
        output_dir=str(output_dir),
        only_indices=only_indices,
    )
    return Path(retry_path)


def merge_retry_into_original(original_path: Path, retry_path: Path) -> None:
    """Replace failed entries in original JSON with results from retry JSON."""
    original = load_json(str(original_path))
    retry_list = load_json(str(retry_path))
    by_qi = {r["question_id"]: r for r in retry_list}
    for i, item in enumerate(original):
        if item.get("question_id") in by_qi:
            original[i] = by_qi[item["question_id"]]
    save_json(original, str(original_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retry failed experiment items and merge results back.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        default=DEFAULT_RESULTS_DIRS,
        help="Directories to scan for result JSONs (default: outputs/results_llama, ...)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "dataset" / "medmcqa_1000.json",
        help="Dataset JSON path (default: dataset/medmcqa_1000.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be retried; do not call API or merge",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Run retries and save retry JSONs but do not merge back into original files",
    )
    args = parser.parse_args()

    results_dirs = [ROOT / d for d in args.results_dirs]
    dataset_path = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset

    if not dataset_path.exists():
        print(f"Error: dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    planned = collect_failed_by_file(results_dirs, dataset_path)
    if not planned:
        print("No failed entries found in any result file.")
        return

    print(f"Found {len(planned)} result file(s) with failed entries.\n")
    for result_path, model, prompt_group, dataset_stem, failed in planned:
        suffix = f"... (total {len(failed)})" if len(failed) > 5 else ""
        sample = failed[:5] if len(failed) > 5 else failed
        print(f"  {result_path.name}: {len(failed)} failed (e.g. {sample}) {suffix}")

    if args.dry_run:
        print("\n[DRY RUN] No API calls or file writes.")
        return

    for result_path, model, prompt_group, dataset_stem, failed in planned:
        output_dir = result_path.parent
        print(f"\nRetrying {len(failed)} items: {model} / {prompt_group} -> {result_path.name}")
        retry_path = run_retry(
            model=model,
            prompt_group=prompt_group,
            dataset_path=dataset_path,
            only_indices=failed,
            output_dir=output_dir,
        )
        if not args.no_merge:
            merge_retry_into_original(result_path, retry_path)
            print(f"  Merged into {result_path.name}")
            try:
                retry_path.unlink()
            except OSError:
                pass

    print("\nDone.")


if __name__ == "__main__":
    main()

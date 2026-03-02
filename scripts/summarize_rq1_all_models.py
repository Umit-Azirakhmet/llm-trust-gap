#!/usr/bin/env python3
"""
Summarize RQ1 (Comparative Analysis by Prompt Group) for all four models in one run.

Models: Llama 3.1 8B, Mistral Small, Gemma 3N 4B, Qwen3 80B.
Outputs one RQ1 table per model.

Uses only json, pathlib, numpy, scipy (no pandas/statsmodels) for portability.

Usage:
  python scripts/summarize_rq1_all_models.py
  python scripts/summarize_rq1_all_models.py --dataset medmcqa_1000

  Results are read from outputs/results_llama, outputs/results_mistral, etc. (under project root).
  To use a different project root, pass an absolute path: --results-base /path/to/project
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

GROUP_STRATEGY = {
    "g0": "Baseline",
    "g1": "Evidence-first",
    "g2": "Counterfactual",
    "g3": "High-stakes",
    "g4": "Skeptical Auditor",
    "g5": "Scoring Rule",
    "g6": "Anchoring",
    "g8": "Group 8",
    "g9": "Group 9",
}
ALL_GROUPS = ["g0", "g1", "g2", "g3", "g4", "g5", "g6", "g8", "g9"]
CER_THRESHOLD = 80

# (model key, display name, results dir relative to ROOT)
MODELS = [
    ("llama-3.1-8b", "Llama 3.1 8B", "outputs/results_llama"),
    ("mistral-small", "Mistral Small", "outputs/results_mistral"),
    ("gemma-3-4b", "Gemma 3N 4B", "outputs/results_gemma"),
    ("qwen-3-80b", "Qwen3 80B", "outputs/results_qwen"),
]


def _cell(s, w):
    """Pad string to width w for aligned table output."""
    return str(s)[:w].ljust(w)


def load_group_results(results_dir: Path, model: str, dataset: str) -> dict:
    """Load latest JSON per prompt group for given model and dataset."""
    results_dir = Path(results_dir)
    pattern = f"{model}_g*_{dataset}_*.json"
    files = list(results_dir.glob(pattern))
    by_group = {}
    valid = tuple(ALL_GROUPS)
    for f in files:
        try:
            parts = f.stem.split("_")
            group = None
            for p in parts:
                if p in valid:
                    group = p
                    break
            if group is None:
                continue
            data = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                continue
            if group not in by_group or f.stat().st_mtime > Path(by_group[group]["_path"]).stat().st_mtime:
                by_group[group] = {"_path": str(f), "data": data}
        except Exception as e:
            print(f"Skip {f}: {e}")
    return {k: v["data"] for k, v in by_group.items()}


def compute_metrics(data: list, cer_threshold: int = CER_THRESHOLD) -> dict | None:
    """Compute RQ1 metrics: Acc, Mean V, Mean P, V-P Gap, vp_diffs, Spearman corr(V,P), p-value."""
    if not data:
        return None
    correct = [1 if r.get("is_correct") else 0 for r in data]
    V = [r["V"] for r in data if r.get("V") is not None]
    P = [r["P"] for r in data if r.get("P") is not None]
    acc = float(np.mean(correct) * 100)
    mean_v = float(np.mean(V)) if V else None
    mean_p = float(np.mean(P)) if P else None
    vp_diffs = [abs(r["V"] - r["P"]) / 100.0 for r in data if r.get("V") is not None and r.get("P") is not None]
    vp_gap = float(np.mean(vp_diffs)) if vp_diffs else None
    v_for_corr = [r["V"] for r in data if r.get("V") is not None and r.get("P") is not None]
    p_for_corr = [r["P"] for r in data if r.get("V") is not None and r.get("P") is not None]
    spearman_corr = None
    spearman_p = None
    if len(p_for_corr) > 1 and len(v_for_corr) > 1:
        rho, p_rho = stats.spearmanr(v_for_corr, p_for_corr)
        if not np.isnan(rho):
            spearman_corr = float(rho)
            spearman_p = float(p_rho)
    return {
        "acc": acc,
        "mean_v": mean_v,
        "mean_p": mean_p,
        "vp_gap": vp_gap,
        "vp_diffs": vp_diffs,
        "spearman_corr": spearman_corr,
        "spearman_p": spearman_p,
    }


def print_rq1_table(metrics: dict) -> None:
    """Print a single RQ1 table (same format as analyze_results.py)."""
    wG, wS, wA, wV, wP, wVP, wPt, wCorr, wPcorr = 5, 20, 5, 7, 7, 7, 10, 10, 10
    print("| " + _cell("Group", wG) + " | " + _cell("Strategy", wS) + " | " + _cell("Acc.", wA) + " | " + _cell("Mean V", wV) + " | " + _cell("Mean P", wP) + " | " + _cell("V-P Gap", wVP) + " | " + _cell("p (t-test vs G0)", wPt) + " | " + _cell("corr(V,P)", wCorr) + " | " + _cell("p (corr)", wPcorr) + " |")
    print("|" + "-" * (wG + 2) + "|" + "-" * (wS + 2) + "|" + "-" * (wA + 2) + "|" + "-" * (wV + 2) + "|" + "-" * (wP + 2) + "|" + "-" * (wVP + 2) + "|" + "-" * (wPt + 2) + "|" + "-" * (wCorr + 2) + "|" + "-" * (wPcorr + 2) + "|")
    for g in ALL_GROUPS:
        if g not in metrics:
            continue
        m = metrics[g]
        acc = f"{m['acc']:.0f}%"
        mv = f"{m['mean_v']:.1f}%" if m["mean_v"] is not None else "—"
        mp = f"{m['mean_p']:.1f}%" if m["mean_p"] is not None else "—"
        vp = f"{m['vp_gap']:.3f}" if m["vp_gap"] is not None else "—"
        corr_str = f"{m['spearman_corr']:.3f}" if m.get("spearman_corr") is not None else "—"
        p_corr_str = f"{m['spearman_p']:.3f}" if m.get("spearman_p") is not None else "—"
        if g == "g0":
            p_str = "—"
        else:
            g0_vp = metrics.get("g0", {}).get("vp_diffs", [])
            gx_vp = m.get("vp_diffs", [])
            if g0_vp and gx_vp:
                _, p_val = stats.ttest_ind(g0_vp, gx_vp)
                p_str = f"{p_val:.3f}"
            else:
                p_str = "—"
        print("| " + _cell(g.upper(), wG) + " | " + _cell(GROUP_STRATEGY.get(g, g), wS) + " | " + _cell(acc, wA) + " | " + _cell(mv, wV) + " | " + _cell(mp, wP) + " | " + _cell(vp, wVP) + " | " + _cell(p_str, wPt) + " | " + _cell(corr_str, wCorr) + " | " + _cell(p_corr_str, wPcorr) + " |")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize RQ1 for Llama 3.1 8B, Mistral Small, Gemma 3N 4B, Qwen3 80B.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="medmcqa_1000",
        help="Dataset name (default: medmcqa_1000)",
    )
    parser.add_argument(
        "--results-base",
        type=Path,
        default=ROOT,
        help="Project root (default: script's parent parent). Use absolute path to override.",
    )
    args = parser.parse_args()
    # If results_base is relative, treat as project root (ROOT) to avoid outputs/outputs/... duplication.
    results_base = args.results_base.resolve() if args.results_base.is_absolute() else ROOT

    print("\n# RQ1: Comparative Analysis by Prompt Group (all models)\n")
    print("Acc = accuracy (%).  Mean V / Mean P = mean verbalized / mean internal confidence.")
    print("V-P Gap = mean |V−P|.  p (t-test) = t-test on |V−P| vs G0.  corr(V,P) = Spearman ρ.  p (corr) = p-value for Spearman test.\n")

    for model_key, display_name, rel_dir in MODELS:
        results_dir = results_base / rel_dir
        if not results_dir.is_dir():
            print(f"## {display_name}\n(Skip: {results_dir} not found.)\n")
            continue
        by_group = load_group_results(results_dir, model_key, args.dataset)
        if not by_group:
            print(f"## {display_name}\n(Skip: no result files for model={model_key}, dataset={args.dataset}.)\n")
            continue
        metrics = {}
        for g in ALL_GROUPS:
            if g not in by_group:
                continue
            m = compute_metrics(by_group[g])
            if m:
                metrics[g] = m
        if not metrics:
            print(f"## {display_name}\n(Skip: no valid metrics.)\n")
            continue
        print(f"## {display_name} ({model_key}, {args.dataset})\n")
        print_rq1_table(metrics)

    print("Done.")


if __name__ == "__main__":
    main()

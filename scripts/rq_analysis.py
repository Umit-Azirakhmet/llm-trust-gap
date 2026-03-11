#!/usr/bin/env python3
"""
Comprehensive analysis script for all research questions with unified metrics.

Research questions:
1. Which structural prompt elements maximize alignment corr(V, P)?
2. Does minimizing Expected Calibration Error (ECE) conflict with maximizing alignment (corr)?
3. Does improved alignment translate to better correctness?
4. How do model size and dataset moderate the elasticity of alignment to prompt structures?

Metrics:
- Alignment (corr): Spearman's rank correlation between P (internal logit-based prob) and V (verbal score).
- Dual Calibration (ECE):
    * ECE_V: Expected Calibration Error of V vs. correctness.
    * ECE_P: Expected Calibration Error of P vs. correctness.
- Trustworthiness (AURC):
    * AURC_V: Area Under the Risk–Coverage curve using V as confidence.
    * AURC_P: Area Under the Risk–Coverage curve using P as confidence.

Usage examples:
    python scripts/rq_analysis.py --dataset medmcqa_1000
    python scripts/rq_analysis.py --dataset both
    python scripts/rq_analysis.py -o outputs/my_report.txt

Output (when run with --dataset both):
    - outputs/rq_analysis_medmcqa.txt   (med report: tables + bar charts)
    - outputs/rq_analysis_law.txt   (law report)
    - outputs/rq_analysis_law+med.txt   (law+med: averaged statistics)
    - outputs/rq_analysis_case_study.txt   (null values + model responses for both, clearly separated)
    - outputs/rq_analysis_<datasets>.json   (machine-readable metrics)
    Use -o/--output to also write a custom report path.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
from datetime import datetime
from scipy import stats
import statsmodels.formula.api as smf
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


GROUP_STRATEGY = {
    "g0": "Baseline",
    "g1": "Evidence-first",
    "g2": "Counterfactual",
    "g3": "High-stakes",
    "g4": "Skeptical Auditor",
    "g5": "Scoring Rule",
    "g6": "Anchoring",
    "g7": "Group 7",
    "g8": "Group 8",
}

ALL_GROUPS = ["g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"]

# (model key, display name, model size in B parameters)
MODELS = [
    ("llama-3.1-8b", "Llama 3.1 8B", 8),
    ("mistral-small", "Mistral Small", 24),
    ("gemma-3-4b", "Gemma 3N 4B", 4),
    ("qwen-3-80b", "Qwen3 80B", 80),
]

# Map model key -> short tag used in folder names like med_results_<tag>, law_results_<tag>
MODEL_TAG = {
    "llama-3.1-8b": "llama",
    "mistral-small": "mistral",
    "gemma-3-4b": "gemma",
    "qwen-3-80b": "qwen",
}


class Tee:
    """Write to multiple streams (e.g., stdout and a file)."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def _ascii_bar_chart(
    labels: List[str],
    values: List[float],
    title: str = "",
    width: int = 40,
    bar_char: str = "█",
    value_fmt: str = ".3f",
    display_values: Optional[List[float]] = None,
) -> str:
    """Produce an ASCII bar chart. values used for bar length (0-1 scale). display_values for label."""
    lines = []
    if title:
        lines.append(title)
        lines.append("")
    if not values or not labels:
        return "\n".join(lines) if lines else ""
    disp = display_values if display_values is not None else values
    valid_vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    v_min = min(valid_vals) if valid_vals else 0
    v_max = max(valid_vals) if valid_vals else 1
    v_range = v_max - v_min if v_max != v_min else 1.0
    for lbl, val, d in zip(labels, values, disp):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            bar_len = 0
            val_str = "—"
        else:
            norm = (float(val) - v_min) / v_range
            bar_len = int(norm * width) if norm >= 0 else 0
            val_str = f"{d:{value_fmt}}" if d is not None and not (isinstance(d, float) and np.isnan(d)) else "—"
        bar = bar_char * bar_len
        lines.append(f"  {lbl:<8} |{bar} {val_str}")
    return "\n".join(lines)


def _cell(s: object, w: int) -> str:
    """Pad string to width w for aligned markdown table output."""
    return str(s)[:w].ljust(w)


def _dataset_short_name(dataset: str) -> str:
    """Human-friendly short name for datasets for column headers."""
    if dataset.startswith("medmcqa"):
        return "Med"
    if dataset.startswith("mmlu"):
        return "Law"
    return dataset


def _ece(conf_list: List[float], correct_list: List[int], n_bins: int = 10) -> Optional[float]:
    """
    Expected Calibration Error: weighted avg of |accuracy_bin - confidence_bin|.
    conf_list: confidence scores on 0–100 scale.
    """
    if not conf_list or not correct_list or len(conf_list) != len(correct_list):
        return None
    conf_arr = np.array(conf_list, dtype=float) / 100.0
    correct_arr = np.array(correct_list, dtype=float)
    n = len(conf_arr)
    if n == 0:
        return None
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (conf_arr >= bin_edges[i]) & (conf_arr < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin = (conf_arr >= bin_edges[i]) & (conf_arr <= bin_edges[i + 1])
        n_bin = int(np.sum(in_bin))
        if n_bin == 0:
            continue
        acc_bin = float(np.mean(correct_arr[in_bin]))
        conf_bin = float(np.mean(conf_arr[in_bin]))
        ece += (n_bin / n) * abs(acc_bin - conf_bin)
    return float(ece)


def _aurc(conf_list: List[float], correct_list: List[int]) -> Optional[float]:
    """
    Area Under the Risk–Coverage curve (AURC).

    - Sort examples by decreasing confidence.
    - At coverage k/n, risk = 1 - accuracy among top-k predictions.
    - Discrete approximation of ∫ risk(coverage) d(coverage).
    """
    if not conf_list or len(conf_list) != len(correct_list):
        return None
    conf = np.array(conf_list, dtype=float) / 100.0
    y = np.array(correct_list, dtype=int)
    n = len(conf)
    if n == 0:
        return None
    order = np.argsort(-conf)  # descending confidence
    y_sorted = y[order]
    cum_errors = np.cumsum(1 - y_sorted)
    k = np.arange(1, n + 1)
    risk = cum_errors / k  # error rate among top-k
    # coverage = k / n; Δcoverage = 1/n; so AURC = mean(risk)
    aurc = float(np.mean(risk))
    return aurc


def load_group_results(results_dir: Path, model: str, dataset: str) -> Dict[str, List[dict]]:
    """
    Load latest JSON per prompt group for given model and dataset.
    Mirrors logic from summarize_rq1_all_models/analyze_results.
    """
    results_dir = Path(results_dir)
    pattern = f"{model}_g*_{dataset}_*.json"
    files = list(results_dir.glob(pattern))
    by_group: Dict[str, dict] = {}
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
            # keep most recent file per group
            if group not in by_group or f.stat().st_mtime > Path(by_group[group]["_path"]).stat().st_mtime:
                by_group[group] = {"_path": str(f), "data": data}
        except Exception as e:
            print(f"Skip {f}: {e}")
    return {k: v["data"] for k, v in by_group.items()}


@dataclass
class GroupMetrics:
    model_key: str
    model_name: str
    model_size: float
    dataset: str
    group: str
    group_label: str
    n: int
    acc: float
    mean_v: Optional[float]
    mean_p: Optional[float]
    corr_vp: Optional[float]
    corr_vp_p: Optional[float]
    ece_v: Optional[float]
    ece_p: Optional[float]
    aurc_v: Optional[float]
    aurc_p: Optional[float]


def compute_group_metrics(
    data: List[dict],
    model_key: str,
    model_name: str,
    model_size: float,
    dataset: str,
    group: str,
) -> Optional[GroupMetrics]:
    """Compute all per-group metrics needed for RQs."""
    if not data:
        return None
    correct_all = [1 if r.get("is_correct") else 0 for r in data]
    if not correct_all:
        return None

    # Basic aggregates (ignore nulls for means)
    V_all = [r["V"] for r in data if r.get("V") is not None]
    P_all = [r["P"] for r in data if r.get("P") is not None]

    acc = float(np.mean(correct_all) * 100.0)
    mean_v = float(np.mean(V_all)) if V_all else None
    mean_p = float(np.mean(P_all)) if P_all else None

    # Alignment corr(V,P)
    v_for_corr = [r["V"] for r in data if r.get("V") is not None and r.get("P") is not None]
    p_for_corr = [r["P"] for r in data if r.get("V") is not None and r.get("P") is not None]
    corr_vp: Optional[float] = None
    corr_vp_p: Optional[float] = None
    if len(v_for_corr) > 1 and len(p_for_corr) > 1:
        rho, p_rho = stats.spearmanr(v_for_corr, p_for_corr)
        if not np.isnan(rho):
            corr_vp = float(rho)
            corr_vp_p = float(p_rho)

    # Dual calibration: ECE(V vs. acc), ECE(P vs. acc)
    # Use only items where the corresponding confidence is non-null.
    v_conf = [r["V"] for r in data if r.get("V") is not None]
    v_corr = [1 if r.get("is_correct") else 0 for r in data if r.get("V") is not None]
    p_conf = [r["P"] for r in data if r.get("P") is not None]
    p_corr = [1 if r.get("is_correct") else 0 for r in data if r.get("P") is not None]

    ece_v = _ece(v_conf, v_corr) if v_conf and v_corr else None
    ece_p = _ece(p_conf, p_corr) if p_conf and p_corr else None

    # Trustworthiness: AURC using V and P as confidences (again dropping nulls)
    aurc_v = _aurc(v_conf, v_corr) if v_conf and v_corr else None
    aurc_p = _aurc(p_conf, p_corr) if p_conf and p_corr else None

    return GroupMetrics(
        model_key=model_key,
        model_name=model_name,
        model_size=model_size,
        dataset=dataset,
        group=group,
        group_label=GROUP_STRATEGY.get(group, group),
        n=len(data),
        acc=acc,
        mean_v=mean_v,
        mean_p=mean_p,
        corr_vp=corr_vp,
        corr_vp_p=corr_vp_p,
        ece_v=ece_v,
        ece_p=ece_p,
        aurc_v=aurc_v,
        aurc_p=aurc_p,
    )


def collect_all_metrics(dataset: str, results_base: Path) -> List[GroupMetrics]:
    """Load all (model, group) results for a dataset and compute metrics."""
    metrics_all: List[GroupMetrics] = []
    # Decide which results subdirectory to use based on dataset name.
    # medmcqa_*  -> outputs/med_results_<tag>
    # mmlu_*     -> outputs/law_results_<tag>   (for mmlu_professional_law_1000)
    for model_key, model_name, model_size in MODELS:
        tag = MODEL_TAG.get(model_key)
        if tag is None:
            continue
        if dataset.startswith("medmcqa"):
            subdir = f"outputs/med_results_{tag}"
        elif dataset.startswith("mmlu"):
            subdir = f"outputs/law_results_{tag}"
        else:
            # Fallback: generic outputs/<dataset>_<tag> if ever needed
            subdir = f"outputs/{dataset}_results_{tag}"

        results_dir = results_base / subdir
        if not results_dir.is_dir():
            print(f"[WARN] Skip {model_name}: {results_dir} not found.")
            continue
        by_group = load_group_results(results_dir, model_key, dataset)
        if not by_group:
            print(f"[WARN] Skip {model_name}: no result files for dataset={dataset}.")
            continue
        for g in ALL_GROUPS:
            if g not in by_group:
                continue
            gm = compute_group_metrics(
                data=by_group[g],
                model_key=model_key,
                model_name=model_name,
                model_size=model_size,
                dataset=dataset,
                group=g,
            )
            if gm:
                metrics_all.append(gm)
    return metrics_all


def average_metrics_across_datasets(metrics_list: List[GroupMetrics], combined_dataset_name: str = "law+med") -> List[GroupMetrics]:
    """
    For each (model_key, group), average numeric metrics across datasets.
    Returns one GroupMetrics per (model_key, group) with dataset=combined_dataset_name.
    """
    key_to_rows: Dict[Tuple[str, str], List[GroupMetrics]] = defaultdict(list)
    for gm in metrics_list:
        key_to_rows[(gm.model_key, gm.group)].append(gm)

    result: List[GroupMetrics] = []
    for (model_key, group), rows in key_to_rows.items():
        if not rows:
            continue
        model_name = rows[0].model_name
        model_size = rows[0].model_size
        group_label = GROUP_STRATEGY.get(group, group)

        def avg_or_none(field: str) -> Optional[float]:
            vals = [getattr(r, field) for r in rows if getattr(r, field) is not None]
            if not vals:
                return None
            return float(np.mean(vals))

        result.append(GroupMetrics(
            model_key=model_key,
            model_name=model_name,
            model_size=model_size,
            dataset=combined_dataset_name,
            group=group,
            group_label=group_label,
            n=int(np.mean([r.n for r in rows])),
            acc=avg_or_none("acc") or 0.0,
            mean_v=avg_or_none("mean_v"),
            mean_p=avg_or_none("mean_p"),
            corr_vp=avg_or_none("corr_vp"),
            corr_vp_p=avg_or_none("corr_vp_p"),
            ece_v=avg_or_none("ece_v"),
            ece_p=avg_or_none("ece_p"),
            aurc_v=avg_or_none("aurc_v"),
            aurc_p=avg_or_none("aurc_p"),
        ))
    return result


def rq1_alignment_tables(metrics_all: List[GroupMetrics], dataset: str) -> None:
    """RQ1: Which prompt elements maximize alignment corr(V,P)?"""
    print(f"\n# RQ1: Alignment by Prompt Group — Dataset = {dataset}\n")
    print("Alignment = Spearman ρ(V,P), i.e., monotonic alignment between verbal confidence (V) and internal logits (P).\n")

    wG, wS, wCorr, wPcorr, wAcc, wV, wP = 5, 22, 10, 10, 7, 7, 7

    by_model: Dict[str, List[GroupMetrics]] = {}
    for gm in metrics_all:
        by_model.setdefault(gm.model_key, []).append(gm)

    for model_key, groups in by_model.items():
        # sort groups by alignment corr descending
        groups_sorted = sorted(
            [g for g in groups if g.corr_vp is not None],
            key=lambda x: x.corr_vp,
            reverse=True,
        )
        if not groups_sorted:
            continue
        model_name = groups_sorted[0].model_name
        print(f"## {model_name} ({model_key})\n")
        print("| " + _cell("Group", wG) + " | " + _cell("Strategy", wS) + " | " +
              _cell("corr(V,P)", wCorr) + " | " + _cell("p (corr)", wPcorr) + " | " +
              _cell("Acc.", wAcc) + " | " + _cell("Mean V", wV) + " | " + _cell("Mean P", wP) + " |")
        print("|" + "-" * (wG + 2) + "|" + "-" * (wS + 2) + "|" + "-" * (wCorr + 2) +
              "|" + "-" * (wPcorr + 2) + "|" + "-" * (wAcc + 2) + "|" + "-" * (wV + 2) +
              "|" + "-" * (wP + 2) + "|")

        best_group = groups_sorted[0].group
        for gm in groups_sorted:
            label = GROUP_STRATEGY.get(gm.group, gm.group)
            corr_str = f"{gm.corr_vp:.3f}" if gm.corr_vp is not None else "—"
            p_corr_str = f"{gm.corr_vp_p:.3f}" if gm.corr_vp_p is not None else "—"
            acc_str = f"{gm.acc:.0f}%"
            mv_str = f"{gm.mean_v:.1f}%" if gm.mean_v is not None else "—"
            mp_str = f"{gm.mean_p:.1f}%" if gm.mean_p is not None else "—"
            mark = " (BEST)" if gm.group == best_group else ""
            print("| " + _cell(gm.group.upper(), wG) + " | " + _cell(label + mark, wS) +
                  " | " + _cell(corr_str, wCorr) + " | " + _cell(p_corr_str, wPcorr) +
                  " | " + _cell(acc_str, wAcc) + " | " + _cell(mv_str, wV) +
                  " | " + _cell(mp_str, wP) + " |")
        print()

    # Bar chart: corr(V,P) per group per model (map [-1,1] to [0,1] for bar length)
    print("\n### RQ1 Bar Chart: Alignment ρ(V,P) by Group\n")
    for model_key, groups in by_model.items():
        usable = sorted(
            [g for g in groups if g.corr_vp is not None],
            key=lambda g: g.corr_vp or -1,
            reverse=True,
        )
        if not usable:
            continue
        labels = [f"{g.group.upper()}" for g in usable]
        vals = [g.corr_vp for g in usable]
        vals_norm = [(v + 1) / 2 for v in vals]  # [0,1] for bar length
        model_name = usable[0].model_name
        chart = _ascii_bar_chart(
            labels, vals_norm,
            title=f"{model_name}: corr(V,P) (bar ∝ (ρ+1)/2)",
            value_fmt=".3f",
            display_values=vals,
        )
        print(chart)
        print()


def rq2_alignment_vs_calibration(metrics_all: List[GroupMetrics], dataset: str) -> None:
    """RQ2: Does minimizing ECE conflict with maximizing alignment (corr)?"""
    print(f"\n# RQ2: Alignment–Calibration Trade-off — Dataset = {dataset}\n")
    print("ECE_V = ECE(V vs. correctness).  ECE_P = ECE(P vs. correctness).  Alignment = Spearman ρ(V,P).\n")

    wG, wS, wEcv, wEcp, wCorr = 5, 22, 10, 10, 10

    by_model: Dict[str, List[GroupMetrics]] = {}
    for gm in metrics_all:
        by_model.setdefault(gm.model_key, []).append(gm)

    for model_key, groups in by_model.items():
        usable = [g for g in groups if g.corr_vp is not None and g.ece_v is not None and g.ece_p is not None]
        if not usable:
            continue
        model_name = usable[0].model_name
        print(f"## {model_name} ({model_key})\n")
        print("| " + _cell("Group", wG) + " | " + _cell("Strategy", wS) + " | " +
              _cell("ECE_V", wEcv) + " | " + _cell("ECE_P", wEcp) + " | " +
              _cell("corr(V,P)", wCorr) + " |")
        print("|" + "-" * (wG + 2) + "|" + "-" * (wS + 2) + "|" + "-" * (wEcv + 2) +
              "|" + "-" * (wEcp + 2) + "|" + "-" * (wCorr + 2) + "|")

        for gm in usable:
            label = GROUP_STRATEGY.get(gm.group, gm.group)
            ece_v_str = f"{gm.ece_v:.4f}" if gm.ece_v is not None else "—"
            ece_p_str = f"{gm.ece_p:.4f}" if gm.ece_p is not None else "—"
            corr_str = f"{gm.corr_vp:.3f}" if gm.corr_vp is not None else "—"
            print("| " + _cell(gm.group.upper(), wG) + " | " + _cell(label, wS) +
                  " | " + _cell(ece_v_str, wEcv) + " | " + _cell(ece_p_str, wEcp) +
                  " | " + _cell(corr_str, wCorr) + " |")

        # Simple correlation between ECE and alignment across groups (within model)
        ece_v_arr = np.array([g.ece_v for g in usable], dtype=float)
        ece_p_arr = np.array([g.ece_p for g in usable], dtype=float)
        corr_arr = np.array([g.corr_vp for g in usable], dtype=float)
        if len(usable) > 2:
            rho_v, p_v = stats.spearmanr(ece_v_arr, corr_arr)
            rho_p, p_p = stats.spearmanr(ece_p_arr, corr_arr)
            print(f"\nSpearman(ECE_V, corr(V,P)) = {rho_v:.3f} (p = {p_v:.3f})")
            print(f"Spearman(ECE_P, corr(V,P)) = {rho_p:.3f} (p = {p_p:.3f})\n")
        else:
            print("\n(Not enough groups to estimate correlations reliably.)\n")

    # Bar chart: ECE_V per group (lower is better; bar ∝ 1-ECE)
    print("\n### RQ2 Bar Chart: ECE_V by Group (lower is better)\n")
    for model_key, groups in by_model.items():
        usable = [g for g in groups if g.ece_v is not None]
        if not usable:
            continue
        usable = sorted(usable, key=lambda g: g.ece_v or 1)
        labels = [f"{g.group.upper()}" for g in usable]
        vals = [g.ece_v for g in usable]
        vals_norm = [1 - v for v in vals]  # invert: lower ECE -> longer bar
        model_name = usable[0].model_name
        chart = _ascii_bar_chart(labels, vals_norm, title=f"{model_name}: ECE_V (bar ∝ 1-ECE, lower better)", value_fmt=".4f", display_values=vals)
        print(chart)
        print()


def rq3_alignment_vs_correctness(metrics_all: List[GroupMetrics], dataset: str) -> None:
    """Supplement: association between alignment and correctness at group level."""
    print(f"\n## RQ3 (Supplement): Alignment vs. Correctness — Dataset = {dataset}\n")
    print("Here we correlate group-level alignment corr(V,P) with group-level accuracy.\n")

    wG, wS, wCorr, wAcc = 5, 22, 10, 7

    by_model: Dict[str, List[GroupMetrics]] = {}
    for gm in metrics_all:
        by_model.setdefault(gm.model_key, []).append(gm)

    for model_key, groups in by_model.items():
        usable = [g for g in groups if g.corr_vp is not None]
        if not usable:
            continue
        model_name = usable[0].model_name
        print(f"## {model_name} ({model_key})\n")
        print("| " + _cell("Group", wG) + " | " + _cell("Strategy", wS) + " | " +
              _cell("corr(V,P)", wCorr) + " | " + _cell("Acc.", wAcc) + " |")
        print("|" + "-" * (wG + 2) + "|" + "-" * (wS + 2) + "|" + "-" * (wCorr + 2) +
              "|" + "-" * (wAcc + 2) + "|")

        for gm in usable:
            label = GROUP_STRATEGY.get(gm.group, gm.group)
            corr_str = f"{gm.corr_vp:.3f}" if gm.corr_vp is not None else "—"
            acc_str = f"{gm.acc:.0f}%"
            print("| " + _cell(gm.group.upper(), wG) + " | " + _cell(label, wS) +
                  " | " + _cell(corr_str, wCorr) + " | " + _cell(acc_str, wAcc) + " |")

        if len(usable) > 2:
            corr_arr = np.array([g.corr_vp for g in usable], dtype=float)
            acc_arr = np.array([g.acc for g in usable], dtype=float)
            rho, p_val = stats.spearmanr(corr_arr, acc_arr)
            print(f"\nSpearman(corr(V,P), accuracy) across groups = {rho:.3f} (p = {p_val:.3f})\n")
        else:
            print("\n(Not enough groups to estimate correlation reliably.)\n")


def rq3_trustworthiness_tables(metrics_all: List[GroupMetrics], dataset: str) -> None:
    """RQ3: Trustworthiness via AURC per group."""
    print(f"\n# RQ3: Trustworthiness via AURC — Dataset = {dataset}\n")
    print("Lower AURC = better selective prediction (lower risk at a given coverage). We report AURC using V and P as confidence.\n")

    wG, wS, wAv, wAp = 5, 22, 12, 12

    by_model: Dict[str, List[GroupMetrics]] = {}
    for gm in metrics_all:
        by_model.setdefault(gm.model_key, []).append(gm)

    for model_key, groups in by_model.items():
        usable = [g for g in groups if g.aurc_v is not None and g.aurc_p is not None]
        if not usable:
            continue
        model_name = usable[0].model_name
        print(f"### {model_name} ({model_key})\n")
        print("| " + _cell("Group", wG) + " | " + _cell("Strategy", wS) + " | " +
              _cell("AURC_V", wAv) + " | " + _cell("AURC_P", wAp) + " |")
        print("|" + "-" * (wG + 2) + "|" + "-" * (wS + 2) + "|" + "-" * (wAv + 2) +
              "|" + "-" * (wAp + 2) + "|")

        for gm in usable:
            label = GROUP_STRATEGY.get(gm.group, gm.group)
            aurc_v_str = f"{gm.aurc_v:.4f}" if gm.aurc_v is not None else "—"
            aurc_p_str = f"{gm.aurc_p:.4f}" if gm.aurc_p is not None else "—"
            print("| " + _cell(gm.group.upper(), wG) + " | " + _cell(label, wS) +
                  " | " + _cell(aurc_v_str, wAv) + " | " + _cell(aurc_p_str, wAp) + " |")
        print()

    # Bar chart: AURC_V per group (lower is better; bar ∝ 1-AURC)
    print("\n### RQ3 Bar Chart: AURC_V by Group (lower is better)\n")
    for model_key, groups in by_model.items():
        usable = [g for g in groups if g.aurc_v is not None]
        if not usable:
            continue
        usable = sorted(usable, key=lambda g: g.aurc_v or 1)
        labels = [f"{g.group.upper()}" for g in usable]
        vals = [g.aurc_v for g in usable]
        vals_norm = [1 - v for v in vals]  # invert: lower AURC -> longer bar
        model_name = usable[0].model_name
        chart = _ascii_bar_chart(labels, vals_norm, title=f"{model_name}: AURC_V (bar ∝ 1-AURC, lower better)", value_fmt=".4f", display_values=vals)
        print(chart)
        print()


def null_values_case_study(dataset: str, results_base: Path) -> None:
    """
    Case study table for all entries with null V or P.
    For each (model, group) we report counts and a few example rows with the raw model output.
    """
    print(f"\n# Null-Value Case Study — Dataset = {dataset}\n")
    print("We list groups where verbal confidence V or internal probability P is missing,")
    print("along with example raw model outputs for inspection.\n")

    for model_key, model_name, model_size in MODELS:
        tag = MODEL_TAG.get(model_key)
        if tag is None:
            continue
        if dataset.startswith("medmcqa"):
            subdir = f"outputs/med_results_{tag}"
        elif dataset.startswith("mmlu"):
            subdir = f"outputs/law_results_{tag}"
        else:
            subdir = f"outputs/{dataset}_results_{tag}"
        results_dir = results_base / subdir
        if not results_dir.is_dir():
            continue

        files = sorted(results_dir.glob(f"{model_key}_g*_{dataset}_*.json"))
        any_for_model = False
        for f in files:
            parts = f.stem.split("_")
            group = None
            for p in parts:
                if p in ALL_GROUPS:
                    group = p
                    break
            if group is None:
                continue

            try:
                with f.open(encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as e:
                print(f"Skip {f}: {e}")
                continue

            null_entries = [r for r in data if r.get("V") is None or r.get("P") is None]
            if not null_entries:
                continue

            if not any_for_model:
                print(f"## {model_name} ({model_key})\n")
                any_for_model = True

            v_null_only = sum(1 for r in null_entries if r.get("V") is None and r.get("P") is not None)
            p_null_only = sum(1 for r in null_entries if r.get("P") is None and r.get("V") is not None)
            both_null = sum(1 for r in null_entries if r.get("V") is None and r.get("P") is None)

            print(f"### Group {group.upper()} — file `{f.name}`")
            print(f"- Total items with null V or P: {len(null_entries)}  "
                  f"(V null only: {v_null_only}, P null only: {p_null_only}, both null: {both_null})")
            print("- Example rows (up to 5):")
            for r in null_entries[:5]:
                qid = r.get("question_id")
                V = r.get("V")
                P = r.get("P")
                is_corr = r.get("is_correct")
                raw = (r.get("model_output") or "").replace("\n", " ")
                if len(raw) > 120:
                    raw = raw[:117] + "..."
                print(f"  - question_id={qid}, V={V}, P={P}, is_correct={is_corr}, output=`{raw}`")
            print()

        if any_for_model:
            print()


def save_metrics_json(metrics_all: List[GroupMetrics], datasets: Iterable[str], results_base: Path) -> None:
    """
    Save a machine-readable JSON with all per-(model,group) metrics.
    """
    out_dir = results_base / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_list = sorted(set(datasets))
    if len(ds_list) == 1:
        label = ds_list[0]
    else:
        label = "+".join(ds_list)
    out_path = out_dir / f"rq_analysis_{label}.json"

    payload = {
        "datasets": ds_list,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "groups": [asdict(gm) for gm in metrics_all],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved JSON metrics to: {out_path}")


def rq4_scaling_moderation(metrics_all: List[GroupMetrics], dataset: str) -> None:
    """RQ4: How do model size and dataset moderate the elasticity of alignment to prompt structures?"""
    print(f"\n# RQ4: Scaling & Moderation — Dataset = {dataset}\n")
    print("We pool all (model, group) cells and regress alignment on group index, model size, and their interaction.\n")

    rows: List[Dict[str, object]] = []
    for gm in metrics_all:
        if gm.corr_vp is None or gm.model_size <= 0:
            continue
        try:
            group_int = int("".join(ch for ch in gm.group if ch.isdigit()))
        except ValueError:
            continue
        rows.append(
            {
                "rho": gm.corr_vp,
                "group_int": float(group_int),
                "model_size": float(gm.model_size),
                "size_log": np.log10(float(gm.model_size)),
                "model": gm.model_key,
                "dataset": gm.dataset,
            }
        )

    if len(rows) < 5:
        print("Not enough (model, group) cells with valid alignment and model size. Skipping RQ4.")
        return

    df = pd.DataFrame(rows)
    num_models = df["model"].nunique()
    if num_models > 1:
        res = smf.ols("rho ~ size_log * group_int + C(dataset)", data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["model"]}
        )
    else:
        res = smf.ols("rho ~ size_log * group_int + C(dataset)", data=df).fit()

    print("Model: rho ~ size_log * group_int + C(dataset)\n")
    print(res.summary().tables[1])
    print(
        "\nInterpretation:\n"
        "- group_int captures movement across prompt structures (e.g., control vs. evidence-first vs. counterfactual).\n"
        "- size_log is log10(model_size in B parameters).\n"
        "- size_log:group_int interaction tests whether the effect of prompt structure on alignment scales with model size.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis for all RQs with alignment, dual ECE, and AURC metrics.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        help="Dataset name (e.g., medmcqa_1000, mmlu_professional_law_1000, or 'both' for joint analysis).",
    )
    parser.add_argument(
        "--results-base",
        type=Path,
        default=ROOT,
        help="Project root (default: script's parent parent). Use absolute path to override.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path for report (txt with tables/graphs). Default: outputs/rq_analysis_<datasets>.txt",
    )
    args = parser.parse_args()

    results_base = args.results_base.resolve() if args.results_base.is_absolute() else ROOT

    # Determine which datasets to analyze.
    if args.dataset.lower() in ("both", "all"):
        datasets = ["medmcqa_1000", "mmlu_professional_law_1000"]
    else:
        datasets = [args.dataset]

    all_metrics: List[GroupMetrics] = []
    per_dataset_metrics: Dict[str, List[GroupMetrics]] = {}
    for ds in datasets:
        metrics_ds = collect_all_metrics(dataset=ds, results_base=results_base)
        if not metrics_ds:
            print(f"No metrics computed for dataset={ds}.")
            continue
        per_dataset_metrics[ds] = metrics_ds
        all_metrics.extend(metrics_ds)

    if not all_metrics:
        print("No metrics computed for any dataset. Check that results exist.")
        return

    # Save all numeric results (possibly across multiple datasets) for later analysis.
    save_metrics_json(all_metrics, datasets=datasets, results_base=results_base)

    out_dir = results_base / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_list = sorted(per_dataset_metrics.keys())

    def write_report(path: Path, title: str, metrics: List[GroupMetrics], dataset_label: str, include_rq4: bool = False, rq4_metrics: Optional[List[GroupMetrics]] = None):
        """Write one report file: header + RQ1–RQ3 (+ optional RQ4) for given metrics. No null section."""
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            with contextlib.redirect_stdout(f):
                print(f"# Comprehensive RQ Analysis Report — {title}")
                print(f"# Generated: {datetime.utcnow().isoformat()}Z")
                print(f"# Dataset: {dataset_label}\n")
                rq1_alignment_tables(metrics, dataset=dataset_label)
                rq2_alignment_vs_calibration(metrics, dataset=dataset_label)
                rq3_alignment_vs_correctness(metrics, dataset=dataset_label)
                rq3_trustworthiness_tables(metrics, dataset=dataset_label)
                if include_rq4 and rq4_metrics:
                    rq4_scaling_moderation(rq4_metrics, dataset=dataset_label)

    # 1) Med report
    if "medmcqa_1000" in per_dataset_metrics:
        med_path = out_dir / "rq_analysis_medmcqa.txt"
        write_report(med_path, "Med", per_dataset_metrics["medmcqa_1000"], "medmcqa_1000", include_rq4=True, rq4_metrics=all_metrics)
        print(f"Report (med) saved to: {med_path}")

    # 2) Law report
    if "mmlu_professional_law_1000" in per_dataset_metrics:
        law_path = out_dir / "rq_analysis_law.txt"
        write_report(law_path, "Law", per_dataset_metrics["mmlu_professional_law_1000"], "mmlu_professional_law_1000", include_rq4=True, rq4_metrics=all_metrics)
        print(f"Report (law) saved to: {law_path}")

    # 3) Law+med report (averaged statistics) — only when both datasets present
    if "medmcqa_1000" in per_dataset_metrics and "mmlu_professional_law_1000" in per_dataset_metrics:
        averaged = average_metrics_across_datasets(all_metrics, combined_dataset_name="law+med")
        law_med_path = out_dir / "rq_analysis_law+med.txt"
        write_report(law_med_path, "Law+Med (averaged)", averaged, "law+med (averaged across medmcqa_1000 & mmlu_professional_law_1000)", include_rq4=True, rq4_metrics=all_metrics)
        print(f"Report (law+med, averaged) saved to: {law_med_path}")

    # 4) Single combined report for backward compatibility when user passes -o or single dataset
    if args.output is not None:
        custom_path = args.output.resolve()
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        if len(ds_list) == 1:
            write_report(custom_path, ds_list[0], per_dataset_metrics[ds_list[0]], ds_list[0], include_rq4=True, rq4_metrics=all_metrics)
        else:
            averaged = average_metrics_across_datasets(all_metrics, combined_dataset_name="law+med")
            write_report(custom_path, "Law+Med (averaged)", averaged, "law+med", include_rq4=True, rq4_metrics=all_metrics)
        print(f"Report (custom -o) saved to: {custom_path}")

    # 5) Case study file: null values and model responses for each dataset, clearly separated
    case_study_path = out_dir / "rq_analysis_case_study.txt"
    with open(case_study_path, "w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f):
            print("# Null Values & Model Response Case Study")
            print(f"# Generated: {datetime.utcnow().isoformat()}Z")
            print("# Lists entries with missing V or P and example raw model outputs.\n")
            for ds in ds_list:
                print("=" * 70)
                if ds.startswith("mmlu"):
                    print("\n## Law dataset (mmlu_professional_law_1000)\n")
                elif ds.startswith("medmcqa"):
                    print("\n## Med dataset (medmcqa_1000)\n")
                else:
                    print(f"\n## Dataset: {ds}\n")
                null_values_case_study(dataset=ds, results_base=results_base)
                print()
    print(f"Case study (null + model responses) saved to: {case_study_path}")


if __name__ == "__main__":
    main()


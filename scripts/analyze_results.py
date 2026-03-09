#!/usr/bin/env python3
"""
Summarize experiment results: Comparative Analysis (alignment = Spearman corr(V,P)), RQ2, RQ3.
Usage:
  python scripts/analyze_results.py --model llama-3.1-8b --dataset medmcqa_10
  python scripts/analyze_results.py --model llama-3.1-8b --dataset medmcqa_100
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

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

# All prompt groups (including g7, g8). Order for tables and iteration.
ALL_GROUPS = ["g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"]

# Confident Error Rate: threshold for "confident" (V >= this %)
CER_THRESHOLD = 80


def _ece(V_list, correct_list, n_bins=10):
    """Expected Calibration Error: weighted avg of |accuracy_bin - confidence_bin|."""
    V_arr = np.array(V_list) / 100.0
    correct_arr = np.array(correct_list, dtype=float)
    n = len(V_arr)
    if n == 0:
        return None
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (V_arr >= bin_edges[i]) & (V_arr < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin = (V_arr >= bin_edges[i]) & (V_arr <= bin_edges[i + 1])
        n_bin = np.sum(in_bin)
        if n_bin == 0:
            continue
        ece += (n_bin / n) * np.abs(np.mean(correct_arr[in_bin]) - np.mean(V_arr[in_bin]))
    return float(ece)


def _brier(V_list, correct_list):
    """Brier score: mean((V/100 - correct)^2). Lower is better."""
    if not V_list or len(V_list) != len(correct_list):
        return None
    V_arr = np.array(V_list) / 100.0
    correct_arr = np.array(correct_list, dtype=float)
    return float(np.mean((V_arr - correct_arr) ** 2))


def _cell(s, w):
    """Pad string to width w for aligned table output."""
    t = str(s)[:w]
    return t.ljust(w)


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


def build_item_level_df(by_group: dict) -> pd.DataFrame:
    """One row per item: group, vp_gap = |V-P|/100, V (0-100), is_correct (0/1)."""
    rows = []
    for group, data in by_group.items():
        for r in data:
            if r.get("V") is None or r.get("P") is None:
                continue
            vp_gap = abs(r["V"] - r["P"]) / 100.0
            is_correct = 1 if r.get("is_correct") else 0
            rows.append({"group": group, "vp_gap": vp_gap, "V": r["V"], "is_correct": is_correct})
    return pd.DataFrame(rows)


def compute_metrics(data: list, cer_threshold: int = CER_THRESHOLD) -> dict:
    """Compute Acc, Mean V, Mean P, Reality Gap, V-P Gap, V-P Gap SE, ECE, Brier, Spearman corr(P,V), p-value, CER(threshold)."""
    if not data:
        return None
    correct = [1 if r.get("is_correct") else 0 for r in data]
    V = [r["V"] for r in data if r.get("V") is not None]
    P = [r["P"] for r in data if r.get("P") is not None]
    acc = np.mean(correct) * 100
    mean_v = np.mean(V) if V else None
    mean_p = np.mean(P) if P else None
    reality_gap = (mean_v / 100.0 - acc / 100.0) if mean_v is not None else None
    vp_diffs = [abs(r["V"] - r["P"]) / 100.0 for r in data if r.get("V") is not None and r.get("P") is not None]
    vp_gap = (np.mean(vp_diffs) if vp_diffs else None)
    vp_gap_se = (float(np.std(vp_diffs) / np.sqrt(len(vp_diffs))) if len(vp_diffs) > 1 else None)
    n_confident_wrong = sum(1 for r in data if r.get("V") is not None and r["V"] >= cer_threshold and not r.get("is_correct"))
    cer_80 = (n_confident_wrong / len(data)) * 100 if data else None
    V_list = [r["V"] for r in data if r.get("V") is not None]
    correct_list = [1 if r.get("is_correct") else 0 for r in data]
    ece = _ece(V_list, correct_list) if len(V_list) == len(correct_list) and V_list else None
    brier = _brier(V_list, correct_list) if V_list else None
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
        "reality_gap": reality_gap,
        "vp_gap": vp_gap,
        "vp_gap_se": vp_gap_se,
        "n": len(data),
        "vp_diffs": vp_diffs,
        "ece": ece,
        "brier": brier,
        "spearman_corr": spearman_corr,
        "spearman_p": spearman_p,
        "cer_80": cer_80,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment results into tables")
    parser.add_argument("--results-dir", type=str, default="outputs/results")
    parser.add_argument("--model", type=str, default="llama-3.1-8b")
    parser.add_argument("--dataset", type=str, default="medmcqa_10")
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    model = args.model
    dataset = args.dataset
    by_group = load_group_results(results_dir, model, dataset)
    if not by_group:
        print("No result files found.")
        return

    metrics = {}
    for g in ALL_GROUPS:
        if g not in by_group:
            continue
        m = compute_metrics(by_group[g])
        if m:
            metrics[g] = m

    # RQ1 table: Acc, Mean V, Mean P, Alignment corr(V,P), p (corr).  Alignment = Spearman ρ.
    wG, wS, wA, wV, wP, wCorr, wPcorr = 5, 20, 5, 7, 7, 10, 10

    title = args.title or f"Preliminary Results - {model}, {dataset}"
    print(f"\n# {title}\n")
    print("## RQ1: Comparative Analysis by Prompt Group\n")
    print("Alignment = Spearman ρ(V,P) (monotonic alignment between verbalized and internal confidence).  Acc = accuracy (%).  Mean V / Mean P = mean verbalized / mean internal confidence.  p (corr) = p-value for Spearman test.\n")
    print("| " + _cell("Group", wG) + " | " + _cell("Strategy", wS) + " | " + _cell("Acc.", wA) + " | " + _cell("Mean V", wV) + " | " + _cell("Mean P", wP) + " | " + _cell("corr(V,P)", wCorr) + " | " + _cell("p (corr)", wPcorr) + " |")
    print("|" + "-" * (wG + 2) + "|" + "-" * (wS + 2) + "|" + "-" * (wA + 2) + "|" + "-" * (wV + 2) + "|" + "-" * (wP + 2) + "|" + "-" * (wCorr + 2) + "|" + "-" * (wPcorr + 2) + "|")

    for g in ALL_GROUPS:
        if g not in metrics:
            continue
        m = metrics[g]
        acc = f"{m['acc']:.0f}%"
        mv = f"{m['mean_v']:.1f}%" if m["mean_v"] is not None else "—"
        mp = f"{m['mean_p']:.1f}%" if m["mean_p"] is not None else "—"
        corr_str = f"{m['spearman_corr']:.3f}" if m.get("spearman_corr") is not None else "—"
        p_corr_str = f"{m['spearman_p']:.3f}" if m.get("spearman_p") is not None else "—"
        print("| " + _cell(g.upper(), wG) + " | " + _cell(GROUP_STRATEGY.get(g, g), wS) + " | " + _cell(acc, wA) + " | " + _cell(mv, wV) + " | " + _cell(mp, wP) + " | " + _cell(corr_str, wCorr) + " | " + _cell(p_corr_str, wPcorr) + " |")

    # RQ2
    wE, wB, wC, wPc = 8, 8, 10, 10
    print("\n## RQ2: Alignment-Calibration Trade-off\n")
    print("ECE = Expected Calibration Error.  Brier = mean((V/100 - correct)^2).  corr = Spearman ρ(P,V).  p (corr) = p-value for corr.\n")
    print("| " + _cell("Group", wG) + " | " + _cell("ECE", wE) + " | " + _cell("Brier", wB) + " | " + _cell("corr(P,V)", wC) + " | " + _cell("p (corr)", wPc) + " |")
    print("|" + "-" * (wG + 2) + "|" + "-" * (wE + 2) + "|" + "-" * (wB + 2) + "|" + "-" * (wC + 2) + "|" + "-" * (wPc + 2) + "|")
    for g in ALL_GROUPS:
        if g not in metrics:
            continue
        m = metrics[g]
        ece = f"{m['ece']:.4f}" if m.get("ece") is not None else "—"
        brier = f"{m['brier']:.4f}" if m.get("brier") is not None else "—"
        corr = f"{m['spearman_corr']:.3f}" if m.get("spearman_corr") is not None else "—"
        p_corr = f"{m['spearman_p']:.3f}" if m.get("spearman_p") is not None else "—"
        print("| " + _cell(g.upper(), wG) + " | " + _cell(ece, wE) + " | " + _cell(brier, wB) + " | " + _cell(corr, wC) + " | " + _cell(p_corr, wPc) + " |")
    print("\n(Plot ECE vs corr: negative slope suggests alignment-calibration trade-off.)")

    # RQ3 (item-level)
    print("\n## RQ3: Alignment vs. Correctness (item-level)\n")
    print("Unit of analysis: each item.  Alignment at item level = |V−P| (lower = better).  Logistic: is_correct ~ vp_gap.\n")
    df = build_item_level_df(by_group)
    if df.empty or df["is_correct"].nunique() < 2:
        print("Not enough item-level data (or no variation in is_correct) to run logistic regression.")
    else:
        model_rq3 = smf.logit("is_correct ~ vp_gap", data=df).fit(disp=0)
        coef = model_rq3.params["vp_gap"]
        se = model_rq3.bse["vp_gap"]
        p_val = model_rq3.pvalues["vp_gap"]
        or_01 = np.exp(coef * 0.1)
        n_items = len(df)
        print(f"N = {n_items} items.  Logistic: P(is_correct) ~ vp_gap (|V−P| on 0–1 scale).")
        print(f"Coefficient (log-odds per unit vp_gap): {coef:.4f},  SE = {se:.4f},  p-value = {p_val:.4f}")
        print(f"OR per 0.1 increase in |V−P|: {or_01:.4f}")
        if p_val < 0.05 and coef < 0:
            print("→ Significant: lower |V−P| (better alignment) is associated with higher probability of being correct.")
        elif p_val < 0.05:
            print("→ Significant: vp_gap is associated with is_correct.")
        else:
            print("→ Not significant at α=0.05.")

    # RQ3: G0 (control) vs G4 (treatment)
    print("\n## RQ3: Alignment vs. Correctness (G0 control vs G4 treatment)\n")
    print("Control = G0 (baseline).  Treatment = G4 (Skeptical Auditor).  Item-level: is_correct ~ vp_gap * treatment.\n")
    df = build_item_level_df(by_group)
    df_g0_g4 = df[df["group"].isin(["g0", "g4"])].copy()
    df_g0_g4["treatment"] = (df_g0_g4["group"] == "g4").astype(int)
    if df_g0_g4.empty or df_g0_g4["group"].nunique() < 2 or df_g0_g4["is_correct"].nunique() < 2:
        print("Need both G0 and G4 with valid item-level data.  Skipping RQ3.")
    else:
        model_rq3 = smf.logit("is_correct ~ vp_gap * treatment", data=df_g0_g4).fit(disp=0)
        n_items = len(df_g0_g4)
        n_g0 = (df_g0_g4["treatment"] == 0).sum()
        n_g4 = (df_g0_g4["treatment"] == 1).sum()
        print(f"N = {n_items} items (G0: {n_g0}, G4: {n_g4}).  Logistic: is_correct ~ vp_gap * treatment.\n")
        print("Coefficients (log-odds):")
        print(f"  vp_gap (effect in control G0):     {model_rq3.params['vp_gap']:.4f},  SE = {model_rq3.bse['vp_gap']:.4f},  p = {model_rq3.pvalues['vp_gap']:.4f}")
        print(f"  treatment (G4 vs G0 level):       {model_rq3.params['treatment']:.4f},  SE = {model_rq3.bse['treatment']:.4f},  p = {model_rq3.pvalues['treatment']:.4f}")
        print(f"  vp_gap:treatment (interaction):   {model_rq3.params['vp_gap:treatment']:.4f},  SE = {model_rq3.bse['vp_gap:treatment']:.4f},  p = {model_rq3.pvalues['vp_gap:treatment']:.4f}")
        inter_coef = model_rq3.params["vp_gap:treatment"]
        inter_p = model_rq3.pvalues["vp_gap:treatment"]
        if inter_p < 0.05 and inter_coef < 0:
            print("\n→ Significant negative interaction: the effect of alignment (lower |V−P|) on correctness is stronger under G4 than under G0.")
        elif inter_p < 0.05:
            print("\n→ Significant interaction: alignment → accuracy effect differs between G0 and G4.")
        else:
            print("\n→ Interaction not significant at α=0.05.")

    print("\n(2) Risk (G0 vs G4): Among items with V≥80, is_correct ~ vp_gap * treatment.\n")
    df_conf = df_g0_g4[df_g0_g4["V"] >= CER_THRESHOLD].copy() if not df_g0_g4.empty else pd.DataFrame()
    if len(df_conf) < 30 or df_conf["is_correct"].nunique() < 2 or df_conf["treatment"].nunique() < 2:
        print("Too few confident items or no variation.  Skipping.")
    else:
        model_conf = smf.logit("is_correct ~ vp_gap * treatment", data=df_conf).fit(disp=0)
        inter_c = model_conf.params.get("vp_gap:treatment", np.nan)
        inter_p_c = model_conf.pvalues.get("vp_gap:treatment", np.nan)
        print(f"N = {len(df_conf)} items with V≥80 (G0+G4).  vp_gap:treatment coef = {inter_c:.4f},  p = {inter_p_c:.4f}")
        if inter_p_c < 0.05 and inter_c < 0:
            print("→ Among confident items, alignment has a stronger effect on correctness under G4.")
        else:
            print("→ Interaction not significant at α=0.05.")
    print()


if __name__ == "__main__":
    main()
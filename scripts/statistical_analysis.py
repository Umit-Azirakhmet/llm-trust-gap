import os, json, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, f_oneway, ttest_ind, norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings

# Suppress warnings for clean academic output
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 0. CONFIGURATION & MAPPING
# ------------------------------------------------------------------------------

TARGET_DATASETS = ['medmcqa_1000', 'mmlu_professional_law_1000']
MODEL_SIZE = {'llama-3.1-8b': 8, 'gemma-3-4b': 4, 'mistral-small': 24, 'qwen-3-80b': 80}
GROUP_LABEL = {
    'g0': 'Control', 'g1': 'Evidence-First', 'g2': 'Counterfactual',
    'g3': 'Inoculation', 'g4': 'Persona', 'g5': 'Scoring-Rule',
    'g6': 'Anchoring', 'g8': 'GEPA-align', 'g9': 'GEPA-calib',
}

# Visualization Themes
DARK = '#0f1117'; PANEL = '#1c1f2e'; TEXT = '#e2e8f0'; GRID = '#2a2d3e'

# ------------------------------------------------------------------------------
# 1. CORE LOGIC: PARETO FRONTIER & PLOTTING
# ------------------------------------------------------------------------------

def compute_pareto_set(data_df, x_col='ece_V', y_col='rho'):
    """Identifies Pareto-optimal groups based on the non-dominance principle."""
    pts = data_df[[x_col, y_col, 'group']].values
    pareto_groups = []
    for i, p1 in enumerate(pts):
        is_dominated = False
        for j, p2 in enumerate(pts):
            if (p2[0] <= p1[0] and p2[1] >= p1[1]) and (p2[0] < p1[0] or p2[1] > p1[1]):
                is_dominated = True; break
        if not is_dominated: pareto_groups.append(p1[2])
    return sorted(pareto_groups)

def save_pareto_plot(data_df, title, filename, out_dir):
    """Generates and saves a professional Pareto Frontier plot."""
    os.makedirs(out_dir, exist_ok=True)
    pareto_indices = []
    points = data_df[['ece_V', 'rho']].values
    for i, p1 in enumerate(points):
        dominated = False
        for j, p2 in enumerate(points):
            if (p2[0] <= p1[0] and p2[1] >= p1[1]) and (p2[0] < p1[0] or p2[1] > p1[1]):
                dominated = True; break
        if not dominated: pareto_indices.append(i)
    
    pareto_points = data_df.iloc[pareto_indices].sort_values('ece_V')
    plt.figure(figsize=(10, 7), facecolor=DARK)
    ax = plt.gca(); ax.set_facecolor(PANEL)
    
    plt.scatter(data_df['ece_V'], data_df['rho'], color='#4f9cf9', alpha=0.4, s=100, label='Other Groups')
    plt.scatter(pareto_points['ece_V'], pareto_points['rho'], color='#ef4444', s=150, edgecolors=TEXT, label='Pareto Frontier', zorder=5)
    plt.plot(pareto_points['ece_V'], pareto_points['rho'], color='#ef4444', linestyle='--', alpha=0.6)

    for _, row in data_df.iterrows():
        plt.annotate(GROUP_LABEL.get(row['group'], row['group']), (row['ece_V'], row['rho']), 
                     textcoords="offset points", xytext=(0,10), ha='center', color=TEXT, fontsize=9)

    plt.title(title, color=TEXT, fontsize=14, fontweight='bold')
    plt.xlabel('Calibration Error (ECE_V) - [Lower Better]', color=TEXT)
    plt.ylabel('Alignment (Spearman Rho) - [Higher Better]', color=TEXT)
    plt.grid(color=GRID, linestyle='--', alpha=0.5); plt.tick_params(colors=TEXT)
    for spine in ax.spines.values(): spine.set_edgecolor(GRID)
    plt.savefig(os.path.join(out_dir, filename), facecolor=DARK, bbox_inches='tight'); plt.close()

# ------------------------------------------------------------------------------
# 2. MAIN ANALYSIS SUITE
# ------------------------------------------------------------------------------

def run_comprehensive_analysis(df, summary, pareto_out):
    # --- SECTION 1: PER-CASE ANALYSIS (8 TABLES & 8 PLOTS) ---
    print("\n" + "="*105)
    print(f"{'SECTION 1. PER-CASE STATISTICAL ANALYSIS (ANOVA & POST-HOC T-TESTS)':^105}")
    print("="*105)

    cases = sorted(df[['model', 'dataset']].drop_duplicates().values.tolist())
    for i, (model, dataset) in enumerate(cases, 1):
        sub_df = df[(df['model'] == model) & (df['dataset'] == dataset)].copy()
        sub_df['pv_prod'] = sub_df['P_norm'] * sub_df['V_norm']
        groups = {g: grp['pv_prod'].values for g, grp in sub_df.groupby('prompt_group') if len(grp) > 1}
        
        if 'g0' not in groups: continue
        F, p_anova = f_oneway(*groups.values())
        sig_a = '***' if p_anova < 0.001 else '**' if p_anova < 0.01 else '*' if p_anova < 0.05 else 'ns'
        
        print(f"\n[TABLE {i}] CASE: {model} | {dataset}")
        print(f"+{'-'*101}+")
        print(f"| {'Omnibus ANOVA':<20} | F-stat: {F:<15.4f} | p-value: {p_anova:<15.2e} | Sig: {sig_a:<8} |")
        print(f"+{'-'*13}+{'-'*25}+{'-'*12}+{'-'*12}+{'-'*15}+{'-'*15}+")
        print(f"| {'Group':<11} | {'Label':<23} | {'Δ Mean':>10} | {'t-stat':>10} | {'p-val':>13} | {'Sig':<13} |")
        print(f"|{'-'*13}|{'-'*25}|{'-'*12}|{'-'*12}|{'-'*15}|{'-'*15}|")
        
        g0_data = groups['g0']
        for g in sorted(groups.keys()):
            if g == 'g0': continue
            t, p_raw = ttest_ind(groups[g], g0_data, equal_var=False)
            delta = groups[g].mean() - g0_data.mean()
            sig_t = '***' if p_raw < 0.001 else '*' if p_raw < 0.05 else 'ns'
            print(f"| {g:<11} | {GROUP_LABEL.get(g,g):<23} | {delta:>10.4f} | {t:>10.2f} | {p_raw:>13.2e} | {sig_t:<13} |")
        
        case_summary = summary[(summary['model'] == model) & (summary['dataset'] == dataset)]
        pareto_set = compute_pareto_set(case_summary)
        print(f"+{'-'*13}+{'-'*25}+{'-'*12}+{'-'*12}+{'-'*15}+{'-'*15}+")
        print(f"| {'Pareto-Optimal Set (Best Balance):':<60} {', '.join(pareto_set):<38} |")
        print(f"+{'-'*101}+")
        
        save_pareto_plot(case_summary, f"Pareto Frontier: {model} | {dataset}", 
                         f"pareto_{model.replace('.','_')}_{dataset}.png", pareto_out)

    # --- SECTION 2: DETAILED POOLED MEDIATION ANALYSIS ---
    print("\n" + "="*105)
    print(f"{'SECTION 2. DETAILED POOLED MEDIATION ANALYSIS (Fixed Effects Controlled)':^105}")
    print("="*105)
    s = summary.copy().dropna(subset=['rho', 'accuracy'])
    s['group_int'] = s['group'].str.extract(r'(\d+)').astype(float)

    res_a = smf.ols("rho ~ group_int + C(model) + C(dataset)", data=s).fit()
    res_c = smf.ols("accuracy ~ group_int + C(model) + C(dataset)", data=s).fit()
    res_b_cp = smf.ols("accuracy ~ rho + group_int + C(model) + C(dataset)", data=s).fit()

    a, b = res_a.params['group_int'], res_b_cp.params['rho']
    z = (a*b) / np.sqrt(b**2 * res_a.bse['group_int']**2 + a**2 * res_b_cp.bse['rho']**2)
    p_sobel = 2 * (1 - norm.cdf(abs(z)))

    print(f"| Path a (X -> M) beta: {a:.4f} (p={res_a.pvalues['group_int']:.2e})")
    print(f"| Path b (M -> Y) beta: {b:.4f} (p={res_b_cp.pvalues['rho']:.2e})")
    print(f"| Path c (Total Effect) beta: {res_c.params['group_int']:.4f}")
    print(f"| Path c' (Direct Effect) beta: {res_b_cp.params['group_int']:.4f}")
    print(f"| Indirect Effect (a*b): {a*b:.4f}")
    print(f"| Sobel Test Result: z={z:.3f}, p-value={p_sobel:.4f}")

    # --- SECTION 3: SCALING LAWS & GLOBAL PARETO ---
    print("\n" + "="*105)
    print(f"{'SECTION 3. SCALING LAWS (MODERATION) & GLOBAL PARETO FRONTIER':^105}")
    print("="*105)
    s['size_log'] = np.log10(s['model_size'])
    num_models = s['model'].nunique()
    res_int = smf.ols("rho ~ size_log * group_int + C(dataset)", data=s).fit(cov_type='cluster', cov_kwds={'groups': s['model']}) if num_models > 1 else smf.ols("rho ~ size_log * group_int + C(dataset)", data=s).fit()
    print(f"[A] Interaction Model Summary:\n{res_int.summary().tables[1]}")

    global_avg = s.groupby('group')[['ece_V', 'rho']].mean().reset_index()
    global_pareto = compute_pareto_set(global_avg)
    print(f"\n[B] Global Pareto-Optimal Groups (Pooled): {', '.join(global_pareto)}")
    save_pareto_plot(global_avg, "Global Pareto Frontier (Pooled Across All Conditions)", "pareto_global.png", pareto_out)
    print(f"✅ All 9 Pareto plots saved to: {pareto_out}")

# ------------------------------------------------------------------------------
# 3. DATA LOAD & UTILS
# ------------------------------------------------------------------------------

def load_data(root):
    records = []
    files = glob.glob(os.path.join(root, '**', '*.json'), recursive=True)
    for f in files:
        try:
            with open(f) as j:
                data = json.load(j)
                for item in data:
                    v, p = item.get('V'), item.get('P')
                    if v is None or p is None: continue
                    meta = item.get('metadata', {})
                    if meta.get('dataset') not in TARGET_DATASETS: continue
                    records.append({'V_norm': float(v)/100, 'P_norm': float(p)/100,
                                    'correct': int(item.get('is_correct', 0)), 'model': meta.get('model'),
                                    'prompt_group': meta.get('prompt_group'), 'dataset': meta.get('dataset'),
                                    'model_size': MODEL_SIZE.get(meta.get('model'), 0)})
        except: continue
    return pd.DataFrame(records).dropna()

def build_summary(df):
    rows = []
    for (m, g, d), sub in df.groupby(['model','prompt_group','dataset']):
        if len(sub) < 5: continue
        rho, _ = spearmanr(sub['P_norm'], sub['V_norm'])
        acc = sub['correct'].mean()
        rows.append({'model': m, 'group': g, 'dataset': d, 'rho': rho, 'accuracy': acc, 
                     'ece_V': np.abs(sub['V_norm'].mean() - acc), 'model_size': MODEL_SIZE.get(m, 0)})
    return pd.DataFrame(rows)

def main():
    root_dir = 'outputs'
    out_dir = 'outputs/results/pareto'
    df = load_data(root_dir)
    if not df.empty:
        summary = build_summary(df)
        run_comprehensive_analysis(df, summary, out_dir)
    else: print("No matching data found.")

if __name__ == '__main__': main()
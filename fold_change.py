import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

# === Load Data ===
file_path = "processed_Species_data.csv"
df = pd.read_csv(file_path)

# === Raman Shift Axis ===
trimmed_shifts = df.select_dtypes(include=[np.number]).columns.astype(float)
df["Species"] = df["Species"].astype(str)

# === Define Groups ===
groups = ["SAME", "NP40", "SPICE"]
base_group = "NP40"

# === Target Raman Bands for Significance Marking ===
panel_targets = {
    "NP40": [497, 586, 732, 825, 873, 952, 1003, 1141, 1172, 1210, 1320, 1431, 1520, 1601]
}
target_lines = panel_targets.get("NP40", [])

# === Bootstrap Function ===
def bootstrap_fold_change(group1, group2, n_boot=1000):
    fc_samples = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        boot1 = group1.sample(frac=1, replace=True)
        boot2 = group2.sample(frac=1, replace=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            fc = np.log2((boot2.mean() + 1e-10) / (boot1.mean() + 1e-10))
        fc_samples.append(fc)
    fc_array = np.vstack(fc_samples)
    ci_lower = np.percentile(fc_array, 2.5, axis=0)
    ci_upper = np.percentile(fc_array, 97.5, axis=0)
    fc_mean = np.mean(fc_array, axis=0)
    return fc_mean, ci_lower, ci_upper

# === Kruskal-Wallis Across All Shifts ===
kruskal_pvals = []
for shift in trimmed_shifts:
    grouped = [df[df["Species"] == g][str(shift)].values for g in groups if g in df["Species"].unique()]
    stat, p = kruskal(*grouped)
    kruskal_pvals.append(p)
overall_kw_p = np.mean(kruskal_pvals)

# === Begin Plot ===
plt.figure(figsize=(7, 4))
base_df = df[df["Species"] == base_group].select_dtypes(include=[np.number])

# Plot fold change curves
for group in groups:
    if group == base_group or group not in df["Species"].unique():
        continue
    comparison_df = df[df["Species"] == group].select_dtypes(include=[np.number])
    fc_mean, ci_lower, ci_upper = bootstrap_fold_change(base_df, comparison_df, n_boot=1000)
    plt.plot(trimmed_shifts, fc_mean, label=f"{group} vs {base_group}")
    plt.fill_between(trimmed_shifts, ci_lower, ci_upper, alpha=0.2)

plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("Raman Shift (cm⁻¹)")
plt.ylabel("log₂(Fold Change)")

# === Add significance vertical lines using post hoc Dunn ===
ymin, ymax = plt.ylim()
for x in target_lines:
    idx = np.argmin(np.abs(trimmed_shifts - x))
    shift_val = trimmed_shifts[idx]
    
    # Prepare for post hoc
    df_post = df[df["Species"].isin(groups)][["Species", str(shift_val)]].copy()
    df_post.columns = ["group", "value"]
    
    try:
        posthoc = posthoc_dunn(df_post, val_col="value", group_col="group", p_adjust="bonferroni")

        # Determine significance
        sig_same_vs_np40 = posthoc.loc["SAME", "NP40"] < 0.05 if "SAME" in posthoc.index and "NP40" in posthoc.columns else False
        sig_spice_vs_np40 = posthoc.loc["SPICE", "NP40"] < 0.05 if "SPICE" in posthoc.index and "NP40" in posthoc.columns else False
        sig_same_vs_spice = posthoc.loc["SAME", "SPICE"] < 0.05 if "SAME" in posthoc.index and "SPICE" in posthoc.columns else False

        if (sig_same_vs_np40 or sig_spice_vs_np40) and sig_same_vs_spice:
            color = "purple"
        elif sig_same_vs_spice:
            color = "red"
        elif sig_same_vs_np40 or sig_spice_vs_np40:
            color = "blue"
        else:
            color = "gray"

    except Exception:
        color = "gray"

    plt.axvline(x=shift_val, color=color, linestyle='--', linewidth=1.0)
    plt.text(
        shift_val, ymax * 1.05, f"{int(shift_val)}", color=color, fontsize=8, rotation=90,
        ha='center', va='top',
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
    )

plt.ylim(top=ymax * 1.15)

# === Add Kruskal-Wallis annotation ===
plt.text(0.01, 0.01, f"Kruskal-Wallis p = {overall_kw_p:.2e}",
         transform=plt.gca().transAxes,
         fontsize=10, color='black', ha='left', va='bottom',
         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

plt.legend(fontsize=10, loc='lower right')
plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import scikit_posthocs as sp

# === Load Dataset ===
file_path = "processed_Species_data.csv"
df = pd.read_csv(file_path)

# === Extract Raman Intensities and Labels ===
X = df.select_dtypes(include=[np.number])
class_wanted = "Species"
y = df[class_wanted]

# === Parameters ===
target_shifts = [837, 784, 685, 920, 1384, 1430, 1512]
target_species = ["NP40", "SAME", "SPICE"]
colors = {
    "NP40": "blue",
    "SAME": "green",
    "SPICE": "red"
}

group_indices = {group: i for i, group in enumerate(target_species)}

# === Find Closest Matching Raman Shifts ===
available_shifts = np.array(X.columns, dtype=float)
closest_shifts = {
    t: available_shifts[np.argmin(np.abs(available_shifts - t))]
    for t in target_shifts
}

# === Plot Setup ===
n_shifts = len(closest_shifts)
cols = 3
rows = (n_shifts + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
axes = axes.flatten()

# === Helper for asterisk labels
def p_to_asterisks(p):
    if p < 0.0001:
        return "****"
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return None #instead of "non-sig" or "NS" to save graph space

# === Analysis and Plotting ===
for idx, (target, matched_shift) in enumerate(closest_shifts.items()):
    shift_str = str(matched_shift)
    ax = axes[idx]

    # Subset Data
    subset_df = df[[class_wanted, shift_str]].dropna()
    values = [subset_df[subset_df[class_wanted] == group][shift_str] for group in target_species]

    # Kruskal-Wallis Test
    stat, p_kw = kruskal(*values)
    print(f"{matched_shift:.2f} cm⁻¹ → Kruskal-Wallis H = {stat:.4f}, p = {p_kw:.4e}")

    # Dunn’s Post-Hoc Test
    posthoc = sp.posthoc_dunn(subset_df, val_col=shift_str, group_col=class_wanted, p_adjust="bonferroni")

    # Boxplot
    sns.boxplot(data=subset_df, x=class_wanted, y=shift_str, hue=class_wanted,
                palette=colors, legend=False, ax=ax)

    ax.set_title(f"{matched_shift:.2f} cm⁻¹")
    ax.set_ylabel("Intensity (arb. u.)")

    # Determine space for annotations
    y_max = subset_df[shift_str].max()
    y_min = subset_df[shift_str].min()
    y_range = y_max - y_min
    y_offset = y_range * 0.05
    annotation_y = y_max + y_offset
    ax.set_ylim(y_min, y_max + 6 * y_offset)

    # Add horizontal lines and asterisks for significant pairs
    pair_idx = 0
    pairs = [("NP40", "SAME"), ("NP40", "SPICE"), ("SAME", "SPICE")]

    for g1, g2 in pairs:
        p_val = posthoc.loc[g1, g2] if g1 in posthoc.index and g2 in posthoc.columns else posthoc.loc[g2, g1]
        asterisk = p_to_asterisks(p_val)
        if asterisk:
            x1, x2 = group_indices[g1], group_indices[g2]
            y = annotation_y + y_offset * pair_idx

            # Draw horizontal line
            ax.plot([x1, x1, x2, x2], [y - y_offset/4, y, y, y - y_offset/4], lw=1.5, c='black')
            ax.text((x1 + x2) / 2, y + y_offset * 0.1, asterisk, ha='center', va='bottom', fontsize=13, color='black')

            pair_idx += 1

# Hide unused axes
for k in range(idx + 1, len(axes)):
    axes[k].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.5, pad=2.0)
fig.suptitle("Kruskal-Wallis + Dunn's Post-Hoc Test", fontsize=16)
plt.show()
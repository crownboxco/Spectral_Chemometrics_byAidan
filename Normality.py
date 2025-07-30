import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest, probplot

# === Load Dataset ===
file_path = "processed_Species_data.csv"
df = pd.read_csv(file_path)

# === Extract Features and Labels ===
X = df.select_dtypes(include=[np.number])  # Raman intensities
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

# === Find closest matching Raman shifts ===
available_shifts = np.array(X.columns, dtype=float)
closest_shifts = {t: available_shifts[np.argmin(np.abs(available_shifts - t))] for t in target_shifts}

# === Setup Q-Q Plot Grid ===
n_shifts = len(target_shifts)
cols = 3
rows = (n_shifts + cols - 1) // cols  # Ceiling division for grid layout
fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
axes = axes.flatten()  # To index easily

print("=== D'Agostino and Pearson Normality Test ===\n")

for idx, (target, matched_shift) in enumerate(closest_shifts.items()):
    ax = axes[idx]
    print(f"--- Target: {target} cm⁻¹ → Closest Match: {matched_shift:.2f} cm⁻¹ ---")

    for species in target_species:
        group_data = df[df[class_wanted] == species][str(matched_shift)].dropna()

        if len(group_data) < 15:
            print(f"{species}: Not enough data for normality test.")
            continue

        # Normality test
        stat, p = normaltest(group_data)
        result = "Normal" if p > 0.05 else "Not Normal"
        print(f"{species}: K² = {stat:.4f}, p = {p:.4e} → {result}")

        # Q-Q Plot
        (osm, osr), (slope, intercept, _) = probplot(group_data, dist="norm")
        ax.plot(osm, osr, 'o', label=f"{species} (p={p:.2g})", color=colors[species])
        ax.plot(osm, slope * osm + intercept, '--', color=colors[species])

    ax.set_title(f"{matched_shift:.2f} cm⁻¹")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.legend()

# Turn off unused axes if any
for k in range(idx + 1, len(axes)):
    axes[k].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.5, pad=2.0)
fig.suptitle("Q-Q plots", fontsize=16)
plt.show()
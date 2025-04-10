# Change group name on lines 14-15.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from adjustText import adjust_text

# Load the preprocessed dataset
file_path = "processed_Raman_data.csv"
df_processed = pd.read_csv(file_path)

# Convert dataframe to numpy array for training
X = df_processed.drop(columns=['Species'])  # Features (Raman intensities)
y = df_processed['Species']  # Labels (species classification)

class_labels = y.unique()
class_1_name = class_labels[1]
class_2_name = class_labels[2]

# Split the data
X1 = X[y == class_1_name]
X2 = X[y == class_2_name]

# Convert Raman shifts to float
raman_shifts = X.columns.astype(float)

# Compute mean spectra
mean_1 = X1.mean().values
mean_2 = X2.mean().values

# Compute difference spectrum
diff_spectrum = mean_2 - mean_1

# Perform t-tests at each wavenumber
t_vals, p_vals = ttest_ind(X2, X1, axis=0, equal_var=False)

# Multiple testing correction (FDR or False Discovery Rate--Benjamini-Hochberg Correction)
reject, pvals_corrected, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
significant_mask = reject  # True where p < 0.05 (FDR corrected)

class_name_1 = y.unique()[1]
class_name_2 = y.unique()[2]

target_shifts = [732, 1003, 1450, 1509, 600, 1100, 1300, 1500, 900, 1050] # Change to your target shifts!
target_indices = [np.argmin(np.abs(raman_shifts - target)) for target in target_shifts]

plt.figure(figsize=(12, 5))
plt.plot(raman_shifts, diff_spectrum, color='black', label=f"Mean Difference ({class_name_2} - {class_name_1})")
plt.axhline(0, color='gray', linestyle='--')

# Highlight significant points
plt.scatter(raman_shifts[significant_mask], diff_spectrum[significant_mask],
            color='orange', label='p < 0.05 (FDR)', zorder=3)

# Highlight and annotate selected peaks (in blue)
texts = []
for idx in target_indices:
    shift = raman_shifts[idx]
    value = diff_spectrum[idx]
    is_significant = significant_mask[idx]
    
    color = 'red' if is_significant else 'blue'
    plt.scatter(shift, value, color=color, zorder=4)
    texts.append(
        plt.text(shift, value, f"{int(shift)}", color=color, fontsize=9)
    )

# Adjust label positions to avoid overlaps
adjust_text(texts, arrowprops=dict(arrowstyle="->", color='blue', lw=0.5))

plt.xlabel('Raman Shift (cm⁻¹)')
plt.ylabel('Relative Intensity Difference')
plt.title('Difference Spectrum Between Classes with Statistical Significance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
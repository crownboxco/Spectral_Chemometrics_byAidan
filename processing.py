# This imports all the necessary packages and libraries to run the code below it. DO NOT CHAGE THESE!!!
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from pybaselines import Baseline
from adjustText import adjust_text

# Load the dataset (class-labelled and background-subtracted, if applicable)
file_path = "Python_Test_Data.csv"
df = pd.read_csv(file_path)

# Extract Raman Shift values from column names (ignore metadata columns)
raman_shifts = np.array([float(col) for col in df.columns[6:]])  # Skip first 6 columns: For example, if Label is the first column and I have 5 classes, I skip the first 6
intensity_data = df.iloc[:, 6:].values  # Extract intensity values

# Trim to 450–1650 cm⁻¹ range
mask = (raman_shifts >= 450) & (raman_shifts <= 1651)
trimmed_shifts = raman_shifts[mask]
trimmed_intensity = intensity_data[:, mask]

# Adaptive Smoothness Least Squares (ASLS), p-value [0.001, 0.5] low = strong, high = weak
baseline_fitter = Baseline()
def baseline_correction(intensities, lam=1e5, p=0.01): #lamda ranges from 1 to 1e9, lower = more powerful baseline
    baseline, _ = baseline_fitter.asls(intensities, lam=lam, p=p)
    return intensities - baseline
baseline_corrected = np.array([baseline_correction(spec) for spec in trimmed_intensity])

# Apply Savitzky-Golay filter (2nd order, window size = # of data points to smooth at a time)
smoothed_intensity = savgol_filter(baseline_corrected, window_length=8, polyorder=1, axis=1) #if more smoothing is required, edit window length

# Area normalization per spectrum
def area_normalize_spectra(intensities):
    area = np.sum(intensities, axis=1, keepdims=True)
    # Avoid division by zero
    area[area == 0] = 1e-10 #to prevent dividing by zero
    normalized = intensities / area
    return normalized

df_normalized = area_normalize_spectra(smoothed_intensity)

# Convert back to DataFrame
df_processed = pd.DataFrame(df_normalized, columns=trimmed_shifts)
df_processed.insert(0, "Species", df["Species"].values)  # Retain IDs

# Save processed data for machine learning scripts to import later
processed_file_path = "processed_Raman_data.csv" # Change the name of this to whatever project you're doing
df_processed.to_csv(processed_file_path, index=False)
print(f"Processed data saved to {processed_file_path}")

##### PLOTTTING ########

# Select a random spectrum (or specify an index)
spectrum_idx = 20  # Change this to visualize a different spectrum

# Extract data for plotting
original_shift = raman_shifts
original_intensity = intensity_data[spectrum_idx]

trimmed_shift = trimmed_shifts
trimmed_original = trimmed_intensity[spectrum_idx]
baseline = trimmed_original - baseline_corrected[spectrum_idx]
processed_intensity = smoothed_intensity[spectrum_idx]

# Do Min-Max normalization
def normalize_single_spectrum(spectrum_idx):
    return (spectrum_idx - np.min(spectrum_idx)) / (np.max(spectrum_idx) - np.min(spectrum_idx))

# Normalize each spectrum independently
normalized_original = normalize_single_spectrum(original_intensity)
normalized_baseline = normalize_single_spectrum(baseline)
normalized_processed = normalize_single_spectrum(processed_intensity)
plt.figure(figsize=(8, 5))

original_up = 0.9
baseline_up = 1.4

# Plot normalized original spectrum
plt.plot(original_shift, normalized_original + original_up, linestyle="--", color="blue", alpha=0.7, label="Original Spectrum")

# # Plot normalized baseline
plt.plot(trimmed_shift, normalized_baseline + baseline_up, linestyle=":", color="red", label="Estimated Baseline")

# Plot normalized preprocessed spectrum
plt.plot(trimmed_shift, normalized_processed, linestyle="-", color="green", label="Processed Spectrum")

spaced_lines = np.arange(450, 1650, 50)  # from 450 to 1650 every 50 cm⁻¹
for x in spaced_lines:
    plt.axvline(x=x, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

# Labels and legend
plt.xlabel("Raman Shift (cm⁻¹)")
plt.ylabel("Offset Intensity (arb.u.)")
plt.title(f"Raman Spectrum Processing (Sample {spectrum_idx})")
plt.legend()

#===== PLOTTING GROUPS IN CLASSES ======
selected_classes = ["SAME", "SPICE", "NP40"] #change to your groups
shifts = {"SPICE": 0.0, "NP40": 0.0} #change to your groups and offsets, if desired, if no, enter "0.0"

class_colors = {
    "SAME": "green", # Change to your groups and colors (say color or find hex code you like)
    "SPICE": "red",
    "NP40": "blue"
}

species_groups = df_processed.groupby("Species").mean()
species_std = df_processed.groupby("Species").std()
group_counts = df_processed.groupby("Species").count().iloc[:, 0]
combined_sem = species_std.div(np.sqrt(group_counts), axis=0)

plt.figure(figsize=(8, 5))

# Loop through each species and plot mean ± SE
for mix in selected_classes:
    if mix in species_groups.index:
        mean_spectrum = species_groups.loc[mix].values
        sem_spectrum = combined_sem.loc[mix].values
        shift_value = shifts.get(mix, 0)
        shifted_mean = mean_spectrum + shift_value
        color = class_colors.get(mix, "gray")

        # Plot mean spectrum
        plt.plot(trimmed_shifts, shifted_mean, label=f"{mix} (Mean)", color=color)

        # Plot shaded standard deviation
        plt.fill_between(trimmed_shifts, shifted_mean - sem_spectrum,
                         shifted_mean + sem_spectrum, alpha=0.2, color=color)

        # Add vertical lines and labels for specific points in NP40
        if mix == "NP40": #change to tallest class
            target_lines = [730, 1003, 1172, 1449, 1603]  # Example target Raman shifts
            ymax = plt.ylim()[1]  # get the current top y-limit
            for x in target_lines:
                plt.axvline(x=x, color='blue', linestyle='--', linewidth=1.5)
                plt.text(
                    x, ymax * 1.2, f"{x}",
                    fontsize=10, color='blue', rotation=90,
                    ha='center', va='top',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
                )
                plt.ylim(top=ymax * 1.22)
## Optional: global vertical lines, if you still want them
# for x in spaced_lines:
#     plt.axvline(x=x, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

# Plot settings
plt.xlabel("Raman Shift (cm⁻¹)")
plt.ylabel("Relative Intensity (arb. units)")
plt.title("")
plt.legend()
plt.tight_layout()
plt.show()
# Change group name on lines 14-15.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend, hilbert
import matplotlib.gridspec as gridspec

# Load the preprocessed dataset
file_path = "processed_Raman_data.csv"
df_processed = pd.read_csv(file_path)

# Convert dataframe to numpy array for training
X = df_processed.drop(columns=['Species'])  # Features (Raman intensities)
y = df_processed['Species']  # Labels (species classification)

# Convert labels to numeric and separate classes
df_class_0 = X[y == y.unique()[0]]
df_class_1 = X[y == y.unique()[1]]

# Get Raman shifts (assumed to be in order)
raman_shifts = X.columns.astype(float)

# Stack both classes together
X_all = pd.concat([df_class_0, df_class_1], axis=0)
X_all = detrend(X_all, axis=0)  # Remove baseline (optional)

# Mean center the spectra (important for 2D-COS)
X_mean_centered = X_all - np.mean(X_all, axis=0)

# --- Synchronous Correlation Matrix ---
sync_matrix = np.dot(X_mean_centered.T, X_mean_centered) / X_mean_centered.shape[0]

# --- Asynchronous Correlation Matrix ---
hilbert_transformed = hilbert(X_mean_centered, axis=0)
async_matrix = np.imag(np.dot(X_mean_centered.T, hilbert_transformed)) / X_mean_centered.shape[0]

# --- Mean spectra per class ---
mean_0 = np.mean(df_class_0.values, axis=0)
mean_1 = np.mean(df_class_1.values, axis=0)

class_name_0 = y.unique()[0]
class_name_1 = y.unique()[1]

# --- Plotting ---
def plot_2d_contour(matrix, title, raman_shifts, top_mean, side_mean, top_label=class_name_1, side_label=class_name_0):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 6], height_ratios=[1, 6],
                           wspace=0.05, hspace=0.05)

    # Main contour plot
    ax_matrix = fig.add_subplot(gs[1, 1])
    levels = np.linspace(-np.max(np.abs(matrix)), np.max(np.abs(matrix)), 20)
    contour = ax_matrix.contourf(raman_shifts, raman_shifts, matrix, levels=levels, cmap='coolwarm')
    ax_matrix.set_xlabel("Raman Shift (1/cm)", fontsize=12)
    ax_matrix.xaxis.set_label_position("bottom")
    ax_matrix.yaxis.set_label_position("right")
    raman_shifts = np.array(sorted(raman_shifts))
    ax_matrix.tick_params(
        left=False,      # Hide left ticks
        labelleft=False, # Hide left labels
        right=True,
        labelright=True,
        top=False,
        labeltop=False,
        bottom=True,
        labelbottom=True
    )
    # Add diagonal
    ax_matrix.plot(raman_shifts, raman_shifts, 'k:', linewidth=0.75)
    # Top mean spectrum
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_matrix)
    ax_top.plot(raman_shifts, top_mean, color='green', linewidth=1, label=top_label)
    ax_top.legend(loc='upper right', frameon=False)
    ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_top.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax_top.set_title(title)

    # Left mean spectrum
    ax_side = fig.add_subplot(gs[1, 0], sharey=ax_matrix)
    ax_side.plot(side_mean, raman_shifts, color='red', linewidth=1, label=side_label)
    ax_side.set_xticks([])
    ax_side.invert_xaxis()
    ax_side.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_side.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax_side.legend(loc='lower left', frameon=False)
    plt.show()

# Call the plotting function
plot_2d_contour(async_matrix, "Asynchronous 2D Raman Correlation Map", raman_shifts, top_mean=mean_1, side_mean=mean_0)
plot_2d_contour(sync_matrix, "Synchronous 2D Raman Correlation Map", raman_shifts, top_mean=mean_1, side_mean=mean_0)


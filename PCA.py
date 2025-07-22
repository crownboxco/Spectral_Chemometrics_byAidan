import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
from scipy.stats import chi2

# === Load Dataset ===
file_path = "processed_Species_data.csv"
df = pd.read_csv(file_path)

# === Extract Features and Labels ===
X = df.select_dtypes(include=[np.number])  # Raman intensities
y = df["Species"]

# === Standardize Spectra ===
X_scaled = StandardScaler().fit_transform(X)

# === Run PCA ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# === Create DataFrame ===
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Species"] = y

# === Color map for fixed colors ===
color_map = {
    "SAME": "green",
    "SPICE": "red",
    "NP40": "blue"
}

# === Ellipse Function ===
def plot_confidence_ellipse(x, y, ax, color, alpha=0.2):
    if len(x) < 2:
        return
    cov = np.cov(x, y)
    if np.linalg.det(cov) == 0:
        return
    mean_x, mean_y = np.mean(x), np.mean(y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)

    scale = np.sqrt(chi2.ppf(0.95, df=2))
    ell_width, ell_height = 2 * scale * lambda_
    angle = np.degrees(np.arctan2(*v[:, 0][::-1]))

    ellipse = Ellipse(
        (mean_x, mean_y),
        width=ell_width,
        height=ell_height,
        angle=angle,
        edgecolor=color,
        facecolor=color,
        lw=1,
        alpha=alpha
    )
    ax.add_patch(ellipse)

# === Plotting ===
plt.figure(figsize=(7, 5))
ax = plt.gca()

groups = pca_df["Species"].unique()
for group in groups:
    subset = pca_df[pca_df["Species"] == group]
    color = color_map.get(group, "gray")

    plt.scatter(subset["PC1"], subset["PC2"], label=group, color=color, edgecolor='k', s=50)
    plot_confidence_ellipse(subset["PC1"], subset["PC2"], ax, color=color)

# Labeling
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Plot of Spectral Embeddings")
plt.legend(title="Species", loc="upper right")

plt.tight_layout()
plt.show()
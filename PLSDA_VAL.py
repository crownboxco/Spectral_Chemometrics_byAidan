# Change group name on lines 19-20.
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold
import joblib

# Load the preprocessed dataset
file_path = "processed_Raman_data.csv"
df_processed = pd.read_csv(file_path)

# Convert dataframe to numpy array for training
X = df_processed.drop(columns=['Species'])  # Features (Raman intensities)
y = df_processed['Species']  # Labels (species classification)

# Encode class labels (e.g., from strings to 0,1,2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-hot encode for PLSRegression target
onehot_encoder = OneHotEncoder(sparse_output=False)
y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))

# Train/test split
X_train, X_test, y_train_enc, y_test_enc = train_test_split(X, y_onehot, stratify=y_encoded, test_size=0.3, random_state=42)
y_train_labels = np.argmax(y_train_enc, axis=1)
y_test_labels = np.argmax(y_test_enc, axis=1)

# Cross-validation to find best n_components
best_n_components = 1
best_score = 0
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for n in range(2, 10):
    fold_scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train_enc[train_idx], y_train_enc[val_idx]
        y_val_labels = np.argmax(y_fold_val, axis=1)

        pls = PLSRegression(n_components=n)
        pls.fit(X_fold_train, y_fold_train)
        y_val_pred = pls.predict(X_fold_val)
        y_val_pred_class = np.argmax(y_val_pred, axis=1)

        acc = accuracy_score(y_val_labels, y_val_pred_class)
        fold_scores.append(acc)

    mean_score = np.mean(fold_scores)
    if mean_score > best_score:
        best_score = mean_score
        best_n_components = n

print(f"Best PLS-DA n_components: {best_n_components}")

# Train best model on full training set
best_pls = PLSRegression(n_components=best_n_components)
best_pls.fit(X_train, y_train_enc)
y_pred_test = best_pls.predict(X_test)
y_pred_labels = np.argmax(y_pred_test, axis=1)

# Save trained model to a .pth file (even though .pkl is more common)
model_path = "best_pls_model.pth"
joblib.dump(best_pls, model_path)

# Also save the label encoder to use during prediction
encoder_path = "label_encoder_pls.pkl"
joblib.dump(label_encoder, encoder_path)

# Evaluate
final_acc = accuracy_score(y_test_labels, y_pred_labels)
print(f"Optimized PLS-DA Accuracy: {final_acc:.4f}")

print(classification_report(y_test_labels, y_pred_labels, target_names=label_encoder.classes_, digits = 4, zero_division=0))

# Create the confusion matrix
cm_pls = confusion_matrix(y_test_labels, y_pred_labels)
cm_normalized = cm_pls.astype('float') / cm_pls.sum(axis=1)[:, np.newaxis]

# Get original class names (e.g., ["Male", "Female", "Unknown"])
class_names = label_encoder.classes_

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=cm_pls, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            vmin=0, vmax=1)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title("PLS-DA Confusion Matrix")

# === Plot PLS-DA Feature Importance Across Raman Shifts with Annotations ===
coef = best_pls.coef_
if coef.shape[1] == X.shape[1]:
    coef = coef.T  # shape becomes (959, 3)

# Compute importance as sum of squared coefficients across classes
pls_importance = np.sum(np.square(coef), axis=1)  # shape: (n_features,)

# Extract Raman shift values from column names
raman_shifts = np.array([float(shift) for shift in X.columns])

# Sort by shift
sorted_idx = np.argsort(raman_shifts)
raman_shifts = raman_shifts[sorted_idx]
pls_importance = pls_importance[sorted_idx]

# Plot
plt.figure(figsize=(10, 4))
plt.plot(raman_shifts, pls_importance, color='purple', linewidth=1.5)
plt.xlabel("Raman Shift (cm⁻¹)")
plt.ylabel("Variable Importance")
plt.title("PLS-DA Variable Importance Plot")
plt.grid(True, linestyle='--', alpha=0.5)

# Add peak annotations
target_lines = [730, 1003, 1172, 1449, 1603]
ymax = pls_importance.max()
for x in target_lines:
    plt.axvline(x=x, color='blue', linestyle='--', linewidth=1.5)
    plt.text(
        x, ymax * 1.2, f"{x}",
        fontsize=10, color='blue', rotation=90,
        ha='center', va='top',
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
    )

plt.ylim(top=ymax * 1.22)
plt.tight_layout()
plt.show()
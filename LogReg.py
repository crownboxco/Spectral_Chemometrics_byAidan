# Change group name on lines 19-20.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Load the preprocessed dataset
file_path = "processed_Raman_data.csv"
df_processed = pd.read_csv(file_path)

# Convert dataframe to numpy array for training
X = df_processed.drop(columns=['Species'])  # Features (Raman intensities)
y = df_processed['Species']  # Labels (species classification)

# Create a LabelEncoder instance
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.3, random_state=42) # random-state is a fixed seed (any model in or outside script using 42 will train and test the same way)

logreg_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [100, 200, 500, 1000]
}

logreg = LogisticRegression()

logreg_grid_search = GridSearchCV(logreg, logreg_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
logreg_grid_search.fit(X_train, y_train)

best_logreg_params = logreg_grid_search.best_params_
print("Best Logistic Regression Parameters:", best_logreg_params)

# Train best model
best_logreg = LogisticRegression(**best_logreg_params)
best_logreg.fit(X_train, y_train)
y_pred_logreg_best = best_logreg.predict(X_test)
logreg_best_accuracy = accuracy_score(y_test, y_pred_logreg_best)
print(f"Optimized Logistic Regression Accuracy: {logreg_best_accuracy:.4f}")

print(classification_report(y_test, y_pred_logreg_best, target_names=label_encoder.classes_, digits = 4))

# Generate Confusion Matrices
cm_logreg = confusion_matrix(y_test, y_pred_logreg_best)
cm_normalized_logreg = cm_logreg.astype('float') / cm_logreg.sum(axis=1)[:, np.newaxis]
class_names = label_encoder.classes_

plt.figure(figsize=(8, 6))
sns.heatmap((cm_normalized_logreg), annot=cm_logreg, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, 
            vmin=0, vmax=1)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# === Plot Logistic Regression Feature Importance Across Raman Shifts ===
# For multi-class, best_logreg.coef_ has shape (n_classes, n_features)
# We'll take the L2-norm of the coefficients across all classes for each feature
coef = best_logreg.coef_
logreg_importance = np.linalg.norm(coef, axis=0)  # shape: (n_features,)

# Raman shifts from column names
raman_shifts = np.array([float(shift) for shift in X.columns])
assert len(raman_shifts) == len(logreg_importance), "Mismatch between features and importances!"

# Sort by shift
sorted_idx = np.argsort(raman_shifts)
raman_shifts = raman_shifts[sorted_idx]
logreg_importance = logreg_importance[sorted_idx]

# Plot
plt.figure(figsize=(10, 4))
plt.plot(raman_shifts, logreg_importance, color='teal', linewidth=1.5)
plt.xlabel("Raman Shift (cm⁻¹)")
plt.ylabel("Variable Importance")
plt.title("Logistic Regression Variable Importance Plot")
plt.grid(True, linestyle='--', alpha=0.5)

# Annotate known Raman shifts
target_lines = [730, 1003, 1172, 1449, 1603]
ymax = logreg_importance.max()
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
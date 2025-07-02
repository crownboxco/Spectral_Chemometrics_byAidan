# Change group name on lines 19-20.

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

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

xgb_param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [1, 3, 6, 10], 
    'n_estimators': [50, 100, 200, 500]
}

xgb_clf = xgb.XGBClassifier(objective='multi:softmax', num_class=len(set(y_train)), eval_metric='mlogloss', random_state=42)

xgb_grid_search = GridSearchCV(xgb_clf, xgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)

best_xgb_params = xgb_grid_search.best_params_
print("Best XGBoost Parameters:", best_xgb_params)

# Train best model
best_xgb_clf = xgb.XGBClassifier(**best_xgb_params, random_state=42)
best_xgb_clf.fit(X_train, y_train)
y_pred_xgb_best = best_xgb_clf.predict(X_test)
xgb_best_accuracy = accuracy_score(y_test, y_pred_xgb_best)
print(f"Optimized XGBoost Accuracy: {xgb_best_accuracy:.4f}")

print(classification_report(y_test, y_pred_xgb_best, target_names=label_encoder.classes_, digits = 4))

cm_xgb = confusion_matrix(y_test, y_pred_xgb_best)
cm_normalized_xgb = cm_xgb.astype('float') / cm_xgb.sum(axis=1)[:, np.newaxis]
class_names = label_encoder.classes_

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
ax = sns.heatmap((cm_normalized_xgb), annot=cm_xgb, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, 
            vmin=0, vmax=1)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title("XGBoost Confusion Matrix")
plt.show()

# === Plot Feature Importance Across Raman Shifts with Annotations (XGBoost) ===
importances = best_xgb_clf.feature_importances_

# Extract Raman shift values from column names
raman_shifts = np.array([float(shift) for shift in X.columns])

# Sort by shift (in case columns are out of order)
sorted_idx = np.argsort(raman_shifts)
raman_shifts = raman_shifts[sorted_idx]
importances = importances[sorted_idx]

# Begin plot
plt.figure(figsize=(10, 4))
plt.plot(raman_shifts, importances, color='darkgreen', linewidth=1.5)
plt.xlabel("Raman Shift (cm⁻¹)")
plt.ylabel("Variable Importance")
plt.title("XGBoost Variable Importance Plot")
plt.grid(True, linestyle='--', alpha=0.5)

# Add vertical lines and labels at specific Raman bands
target_lines = [730, 1003, 1172, 1449, 1603]
ymax = importances.max()
for x in target_lines:
    plt.axvline(x=x, color='blue', linestyle='--', linewidth=1.5)
    plt.text(
        x, ymax * 1.2, f"{x}",
        fontsize=10, color='blue', rotation=90,
        ha='center', va='top',
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
    )

# Adjust y-axis to accommodate label height
plt.ylim(top=ymax * 1.22)
plt.tight_layout()
plt.show()
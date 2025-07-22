import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
import joblib

# Load the preprocessed dataset
file_path = "processed_Species_data.csv"
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

# Compute sample weights for training data
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Train best model
best_xgb_clf = xgb.XGBClassifier(**best_xgb_params, random_state=42)
best_xgb_clf.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_xgb_best = best_xgb_clf.predict(X_test)
xgb_best_accuracy = accuracy_score(y_test, y_pred_xgb_best)
print(f"Optimized XGBoost Accuracy: {xgb_best_accuracy:.4f}")

print(classification_report(y_test, y_pred_xgb_best, target_names=label_encoder.classes_, digits = 4))

cm_xgb = confusion_matrix(y_test, y_pred_xgb_best)
cm_normalized_xgb = cm_xgb.astype('float') / cm_xgb.sum(axis=1)[:, np.newaxis]
class_names = label_encoder.classes_

# Save trained model to a .pth file (even though .pkl is more common)
model_path = "best_xgb_model.pth"
joblib.dump(best_xgb_clf, model_path)

# Also save the label encoder to use during prediction
encoder_path = "label_encoder_xgb.pkl"
joblib.dump(label_encoder, encoder_path)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
ax = sns.heatmap((cm_normalized_xgb), annot=cm_xgb, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, 
            vmin=0, vmax=1)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title("XGBDA Confusion Matrix")

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
plt.title("XGBDA Variable Importance Plot")
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

# === Multiclass ROC Curve for XGBoost ===
# Binarize test labels
y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))
n_classes = y_test_bin.shape[1]

# Get class probabilities
y_score = best_xgb_clf.predict_proba(X_test)

# Define optional color map (manually or automatically)
custom_colors = {
    "NP40": "red",
    "SAME": "green",
    "SPICE": "blue"
}

# Get class name strings for plotting
class_names_str = label_encoder.inverse_transform(np.arange(n_classes))

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure(figsize=(8, 6))

for i in range(n_classes):
    class_name = class_names_str[i]
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

    color = custom_colors.get(class_name, None)  # fallback = default
    plt.plot(fpr[i], tpr[i], lw=2, label=f"{class_name} (AUC = {roc_auc[i]:.2f})", color=color)

# Plot baseline
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBDA Multiclass ROC Curve')
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
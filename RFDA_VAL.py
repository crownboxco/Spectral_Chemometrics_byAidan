import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import GridSearchCV
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
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify= y, test_size=0.3, random_state=42) # random-state is a fixed seed (any model in or outside script using 42 will train and test the same way)

# # Define parameter grid for Random Forest
# rf_param_grid = {
#     'n_estimators': [50, 100, 200, 500],  # Number of trees
#     'max_depth': [1, 3, 5, 10, 20],  # Maximum depth of trees
#     'min_samples_split': [2, 5, 10]  # Minimum samples to split
# }
# Define parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50],  # Number of trees
    'max_depth': [1, 3],  # Maximum depth of trees
    'min_samples_split': [2]  # Minimum samples to split
}

# Initialize Random Forest model
rf_clf = RandomForestClassifier()

# Grid search with cross-validation
rf_grid_search = GridSearchCV(rf_clf, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

# Get best parameters
best_rf_params = rf_grid_search.best_params_
print("Best Random Forest Parameters:", best_rf_params)

# Train best model
best_rf_clf = RandomForestClassifier(**best_rf_params, class_weight='balanced', random_state=42)
best_rf_clf.fit(X_train, y_train)
y_pred_rf_best = best_rf_clf.predict(X_test)
rf_best_accuracy = accuracy_score(y_test, y_pred_rf_best)
print(f"Optimized Random Forest Accuracy: {rf_best_accuracy:.4f}")

# Save trained model to a .pth file (even though .pkl is more common)
model_path = "best_rf_model.pth"
joblib.dump(best_rf_clf, model_path)

# Also save the label encoder to use during prediction
encoder_path = "label_encoder_rf.pkl"
joblib.dump(label_encoder, encoder_path)

print(classification_report(y_test, y_pred_rf_best, target_names=label_encoder.classes_, digits = 4))
# Plot Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf_best)
cm_normalized_rf = cm_rf.astype('float') / cm_rf.sum(axis=1)[:, np.newaxis]
class_names = label_encoder.classes_

plt.figure(figsize=(8, 6))
sns.heatmap((cm_normalized_rf), annot=cm_rf, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, 
            vmin=0, vmax=1)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title("RFDA Confusion Matrix")
plt.show()

# === Plot Feature Importance Across Raman Shifts ===
importances = best_rf_clf.feature_importances_

# Extract Raman shift values from column names
raman_shifts = np.array([float(shift) for shift in X.columns])

# Sort by shift (optional, if not already in order)
sorted_idx = np.argsort(raman_shifts)
raman_shifts = raman_shifts[sorted_idx]
importances = importances[sorted_idx]

# Begin plot
plt.figure(figsize=(10, 4))
plt.plot(raman_shifts, importances, color='darkorange', linewidth=1.5)
plt.xlabel("Raman Shift (cm⁻¹)")
plt.ylabel("Variable Importance")
plt.title("RFDA Variable Importance Plot")
plt.grid(True, linestyle='--', alpha=0.5)

# Add vertical lines and labels at key Raman shifts
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

# Adjust y-axis limit to make room for the labels
plt.ylim(top=ymax * 1.22)
plt.tight_layout()

# === ROC Curve for Multiclass Random Forest ===
# Binarize test labels
y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))
n_classes = y_test_bin.shape[1]

# Predict class probabilities
y_score = best_rf_clf.predict_proba(X_test)

# Define manual colors (optional)
custom_colors = {
    "NP40": "red",
    "SAME": "green",
    "SPICE": "blue"
}

# Use inverse transform to get class name strings
class_names_str = label_encoder.inverse_transform(np.arange(n_classes))

# Compute ROC and AUC per class
fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure(figsize=(8, 6))

for i in range(n_classes):
    class_name = class_names_str[i]
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

    color = custom_colors.get(class_name, None)
    plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_name} (AUC = {roc_auc[i]:.2f})', color=color)

# Plot chance line
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RFDA Multiclass ROC Curve')
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
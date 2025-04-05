# Change group name on lines 19-20.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
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
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify= y, test_size=0.3, random_state=42) # random-state is a fixed seed (any model in or outside script using 42 will train and test the same way)

# Define parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200, 500],  # Number of trees
    'max_depth': [None, 1, 5, 10, 20],  # Maximum depth of trees
    'min_samples_split': [2, 5, 10]  # Minimum samples to split
}

# Initialize Random Forest model
rf_clf = RandomForestClassifier(random_state=42)

# Grid search with cross-validation
rf_grid_search = GridSearchCV(rf_clf, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

# Get best parameters
best_rf_params = rf_grid_search.best_params_
print("Best Random Forest Parameters:", best_rf_params)

# Train best model
best_rf_clf = RandomForestClassifier(**best_rf_params, random_state=42)
best_rf_clf.fit(X_train, y_train)
y_pred_rf_best = best_rf_clf.predict(X_test)
rf_best_accuracy = accuracy_score(y_test, y_pred_rf_best)
print(f"Optimized Random Forest Accuracy: {rf_best_accuracy:.4f}")

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
plt.title("Random Forest Confusion Matrix")
plt.show()
# This script is for testing replicate data on the model built from XGBDA_VAL.py
# MAKE SURE YOUR EXTERNAL DATA HAS GONE THROUGH processing.py before analysis!
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# === Load external Raman test data ===
external_file_path = "external_data.csv"  # Replace with your actual external file
df_external = pd.read_csv(external_file_path)
df_external.columns = [
    float(col) if col.replace('.', '', 1).isdigit() else col
    for col in df_external.columns
]

# Extract features (assumes same columns as training data)
X_external = df_external.select_dtypes(include=[np.number])  # Ignore if Species not present

# === Load trained model and label encoder ===
model_path = "best_xgb_model.pth"
encoder_path = "label_encoder_xgb.pkl"

xgb_model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

# === Make predictions ===
y_pred = xgb_model.predict(X_external)
predicted_labels = label_encoder.inverse_transform(y_pred)

# === Output predictions ===
df_external['Predicted_Label'] = predicted_labels

# === Optional: Evaluate if true labels are available ===
if 'Species' in df_external.columns:
    y_true = label_encoder.transform(df_external['Species'])
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nExternal Data Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, digits=4, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_, vmin=0, vmax=1)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (External Data)")
    plt.tight_layout()
    plt.show()
else:
    print("\nNo true labels found in external dataset. Only predictions were saved.")
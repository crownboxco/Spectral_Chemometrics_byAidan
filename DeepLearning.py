# Change group name on lines 21-22.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

# Convert datatypes to tensor
X_train_torch = torch.from_numpy(X_train.to_numpy()).float()
Y_train_torch = torch.from_numpy(y_train).long()
X_test_torch = torch.from_numpy(X_test.to_numpy()).float()
Y_test_torch = torch.from_numpy(y_test).long()

# Define the model-- DL ANNs contain 3 or more hidden layers
class DNNClassifier(nn.Module):
    def __init__(self):
        super(DNNClassifier, self).__init__()
        self.layer1 = nn.Linear(X_train.shape[1], 1024) # input layer
        self.bn1 = nn.BatchNorm1d(1024)
        self.layer2 = nn.Linear(1024, 512) # first hidden layer
        self.bn2 = nn.BatchNorm1d(512)
        self.layer3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.layer4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.layer5 = nn.Linear(128, 64) 
        self.bn5 = nn.BatchNorm1d(64)
        num_classes = len(np.unique(y_train))
        self.layer6 = nn.Linear(64, num_classes) # This is ouput layer, not considered hidden
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)  # Prevents overfitting
    
    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.layer3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.layer4(x)))
        x = self.dropout(x)
        x = self.relu(self.bn5(self.layer5(x)))
        x = self.layer6(x) # sigmoid activation
        return x

# Instantiate the model
model = DNNClassifier()
criterion = nn.CrossEntropyLoss()  # Cross Entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001) #faster learning rate (0.001) was better (0.0005)

# Training loop
for epoch in range(120):
    optimizer.zero_grad()
    predictions = model(X_train_torch)
    loss = criterion(predictions, Y_train_torch)
    loss.backward()
    optimizer.step()
    if epoch % 30 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluate accuracy
with torch.no_grad():
    logits = model(X_test_torch)
    test_preds = torch.argmax(logits, dim=1)  # Get class predictions
    accuracy_DNNs = accuracy_score(Y_test_torch.cpu().numpy(), test_preds.cpu().numpy())
    print(f"Deep Learning Accuracy: {accuracy_DNNs:.4f}")

# Plot Confusion Matrix for DL ANNs
cm_DNNs = confusion_matrix(y_test, test_preds)
cm_normalized_DNNs = cm_DNNs.astype('float') / cm_DNNs.sum(axis=1)[:, np.newaxis]
class_names = label_encoder.classes_

plt.figure(figsize=(8, 6))
sns.heatmap((cm_normalized_DNNs), annot=cm_DNNs, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            vmin=0, vmax=1)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("DNN Confusion Matrix")
plt.show()

# Print Classification Reports
print("DNN Classification Report:\n", classification_report(y_test, test_preds, digits=4))
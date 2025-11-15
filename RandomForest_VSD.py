# Import libraries
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Function to extract features from image
def image_to_features(image_path, size=(64, 64)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size)  # Resize to fixed size
    img_array = np.array(img)
    features = img_array.flatten()  # Flatten to 1D array
    return features

# Load data
sehat_paths = [os.path.join('Sehat', f) for f in os.listdir('Sehat') if f.endswith('.jpg')]
vsd_paths = [os.path.join('VSD', f) for f in os.listdir('VSD') if f.endswith('.jpg')]
all_paths = sehat_paths + vsd_paths
labels = [0] * len(sehat_paths) + [1] * len(vsd_paths)

# Extract features
features = []
for path in all_paths:
    feat = image_to_features(path)
    features.append(feat)
features = np.array(features)
labels = np.array(labels)

# Train-val split
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Predict probabilities for ROC-AUC
y_pred_proba = model.predict_proba(X_val)[:, 1]

# Evaluate
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:')
print(classification_report(y_val, y_pred, target_names=['Sehat', 'VSD']))

# ROC-AUC
roc_auc = roc_auc_score(y_val, y_pred_proba)
print(f'ROC-AUC: {roc_auc:.2f}')

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Sehat', 'VSD'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
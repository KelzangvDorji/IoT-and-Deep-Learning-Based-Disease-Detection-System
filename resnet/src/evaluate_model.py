import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Load features and labels
X_test = np.load("features.npy")
y_test = np.load("labels.npy")

# Load trained model
clf = joblib.load("potato_classifier.pkl")

# Make predictions
y_pred = clf.predict(X_test)

# Print evaluation metrics
print(f"Model Accuracy: {clf.score(X_test, y_test) * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

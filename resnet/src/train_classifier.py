import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load extracted features
features = np.load("features.npy")
labels = np.load("labels.npy")

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train an MLP Classifier
clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# Print detailed evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save evaluation results
with open("evaluation.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))

print("Evaluation results saved in 'evaluation.txt'")

# Save Model
joblib.dump(clf, "potato_classifier.pkl")
print("Model Saved as 'potato_classifier.pkl'")

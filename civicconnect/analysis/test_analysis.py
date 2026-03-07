import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import os

# -----------------------------
# Load TEST data outputs
# -----------------------------
y_true = np.load("analysis/eval_labels.npy")
y_pred = np.load("analysis/eval_preds.npy")
y_prob = np.load("analysis/eval_probs.npy")

os.makedirs("analysis/test_plots", exist_ok=True)

# -----------------------------
# 1️⃣ Test Accuracy & F1
# -----------------------------
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")

print(f"✅ Test Accuracy: {acc:.4f}")
print(f"✅ Test F1-score: {f1:.4f}")

# -----------------------------
# 2️⃣ Confusion Matrix (TEST)
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Test Data Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("analysis/test_plots/test_confusion_matrix.png")
plt.close()

# -----------------------------
# 3️⃣ Class-wise Accuracy (TEST)
# -----------------------------
class_acc = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(8,5))
plt.bar(range(len(class_acc)), class_acc)
plt.xlabel("Class Index")
plt.ylabel("Accuracy")
plt.title("Class-wise Test Accuracy")
plt.ylim(0, 1)
plt.savefig("analysis/test_plots/test_class_accuracy.png")
plt.close()

# -----------------------------
# 4️⃣ Probability Confidence Histogram
# -----------------------------
max_probs = y_prob.max(axis=1)

plt.figure(figsize=(8,5))
plt.hist(max_probs, bins=20)
plt.xlabel("Prediction Confidence")
plt.ylabel("Number of Samples")
plt.title("Test Data Prediction Confidence")
plt.savefig("analysis/test_plots/test_probability_hist.png")
plt.close()

# -----------------------------
# 5️⃣ Error Confidence Analysis
# -----------------------------
errors = y_true != y_pred

plt.figure(figsize=(8,5))
plt.hist(max_probs[errors], bins=15)
plt.xlabel("Confidence")
plt.ylabel("Misclassified Samples")
plt.title("Confidence of Wrong Predictions (Test Data)")
plt.savefig("analysis/test_plots/test_error_confidence.png")
plt.close()

# -----------------------------
# 6️⃣ Classification Report
# -----------------------------
print("\n📊 Test Classification Report:")
print(classification_report(y_true, y_pred))

print("\n📈 All TEST analysis graphs saved in: analysis/test_plots/")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import os

# ------------------------------
# 1️⃣ Load evaluation data
# ------------------------------
eval_labels = np.load("analysis/eval_labels.npy")
eval_preds = np.load("analysis/eval_preds.npy")
eval_probs = np.load("analysis/eval_probs.npy")  # Optional for probability plots

# ------------------------------
# 2️⃣ Define labels
# ------------------------------
label_names = ["electricity", "garbage", "road", "water"]  # same order as training

# ------------------------------
# 3️⃣ Confusion Matrix
# ------------------------------
cm = confusion_matrix(eval_labels, eval_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("analysis/confusion_matrix_full.png")
plt.close()
print("✅ Confusion matrix saved as analysis/confusion_matrix_full.png")

# ------------------------------
# 4️⃣ Classification Report
# ------------------------------
report = classification_report(eval_labels, eval_preds, target_names=label_names, output_dict=True)
print("\n📊 Classification Report:")
for label in label_names:
    print(f"{label}: Precision: {report[label]['precision']:.4f}, Recall: {report[label]['recall']:.4f}, F1: {report[label]['f1-score']:.4f}")
print(f"\nOverall Accuracy: {accuracy_score(eval_labels, eval_preds):.4f}")
print(f"Weighted F1: {f1_score(eval_labels, eval_preds, average='weighted'):.4f}")

# ------------------------------
# 5️⃣ Class-wise metrics plot
# ------------------------------
classes = label_names
precision = [report[c]['precision'] for c in classes]
recall = [report[c]['recall'] for c in classes]
f1 = [report[c]['f1-score'] for c in classes]

plt.figure(figsize=(10,6))
plt.plot(classes, precision, marker='o', label='Precision')
plt.plot(classes, recall, marker='o', label='Recall')
plt.plot(classes, f1, marker='o', label='F1 Score')
plt.title("Class-wise Metrics")
plt.ylabel("Score")
plt.ylim(0,1.05)
plt.legend()
plt.savefig("analysis/class_metrics.png")
plt.close()
print("✅ Class-wise metrics plot saved as analysis/class_metrics.png")

# ------------------------------
# 6️⃣ Probability distribution per class
# ------------------------------
plt.figure(figsize=(10,6))
for i, label in enumerate(classes):
    plt.hist(eval_probs[:, i], bins=20, alpha=0.5, label=label)
plt.title("Predicted Probability Distribution per Class")
plt.xlabel("Probability")
plt.ylabel("Count")
plt.legend()
plt.savefig("analysis/probabilities_hist.png")
plt.close()
print("✅ Probability distribution histogram saved as analysis/probabilities_hist.png")

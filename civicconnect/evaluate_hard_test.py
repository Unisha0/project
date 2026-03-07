import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support

# ------------------------------
# 1️⃣ Load model and tokenizer
# ------------------------------
model_dir = "best_model"
model = DistilBertForSequenceClassification.from_pretrained(model_dir)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)

with open(os.path.join(model_dir, "labels.txt")) as f:
    label_names = [line.strip() for line in f]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ------------------------------
# 2️⃣ Load hard test data
# ------------------------------
TEST_FILE = "data/hard_test.csv"
df = pd.read_csv(TEST_FILE)

# Ensure your CSV has columns: 'text' and 'label'
texts = df['text'].tolist()
true_labels_text = df['label'].tolist()

# Map labels to indices
label2idx = {label: i for i, label in enumerate(label_names)}
true_labels = np.array([label2idx[l] for l in true_labels_text])

# ------------------------------
# 3️⃣ Predict
# ------------------------------
enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
with torch.no_grad():
    outputs = model(**enc)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
    preds = np.argmax(probs, axis=1)

# ------------------------------
# 4️⃣ Metrics
# ------------------------------
acc = accuracy_score(true_labels, preds)
f1 = f1_score(true_labels, preds, average='weighted')
print(f"✅ Hard Test Accuracy: {acc:.4f}")
print(f"✅ Hard Test F1-score: {f1:.4f}\n")
print("📊 Hard Test Classification Report:")
report = classification_report(true_labels, preds, target_names=label_names)
print(report)

# ------------------------------
# 5️⃣ Confusion Matrix
# ------------------------------
os.makedirs("analysis/hard_test_plots", exist_ok=True)
cm = confusion_matrix(true_labels, preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names, cmap="Reds")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Hard Test Confusion Matrix")
plt.savefig("analysis/hard_test_plots/confusion_matrix.png")
plt.close()

# ------------------------------
# 6️⃣ Class-wise metrics plot (Precision, Recall, F1)
# ------------------------------
precision, recall, f1_scores, support = precision_recall_fscore_support(true_labels, preds, labels=range(len(label_names)))
x = np.arange(len(label_names))

plt.figure(figsize=(10,6))
plt.bar(x-0.2, precision, 0.2, label='Precision', color='blue')
plt.bar(x, recall, 0.2, label='Recall', color='green')
plt.bar(x+0.2, f1_scores, 0.2, label='F1-score', color='red')
plt.xticks(x, label_names, rotation=45)
plt.ylabel("Score")
plt.ylim(0,1)
plt.title("Hard Test Class-wise Metrics")
plt.legend()
plt.tight_layout()
plt.savefig("analysis/hard_test_plots/class_metrics.png")
plt.close()

# ------------------------------
# 7️⃣ Probability histogram per class
# ------------------------------
plt.figure(figsize=(10,6))
for i, label in enumerate(label_names):
    plt.hist(probs[:, i], bins=20, alpha=0.6, label=label)
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Hard Test Predicted Probabilities per Class")
plt.legend()
plt.savefig("analysis/hard_test_plots/probabilities_hist.png")
plt.close()

print("📈 All HARD TEST analysis plots saved in: analysis/hard_test_plots/")

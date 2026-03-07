import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ------------------------------
# 1️⃣ Load and preprocess dataset
# ------------------------------
data_file = "data/merged_complaints.csv"  # path to your CSV
df = pd.read_csv(data_file)
df = df.dropna(subset=['text', 'label'])

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
label_names = le.classes_.tolist()

# Save temp CSV for HuggingFace dataset
os.makedirs("data", exist_ok=True)
temp_csv = "data/temp.csv"
df.to_csv(temp_csv, index=False)

dataset = load_dataset('csv', data_files=temp_csv, split='train')
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# ------------------------------
# 2️⃣ Tokenization
# ------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

encoded_dataset = dataset.map(tokenize, batched=True)
encoded_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# ------------------------------
# 3️⃣ Load DistilBERT model
# ------------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(label_names)
)

# ------------------------------
# 4️⃣ Training arguments
# ------------------------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2
)

# ------------------------------
# 5️⃣ Compute metrics function
# ------------------------------
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

# ------------------------------
# 6️⃣ Trainer setup
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    compute_metrics=compute_metrics
)

# ------------------------------
# 7️⃣ Train the model
# ------------------------------
print("🚀 Starting training...")
trainer.train()

# ------------------------------
# 8️⃣ Save best model and tokenizer
# ------------------------------
model_dir = "best_model"
os.makedirs(model_dir, exist_ok=True)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
with open(os.path.join(model_dir, "labels.txt"), "w") as f:
    for label in label_names:
        f.write(f"{label}\n")

print("✅ Training completed and best model saved!")


# ------------------------------
## ------------------------------
# 9️⃣ Evaluate model & save for analysis
# ------------------------------
preds_output = trainer.predict(encoded_dataset['test'])
preds = np.argmax(preds_output.predictions, axis=1)
true_labels = preds_output.label_ids
probs = torch.nn.functional.softmax(torch.tensor(preds_output.predictions), dim=-1).numpy()

# Save evaluation outputs for separate analysis
os.makedirs("analysis", exist_ok=True)
np.save("analysis/eval_labels.npy", true_labels)
np.save("analysis/eval_preds.npy", preds)
np.save("analysis/eval_probs.npy", probs)

print("✅ Evaluation outputs saved in 'analysis/' folder")

# ------------------------------
# 🔟 Confusion matrix
# ------------------------------
cm = confusion_matrix(true_labels, preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("analysis/confusion_matrix.png")
plt.close()

# ------------------------------
# 1️⃣1️⃣ Plot metrics per epoch
# ------------------------------
metrics_history = trainer.state.log_history

epochs, train_loss, eval_loss, accuracy_list, f1_list = [], [], [], [], []
for log in metrics_history:
    if 'epoch' in log:
        if 'loss' in log:
            epochs.append(log['epoch'])
            train_loss.append(log['loss'])
        if 'eval_loss' in log:
            eval_loss.append(log['eval_loss'])
        if 'eval_accuracy' in log:
            accuracy_list.append(log['eval_accuracy'])
        if 'eval_f1' in log:
            f1_list.append(log['eval_f1'])

sns.set(style="whitegrid")

# Loss plot
plt.figure(figsize=(10,6))
plt.plot(epochs, train_loss, label="Train Loss", marker='o')
plt.plot(epochs, eval_loss, label="Eval Loss", marker='o')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Evaluation Loss per Epoch")
plt.legend()
plt.savefig("analysis/loss_plot.png")
plt.close()

# Accuracy & F1 plot
plt.figure(figsize=(10,6))
plt.plot(epochs, accuracy_list, label="Accuracy", marker='o', color='green')
plt.plot(epochs, f1_list, label="F1 Score", marker='o', color='red')
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.title("Accuracy & F1 Score per Epoch")
plt.legend()
plt.savefig("analysis/score_plot.png")
plt.close()

print("📈 All analysis plots saved in 'analysis/' folder.")

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

# Remove empty texts
df = df.dropna(subset=['text', 'label'])

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
label_names = le.classes_.tolist()

# Save temp CSV for HuggingFace
os.makedirs("data", exist_ok=True)
df.to_csv("data/temp.csv", index=False)
dataset = load_dataset('csv', data_files='data/temp.csv', split='train')
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
    num_train_epochs=5,                  # train longer for better accuracy
    per_device_train_batch_size=16,      # bigger batch if memory allows
    per_device_eval_batch_size=16,
    learning_rate=5e-5,                  # typical fine-tuning learning rate
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
# 7️⃣ Train
# ------------------------------
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
# 9️⃣ Evaluate best model
# ------------------------------
preds_output = trainer.predict(encoded_dataset['test'])
preds = np.argmax(preds_output.predictions, axis=1)
true_labels = preds_output.label_ids

print("\n📊 Classification Report:")
print(classification_report(true_labels, preds, target_names=label_names))

cm = confusion_matrix(true_labels, preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ------------------------------
# 🔟 Prediction function for new data
# ------------------------------
def predict(texts):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
    return [label_names[i] for i in preds], probs.cpu().numpy()

# Example usage
examples = ["Power outage in my neighborhood", "Water supply issue for days"]
pred_labels, pred_probs = predict(examples)
for t, l, p in zip(examples, pred_labels, pred_probs):
    print(f"Text: {t}\nPredicted: {l}\nProbabilities: {p}\n")

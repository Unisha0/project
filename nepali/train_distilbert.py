#!/usr/bin/env python3
"""
CivicConnect — Nepali Civic Complaint Classification
Fine-tune distilbert-base-multilingual-cased on merged Nepali complaint dataset.

Improvements over English pipeline:
  - Multilingual DistilBERT (supports Devanagari)
  - Uses pre-defined train/val/test splits from CSV
  - Early stopping (patience=2)
  - Learning rate scheduler with warmup
  - Saves confusion matrix, loss plot, accuracy/F1 plot, classification report
  - Saves model, tokenizer, and label mapping
"""

import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import Dataset, DatasetDict
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "distilbert-base-multilingual-cased"
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "merged_nepali.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "best_model")
ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), "analysis")

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 2
SEED = 42

LABEL_ORDER = ["electricity", "water", "road", "garbage"]

# ============================================================
# 1. LOAD & PREPARE DATA
# ============================================================
print("=" * 60)
print("1. Loading merged dataset...")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
df = df.dropna(subset=["text", "category", "split"])

# Clean text: strip whitespace
df["text"] = df["text"].astype(str).str.strip()

# Label encoding (fixed order)
label2id = {label: idx for idx, label in enumerate(LABEL_ORDER)}
id2label = {idx: label for label, idx in label2id.items()}
df["label"] = df["category"].map(label2id)

print(f"  Total samples: {len(df)}")
print(f"  Categories: {LABEL_ORDER}")
print(f"  Label mapping: {label2id}")
print()

# Split using the 'split' column from CSV
train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df = df[df["split"] == "val"].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)

print(f"  Train: {len(train_df)}")
print(f"  Val:   {len(val_df)}")
print(f"  Test:  {len(test_df)}")
print()

# Category distribution
print("  Category distribution:")
for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    counts = split_df["category"].value_counts().to_dict()
    print(f"    {split_name}: {counts}")
print()

# ============================================================
# 2. TOKENIZATION
# ============================================================
print("=" * 60)
print("2. Tokenizing with multilingual DistilBERT...")
print("=" * 60)

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

# Convert to HuggingFace Datasets
def df_to_dataset(dataframe):
    return Dataset.from_pandas(dataframe[["text", "label"]])

dataset = DatasetDict({
    "train": df_to_dataset(train_df),
    "val": df_to_dataset(val_df),
    "test": df_to_dataset(test_df),
})

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

encoded = dataset.map(tokenize_fn, batched=True, batch_size=256)
encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Show tokenization example
sample_text = train_df["text"].iloc[0]
sample_tokens = tokenizer.tokenize(sample_text)
print(f"\n  Sample text: {sample_text[:80]}...")
print(f"  Tokens ({len(sample_tokens)}): {sample_tokens[:10]}...")
print()

# ============================================================
# 3. LOAD MODEL
# ============================================================
print("=" * 60)
print("3. Loading distilbert-base-multilingual-cased...")
print("=" * 60)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_ORDER),
    id2label=id2label,
    label2id=label2id,
)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print()

# ============================================================
# 4. TRAINING ARGUMENTS
# ============================================================
print("=" * 60)
print("4. Setting up training...")
print("=" * 60)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    seed=SEED,
    report_to="none",  # disable wandb/tensorboard
)

print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Warmup ratio: {WARMUP_RATIO}")
print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
print()

# ============================================================
# 5. METRICS
# ============================================================
def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1, "f1_macro": f1_macro}

# ============================================================
# 6. TRAINER
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["val"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

# ============================================================
# 7. TRAIN
# ============================================================
print("=" * 60)
print("7. Starting training...")
print("=" * 60)

trainer.train()

print("\n✅ Training completed!")

# ============================================================
# 8. SAVE MODEL & TOKENIZER
# ============================================================
print("\n" + "=" * 60)
print("8. Saving best model...")
print("=" * 60)

os.makedirs(MODEL_DIR, exist_ok=True)
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

# Save label mapping
with open(os.path.join(MODEL_DIR, "labels.txt"), "w") as f:
    for label in LABEL_ORDER:
        f.write(f"{label}\n")

print(f"  Model saved to: {MODEL_DIR}")
print(f"  Labels: {LABEL_ORDER}")

# ============================================================
# 9. EVALUATE ON TEST SET
# ============================================================
print("\n" + "=" * 60)
print("9. Evaluating on test set...")
print("=" * 60)

test_output = trainer.predict(encoded["test"])
test_preds = np.argmax(test_output.predictions, axis=1)
test_labels = test_output.label_ids
test_probs = torch.nn.functional.softmax(
    torch.tensor(test_output.predictions), dim=-1
).numpy()

test_acc = accuracy_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds, average="weighted")
test_f1_macro = f1_score(test_labels, test_preds, average="macro")

print(f"\n  Test Accuracy:     {test_acc:.4f}")
print(f"  Test F1 (weighted): {test_f1:.4f}")
print(f"  Test F1 (macro):    {test_f1_macro:.4f}")
print()

report = classification_report(
    test_labels, test_preds,
    target_names=LABEL_ORDER,
    digits=4
)
print("  Classification Report:")
print(report)

# Save report to file
os.makedirs(ANALYSIS_DIR, exist_ok=True)
with open(os.path.join(ANALYSIS_DIR, "classification_report.txt"), "w") as f:
    f.write("CivicConnect — Nepali Complaint Classification Report\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Test F1 (weighted): {test_f1:.4f}\n")
    f.write(f"Test F1 (macro): {test_f1_macro:.4f}\n\n")
    f.write(report)

# Save numpy arrays for later analysis
np.save(os.path.join(ANALYSIS_DIR, "test_labels.npy"), test_labels)
np.save(os.path.join(ANALYSIS_DIR, "test_preds.npy"), test_preds)
np.save(os.path.join(ANALYSIS_DIR, "test_probs.npy"), test_probs)

# ============================================================
# 10. CONFUSION MATRIX
# ============================================================
print("=" * 60)
print("10. Generating analysis plots...")
print("=" * 60)

sns.set(style="whitegrid")

cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d",
    xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Nepali Complaint Classification — Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("  ✓ Confusion matrix saved")

# ============================================================
# 11. CLASS-WISE METRICS BAR CHART
# ============================================================
precision, recall, f1_scores, support = precision_recall_fscore_support(
    test_labels, test_preds, labels=range(len(LABEL_ORDER))
)

x = np.arange(len(LABEL_ORDER))
width = 0.2

plt.figure(figsize=(10, 6))
plt.bar(x - width, precision, width, label="Precision", color="#2196F3")
plt.bar(x, recall, width, label="Recall", color="#4CAF50")
plt.bar(x + width, f1_scores, width, label="F1-score", color="#F44336")
plt.xticks(x, LABEL_ORDER)
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title("Per-class Precision / Recall / F1")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "class_metrics.png"), dpi=150)
plt.close()
print("  ✓ Class metrics plot saved")

# ============================================================
# 12. TRAINING LOSS & METRICS OVER EPOCHS
# ============================================================
log_history = trainer.state.log_history

train_epochs, train_losses = [], []
eval_epochs, eval_losses, eval_accs, eval_f1s = [], [], [], []

for entry in log_history:
    if "loss" in entry and "eval_loss" not in entry:
        train_epochs.append(entry.get("epoch", 0))
        train_losses.append(entry["loss"])
    if "eval_loss" in entry:
        eval_epochs.append(entry.get("epoch", 0))
        eval_losses.append(entry["eval_loss"])
        if "eval_accuracy" in entry:
            eval_accs.append(entry["eval_accuracy"])
        if "eval_f1" in entry:
            eval_f1s.append(entry["eval_f1"])

# Loss plot
plt.figure(figsize=(10, 6))
if train_epochs:
    plt.plot(train_epochs, train_losses, label="Train Loss", marker="o", alpha=0.7)
if eval_epochs:
    plt.plot(eval_epochs, eval_losses, label="Val Loss", marker="s", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "loss_plot.png"), dpi=150)
plt.close()
print("  ✓ Loss plot saved")

# Accuracy & F1 plot
if eval_accs and eval_f1s:
    plt.figure(figsize=(10, 6))
    plt.plot(eval_epochs[:len(eval_accs)], eval_accs, label="Accuracy", marker="o", color="green")
    plt.plot(eval_epochs[:len(eval_f1s)], eval_f1s, label="F1 (weighted)", marker="s", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Accuracy & F1 per Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "score_plot.png"), dpi=150)
    plt.close()
    print("  ✓ Score plot saved")

# ============================================================
# 13. PROBABILITY DISTRIBUTION HISTOGRAM
# ============================================================
plt.figure(figsize=(10, 6))
for i, label in enumerate(LABEL_ORDER):
    plt.hist(test_probs[:, i], bins=30, alpha=0.5, label=label)
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Test Set — Predicted Probability Distribution per Class")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "probability_hist.png"), dpi=150)
plt.close()
print("  ✓ Probability histogram saved")

# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 60)
print("✅ ALL DONE!")
print("=" * 60)
print(f"  Model:    {MODEL_DIR}/")
print(f"  Analysis: {ANALYSIS_DIR}/")
print(f"  Results:  {OUTPUT_DIR}/")
print(f"\n  Test Accuracy: {test_acc:.4f}")
print(f"  Test F1:       {test_f1:.4f}")
print("=" * 60)

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import os

# ------------------------------
# Load model & tokenizer
# ------------------------------
model_dir = "best_model"
model = DistilBertForSequenceClassification.from_pretrained(model_dir)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)

# Load labels
with open(os.path.join(model_dir, "labels.txt"), "r") as f:
    label_names = [line.strip() for line in f.readlines()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ------------------------------
# Prediction function
# ------------------------------
def predict(text):
    enc = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        return label_names[pred_idx], probs[0, pred_idx].item()

# ------------------------------
# Interactive loop
# ------------------------------
if __name__ == "__main__":
    print("💡 Complaint Classifier Ready. Type 'q' to quit.")
    while True:
        complaint = input("Enter complaint: ")
        if complaint.lower() == 'q':
            break
        label, prob = predict(complaint)
        print(f"Predicted Category: {label} (Probability: {prob:.4f})\n")

# predict_distilbert.py

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import joblib
import os

# Load everything
model = DistilBertForSequenceClassification.from_pretrained('model/distilbert_model')
tokenizer = DistilBertTokenizerFast.from_pretrained('model/tokenizer')
label_encoder = joblib.load('model/label_encoder.pkl')

model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([predicted])[0]

# Loop for interactive testing
if __name__ == "__main__":
    print("Complaint Classifier Ready. Type 'q' to quit.")
    while True:
        complaint = input("Enter complaint: ")
        if complaint.lower() == 'q':
            break
        label = predict(complaint)
        print(f"Predicted Category: {label}\n")

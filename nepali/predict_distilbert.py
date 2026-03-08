#!/usr/bin/env python3
"""
CivicConnect — Nepali Complaint Prediction Script
Interactive terminal-based prediction using the trained multilingual DistilBERT model.
"""

import os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "best_model")
CONFIDENCE_THRESHOLD = 0.70  # minimum confidence to show as "confident"

# ============================================================
# LOAD MODEL AND TOKENIZER
# ============================================================
print("=" * 60)
print("Loading Nepali Complaint Classifier...")
print("=" * 60)

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Load label mapping
labels_file = os.path.join(MODEL_DIR, "labels.txt")
if os.path.exists(labels_file):
    with open(labels_file, "r") as f:
        LABELS = [line.strip() for line in f.readlines()]
else:
    LABELS = ["electricity", "water", "road", "garbage"]

print(f"  Model loaded from: {MODEL_DIR}")
print(f"  Categories: {LABELS}")
print()

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict(text):
    """Predict the category for a Nepali complaint text."""
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    # Get prediction
    pred_idx = torch.argmax(probs).item()
    pred_label = LABELS[pred_idx]
    confidence = probs[pred_idx].item()
    
    return pred_label, confidence, probs.tolist()

# ============================================================
# INTERACTIVE LOOP
# ============================================================
def main():
    print("=" * 60)
    print("Nepali Civic Complaint Classifier")
    print("=" * 60)
    print("Enter a Nepali complaint to classify.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    print()
    
    while True:
        try:
            text = input("Enter complaint: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nExiting...")
            break
        
        if not text:
            continue
        
        if text.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        # Predict
        pred_label, confidence, all_probs = predict(text)
        
        # Display results
        print()
        print("-" * 40)
        
        if confidence >= CONFIDENCE_THRESHOLD:
            print(f"Predicted Category: {pred_label.upper()}")
        else:
            print(f"Predicted Category: {pred_label.upper()} (low confidence)")
        
        print(f"Confidence Score:   {confidence:.1%}")
        print()
        print("Class Probabilities:")
        for i, label in enumerate(LABELS):
            bar = "#" * int(all_probs[i] * 20)
            print(f"  {label:12} {all_probs[i]:6.1%} [{bar:<20}]")
        
        print("-" * 40)
        print()

# ============================================================
# QUICK TEST MODE
# ============================================================
def quick_test():
    """Run a few sample predictions."""
    samples = [
        "हाम्रो टोलमा बिजुली बारम्बार जान्छ",
        "पानीको पाइपबाट फोहोर पानी आउँछ",
        "सडकमा ठूला खाल्डाखुल्डी छन्",
        "फोहोर संकलन हुँदैन",
        "ट्रान्सफर्मर पड्किएको छ",
        "धारामा पानी आउँदैन",
        "बाटो भत्किएको छ",
        "कचरा व्यवस्थापन हुँदैन"
    ]
    
    print("=" * 60)
    print("Quick Test - Sample Predictions")
    print("=" * 60)
    print()
    print(f"{'Status':<8} {'Category':<12} {'Confidence':<10} Text")
    print("-" * 60)
    
    for text in samples:
        pred_label, confidence, _ = predict(text)
        status = "[OK]" if confidence >= CONFIDENCE_THRESHOLD else "[LOW]"
        print(f"{status:<8} {pred_label:<12} {confidence:>6.1%}     {text[:40]}")
    
    print()
    print("=" * 60)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] in ["--test", "-t"]:
        quick_test()
    else:
        quick_test()  # show samples first
        print()
        main()  # then interactive mode

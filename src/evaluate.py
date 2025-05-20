import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from tqdm import tqdm
import utils

model = AutoModelForSequenceClassification.from_pretrained(utils.MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(utils.MODEL_PATH)

df = pd.read_csv(utils.DATASET_PATH, nrows=100)
texts = df["text"].tolist()
labels = df["label"].tolist()

model.eval()
predictions = []

with torch.no_grad():
    for text in tqdm(texts, desc="Evaluating"):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        predictions.append(pred)

accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions, average="weighted", zero_division=0)
recall = recall_score(labels, predictions, average="weighted", zero_division=0)
f1 = f1_score(labels, predictions, average="weighted", zero_division=0)

print(f"\nEvaluation Metrics:")
print(f"  Accuracy : {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall   : {recall:.4f}")
print(f"  F1 Score : {f1:.4f}")



import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from tqdm import tqdm

# === Configurazione ===
MODEL_PATH = "models"  # path al modello fine-tuned
DATA_PATH = "data/processed/train.csv"  # dati di test

# === Caricamento modello/tokenizer ===
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# === Caricamento dati ===
df = pd.read_csv(DATA_PATH, nrows=100)
texts = df["text"].tolist()
labels = df["label"].tolist()

# === Inferenza ===
model.eval()
predictions = []

with torch.no_grad():
    for text in tqdm(texts, desc="Evaluating"):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        predictions.append(pred)

# === Metriche ===
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions, average="weighted")
recall = recall_score(labels, predictions, average="weighted")
f1 = f1_score(labels, predictions, average="weighted")

print(f"\n📊 Evaluation Metrics:")
print(f"  Accuracy : {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall   : {recall:.4f}")
print(f"  F1 Score : {f1:.4f}")



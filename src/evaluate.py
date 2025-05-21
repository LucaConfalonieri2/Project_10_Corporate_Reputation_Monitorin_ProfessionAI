import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from tqdm import tqdm
import utils
from datetime import datetime
import csv
import os

model = AutoModelForSequenceClassification.from_pretrained(utils.MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(utils.MODEL_PATH)

df = pd.read_csv(utils.TEST_DATASET_TEMP)
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

log_file = "logs/eval_log.csv"
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
headers = ["timestamp", "test_size", "accuracy", 
           "precision", "recall", "f1"]

row = {"timestamp": now,
       "test_size": len(texts),
       "accuracy": accuracy,
       "precision": precision,
       "recall": recall,
       "f1": f1
        }

file_exists = os.path.isfile(log_file)
with open(log_file, "a", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    if not file_exists:
        writer.writeheader()
    writer.writerow(row)






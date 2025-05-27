from transformers import pipeline
import utils
import pandas as pd 
from datetime import datetime
import csv
import os

pipe = pipeline(
    "sentiment-analysis",
    model="confa3452/fasttext-sentiment-it-ProfectionAI",
    tokenizer="confa3452/fasttext-sentiment-it-ProfectionAI"
    )

df = pd.read_csv(utils.TEST_DATASET_TEMP)
texts = df["text"].tolist()

results = []
for i in range(len(texts)):
    results.append(pipe(texts[i])[0]["label"])

headers = ["timestamp", "positive", "negative", "neutral"]

row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "positive": results.count("positive"),
        "negative": results.count("negative"),
        "neutral": results.count("neutral")
}

file_exists = os.path.isfile(utils.COMM_FILE)
with open(utils.COMM_FILE, "a", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    if not file_exists:
        writer.writeheader()
    writer.writerow(row)






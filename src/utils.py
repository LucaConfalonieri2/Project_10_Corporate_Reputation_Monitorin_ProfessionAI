from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
import os

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_PATH = "models/sentiment_model"
ORIGINAL_DATASET_FILE = "data/raw/training.1600000.processed.noemoticon.csv"
DATASET_FILE = "data/processed/train.csv"
PROGRESS_FILE = "data/batch_progress.txt"
DATASET_FILE_TEMP = "data/new/new_data.csv"

# Metriche di valutazione
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }

def create_batch_data(batch_size):
    df = pd.read_csv(DATASET_FILE)

    start = 0

    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            start = int(f.read().strip())

    end = start + batch_size

    # Estrai blocco
    df_batch = df.iloc[start:end]

    if df_batch.empty:
        print("Tutti i dati sono già stati usati.")
        exit(0)

    # Salva batch per retraining
    os.makedirs(os.path.dirname(DATASET_FILE_TEMP), exist_ok=True)
    df_batch.to_csv(DATASET_FILE_TEMP, index=False)

    # Aggiorna progress
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(end))

    return df_batch




from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
import os
import json

LOG_FILE = "logs/eval_log.csv"
COMM_FILE = "logs/comm_log.csv"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_PATH = "models/sentiment_model"
ORIGINAL_DATASET_FILE = "data/raw/training.1600000.processed.noemoticon.csv"
TRAIN_DATASET_FILE = "data/processed/train.csv"
TEST_DATASET_FILE = "data/processed/test.csv"
PROGRESS_FILE = "data/batch_progress.json"
TRAIN_DATASET_TEMP = "data/new/new_train_data.csv"
TEST_DATASET_TEMP = "data/new/new_test_data.csv"
COMM_DATASET_TEMP = "data/new/new_comm.csv"
REPO_ID = "confa3452/fasttext-sentiment-it-ProfectionAI"

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

# Prende batch_size esempi dal dataset principale
# e li salva su un dataset temporaneo
def create_batch_data(batch_size, file_in, file_out):
    df = pd.read_csv(file_in)

    data = {TRAIN_DATASET_FILE: 0, TEST_DATASET_FILE: 0}
    start = 0
    print("Progress file path:", os.path.abspath(PROGRESS_FILE))

    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            data = json.load(f)
            start = data[file_in]

    end = start + batch_size

    # Estrai blocco
    df_batch = df.iloc[start:end]

    if df_batch.empty:
        print("Tutti i dati sono già stati usati.")
        exit(0)

    # Salva batch per retraining
    os.makedirs(os.path.dirname(file_out), exist_ok=True)
    df_batch.to_csv(file_out, index=False)

    # Aggiorna progress
    with open(PROGRESS_FILE, "w") as f:
        data[file_in] = end
        json.dump(data, f, indent=4)




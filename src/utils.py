from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
import os
import json
from huggingface_hub import HfApi, snapshot_download

from pathlib import Path



LOG_FILE = "logs/eval_log.csv"
COMM_FILE = "logs/comm_log.csv"
PROGRESS_FILE = "logs/batch_progress.json"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_PATH = "models/sentiment_model"
ORIGINAL_DATASET_FILE = "data/raw/training.1600000.processed.noemoticon.csv"
TRAIN_DATASET_FILE = "data/processed/train.csv"
TEST_DATASET_FILE = "data/processed/test.csv"
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


def upload_folder_to_hf(local_folder_path = "/data", repo_id=REPO_ID):
    """
    Carica una cartella su un repository Hugging Face.

    :param local_folder_path: Cartella da caricare.
    :param repo_id: Nome del repository HF
    """
    api = HfApi()
    print(f"Uploading '{local_folder_path}' to HF...")

    api.upload_folder(
        folder_path=local_folder_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload folder",
    )
    print("Upload completato...")

def download_folder_from_hf(folder_path="logs/", local_dir = "./logs", repo_id=REPO_ID):
    """
    Scarica un file da Hugging Face Hub.

    :folder_path: Cartella da scaricare
    :param local_dir: Cartella locale di destinazione
    :param repo_id: Nome del repository
    :return: Percorso locale al file scaricato
    """
    print(f"Scaricando '{folder_path}' da repo '{repo_id}'...")

    downloaded_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=[f"{folder_path}*"]
    )

    print("Download completato...")
    return downloaded_path



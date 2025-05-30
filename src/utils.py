from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
import os
import json
from huggingface_hub import HfApi, snapshot_download
import shutil

from pathlib import Path

# Variabili utili ai vari file
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

# Calcola le metriche di valutazione
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

def upload_folder_to_hf(local_folder_path = "logs", repo_id=REPO_ID):
    """
    Carica una cartella su un repository Hugging Face.

    :param local_folder_path: Cartella da caricare.
    :param repo_id: Nome del repository HF
    """
    api = HfApi()
    print(f"Uploading '{local_folder_path}' to HF...")

    for root, _, files in os.walk(local_folder_path):
        if '.cache' in root:
            continue
        for file in files:
            full_path = os.path.join(root, file)
            # Crea il path relativo da inserire nel repo
            relative_path = os.path.relpath(full_path, start=local_folder_path)
            path_in_repo = os.path.join(local_folder_path, relative_path)

            print(f" - Uploading {relative_path}")
            api.upload_file(
                path_or_fileobj=full_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {relative_path}"
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

    tmp_dir = "./_hf_tmp_download"
    os.makedirs(local_dir, exist_ok=True)

    downloaded_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=tmp_dir,
        local_dir_use_symlinks=False,
        allow_patterns=[f"{folder_path}*.csv", f"{folder_path}*.json"]
    )

    # Copia solo i file .csv e .json da downloaded_path/folder_path a local_dir
    source_path = os.path.join(downloaded_path, folder_path)

    for filename in os.listdir(source_path):
        if filename.endswith(".csv") or filename.endswith(".json"):
            full_source = os.path.join(source_path, filename)
            full_target = os.path.join(local_dir, filename)
            shutil.copy2(full_source, full_target)
            print(f" - Copiato: {filename} → {local_dir}")

    print("Download completato...")
    return local_dir


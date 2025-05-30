from transformers import pipeline
import utils
import pandas as pd 
from datetime import datetime
import csv
import os

# Scarica la cartella contenente dei file utili per tenere traccia dei dati usati e delle metriche salvate
utils.download_folder_from_hf()

# Crea un nuovo file csv di commenti per classificarli come se fossero dei dati reali
utils.create_batch_data(100, utils.TEST_DATASET_FILE, utils.COMM_DATASET_TEMP)

# Carica il modello da HF
pipe = pipeline(
    "sentiment-analysis",
    model="confa3452/fasttext-sentiment-it-ProfectionAI",
    tokenizer="confa3452/fasttext-sentiment-it-ProfectionAI"
    )

# Carica il dataset appena creato
df = pd.read_csv(utils.COMM_DATASET_TEMP)
texts = df["text"].tolist()

# Valuta i commenti e salva i risultati
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

# DISABLE_UPLOAD è 1 se sto eseguente un test e quindi non ha senso aggiornare HF
if os.getenv("DISABLE_UPLOAD") != "1":
    utils.upload_folder_to_hf()



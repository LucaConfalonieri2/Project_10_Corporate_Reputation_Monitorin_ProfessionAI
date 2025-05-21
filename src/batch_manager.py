import pandas as pd
import os
import utils

BATCH_SIZE = 500

df = pd.read_csv(utils.DATASET_FILE)

start = 0

if os.path.exists(utils.PROGRESS_FILE):
    with open(utils.PROGRESS_FILE, "r") as f:
        start = int(f.read().strip())

end = start + BATCH_SIZE

# Estrai blocco
df_batch = df.iloc[start:end]

if df_batch.empty:
    print("Tutti i dati sono già stati usati.")
    exit(0)

# Salva batch per retraining
os.makedirs(os.path.dirname(utils.DATASET_FILE_TEMP), exist_ok=True)
df_batch.to_csv(utils.DATASET_FILE_TEMP, index=False)

# Aggiorna progress
with open(utils.PROGRESS_FILE, "w") as f:
    f.write(str(end))

print(f"Creato batch da {start} a {end}, salvato in: {utils.DATASET_FILE_TEMP}")

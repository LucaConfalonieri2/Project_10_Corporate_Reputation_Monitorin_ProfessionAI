import utils

# Scarica la cartella contenente dei file utili per tenere traccia dei dati usati e delle metriche salvate
utils.download_folder_from_hf()

# Crea un nuovo batch dal dataset originale di training e li salva in un file temporaneo
utils.create_batch_data(500, utils.TRAIN_DATASET_FILE, utils.TRAIN_DATASET_TEMP)

# Crea un nuovo batch dal dataset originale di test e li salva in un file temporaneo
utils.create_batch_data(100, utils.TEST_DATASET_FILE, utils.TEST_DATASET_TEMP)



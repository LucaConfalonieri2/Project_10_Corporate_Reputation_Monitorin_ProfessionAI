import utils

utils.download_folder_from_hf()

utils.create_batch_data(500, utils.TRAIN_DATASET_FILE, utils.TRAIN_DATASET_TEMP)

utils.create_batch_data(100, utils.TEST_DATASET_FILE, utils.TEST_DATASET_TEMP)



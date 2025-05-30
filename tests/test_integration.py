import os
import subprocess
import src.utils as utils

def test_batch_manager_crea_csv():
    assert os.path.exists(utils.TRAIN_DATASET_FILE), f"{utils.TRAIN_DATASET_FILE} not exits"
    assert os.path.exists(utils.TEST_DATASET_FILE), f"{utils.TEST_DATASET_FILE} not exits"

    with open(utils.TRAIN_DATASET_TEMP) as f:
        old_content_train = f.read()
    with open(utils.TEST_DATASET_TEMP) as f:
        old_content_test = f.read()

    subprocess.run(["python", "src/batch_manager.py"], check=True)
    
    with open(utils.TRAIN_DATASET_TEMP) as f:
        new_content_train = f.read()
    with open(utils.TEST_DATASET_TEMP) as f:
        new_content_test = f.read()

    assert old_content_train != new_content_train, f"{utils.TRAIN_DATASET_TEMP} not change"
    assert old_content_test != new_content_test, f"{utils.TEST_DATASET_TEMP} not change"

def test_train_and_test_model():
    subprocess.run(["python", "src/train.py"], check=True)
    expected_files = ["config.json", "merges.txt", "model.safetensors", 
                      "special_tokens_map.json", "tokenizer.json", 
                      "tokenizer_config.json", "vocab.json"
                      ]
    for f in expected_files:
        full_path = os.path.join(utils.MODEL_PATH, f)
        assert os.path.isfile(full_path), f"File mancante nel modello: {f}"

    assert os.path.exists(utils.LOG_FILE)

    with open(utils.LOG_FILE, 'r') as f:
        old_content = f.read()

    subprocess.run(
        ["python", "src/evaluate.py"], check=True,
        env={**os.environ, "DISABLE_UPLOAD": "1"}
    )
    
    with open(utils.LOG_FILE, 'r') as f:
        new_content = f.read()

    assert old_content != new_content, "Unsaved metrics"
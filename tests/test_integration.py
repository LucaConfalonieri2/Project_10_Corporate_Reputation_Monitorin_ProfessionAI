import os
import subprocess

def test_batch_manager():
    assert os.path.exists("data/processed/train.csv"), f"data/processed/train.csv doesn't exits"
    assert os.path.exists("data/processed/train.csv"), f"data/processed/train.csv doesn't exits"

    with open("data/new/new_train_data.csv") as f:
        old_content_train = f.read()
    with open("data/new/new_test_data.csv") as f:
        old_content_test = f.read()
    with open("logs/batch_progress.json") as f:
        old_batch_progress = f.read()

    subprocess.run(["python", "src/batch_manager.py"], check=True)
    
    with open("data/new/new_train_data.csv") as f:
        new_content_train = f.read()
    with open("data/new/new_test_data.csv") as f:
        new_content_test = f.read()
    with open("logs/batch_progress.json") as f:
        new_batch_progress = f.read()

    assert old_content_train != new_content_train, f"data/new/new_train_data.csv doesn't change"
    assert old_content_test != new_content_test, f"data/new/new_test_data.csv doesn't change"
    assert old_batch_progress != new_batch_progress, f"logs/batch_progress.json doesn't change"


def test_train_and_test_model():
    subprocess.run(["python", "src/train.py"], check=True)
    expected_files = ["config.json", "merges.txt", "model.safetensors", 
                      "special_tokens_map.json", "tokenizer.json", 
                      "tokenizer_config.json", "vocab.json"
                      ]
    for f in expected_files:
        full_path = os.path.join("models/sentiment_model", f)
        assert os.path.isfile(full_path), f"{f} not exits"

    assert os.path.exists("logs/eval_log.csv")

    with open("logs/eval_log.csv", 'r') as f:
        old_content = f.read()

    subprocess.run(
        ["python", "src/evaluate.py"], check=True,
        env={**os.environ, "DISABLE_UPLOAD": "1"}
    )
    
    with open("logs/eval_log.csv", 'r') as f:
        new_content = f.read()

    assert old_content != new_content, "Unsaved metrics in logs/eval_log.csv"


def test_predict():
    with open("logs/batch_progress.json", "r") as f:
        old_batch_progress = f.read()
    with open("logs/comm_log.csv", 'r') as f:
        old_comm_logs = f.read()

    subprocess.run(["python", "src/predict.py"], check=True,
        env={**os.environ, "DISABLE_UPLOAD": "1"}
    )

    with open("logs/batch_progress.json", "r") as f:
        new_batch_progress = f.read()
    with open("logs/comm_log.csv", 'r') as f:
        new_comm_logs = f.read()

    assert old_batch_progress!=new_batch_progress, "logs/batch_progress.json doesn't change"
    assert old_comm_logs!=new_comm_logs, "logs/batch_progress.json doesn't change"



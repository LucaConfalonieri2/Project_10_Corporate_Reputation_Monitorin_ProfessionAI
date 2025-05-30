import os
import subprocess

def test_batch_manager_crea_csv():
    assert os.path.exists("data/processed/train.csv"), f"data/processed/train.csv not exits"
    assert os.path.exists("data/processed/train.csv"), f"data/processed/train.csv not exits"

    with open("data/new/new_train_data.csv") as f:
        old_content_train = f.read()
    with open("data/new/new_test_data.csv") as f:
        old_content_test = f.read()

    subprocess.run(["python", "src/batch_manager.py"], check=True)
    
    with open("data/new/new_train_data.csv") as f:
        new_content_train = f.read()
    with open("data/new/new_test_data.csv") as f:
        new_content_test = f.read()

    assert old_content_train != new_content_train, f"data/new/new_train_data.csv not change"
    assert old_content_test != new_content_test, f"data/new/new_test_data.csv not change"

def test_train_and_test_model():
    subprocess.run(["python", "src/train.py"], check=True)
    expected_files = ["config.json", "merges.txt", "model.safetensors", 
                      "special_tokens_map.json", "tokenizer.json", 
                      "tokenizer_config.json", "vocab.json"
                      ]
    for f in expected_files:
        full_path = os.path.join("models/sentiment_model", f)
        assert os.path.isfile(full_path), f"File mancante nel modello: {f}"

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
import os
import subprocess

def test_batch_manager_crea_csv():
    subprocess.run(["python", "src/batch_manager.py"], check=True)
    assert os.path.exists("data/new/new_test_data.csv")
    assert os.path.exists("data/new/new_traing_data.csv")

def test_train_crea_modello():
    subprocess.run(["python", "src/train.py"], check=True)
    assert os.path.exists("models/sentiment_model")

def test_evaluate_logga_metriche():
    subprocess.run(["python", "src/evaluate.py"], check=True)
    assert os.path.exists("logs/eval_log.csv")

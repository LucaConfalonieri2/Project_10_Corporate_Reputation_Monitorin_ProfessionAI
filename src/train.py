from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Carica il dataset CSV
dataset = load_dataset("csv", data_files={"train": "data/processed/train.csv"}, delimiter=",")

# Tokenizzazione
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding=True, truncation=True)
def tokenize(example): return tokenizer(example["text"], truncation=True, padding="max_length")
tokenized = dataset.map(tokenize, batched=True)

# Modello con 3 classi
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Metriche di valutazione
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# Configurazione training
args = TrainingArguments(
    output_dir="./models/sentiment_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"].shuffle(seed=42).select(range(1)),  # usa subset per test iniziale
    eval_dataset=tokenized["train"].shuffle(seed=42).select(range(100)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("models")

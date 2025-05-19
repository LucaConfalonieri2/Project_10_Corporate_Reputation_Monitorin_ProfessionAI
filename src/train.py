from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import utils

MODEL_NAME = utils.MODEL_NAME

# Carica il dataset CSV
dataset = load_dataset("csv", data_files={"train": utils.TRAIN_DATASET_PATH}, delimiter=",")

# Tokenizzazione
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding=True, truncation=True)
def tokenize(example): return tokenizer(example["text"], truncation=True, padding="max_length")
tokenized = dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Metriche di valutazione
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision_weighted": precision_score(labels, preds, average="weighted"),
        "recall_weighted": recall_score(labels, preds, average="weighted"),
    }

# Configurazione training
args = TrainingArguments(
    output_dir="./models/training_arg",
    evaluation_strategy="epoch",
    save_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    load_best_model_at_end=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"].shuffle(seed=42).select(range(10)),
    eval_dataset=tokenized["train"].shuffle(seed=42).select(range(100)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("models/sentiment_model")
    tokenizer.save_pretrained("models/sentiment_model")



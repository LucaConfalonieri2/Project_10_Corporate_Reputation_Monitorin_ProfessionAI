from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import utils
import os

# Carica il dataset
dataset = load_dataset("csv", data_files={"train": utils.TRAIN_DATASET_TEMP}, delimiter=",")
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

# Carica il modello da locale (se presente) o da HF in caso contrario. Utile per lavorare sia con Active di git che in locale.
os.makedirs(utils.MODEL_PATH, exist_ok=True)
if not os.listdir(utils.MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(utils.REPO_ID)
    model = AutoModelForSequenceClassification.from_pretrained(utils.REPO_ID)
else:
    tokenizer = AutoTokenizer.from_pretrained(utils.MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(utils.MODEL_PATH)

def tokenize(example): 
    return tokenizer(example["text"], truncation=True, padding="max_length")

tokenized_dataset = split_dataset.map(tokenize, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Configurazione training
args = TrainingArguments(
    output_dir="models",
    evaluation_strategy="epoch",
    save_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    load_best_model_at_end=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=utils.compute_metrics,
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained(utils.MODEL_PATH)
    tokenizer.save_pretrained(utils.MODEL_PATH)


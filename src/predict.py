from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "models/sentiment_model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    label = torch.argmax(probs, dim=1).item()
    return label, probs.tolist()

if __name__ == "__main__":
    sample = input("Testo da analizzare: ")
    label, probs = predict_sentiment(sample)
    print(f"Predizione: {label} (probabilità: {probs})")

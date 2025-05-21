from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from src import utils

model = AutoModelForSequenceClassification.from_pretrained(utils.MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(utils.MODEL_PATH)

app = FastAPI(
    title="Sentiment Analysis API",
    description="Analizza il sentiment di un testo (0=negativo, 1=neutro, 2=positivo)",
    version="1.0"
)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        label = torch.argmax(probs, dim=1).item()
    
    return {
        "label": label,
        "prob": probs[0].tolist()
    }

@app.get("/")
def root():
    return {"message": "API Sentiment pronta"}





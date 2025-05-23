from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

#uvicorn src.predict:app --reload

pipe = pipeline(
    "sentiment-analysis",
    model="confa3452/fasttext-sentiment-it-ProfectionAI",
    tokenizer="confa3452/fasttext-sentiment-it-ProfectionAI"
    )

app = FastAPI(
    title="Sentiment Analysis API",
    description="Analizza il sentiment di un testo (0=negativo, 1=neutro, 2=positivo)",
    version="1.0"
)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input: TextInput):
    result = pipe(input.text)[0]
    return { "label": result["label"], "score": result["score"] }

@app.get("/")
def root():
    return {"message": "API Sentiment pronta"}

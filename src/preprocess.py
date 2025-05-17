import pandas as pd

def preprocess_sentiment140(input_path, output_path):
    df = pd.read_csv(input_path, encoding="latin-1", header=None)
    df.columns = ["target", "id", "date", "flag", "user", "text"]

    # Mappatura label 0, 2, 4 → 0, 1, 2
    df["label"] = df["target"].map({0: 0, 2: 1, 4: 2})
    df = df[["text", "label"]]

    df.to_csv(output_path, index=False)
    print(f"✅ Dati preprocessati salvati in: {output_path}")

if __name__ == "__main__":
    preprocess_sentiment140("data/raw/training.1600000.processed.noemoticon.csv", "data/processed/train.csv")

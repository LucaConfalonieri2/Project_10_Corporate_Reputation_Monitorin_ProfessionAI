import pandas as pd
from sklearn.model_selection import train_test_split
import utils
import os

def preprocess_sentiment140(input_path, train_output_path, test_output_path, test_size=0.2, random_state=42):
    df = pd.read_csv(input_path, encoding="latin-1", header=None)
    df.columns = ["target", "id", "date", "flag", "user", "text"]

    # Mappatura label 0, 2, 4 → 0, 1, 2
    df["label"] = df["target"].map({0: 0, 2: 1, 4: 2})
    df = df[["text", "label"]]

    # Suddivisione in train/test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["label"])

    # Salvataggio
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print(f"Train salvato in: {train_output_path}")
    print(f"Test salvato in: {test_output_path}")

if __name__ == "__main__":
    preprocess_sentiment140(utils.ORIGINAL_DATASET_FILE, utils.TRAIN_DATASET_FILE, utils.TEST_DATASET_FILE)




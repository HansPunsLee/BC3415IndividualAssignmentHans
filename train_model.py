from pathlib import Path
import re

import joblib
import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "London_hotel_reviews.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_MODEL_PATH = (
    MODEL_DIR / "sentence_transformer"
)

CLASSIFIER_PATH = (
    MODEL_DIR / "sentiment_classifier.pkl"
)

LABELS = [
    "negative",
    "neutral",
    "positive"
]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rating_to_sentiment(rating):
    if rating <= 2:
        return "negative"

    if rating == 3:
        return "neutral"

    return "positive"


def main():
    df = pd.read_csv(
        DATA_PATH,
        encoding="cp1252"
    )

    df = df[
        ["Review Rating", "Review Text"]
    ].dropna()

    df["Review Rating"] = pd.to_numeric(
        df["Review Rating"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Review Rating"]
    )

    df = df[
        df["Review Rating"].between(1, 5)
    ]

    df["review"] = df["Review Text"].apply(
        clean_text
    )

    df["sentiment"] = df["Review Rating"].apply(
        rating_to_sentiment
    )

    df = df[
        df["review"].str.len() > 0
    ]

    print("Dataset size:", len(df))

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"]
    )

    print("\nLoading Sentence Transformer model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("\nEncoding training reviews...")

    X_train_embeddings = embedding_model.encode(
        X_train.tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("\nEncoding test reviews...")

    X_test_embeddings = embedding_model.encode(
        X_test.tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    classifier.fit(
        X_train_embeddings,
        y_train
    )

    predictions = classifier.predict(
        X_test_embeddings
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=LABELS,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=LABELS
        )
    )

    embedding_model.save(
        str(EMBEDDING_MODEL_PATH)
    )

    joblib.dump(
        classifier,
        CLASSIFIER_PATH
    )

    print(
        "\nSaved Sentence Transformer model to:"
    )
    print(EMBEDDING_MODEL_PATH)

    print(
        "\nSaved sentiment classifier to:"
    )
    print(CLASSIFIER_PATH)


if __name__ == "__main__":
    main()
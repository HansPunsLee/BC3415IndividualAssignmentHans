from pathlib import Path
import re
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR/ "data"/"London_hotel_reviews.csv"
MODEL_DIR = BASE_DIR/ "models"


MODEL_DIR.mkdir(exist_ok=True)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rating_to_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"
    
def main():
    df = pd.read_csv(DATA_PATH, encoding="cp1252") 
    df = df[
        ["Review Rating", "Review Text"]
    ].dropna()


    df["Review Rating"] = pd.to_numeric(
        df["Review Rating"],
        errors="coerce"
    )


    df = df.dropna(subset=["Review Rating"])
    df = df[df["Review Rating"].between(1, 5)]


    df["review"] = df["Review Text"].apply(clean_text)
    df["sentiment"] = df["Review Rating"].apply(
        rating_to_sentiment
    )


    df = df[df["review"].str.len() > 0]


    print("Dataset size:", len(df))
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())
    
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"],df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"])
    
    pipeline = Pipeline([
        ("tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                max_features=100000)),
        ("classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ])
    
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy:.4f}")


    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["negative", "neutral", "positive"],
            zero_division=0
        )
    )


    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=["negative", "neutral", "positive"]
        )
    )


    model_path = MODEL_DIR / "sentiment_pipeline.pkl"


    joblib.dump(
        pipeline,
        model_path
    )


    print(f"\nSaved model to: {model_path}")



if __name__ == "__main__":
    main()
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent

EMBEDDING_MODEL_PATH = (
    BASE_DIR / "models" / "sentence_transformer"
)

CLASSIFIER_PATH = (
    BASE_DIR / "models" / "sentiment_classifier.pkl"
)


app = Flask(__name__)


EMBEDDING_MODEL = SentenceTransformer(
    str(EMBEDDING_MODEL_PATH)
)

CLASSIFIER = joblib.load(
    CLASSIFIER_PATH
)


def predict_sentiment(review):
    embedding = EMBEDDING_MODEL.encode(
        [review],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    probabilities = CLASSIFIER.predict_proba(
        embedding
    )[0]

    classes = CLASSIFIER.classes_

    best_index = probabilities.argmax()

    sentiment = str(
        classes[best_index]
    )

    confidence = float(
        probabilities[best_index]
    )

    probability_dict = {
        str(label): round(
            float(probability),
            4
        )
        for label, probability in zip(
            classes,
            probabilities
        )
    }

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": probability_dict
    }


def get_recommendation(
    sentiment,
    confidence
):
    if confidence < 0.60:
        return {
            "risk_level": "medium",
            "action": "Send for manual review",
            "reason": (
                "The model confidence is low, so the "
                "prediction should be checked by a "
                "hotel employee."
            )
        }

    if sentiment == "negative":
        return {
            "risk_level": "high",
            "action": (
                "Prioritise for service-recovery review"
            ),
            "reason": (
                "The review was classified as negative "
                "with reasonable confidence."
            )
        }

    if sentiment == "neutral":
        return {
            "risk_level": "low",
            "action": "No immediate escalation",
            "reason": (
                "The review does not contain strong "
                "positive or negative sentiment."
            )
        }

    return {
        "risk_level": "low",
        "action": "No immediate escalation",
        "reason": (
            "The review was classified as positive "
            "with reasonable confidence."
        )
    }


@app.route("/")
def home():
    return render_template(
        "index.html"
    )


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "model_type": (
            "Sentence Transformer embeddings "
            "plus Logistic Regression"
        )
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    review = data["review"].strip()

    prediction = predict_sentiment(
        review
    )

    recommendation = get_recommendation(
        prediction["sentiment"],
        prediction["confidence"]
    )

    return jsonify({
        "sentiment": prediction["sentiment"],
        "confidence": round(
            prediction["confidence"],
            4
        ),
        "probabilities": prediction[
            "probabilities"
        ],
        "explanation": {
            "keywords": [],
            "message": (
                "The sentiment was predicted using "
                "a pretrained Sentence Transformer "
                "neural network and a trained "
                "classification layer."
            )
        },
        "recommendation": recommendation
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
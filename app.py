from pathlib import Path
import re

import joblib
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR/ "models" / "sentiment_pipeline.pkl"
TEMPLATE_PATH = BASE_DIR/ "template" / "index.html"

app = Flask(__name__)

MODEL = joblib.load(MODEL_PATH)

def clean_text(text):
    text=str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_sentiment_explanation(review, sentiment):
    keyword_groups = {
        "positive": [
            "good",
            "great",
            "excellent",
            "amazing",
            "love",
            "happy",
            "fast",
            "perfect",
            "satisfied",
            "recommend",
            "friendly",
            "clean",
            "comfortable",
            "helpful",
            "beautiful",
            "wonderful"
        ],
        "negative": [
            "bad",
            "broken",
            "damaged",
            "poor",
            "late",
            "wrong",
            "disappointed",
            "terrible",
            "unusable",
            "refund",
            "dirty",
            "small",
            "expensive",
            "rude",
            "noisy",
            "uncomfortable",
            "awful",
            "disappointing"
        ],
        "neutral": [
            "arrived",
            "received",
            "ordered",
            "delivery",
            "product",
            "item",
            "room",
            "hotel",
            "stayed",
            "night",
            "breakfast",
            "location"
        ]
    }

    cleaned_review = clean_text(review)
    words_found = []

    for keyword in keyword_groups.get(sentiment, []):
        if keyword in cleaned_review:
            words_found.append(keyword)

    if sentiment == "positive":
        message = (
            "The review contains language associated with "
            "satisfaction or a favourable hotel experience."
        )
    elif sentiment == "negative":
        message = (
            "The review contains language associated with "
            "dissatisfaction, inconvenience or service problems."
        )
    else:
        message = (
            "The review appears relatively factual and does not "
            "strongly express positive or negative emotion."
        )

    return {
        "keywords": words_found,
        "message": message
    }


def get_recommendation(sentiment, confidence):
    if confidence < 0.60:
        return {
            "risk_level": "medium",
            "action": "Send for manual review",
            "reason": (
                "The model confidence is low, so the prediction "
                "should be checked by a hotel employee."
            )
        }

    if sentiment == "negative":
        return {
            "risk_level": "high",
            "action": "Prioritise for service-recovery review",
            "reason": (
                "The review is classified as negative with "
                "reasonable confidence."
            )
        }

    if sentiment == "neutral":
        return {
            "risk_level": "low",
            "action": "No immediate escalation",
            "reason": (
                "The review does not contain strong positive "
                "or negative sentiment."
            )
        }

    return {
        "risk_level": "low",
        "action": "No immediate escalation",
        "reason": (
            "The review is classified as positive with "
            "reasonable confidence."
        )
    }


@app.route("/")
def home():
    return render_template(TEMPLATE_PATH)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data["review"].strip()

    cleaned_review = clean_text(review)

    probabilities = MODEL.predict_proba(
        [cleaned_review]
    )[0]

    classes = MODEL.classes_

    best_index = probabilities.argmax()
    sentiment = str(classes[best_index])
    confidence = float(probabilities[best_index])

    probability_dict = {
        str(label): round(float(probability), 4)
        for label, probability in zip(classes, probabilities)
    }

    explanation = get_sentiment_explanation(
        review,
        sentiment
    )

    recommendation = get_recommendation(
        sentiment,
        confidence
    )

    return jsonify({
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "probabilities": probability_dict,
        "explanation": explanation,
        "recommendation": recommendation
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
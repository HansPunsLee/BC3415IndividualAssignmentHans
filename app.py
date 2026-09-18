from pathlib import Path
import json

import joblib
import ollama

from flask import (
    Flask,
    jsonify,
    render_template,
    request
)

from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent

EMBEDDING_MODEL_PATH = (
    BASE_DIR / "models" / "sentence_transformer"
)

CLASSIFIER_PATH = (
    BASE_DIR / "models" / "sentiment_classifier.pkl"
)

OLLAMA_MODEL = "llama3.2"


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


def generate_recommendation(
    review,
    sentiment,
    confidence
):
    prompt = f"""
You are a hotel customer-service triage assistant.

Customer review:
{review}

Machine-learning sentiment prediction:
{sentiment}

Machine-learning confidence:
{confidence:.4f}

Return only valid JSON with exactly these keys:
{{
  "risk_level": "low",
  "action": "one concise recommended action",
  "reason": "one or two concise sentences"
}}

The allowed risk_level values are only:
- low
- medium
- high

Rules:
- Do not invent facts.
- Do not accuse the customer or an employee.
- Do not recommend automatic refunds.
- Recommend human follow-up when the review includes a complaint.
- Recommend medium or high priority for serious service problems,
  safety issues, repeated complaints, or strongly negative experiences.
- Keep the action concise.
"""

    response = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        format="json",
        options={
            "temperature": 0.2
        }
    )

    return json.loads(
        response["response"]
    )


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
        "embedding_model": "all-MiniLM-L6-v2",
        "recommendation_model": OLLAMA_MODEL
    })


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():
    data = request.get_json()

    review = data["review"].strip()

    prediction = predict_sentiment(
        review
    )

    recommendation = generate_recommendation(
        review,
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
        "recommendation": recommendation
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
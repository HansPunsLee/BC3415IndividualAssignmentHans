import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request
import joblib
import ollama
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent

EMBEDDING_MODEL_PATH = BASE_DIR / "models" / "sentence_transformer"
CLASSIFIER_PATH = BASE_DIR / "models" / "sentiment_classifier.pkl"

OLLAMA_MODEL = "llama3.2"
OLLAMA_CLIENT = ollama.Client(host="http://127.0.0.1:11434", timeout=300)

app = Flask(__name__)

EMBEDDING_MODEL = SentenceTransformer(str(EMBEDDING_MODEL_PATH))
CLASSIFIER = joblib.load(CLASSIFIER_PATH)


@app.route("/")
def index():
    return render_template("index.html")


def predict_sentiment(review):
    embedding = EMBEDDING_MODEL.encode(
        [review], convert_to_numpy=True, normalize_embeddings=True
    )

    probabilities = CLASSIFIER.predict_proba(embedding)[0]
    classes = CLASSIFIER.classes_
    best_index = probabilities.argmax()

    sentiment = str(classes[best_index])
    confidence = float(probabilities[best_index])

    probability_dict = {
        str(label): round(float(probability), 4)
        for label, probability in zip(classes, probabilities)
    }

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": probability_dict,
    }


def generate_recommendation(review, sentiment, confidence):
    prompt = f"""You are a hotel customer-service assistant analyzing a guest review.

REVIEW TO ANALYZE:
"{review}"

PREDICTED CLASSIFICATION:
Sentiment: {sentiment}
Confidence: {confidence:.2f}

INSTRUCTIONS:
1. Extract the most praised aspect/feature in "best_feature" (e.g., "Bed comfort", "Location", "Staff", "Cleanliness"). Only output "None identified" if the review is 100% negative with zero positive points.
2. Extract the main area needing improvement in "needs_work" (e.g., "Check-in time", "Air conditioning", "Noise", "Price"). Only output "None identified" if the review is 100% glowing positive with zero complaints.
3. Assign "positive", "neutral", or "negative" to "feature_sentiment" and "needs_work_sentiment".
4. Determine "overall_sentiment" ("positive", "neutral", or "negative").
5. Write a concise 1-sentence explanation in "reason".

OUTPUT FORMAT REQUIREMENTS:
Return strictly a valid JSON object matching this schema:
{{
  "overall_sentiment": "positive",
  "best_feature": "Aspect name",
  "needs_work": "Aspect name or None identified",
  "feature_sentiment": "positive",
  "needs_work_sentiment": "negative",
  "reason": "1-sentence summary"
}}"""

    try:
        response = OLLAMA_CLIENT.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            format="json",
            options={"temperature": 0.2, "num_ctx": 2048, "num_predict": 250},
        )

        response_text = response.message.content.strip()
        return json.loads(response_text)

    except Exception as error:
        print("OLLAMA ERROR:", repr(error))
        return {
            "overall_sentiment": sentiment,
            "best_feature": "None identified",
            "needs_work": "None identified",
            "feature_sentiment": "neutral",
            "needs_work_sentiment": "neutral",
            "reason": "The aspect analysis model encountered an error."
        }

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "review" not in data:
        return jsonify({"error": "No review content provided."}), 400

    review = data["review"].strip()

    prediction = predict_sentiment(review)
    analysis = generate_recommendation(
        review, prediction["sentiment"], prediction["confidence"]
    )

    return jsonify({
        "sentiment": analysis.get("overall_sentiment", prediction["sentiment"]),
        "confidence": round(prediction["confidence"], 4),
        "probabilities": prediction["probabilities"],
        "best_feature": analysis.get("best_feature", "None identified"),
        "needs_work": analysis.get("needs_work", "None identified"),
        "feature_sentiment": analysis.get("feature_sentiment", "neutral"),
        "needs_work_sentiment": analysis.get("needs_work_sentiment", "neutral"),
        "reason": analysis.get("reason", ""),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
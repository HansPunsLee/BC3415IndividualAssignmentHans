const form = document.getElementById("review-form");
const reviewInput = document.getElementById("review");
const characterCount = document.getElementById(
    "character-count"
);
const analyseButton = document.getElementById(
    "analyse-button"
);

const results = document.getElementById("results");
const errorBox = document.getElementById("error-box");
const loadingBox = document.getElementById(
    "loading-box"
);

const sentimentValue = document.getElementById(
    "sentiment-value"
);
const confidenceValue = document.getElementById(
    "confidence-value"
);
const riskValue = document.getElementById(
    "risk-value"
);
const riskBadge = document.getElementById(
    "risk-badge"
);

const probabilityList = document.getElementById(
    "probability-list"
);
const explanationMessage = document.getElementById(
    "explanation-message"
);
const keywordSection = document.getElementById(
    "keyword-section"
);
const keywordList = document.getElementById(
    "keyword-list"
);

const recommendationAction = document.getElementById(
    "recommendation-action"
);
const recommendationReason = document.getElementById(
    "recommendation-reason"
);


reviewInput.addEventListener("input", () => {
    characterCount.textContent =
        `${reviewInput.value.length} / 5000`;
});


form.addEventListener("submit", async (event) => {
    event.preventDefault();

    clearMessages();
    setLoading(true);

    const review = reviewInput.value.trim();

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                review
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || "The analysis failed."
            );
        }

        renderResults(data);
    } catch (error) {
        showError(error.message);
    } finally {
        setLoading(false);
    }
});


function renderResults(data) {
    results.classList.remove("hidden");

    const sentiment = formatLabel(data.sentiment);
    const confidence =
        `${(data.confidence * 100).toFixed(1)}%`;

    const riskLevel = formatLabel(
        data.recommendation.risk_level
    );

    sentimentValue.textContent = sentiment;
    confidenceValue.textContent = confidence;
    riskValue.textContent = riskLevel;

    riskBadge.textContent = riskLevel;
    riskBadge.className =
        `badge ${data.recommendation.risk_level}`;

    renderProbabilities(data.probabilities);
    renderExplanation(data.explanation);

    recommendationAction.textContent =
        data.recommendation.action;

    recommendationReason.textContent =
        data.recommendation.reason;
}


function renderProbabilities(probabilities) {
    probabilityList.innerHTML = "";

    Object.entries(probabilities).forEach(
        ([label, probability]) => {
            const percentage = probability * 100;

            const row = document.createElement("div");
            row.className = "probability-row";

            row.innerHTML = `
                <div class="probability-heading">
                    <span>
                        ${formatLabel(label)}
                    </span>

                    <span>
                        ${percentage.toFixed(1)}%
                    </span>
                </div>

                <div class="progress-background">
                    <div
                        class="progress-bar"
                        style="width: ${percentage}%"
                    ></div>
                </div>
            `;

            probabilityList.appendChild(row);
        }
    );
}


function renderExplanation(explanation) {
    explanationMessage.textContent =
        explanation.message;

    keywordList.innerHTML = "";

    if (!explanation.keywords.length) {
        keywordSection.classList.add("hidden");
        return;
    }

    keywordSection.classList.remove("hidden");

    explanation.keywords.forEach((keyword) => {
        const tag = document.createElement("span");
        tag.className = "keyword";
        tag.textContent = keyword;
        keywordList.appendChild(tag);
    });
}


function formatLabel(value) {
    return value
        .replaceAll("_", " ")
        .replace(/\b\w/g, (character) =>
            character.toUpperCase()
        );
}


function clearMessages() {
    errorBox.classList.add("hidden");
    results.classList.add("hidden");
}


function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}


function setLoading(isLoading) {
    loadingBox.classList.toggle(
        "hidden",
        !isLoading
    );

    analyseButton.disabled = isLoading;

    analyseButton.textContent = isLoading
        ? "Analysing..."
        : "Analyse review";
}
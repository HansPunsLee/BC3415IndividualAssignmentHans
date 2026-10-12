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

const bestFeature = document.getElementById(
    "best-feature"
);

const featureSentiment = document.getElementById(
    "feature-sentiment"
);

const needsWork = document.getElementById(
    "needs-work"
);

const needsWorkSentiment = document.getElementById(
    "needs-work-sentiment"
);

const analysisReason = document.getElementById(
    "analysis-reason"
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

        const responseText = await response.text();

        let data;

        try {
            data = JSON.parse(responseText);
        } catch (error) {
            throw new Error(
                "The server returned an invalid response. " +
                "Check the Flask terminal."
            );
        }

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

    const sentiment = formatLabel(
        data.sentiment
    );

    const confidence =
        `${(data.confidence * 100).toFixed(1)}%`;

    sentimentValue.textContent = sentiment;
    confidenceValue.textContent = confidence;
    riskValue.textContent = sentiment;

    riskBadge.textContent = sentiment;
    riskBadge.className =
        `badge ${data.sentiment}`;

    renderProbabilities(
        data.probabilities
    );

    bestFeature.textContent =
        data.best_feature;

    featureSentiment.textContent =
        `Sentiment: ${formatLabel(
            data.feature_sentiment
        )}`;

    needsWork.textContent =
        data.needs_work;

    needsWorkSentiment.textContent =
        `Sentiment: ${formatLabel(
            data.needs_work_sentiment
        )}`;

    analysisReason.textContent =
        data.reason;
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
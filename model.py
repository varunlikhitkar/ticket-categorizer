"""
model.py
---------
Defines the NLP classification pipeline (TF-IDF + Logistic Regression),
handles training/evaluation (including a confusion matrix), and exposes
a routing function that returns a category, a confidence score, and a
human-review flag for low-confidence / edge-case tickets.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

from preprocess import clean_series, clean_text

# Below this confidence, a ticket is routed to manual review instead of
# being auto-assigned to a category. Mirrors real triage tools that
# don't trust low-confidence model output on live queues.
REVIEW_THRESHOLD = 0.60


def build_pipeline() -> Pipeline:
    """
    Construct the lightweight sklearn text-classification pipeline:
    TF-IDF vectorizer -> Logistic Regression classifier.

    Note: text is already cleaned via preprocess.clean_text() before it
    reaches this pipeline, so the vectorizer's own lowercase/stopword
    handling acts as a second safety net, not the primary cleaning step.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )),
        # Higher C = less regularization = sharper, more confident
        # probability estimates. Default C=1.0 spreads probability mass
        # too evenly across 4 classes on short ticket text.
        ("clf", LogisticRegression(max_iter=1000, C=10.0)),
    ])


def train_and_evaluate(df, test_size: float = 0.25, random_state: int = 42):
    """
    Clean the text, split the dataset, train the pipeline, and print a
    detailed evaluation: accuracy, classification report, and confusion
    matrix on the held-out test split.

    Args:
        df: DataFrame with 'text' and 'category' columns.
        test_size: fraction of data reserved for testing.
        random_state: seed for reproducible splits.

    Returns:
        The trained sklearn Pipeline (fit on the training split).
    """
    cleaned_text = clean_series(df["text"])

    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_text, df["category"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["category"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    labels = sorted(df["category"].unique())

    print("=" * 60)
    print("MODEL EVALUATION — Held-out Test Set")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(f"Rows = actual, Columns = predicted, Order = {labels}")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(cm)
    print()

    return pipeline


def route_ticket(pipeline: Pipeline, ticket_text: str) -> dict:
    """
    Predict the routing category for a single incoming ticket, with a
    confidence score and a human-review flag for low-confidence /
    edge-case tickets that don't clearly fit any category.

    Args:
        pipeline: a trained sklearn Pipeline.
        ticket_text: raw, unseen ticket text to classify.

    Returns:
        dict with keys:
            category: predicted label, or "NEEDS_REVIEW" if below threshold
            confidence: top predicted probability (0-1)
            needs_review: bool, True if confidence < REVIEW_THRESHOLD
    """
    cleaned = clean_text(ticket_text)
    probabilities = pipeline.predict_proba([cleaned])[0]
    classes = pipeline.classes_

    top_idx = probabilities.argmax()
    predicted_category = classes[top_idx]
    confidence = float(probabilities[top_idx])
    needs_review = confidence < REVIEW_THRESHOLD

    return {
        "category": "NEEDS_REVIEW" if needs_review else predicted_category,
        "confidence": confidence,
        "needs_review": needs_review,
    }
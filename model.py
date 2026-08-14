"""
model.py
---------
Defines the NLP classification pipeline (TF-IDF + Logistic Regression),
handles training/evaluation, and exposes a simple predict function for
routing new, unseen support tickets.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    """
    Construct the lightweight sklearn text-classification pipeline:
    TF-IDF vectorizer -> Logistic Regression classifier.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def train_and_evaluate(df, test_size: float = 0.25, random_state: int = 42):
    """
    Split the dataset, train the pipeline, and print a detailed
    classification report on the held-out test split.

    Args:
        df: DataFrame with 'text' and 'category' columns.
        test_size: fraction of data reserved for testing.
        random_state: seed for reproducible splits.

    Returns:
        The trained sklearn Pipeline (fit on the training split).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["category"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["category"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("=" * 60)
    print("MODEL EVALUATION — Held-out Test Set")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return pipeline


def route_ticket(pipeline: Pipeline, ticket_text: str) -> str:
    """
    Predict the routing category for a single incoming ticket.

    Args:
        pipeline: a trained sklearn Pipeline.
        ticket_text: raw ticket text to classify.

    Returns:
        Predicted category label as a string.
    """
    return pipeline.predict([ticket_text])[0]

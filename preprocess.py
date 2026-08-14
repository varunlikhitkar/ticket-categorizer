"""
preprocess.py
--------------
Explicit text-cleaning step applied to raw ticket text before it reaches
the vectorizer. Kept separate from model.py so the cleaning logic is
visible and testable on its own (rather than hidden inside sklearn's
TfidfVectorizer defaults).
"""

import re

# Lightweight stopword list — common English filler words that add noise
# but no category signal for short support-ticket text.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "this", "that",
    "these", "those", "and", "or", "but", "if", "so", "to", "of", "in",
    "on", "at", "for", "with", "about", "please", "can", "could", "would",
    "do", "does", "did", "have", "has", "had",
}


def clean_text(text: str) -> str:
    """
    Normalize a raw ticket string before vectorization:
      1. Lowercase everything.
      2. Strip punctuation / non-alphanumeric noise.
      3. Collapse extra whitespace.
      4. Remove common stopwords.

    Args:
        text: raw ticket subject/body text.

    Returns:
        Cleaned, lowercase text ready for TF-IDF vectorization.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # strip punctuation/symbols
    text = re.sub(r"\s+", " ", text).strip()   # collapse whitespace

    tokens = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(tokens)


def clean_series(text_series):
    """Apply clean_text to a pandas Series of ticket text."""
    return text_series.apply(clean_text)

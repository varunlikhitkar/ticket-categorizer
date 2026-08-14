# Auto Email / Ticket Categorizer

**AI/ML Intern Assessment — Fobes Skill Itech Pvt Ltd**

Lightweight NLP classifier that reads an incoming support ticket and routes it to the correct department in real time — `BILLING`, `TECHNICAL`, `HR`, or `GENERAL` — using TF-IDF + Logistic Regression. Includes confidence-based human-review fallback and simple keyword-based priority tagging, mirroring how a real triage tool would sit in front of a live ticket queue.

## Project Structure

    ticket_categorizer/
    ├── data_generator.py   # Builds the self-created dummy dataset (pandas)
    ├── preprocess.py        # Explicit text cleaning (lowercase, noise/stopword removal)
    ├── priority.py           # Keyword-rule based URGENT / NORMAL tagging
    ├── model.py               # sklearn pipeline: training, evaluation, confidence-based routing
    ├── main.py                 # Runs the full flow, live routing demo, and interactive CLI
    ├── requirements.txt
    └── README.md

## How to Run

1. Install dependencies:

       pip install -r requirements.txt

2. Run the pipeline:

       python main.py

This will:
- Generate a ~160-row dummy ticket dataset (40 samples/category)
- Clean the text (lowercase, strip punctuation/noise, remove stopwords)
- Train a TF-IDF + Logistic Regression pipeline (75/25 stratified split)
- Print a full evaluation: accuracy, precision/recall/F1, and a confusion matrix
- Show live routing predictions on 7 brand-new, unseen sample tickets — each with a predicted category, a confidence score, and a priority tag
- Flag low-confidence or ambiguous tickets as `NEEDS_REVIEW` instead of guessing
- Print a short reflection note on what would improve with more data/time
- Optionally launch an interactive CLI demo where you can type any ticket and see it routed instantly

## Key Design Decisions

- **TF-IDF + Logistic Regression**: lightweight, fast to train, and well-suited to short, structured text like support tickets — easy to explain and debug compared to heavier models.
- **Separate preprocessing step** (`preprocess.py`): text cleaning is explicit and testable on its own, rather than hidden inside the vectorizer's defaults.
- **Stratified train/test split**: keeps class balance consistent across the split so evaluation metrics aren't skewed by an uneven category distribution.
- **Confidence-based review threshold (60%)**: if the model's top prediction probability falls below this, the ticket is routed to `NEEDS_REVIEW` instead of being auto-assigned — a deliberate edge-case safeguard for tickets that don't clearly fit any category.
- **Priority tagging is rule-based, not ML-based**: urgency should be fast, deterministic, and auditable, so it's handled with explicit keyword matching (`priority.py`) rather than folded into the category model.

## Bonus Features Implemented

- ✅ Confidence score returned alongside every prediction
- ✅ "Needs human review" fallback below 60% confidence
- ✅ Priority tagging (URGENT / NORMAL) via keyword rules
- ✅ Interactive CLI live demo — type a ticket, get an instant routing decision
- ✅ Reflection note on potential improvements with more data/time

## Notes

- Swap `LogisticRegression` for `MultinomialNB` in `model.py` (one-line change in `build_pipeline`) if you want to compare classifiers.
- Increase `samples_per_category` in `main.py`'s `run_pipeline()` call for a larger dataset.
- Adjust `REVIEW_THRESHOLD` in `model.py` to make the human-review fallback stricter or looser.
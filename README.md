# Auto Email / Ticket Categorizer

Lightweight NLP classifier that routes support tickets into `BILLING`,
`TECHNICAL`, `HR`, or `GENERAL` using TF-IDF + Logistic Regression.

## Project Structure
```
ticket_categorizer/
├── data_generator.py   # Builds the dummy dataset (pandas)
├── model.py            # sklearn pipeline: training, evaluation, inference
├── main.py             # Runs the full flow + live routing demo
└── requirements.txt
```

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the pipeline:
   ```
   python main.py
   ```

This will:
- Generate a ~160-row dummy ticket dataset (40 samples/category)
- Train a TF-IDF + Logistic Regression pipeline (75/25 stratified split)
- Print a full classification report (accuracy, precision, recall, F1)
- Show live predictions on 6 brand-new, unseen sample tickets

## Notes
- Swap `LogisticRegression` for `MultinomialNB` in `model.py` (one-line
  change in `build_pipeline`) if you want to compare classifiers.
- Increase `samples_per_category` in `main.py` for a larger dataset.

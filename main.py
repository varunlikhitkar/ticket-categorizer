"""
main.py
--------
Entry point: builds the dummy dataset, trains the classifier, prints
the evaluation report, and demonstrates live routing predictions on
brand-new, unseen sample tickets.

Run with:  python main.py
"""

from data_generator import generate_dataset
from model import train_and_evaluate, route_ticket

# Unseen sample tickets to showcase real-time routing after training.
# Ticket IDs follow the TCK-#### format for readability in the demo output.
LIVE_SAMPLES = [
    ("TCK-2295", "I got charged twice on my card this month, please refund me."),
    ("TCK-2296", "The login page throws a 404 error every time I try to sign in."),
    ("TCK-2297", "Can you tell me about the maternity leave policy?"),
    ("TCK-2298", "What time does your support team come online?"),
    ("TCK-2299", "My invoice total doesn't match the plan I selected."),
    ("TCK-2300", "The app crashes whenever I open the reports tab."),
]


def main():
    print("Generating dummy ticket dataset...")
    df = generate_dataset(samples_per_category=40)
    print(f"Dataset ready: {len(df)} tickets across {df['category'].nunique()} categories.\n")

    print("Training TF-IDF + Logistic Regression pipeline...\n")
    pipeline = train_and_evaluate(df)

    print("=" * 60)
    print("LIVE ROUTING DEMO — Unseen Tickets")
    print("=" * 60)
    for ticket_id, ticket_text in LIVE_SAMPLES:
        predicted_category = route_ticket(pipeline, ticket_text)
        print(f"{ticket_id} · [{predicted_category:^10}] \"{ticket_text}\"")


if __name__ == "__main__":
    main()
"""
main.py
--------
Entry point: builds the dummy dataset, trains the classifier, prints
the evaluation report (including confusion matrix), and demonstrates
live routing predictions — with confidence scores, a human-review
fallback for low-confidence/edge-case tickets, and simple priority
tagging — on brand-new, unseen sample tickets.

Run with:  python main.py
Then optionally try the interactive CLI demo when prompted.
"""

from data_generator import generate_dataset
from model import train_and_evaluate, route_ticket
from priority import tag_priority

# Unseen sample tickets to showcase real-time routing after training.
# Includes one deliberately ambiguous ticket (TCK-2301) to demonstrate
# the low-confidence / human-review fallback path.
LIVE_SAMPLES = [
    ("TCK-2295", "I got charged twice on my card this month, please refund me."),
    ("TCK-2296", "The login page throws a 404 error every time I try to sign in, system is down."),
    ("TCK-2297", "Can you tell me about the maternity leave policy?"),
    ("TCK-2298", "What time does your support team come online?"),
    ("TCK-2299", "My invoice total doesn't match the plan I selected."),
    ("TCK-2300", "The app crashes whenever I open the reports tab, this is urgent."),
    ("TCK-2301", "Hey, just checking in about the thing we talked about."),  # ambiguous / edge case
]


def print_ticket_result(ticket_id: str, ticket_text: str, result: dict, priority: str):
    """Format and print a single routed ticket result."""
    label = result["category"]
    confidence_pct = f"{result['confidence'] * 100:.1f}%"
    flag = " ⚠ MANUAL REVIEW" if result["needs_review"] else ""
    print(f'{ticket_id} · [{label:^12}] ({confidence_pct} conf) · priority={priority}{flag}')
    print(f'    "{ticket_text}"')


def run_pipeline():
    """Generate data, train, evaluate, and demo routing. Returns the trained pipeline."""
    print("Generating dummy ticket dataset...")
    df = generate_dataset(samples_per_category=40)
    print(f"Dataset ready: {len(df)} tickets across {df['category'].nunique()} categories.\n")

    print("Training TF-IDF + Logistic Regression pipeline...\n")
    pipeline = train_and_evaluate(df)

    print("=" * 60)
    print("LIVE ROUTING DEMO — Unseen Tickets")
    print("=" * 60)
    for ticket_id, ticket_text in LIVE_SAMPLES:
        result = route_ticket(pipeline, ticket_text)
        priority = tag_priority(ticket_text)
        print_ticket_result(ticket_id, ticket_text, result, priority)

    return pipeline


def interactive_demo(pipeline):
    """Optional CLI demo: type a ticket, get an instant routing decision."""
    print("\n" + "=" * 60)
    print("INTERACTIVE DEMO — type a ticket to route it live (or 'quit')")
    print("=" * 60)
    while True:
        ticket_text = input("\nEnter ticket text: ").strip()
        if ticket_text.lower() in ("quit", "exit", ""):
            print("Exiting interactive demo.")
            break
        result = route_ticket(pipeline, ticket_text)
        priority = tag_priority(ticket_text)
        print_ticket_result("LIVE", ticket_text, result, priority)


def main():
    pipeline = run_pipeline()

    print("\nReflection: With more data/time, the biggest improvement would be")
    print("collecting real historical tickets instead of templated dummy text —")
    print("real data has messier phrasing and would reveal whether the ~90%")
    print("accuracy holds up, or whether the model is over-fitting to the")
    print("phrase-bank patterns used to generate this dummy dataset.")

    try:
        choice = input("\nTry the interactive live-routing demo? (y/n): ").strip().lower()
        if choice == "y":
            interactive_demo(pipeline)
    except EOFError:
        # No interactive stdin available (e.g. running in a non-interactive
        # environment) — skip gracefully instead of crashing.
        pass


if __name__ == "__main__":
    main()

"""
data_generator.py
------------------
Generates a self-created dummy dataset of support tickets labeled with
one of four routing categories: BILLING, TECHNICAL, HR, GENERAL.

The dataset is built from category-specific phrase templates that are
combined to produce varied, realistic-sounding tickets, then shuffled
into a pandas DataFrame.
"""

import random
import pandas as pd

RANDOM_SEED = 42

# Seed phrase banks per category. Each list holds realistic ticket
# snippets a user might submit to a helpdesk.
TICKET_BANK = {
    "BILLING": [
        "I was charged twice for my subscription this month.",
        "Can you explain the extra charge on my latest invoice?",
        "My credit card was declined during checkout, please help.",
        "I need a refund for an accidental purchase.",
        "The pricing plan I signed up for doesn't match what I'm being billed.",
        "Please cancel my subscription and stop auto-renewal.",
        "I haven't received my invoice for last month yet.",
        "Why did my monthly bill suddenly increase?",
        "I want to update my payment method on file.",
        "There is a discrepancy between the quoted price and the amount charged.",
        "Can I get a receipt for my last transaction?",
        "My coupon code didn't apply at checkout.",
        "I was billed after I already cancelled my plan.",
        "How do I downgrade my plan to reduce my monthly cost?",
        "I'd like to dispute a charge that appears fraudulent.",
    ],
    "TECHNICAL": [
        "The app keeps crashing every time I try to log in.",
        "I'm getting a 500 error when I upload a file.",
        "The website is loading extremely slowly today.",
        "My password reset link isn't working.",
        "The API is returning a timeout error on every request.",
        "I can't sync my data across devices anymore.",
        "The dashboard shows a blank screen after the recent update.",
        "Two-factor authentication codes are not being delivered.",
        "The mobile app freezes on the settings page.",
        "I'm unable to connect to the server from my network.",
        "Search results are not loading correctly.",
        "The export to CSV feature is broken.",
        "I keep getting logged out randomly.",
        "There's a bug causing duplicate entries in my records.",
        "The integration with our third-party tool stopped working.",
    ],
    "HR": [
        "I have a question about my paid leave balance.",
        "How do I update my emergency contact information?",
        "I need clarification on the company's remote work policy.",
        "Can you send me a copy of my offer letter?",
        "I'd like to report a workplace conduct concern.",
        "What is the process for requesting parental leave?",
        "I haven't received my payslip for this month.",
        "How do I enroll in the company health insurance plan?",
        "I want to know the procedure for resigning from my role.",
        "Can someone explain the performance review timeline?",
        "I need help updating my bank details for salary deposit.",
        "What documents are required for onboarding a new employee?",
        "I'd like to request a transfer to a different department.",
        "Is there a policy on sick leave carryover to next year?",
        "I need to schedule a meeting with HR about my benefits.",
    ],
    "GENERAL": [
        "What are your customer support working hours?",
        "Do you have a physical office I can visit?",
        "I just wanted to say thanks for the great service.",
        "Where can I find your terms and conditions?",
        "How do I get in touch with your sales team?",
        "Can you tell me more about your company's mission?",
        "I'm interested in a partnership opportunity, who should I contact?",
        "Do you offer discounts for students?",
        "What languages does your support team speak?",
        "I'd like to give some general feedback about the product.",
        "How can I unsubscribe from your newsletter?",
        "Is there a mobile app version of your service?",
        "Can you point me to your community forum?",
        "What is your company's privacy policy on data sharing?",
        "I have a general inquiry not related to billing or tech issues.",
    ],
}

# Small prefix variations to add lexical diversity without changing intent.
PREFIXES = ["", "Hi, ", "Hello team, ", "Hi there, ", "Support, ", "Hey, "]


def generate_dataset(samples_per_category: int = 40) -> pd.DataFrame:
    """
    Build a shuffled DataFrame of dummy support tickets.

    Args:
        samples_per_category: number of ticket samples to generate per
            category (sampled with replacement from the phrase bank,
            each combined with a random prefix for variety).

    Returns:
        pd.DataFrame with columns ['text', 'category'].
    """
    random.seed(RANDOM_SEED)
    rows = []

    for category, phrases in TICKET_BANK.items():
        for _ in range(samples_per_category):
            phrase = random.choice(phrases)
            prefix = random.choice(PREFIXES)
            rows.append({"text": f"{prefix}{phrase}", "category": category})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


if __name__ == "__main__":
    # Quick manual check when run directly.
    data = generate_dataset()
    print(data.head(10))
    print("\nCategory distribution:\n", data["category"].value_counts())

# verified

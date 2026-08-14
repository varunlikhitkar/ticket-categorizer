"""
priority.py
------------
Simple keyword-rule based priority tagger. Runs alongside category
classification to flag tickets that sound urgent, independent of which
department they route to.
"""

URGENT_KEYWORDS = {
    "urgent", "asap", "immediately", "down", "outage", "not working",
    "broken", "critical", "emergency", "can't access", "cannot access",
    "blocked", "failing", "failed", "crash", "crashing", "lost",
}


def tag_priority(ticket_text: str) -> str:
    """
    Assign a simple URGENT / NORMAL priority tag based on keyword match.

    This is intentionally rule-based (not ML) — priority is a fast,
    explainable triage signal that should stay deterministic and
    auditable, unlike the category prediction.

    Args:
        ticket_text: raw ticket text.

    Returns:
        "URGENT" if any urgent keyword is found, else "NORMAL".
    """
    lowered = ticket_text.lower()
    for keyword in URGENT_KEYWORDS:
        if keyword in lowered:
            return "URGENT"
    return "NORMAL"

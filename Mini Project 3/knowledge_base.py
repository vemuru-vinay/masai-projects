"""
Task 1 - FAQ Knowledge Base Foundation
SupportAI Mini Project

This module builds a small FAQ knowledge base and implements
keyword-based search over it (no LLM / API calls used here).
"""

# ---------------------------------------------------------------------------
# 1. FAQ Knowledge Base
# ---------------------------------------------------------------------------
# Each FAQ is a dictionary with: id, category, question, answer, keywords
faqs = [
    {
        "id": "faq-001",
        "category": "Account",
        "question": "How do I reset my password?",
        "answer": "Click 'Forgot Password' on the login page. Enter your "
                  "registered email address and check your inbox for a "
                  "reset link valid for 24 hours.",
        "keywords": ["password", "reset", "forgot", "login", "account"]
    },
    {
        "id": "faq-002",
        "category": "Billing",
        "question": "What is your refund policy?",
        "answer": "We offer full refunds within 30 days of purchase for "
                  "unused subscriptions. Partial refunds are available for "
                  "annual plans cancelled after 30 days.",
        "keywords": ["refund", "money back", "cancel", "billing", "policy"]
    },
    {
        "id": "faq-003",
        "category": "Billing",
        "question": "How do I update my payment method?",
        "answer": "Go to Account Settings > Billing > Payment Methods, then "
                  "click 'Add New Card' or 'Edit' next to an existing card.",
        "keywords": ["payment", "card", "billing", "update", "subscription"]
    },
    {
        "id": "faq-004",
        "category": "Technical",
        "question": "Why is the app crashing on startup?",
        "answer": "Try clearing the app cache, ensure you're on the latest "
                  "version, and restart your device. If the issue persists, "
                  "reinstall the app.",
        "keywords": ["crash", "app", "startup", "bug", "error", "technical"]
    },
    {
        "id": "faq-005",
        "category": "Account",
        "question": "How do I delete my account?",
        "answer": "Go to Account Settings > Privacy > Delete Account. Note "
                  "this action is permanent and cannot be undone.",
        "keywords": ["delete", "account", "close", "remove", "privacy"]
    },
    {
        "id": "faq-006",
        "category": "Shipping",
        "question": "How can I track my order?",
        "answer": "Once your order ships, you'll receive a tracking number "
                  "via email. You can also view tracking info under "
                  "'My Orders' in your account.",
        "keywords": ["track", "order", "shipping", "delivery", "package"]
    },
]


# ---------------------------------------------------------------------------
# 2. Search Functions
# ---------------------------------------------------------------------------

def search_by_keyword(faqs, query):
    """
    Search FAQs by matching query words against keywords, question,
    and category fields (case-insensitive).

    Ranking: FAQs are ordered by number of matching words (most first).
    Only FAQs with at least 1 match are returned.

    Args:
        faqs (list): list of FAQ dicts
        query (str): user's search text

    Returns:
        list: matching FAQ dicts, sorted by relevance (hit count desc)
    """
    # Common filler words that shouldn't count as real matches on their own
    stop_words = {"my", "is", "the", "a", "an", "to", "do", "i", "how",
                  "what", "for", "on", "in", "of", "and", "am"}

    # Break the query into lowercase, meaningful words for comparison
    query_words = [w for w in query.lower().split() if w not in stop_words]

    results = []  # will hold tuples of (hit_count, faq)

    for faq in faqs:
        # Build one combined lowercase list of searchable words per FAQ
        searchable_words = set(faq["category"].lower().split())
        searchable_words.update(faq["question"].lower().replace("?", "").split())
        for keyword in faq["keywords"]:
            searchable_words.update(keyword.lower().split())

        # Count how many query words appear in the FAQ's searchable words
        hit_count = 0
        for word in query_words:
            if word in searchable_words:
                hit_count += 1

        if hit_count > 0:
            results.append((hit_count, faq))

    # Sort by hit_count descending (most relevant first)
    results.sort(key=lambda pair: pair[0], reverse=True)

    # Return just the FAQ dicts, without the hit counts
    return [faq for hit_count, faq in results]


def get_faq_by_id(faqs, faq_id):
    """
    Return the FAQ dict matching the given id, or None if not found.

    Args:
        faqs (list): list of FAQ dicts
        faq_id (str): the id to look up, e.g. "faq-001"

    Returns:
        dict | None: the matching FAQ, or None
    """
    for faq in faqs:
        if faq["id"] == faq_id:
            return faq
    return None


def get_faqs_by_category(faqs, category):
    """
    Return all FAQs belonging to the given category (case-insensitive).

    Args:
        faqs (list): list of FAQ dicts
        category (str): category name, e.g. "Billing"

    Returns:
        list: FAQ dicts matching that category
    """
    category_lower = category.lower()
    return [faq for faq in faqs if faq["category"].lower() == category_lower]


# ---------------------------------------------------------------------------
# 3. Demonstration
# ---------------------------------------------------------------------------

def print_search_results(query):
    """Run a search for `query` and print results in a readable format."""
    print(f"Query: {query}")
    results = search_by_keyword(faqs, query)

    if not results:
        print("  No matching FAQs found.")
    else:
        for faq in results:
            print(f"  [{faq['category']}] {faq['question']}")
            print(f"  \u2192 {faq['answer']}")
    print()  # blank line for readability


if __name__ == "__main__":
    # Test queries as required by the task
    test_queries = ["forgot my password", "refund", "weather today"]

    for q in test_queries:
        print_search_results(q)

    # Extra demo: get_faq_by_id and get_faqs_by_category
    print("get_faq_by_id('faq-002'):")
    print(" ", get_faq_by_id(faqs, "faq-002"))
    print()

    print("get_faqs_by_category('billing'):")
    for faq in get_faqs_by_category(faqs, "billing"):
        print(" -", faq["question"])
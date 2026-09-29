"""
Task 3 - Intelligent FAQ Matching
SupportAI Mini Project

Builds on Task 1: reuses `faqs` and `search_by_keyword()` from
task1_faq_search.py. This module adds TF-IDF based semantic matching
(FAQMatcher) and a hybrid_search() that combines keyword search with
TF-IDF similarity -- so paraphrased questions (e.g. "I forgot my login
credentials") can still find the right FAQ even without exact keyword
overlap.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Reuse everything from Task 1 instead of redefining it
from knowledge_base import faqs, search_by_keyword


# ---------------------------------------------------------------------------
# 1. FAQMatcher Class
# ---------------------------------------------------------------------------
class FAQMatcher:
    """
    TF-IDF based semantic matcher for FAQs.

    Builds a TF-IDF index over each FAQ's question + keywords, then
    matches new queries against that index using cosine similarity.
    This catches paraphrases that plain keyword search misses, because
    TF-IDF works on word importance/overlap across the whole vocabulary
    rather than requiring exact keyword hits.
    """

    def __init__(self, faqs):
        """
        Build the TF-IDF index.

        Args:
            faqs (list): list of FAQ dicts (id, category, question,
                         answer, keywords).
        """
        self.faqs = faqs

        # Combine question + keywords into one text blob per FAQ.
        # This gives TF-IDF richer vocabulary to match against than
        # the question alone.
        self.corpus = [
            f"{faq['question']} {' '.join(faq['keywords'])}" for faq in faqs
        ]

        # Fit the vectorizer on our FAQ corpus once, up front.
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.faq_vectors = self.vectorizer.fit_transform(self.corpus)

    def match(self, query, top_k=3):
        """
        Find the top_k FAQs most similar to the query, by cosine similarity.

        Args:
            query (str): the user's question.
            top_k (int): how many results to return.

        Returns:
            list[tuple]: list of (faq_dict, confidence_score) tuples,
                         sorted by score descending. Scores are floats in
                         [0.0, 1.0], rounded to 4 decimal places.
        """
        # Transform the query using the SAME fitted vectorizer/vocabulary
        query_vector = self.vectorizer.transform([query])

        # Cosine similarity between the query and every FAQ vector
        similarities = cosine_similarity(query_vector, self.faq_vectors)[0]

        # Pair each FAQ with its similarity score
        scored_faqs = [
            (self.faqs[i], round(float(similarities[i]), 4))
            for i in range(len(self.faqs))
        ]

        # Sort by score descending, take the top_k
        scored_faqs.sort(key=lambda pair: pair[1], reverse=True)
        return scored_faqs[:top_k]

    def best_match(self, query, threshold=0.15):
        """
        Return the single best-matching FAQ if its score meets the
        threshold, otherwise None.

        Args:
            query (str): the user's question.
            threshold (float): minimum confidence score required.

        Returns:
            tuple | None: (faq_dict, confidence_score), or None if no
                          FAQ scores high enough.
        """
        top_matches = self.match(query, top_k=1)
        if not top_matches:
            return None

        best_faq, best_score = top_matches[0]
        if best_score >= threshold:
            return (best_faq, best_score)
        return None

    def explain_match(self, query):
        """
        Return a human-readable string showing the top 3 matches and
        their scores, for debugging / demonstration purposes.

        Args:
            query (str): the user's question.

        Returns:
            str: formatted explanation of the top matches.
        """
        top_matches = self.match(query, top_k=3)

        if not top_matches:
            return "No matches found."

        lines = [f"Top matches for: '{query}'"]
        for rank, (faq, score) in enumerate(top_matches, start=1):
            lines.append(f"  {rank}. [{score:.4f}] {faq['question']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Hybrid Search
# ---------------------------------------------------------------------------
def hybrid_search(faqs, query, top_k=3):
    """
    Combine keyword search (Task 1) and TF-IDF matching (Task 3) into a
    single ranked result list.

    Approach:
        - Every FAQ returned by keyword search gets a base score of 0.5
          (a simple, fixed signal that "this FAQ shares real keywords
          with the query").
        - Every FAQ gets a TF-IDF cosine-similarity score too.
        - For each FAQ, we keep the HIGHER of the two scores -- so a
          strong TF-IDF match isn't diluted by a weak keyword score,
          and vice versa.
        - Results are deduplicated by FAQ id, sorted by score descending,
          and the top_k are returned.

    Args:
        faqs (list): list of FAQ dicts.
        query (str): the user's question.
        top_k (int): how many results to return.

    Returns:
        list[tuple]: list of (faq_dict, confidence_score) tuples, sorted
                     by score descending.
    """
    # A fresh matcher each call keeps this function self-contained; for
    # heavy repeated use you'd build the FAQMatcher once and reuse it,
    # but for our small FAQ set the rebuild cost is negligible.
    matcher = FAQMatcher(faqs)

    # Track the best score seen per FAQ id
    scores_by_id = {}

    # --- Keyword search contribution ---
    keyword_results = search_by_keyword(faqs, query)
    for faq in keyword_results:
        scores_by_id[faq["id"]] = (faq, 0.5)

    # --- TF-IDF contribution ---
    tfidf_results = matcher.match(query, top_k=len(faqs))
    for faq, score in tfidf_results:
        if score <= 0:
            continue  # skip zero-similarity noise
        existing = scores_by_id.get(faq["id"])
        if existing is None or score > existing[1]:
            scores_by_id[faq["id"]] = (faq, score)

    # Sort merged results by score descending
    merged = list(scores_by_id.values())
    merged.sort(key=lambda pair: pair[1], reverse=True)

    return merged[:top_k]


# ---------------------------------------------------------------------------
# 3. Comparison Demonstration
# ---------------------------------------------------------------------------
def print_comparison(matcher, query):
    """
    Run the same query through keyword search, TF-IDF matching, and
    hybrid search, and print all three results side by side.
    """
    print(f"Query: {query}")

    # --- Keyword search (Task 1) ---
    print("[Keyword Search]")
    keyword_results = search_by_keyword(faqs, query)
    if not keyword_results:
        print("  (no results)")
    else:
        for i, faq in enumerate(keyword_results, start=1):
            print(f"  {i}. {faq['question']}")

    # --- TF-IDF matching (Task 3) ---
    print("[TF-IDF Matching]")
    tfidf_results = matcher.match(query, top_k=3)
    if not tfidf_results:
        print("  (no results)")
    else:
        for i, (faq, score) in enumerate(tfidf_results, start=1):
            print(f"  {i}. [{score:.4f}] {faq['question']}")

    # --- Hybrid search ---
    print("[Hybrid Search]")
    hybrid_results = hybrid_search(faqs, query, top_k=3)
    if not hybrid_results:
        print("  (no results)")
    else:
        for i, (faq, score) in enumerate(hybrid_results, start=1):
            print(f"  {i}. [{score:.4f}] {faq['question']}")

    # --- Best match summary ---
    best = matcher.best_match(query)
    if best:
        best_faq, best_score = best
        print(f"Best match: {best_faq['question']} (confidence: {best_score:.4f})")
    else:
        print("Best match: none (below confidence threshold)")

    print()


if __name__ == "__main__":
    # Build the TF-IDF matcher once, over Task 1's FAQ set
    matcher = FAQMatcher(faqs)

    # The 3 required comparison queries
    test_queries = [
        "I forgot my login credentials",
        "Can I get my money back?",
        "package delivery time",
    ]

    for query in test_queries:
        print_comparison(matcher, query)

    # Extra demo of explain_match()
    print(matcher.explain_match("app keeps crashing"))
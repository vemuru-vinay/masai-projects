# SupportAI — Project README

A conversational helpdesk agent built in four stages: a static FAQ knowledge
base, an LLM layer that rephrases answers naturally, a smarter semantic
matcher, and finally a full conversational agent that ties everything
together. Each task builds directly on the one before it — nothing is
reimplemented twice.

---

## Overall Workflow

```
User's message
      │
      ▼
Task 3: hybrid_search()
   ├── uses Task 1: search_by_keyword()   (exact keyword overlap)
   └── uses Task 3: FAQMatcher (TF-IDF)   (semantic similarity)
      │
      ▼
Best matching FAQ + confidence score
      │
      ▼
Task 2: LLMClient.generate_faq_response()
   → sends the matched FAQ + user's question to Groq's LLM
   → LLM rephrases the official answer naturally, staying grounded in it
      │
      ▼
Task 4: SupportAgent
   → wraps the whole pipeline above
   → tracks conversation history, confidence, and escalation state
   → falls back to "connect me to a human" when nothing matches well
```

In short: **Task 1 finds candidates → Task 3 ranks them intelligently →
Task 2 makes the answer sound human → Task 4 wraps it all into an agent
you can actually talk to.**

---

## File 1: `task1_faq_search.py` — FAQ Knowledge Base & Keyword Search

**What it does:** This is the data foundation. Nothing here calls an LLM or
any external API — it's pure Python data + basic string matching.

**What's inside:**
- `faqs` — a list of 6 FAQ entries (dictionaries), each with an `id`,
  `category`, `question`, `answer`, and a list of `keywords`. Covers
  Account, Billing, Technical, and Shipping topics.
- `search_by_keyword(faqs, query)` — splits the query into words, strips
  out filler words (like "what", "your", "is") so they don't cause false
  matches, then counts how many meaningful words appear in each FAQ's
  category/question/keywords. Returns FAQs sorted by number of matching
  words, most relevant first.
- `get_faq_by_id(faqs, faq_id)` — simple lookup by unique ID.
- `get_faqs_by_category(faqs, category)` — returns every FAQ in a given
  category, case-insensitive.
- Demo block — runs 3 test queries and prints results.

**Why it matters:** Every other file imports `faqs` and
`search_by_keyword` from here instead of redefining them. This is the
single source of truth for the FAQ data.

**Limitation this file has (fixed by Task 3):** keyword search only works
if the user's words literally overlap with an FAQ's keywords. A
paraphrased question like "I forgot my login credentials" won't match
"password reset" unless the exact word "forgot" happens to appear as a
keyword.

---

## File 2: `task2_llm_integration.py` — LLM Integration (Groq)

**What it does:** Adds a "voice" to the raw FAQ answers. Instead of
dumping the literal FAQ text at the user, it asks an LLM to rephrase it
in a warm, natural, support-agent tone — without letting the LLM invent
facts that aren't in the FAQ.

**What's inside:**
- `LLMClient` class — a thin wrapper around Groq's chat-completions API
  (`https://api.groq.com/openai/v1/chat/completions`), which is
  OpenAI-compatible.
  - `__init__(api_key, model)` — stores credentials and the model name
    (defaults to `openai/gpt-oss-20b`).
  - `generate(prompt, system_message, max_tokens)` — the low-level call:
    builds the request, sends it, and returns the assistant's text.
    Wraps failures (bad key, no internet, malformed response) in clear
    `RuntimeError` messages instead of letting the program crash.
  - `generate_faq_response(user_question, faq_entry)` — the high-level
    call used everywhere else. Builds a prompt containing the matched
    FAQ's question + official answer + the user's actual question, and
    sends it with a system prompt instructing the LLM to:
    - answer **only** from the given FAQ content (no hallucination),
    - use a friendly, professional tone,
    - stay under 150 words,
    - and if the FAQ doesn't fully answer the question, say so and
      point the user to human support instead of guessing.
- Demo block — runs 2 different questions through the full pipeline
  (search → LLM rephrase) and prints the results.

**Why it matters:** This is the only file that talks to an external LLM.
Everything else either produces FAQ matches (Tasks 1 & 3) or consumes
LLM output (Task 4) — the actual API call logic lives only here.

**Note:** The original brief suggested OpenRouter; this version uses
Groq instead (a separate, free, OpenAI-compatible LLM API) — only the
base URL, API key env var, and model name differ. The class structure
and prompt logic are unchanged.

---

## File 3: `task3_intelligent_matching.py` — Intelligent FAQ Matching

**What it does:** Fixes Task 1's biggest weakness — paraphrased
questions. Uses TF-IDF (a way of scoring how important each word is
across the FAQ set) and cosine similarity (a way of measuring how close
two pieces of text are) to catch matches that share *meaning* even when
they don't share exact words.

**What's inside:**
- `FAQMatcher` class — builds and queries a TF-IDF index.
  - `__init__(faqs)` — combines each FAQ's question + keywords into one
    text blob, then fits a `TfidfVectorizer` over all of them.
  - `match(query, top_k)` — converts the query into the same TF-IDF
    space, computes cosine similarity against every FAQ, and returns the
    top-k `(faq, score)` pairs, scores rounded to 4 decimal places.
  - `best_match(query, threshold)` — returns just the single best match,
    or `None` if it doesn't clear the confidence threshold (default
    0.15).
  - `explain_match(query)` — human-readable printout of the top 3
    matches and their scores, useful for debugging/demoing.
- `hybrid_search(faqs, query, top_k)` — the function Task 4 actually
  uses. Combines both matching strategies:
  - keyword hits (Task 1) get a flat base score of 0.5,
  - TF-IDF hits get their real similarity score,
  - for each FAQ, the **higher** of the two scores wins,
  - results are deduplicated and sorted, top-k returned.
- Demo block — compares keyword search vs. TF-IDF vs. hybrid search
  side-by-side on 3 test queries, showing how TF-IDF catches matches
  keyword search misses entirely.

**Why it matters:** `hybrid_search()` becomes the single matching
function that Task 4's agent calls — it's strictly better than either
method alone, since it keeps keyword search's precision on exact
matches while adding TF-IDF's ability to generalize to paraphrases.

---

## File 4: `task4_helpdesk_agent.py` — Complete Helpdesk Agent

**What it does:** The finished product. Wraps Tasks 1–3 into a stateful
conversational agent that can hold a multi-turn conversation, track how
confident it is in each answer, and hand off to a human when it's stuck.

**What's inside:**
- `ConversationTurn` — a small dataclass representing one message in the
  conversation (`role`, `content`, and optionally `faq_id` /
  `confidence` for assistant turns).
- `SupportAgent` class — the orchestrator. Does **not** reimplement any
  search or LLM logic; it only calls into Tasks 1–3.
  - `__init__(faqs, llm_client, confidence_threshold)` — holds the FAQ
    data, the `LLMClient` from Task 2, and the minimum score required to
    trust a match.
  - `handle_message(user_message)` — the core loop:
    1. logs the user's message,
    2. calls `hybrid_search()` (Task 3) to find the best FAQ,
    3. if the score clears the threshold, calls
       `generate_faq_response()` (Task 2) to phrase a natural answer,
    4. if not, returns a fallback message offering escalation,
    5. logs the assistant's reply (with FAQ ID + confidence) and
       returns it.
  - `escalate(reason)` — marks the session escalated and returns a
    confirmation message with a randomly generated mock ticket ID
    (e.g. `TICKET-48291`) and an estimated response time.
  - `get_conversation_summary()` — prints the full conversation so far,
    including which FAQ (if any) and what confidence score backed each
    answer.
  - `reset()` — clears history and escalation state for a fresh session.
- `run_chat(agent)` — an interactive command-line loop supporting free
  text, `history`, `escalate`, `reset`, and `quit`, with a welcome
  banner and an automatic escalation suggestion after 3 consecutive
  low-confidence answers.
- `run_demo(agent)` — a scripted, non-interactive walkthrough of the 4
  required scenarios (clear question, paraphrased question, out-of-scope
  question, escalation request) — useful for a clean, repeatable
  demonstration without manual typing.

**Why it matters:** This is where the project stops being separate
building blocks and becomes an actual product — a single `SupportAgent`
object that a real chat UI (or this CLI) can sit on top of.

---

## Bug fix along the way

While testing Task 4 end-to-end, a false-positive match surfaced: the
query *"What are your office hours in Tokyo?"* was incorrectly matching
the refund FAQ, because the word **"your"** appeared in both the query
and in "What is **your** refund policy?" and wasn't being filtered out
as a filler word. Task 1's `search_by_keyword()` stopword list was
expanded to catch this (adding words like "your", "are", "you", "can",
etc.), which fixed the false match across all downstream tasks (3 and 4)
without needing to touch their code.

---

## Summary Table

| File | Role | Talks to LLM? | Depends on |
|---|---|---|---|
| `task1_faq_search.py` | FAQ data + keyword search | No | — |
| `task2_llm_integration.py` | Natural-language answer generation | Yes (Groq) | Task 1 |
| `task3_intelligent_matching.py` | Smarter matching (TF-IDF + hybrid) | No | Task 1 |
| `task4_helpdesk_agent.py` | Full conversational agent | Yes (via Task 2) | Tasks 1, 2, 3 |
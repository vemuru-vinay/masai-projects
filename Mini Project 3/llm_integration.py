"""
Task 2 - Groq LLM Integration
SupportAI Mini Project

Builds on Task 1: reuses `faqs` and `search_by_keyword()` from
task1_faq_search.py. This module adds an LLMClient that calls the
Groq API to turn raw FAQ answers into natural, conversational
support responses -- while staying grounded in the official FAQ text.

Note on provider choice: the original task brief suggested OpenRouter,
but this implementation uses Groq (https://console.groq.com) instead --
a separate, free, OpenAI-compatible LLM API. Only the base URL, auth
env var, and model name differ; the request/response format and all
class logic are unchanged.

Setup:
    1. pip install requests
    2. Get a free API key at https://console.groq.com/keys
    3. Set your API key as an environment variable (never hard-code it):
         export GROQ_API_KEY="gsk_..."      (macOS/Linux)
         setx GROQ_API_KEY "gsk_..."        (Windows)
    4. python task2_llm_integration.py
"""

import os
import requests
from pathlib import Path

# Reuse everything from Task 1 instead of redefining it
from knowledge_base import faqs, search_by_keyword



# ---------------------------------------------------------------------------
# 1. API Configuration
# ---------------------------------------------------------------------------
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# llama-3.3-70b-versatile was deprecated by Groq; gpt-oss-20b is the
# recommended, fast, free-tier-friendly replacement.
DEFAULT_MODEL = "openai/gpt-oss-20b"


def load_env_file(path=".env"):
    """
    Load simple KEY=VALUE pairs from a local .env file into os.environ.
    Existing environment variables are left unchanged.
    """
    env_path = Path(__file__).resolve().parent / path
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key:
            os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# 2. LLMClient Class
# ---------------------------------------------------------------------------
class LLMClient:
    """
    A small wrapper around the Groq chat-completions API.

    Groq uses an OpenAI-compatible request/response format, so this
    class can be pointed at any other Groq-hosted model simply by
    changing `model`.
    """

    def __init__(self, api_key, model=DEFAULT_MODEL):
        """
        Args:
            api_key (str): Groq API key (read from env var by caller,
                            never hard-coded here).
            model (str): Groq model identifier, e.g. "openai/gpt-oss-20b".
        """
        self.api_key = api_key
        self.model = model

    def generate(self, prompt, system_message=None, max_tokens=512):
        """
        Send a chat completion request to Groq and return the
        assistant's reply text.

        Args:
            prompt (str): the user-role message content.
            system_message (str | None): optional system-role instruction.
            max_tokens (int): cap on response length.

        Returns:
            str: the assistant's response text.

        Raises:
            RuntimeError: with a clear, descriptive message if the request
                          fails (network error, bad API key, bad response, etc).
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                GROQ_URL, headers=headers, json=payload, timeout=30
            )
            response.raise_for_status()  # raises HTTPError for 4xx/5xx
        except requests.exceptions.HTTPError as e:
            # e.g. 401 invalid API key, 429 rate limit, 400 bad request
            raise RuntimeError(
                f"Groq API request failed "
                f"(status {response.status_code}): {response.text}"
            ) from e
        except requests.exceptions.RequestException as e:
            # e.g. no internet connection, timeout, DNS failure
            raise RuntimeError(f"Could not reach Groq API: {e}") from e

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(
                f"Unexpected response format from Groq: {e}"
            ) from e

    def generate_faq_response(self, user_question, faq_entry):
        """
        Turn a matched FAQ entry into a natural, conversational answer to
        the user's specific question, grounded strictly in the FAQ content.

        Args:
            user_question (str): what the user actually typed/asked.
            faq_entry (dict): the matched FAQ dict (id, category, question,
                               answer, keywords).

        Returns:
            str: the LLM-generated support response.
        """
        # System prompt: see "Prompt Engineering" notes below for rationale.
        system_message = (
            "You are SupportAI, a friendly and professional customer support "
            "agent. You must answer the user's question using ONLY the "
            "information in the FAQ_ANSWER provided below -- do not invent, "
            "assume, or add any facts that are not stated there. "
            "Rephrase the FAQ content in a warm, conversational, helpful tone, "
            "as if speaking directly to the customer. "
            "Keep your response under 150 words. "
            "If the FAQ_ANSWER does not fully address the user's specific "
            "question, answer the part you can from the FAQ, then politely "
            "say that for anything beyond that they should contact human "
            "support -- do not guess or fill the gap with invented details."
        )

        prompt = (
            f"FAQ_QUESTION: {faq_entry['question']}\n"
            f"FAQ_ANSWER: {faq_entry['answer']}\n\n"
            f"USER_QUESTION: {user_question}\n\n"
            "Write the support response now."
        )

        return self.generate(prompt, system_message=system_message, max_tokens=200)


# ---------------------------------------------------------------------------
# 3. Prompt Engineering Notes (documentation, not executed)
# ---------------------------------------------------------------------------
"""
System prompt design rationale:

1. Grounding / anti-hallucination:
   - The system message explicitly restricts the model to "ONLY the
     information in the FAQ_ANSWER provided" and forbids inventing or
     assuming facts. The FAQ question/answer are clearly labelled
     (FAQ_QUESTION / FAQ_ANSWER) so the model can distinguish "official
     source material" from the "user's question", reducing the chance it
     blends in outside knowledge.

2. Tone and length:
   - Explicit instruction for a "friendly and professional" tone and a
     hard cap ("under 150 words") keeps responses concise and consistent,
     and max_tokens=200 acts as a hard backstop on the API side too.

3. Handling partial/insufficient FAQ coverage:
   - If the FAQ doesn't fully answer the user's specific question, the
     model is instructed to answer only the part it can support from the
     FAQ, then direct the user to human support rather than guessing.
     This prevents confident-sounding hallucinated answers when the
     knowledge base has a gap.
"""


# ---------------------------------------------------------------------------
# 4. Demonstration
# ---------------------------------------------------------------------------
def answer_user_question(client, user_question):
    """
    Full pipeline: search FAQs -> pick best match -> generate LLM response.
    Prints the matched FAQ and the SupportAI response.
    """
    print(f"Question: {user_question}")

    matches = search_by_keyword(faqs, user_question)
    if not matches:
        print("No matching FAQ found -- cannot generate a grounded answer.\n")
        return

    best_match = matches[0]  # top-ranked result
    print(f"Matched FAQ: {best_match['question']}")

    try:
        answer = client.generate_faq_response(user_question, best_match)
        print("SupportAI Response:")
        print(answer)
    except RuntimeError as e:
        # Clear, descriptive error instead of a crash
        print(f"[Error generating response: {e}]")

    print()


if __name__ == "__main__":
    load_env_file()

    # Read the API key from an environment variable -- never hard-coded
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "ERROR: Please set the GROQ_API_KEY environment variable "
            "before running this script."
        )

    client = LLMClient(api_key=api_key, model=DEFAULT_MODEL)

    # Test with at least 2 different questions
    test_questions = [
        "I can't remember my login password",
        "Can I get my money back for a subscription I don't use anymore?",
    ]

    for question in test_questions:
        answer_user_question(client, question)

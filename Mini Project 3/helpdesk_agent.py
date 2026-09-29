"""
Task 4 - Complete Helpdesk Agent
SupportAI Mini Project

Final task: integrates everything built so far.
    - Task 1 (task1_faq_search.py)        -> faqs, search_by_keyword
    - Task 2 (task2_llm_integration.py)    -> LLMClient (Groq-backed)
    - Task 3 (task3_intelligent_matching.py) -> hybrid_search, FAQMatcher

SupportAgent purely orchestrates these -- it does not reimplement
search, matching, or LLM logic.
"""

import os
import random
from dataclasses import dataclass

from knowledge_base import faqs
from llm_integration import LLMClient, load_env_file
from intelligent_matching import hybrid_search


# ---------------------------------------------------------------------------
# 1. ConversationTurn dataclass
# ---------------------------------------------------------------------------
@dataclass
class ConversationTurn:
    """A single turn in the conversation (either the user or the agent)."""
    role: str        # "user" or "assistant"
    content: str
    faq_id: str = None
    confidence: float = None


# ---------------------------------------------------------------------------
# 2. SupportAgent Class
# ---------------------------------------------------------------------------
class SupportAgent:
    """
    Orchestrates the full SupportAI pipeline:
    hybrid_search (Task 3) -> LLMClient.generate_faq_response (Task 2),
    with conversation tracking and escalation to human support.
    """

    def __init__(self, faqs, llm_client, confidence_threshold=0.15):
        """
        Args:
            faqs (list): FAQ knowledge base from Task 1.
            llm_client (LLMClient): configured LLM client from Task 2.
            confidence_threshold (float): minimum hybrid_search score
                                           required to trust a match.
        """
        self.faqs = faqs
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold

        self.history = []          # list[ConversationTurn]
        self.escalated = False
        self.low_confidence_streak = 0  # for chat-interface escalation nudge

    def handle_message(self, user_message):
        """
        Process one user message end-to-end and return the agent's
        text response.

        Flow:
            1. Record the user's message.
            2. Search for the best FAQ match via hybrid_search (Task 3).
            3. If confident enough, ask the LLM (Task 2) to phrase a
               grounded answer.
            4. Otherwise, return a fallback + escalation offer.
            5. Record the assistant's turn (with faq_id/confidence) and
               return the response text.

        Args:
            user_message (str): what the user typed.

        Returns:
            str: the agent's response text.
        """
        # 1. Record user turn
        self.history.append(ConversationTurn(role="user", content=user_message))

        # 2. Search using hybrid search from Task 3
        results = hybrid_search(self.faqs, user_message, top_k=1)

        if results and results[0][1] >= self.confidence_threshold:
            # 3. Confident match -> generate a grounded LLM response
            best_faq, confidence = results[0]
            try:
                response_text = self.llm_client.generate_faq_response(
                    user_message, best_faq
                )
            except RuntimeError as e:
                # LLM call failed -- degrade gracefully instead of crashing
                response_text = (
                    f"(LLM unavailable, showing raw FAQ answer)\n"
                    f"{best_faq['answer']}\n[Error: {e}]"
                )

            faq_id = best_faq["id"]
            self.low_confidence_streak = 0  # reset streak on a good answer
        else:
            # 4. No confident match -> fallback + escalation offer
            response_text = (
                "I don't have information about that in my knowledge base. "
                "Would you like me to connect you with a human support agent? "
                "(type 'escalate' to do so)"
            )
            faq_id = None
            confidence = 0.0
            self.low_confidence_streak += 1

        # 5. Record assistant turn
        self.history.append(
            ConversationTurn(
                role="assistant",
                content=response_text,
                faq_id=faq_id,
                confidence=round(confidence, 4),
            )
        )

        return response_text

    def escalate(self, reason="User requested human support"):
        """
        Mark the session as escalated and return a confirmation message
        with a mock ticket ID.

        Args:
            reason (str): why the escalation happened (for logging/summary).

        Returns:
            str: confirmation message including ticket ID and ETA.
        """
        self.escalated = True
        ticket_id = f"TICKET-{random.randint(10000, 99999)}"

        confirmation = (
            "Your request has been escalated to our support team.\n"
            f"Ticket ID: {ticket_id}\n"
            "Estimated response time: within 4 business hours."
        )

        self.history.append(
            ConversationTurn(role="assistant", content=confirmation)
        )

        return confirmation

    def get_conversation_summary(self):
        """
        Return a formatted summary of the whole conversation so far,
        including FAQ IDs and confidence scores for assistant turns.

        Returns:
            str: formatted conversation history.
        """
        if not self.history:
            return "No conversation yet."

        lines = ["--- Conversation Summary ---"]
        for turn in self.history:
            if turn.role == "user":
                lines.append(f"User: {turn.content}")
            else:
                meta = ""
                if turn.faq_id is not None:
                    meta = f" [faq: {turn.faq_id}, confidence: {turn.confidence}]"
                lines.append(f"Assistant{meta}: {turn.content}")

        lines.append(f"Escalated: {self.escalated}")
        return "\n".join(lines)

    def reset(self):
        """Clear conversation history and reset escalation/streak state."""
        self.history = []
        self.escalated = False
        self.low_confidence_streak = 0


# ---------------------------------------------------------------------------
# 3. Interactive Chat Interface
# ---------------------------------------------------------------------------
def print_banner():
    """Display the welcome banner."""
    print("╔══════════════════════════════════════════╗")
    print("║       SupportAI — Helpdesk Agent          ║")
    print("╚══════════════════════════════════════════╝")


def run_chat(agent):
    """
    Run the interactive command-line chat loop.

    Supported inputs:
        - any text  -> sent to agent.handle_message()
        - "history" -> print agent.get_conversation_summary()
        - "escalate"-> call agent.escalate()
        - "reset"   -> call agent.reset()
        - "quit"    -> exit the loop
    """
    print_banner()

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        command = user_input.lower()

        if command == "quit":
            print("Thank you for using SupportAI. Goodbye!")
            break

        elif command == "history":
            print(agent.get_conversation_summary())

        elif command == "escalate":
            print("SupportAI:")
            print(agent.escalate())

        elif command == "reset":
            agent.reset()
            print("SupportAI: Conversation has been reset.")

        else:
            response = agent.handle_message(user_input)

            # Find the confidence/faq_id of the turn we just created
            last_turn = agent.history[-1]
            if last_turn.faq_id:
                print(
                    f"SupportAI [confidence: {last_turn.confidence:.2f}, "
                    f"faq: {last_turn.faq_id}]:"
                )
            else:
                print(f"SupportAI [confidence: 0.00]:")
            print(response)

            # After 3 consecutive low-confidence responses, nudge escalation
            if agent.low_confidence_streak >= 3:
                print(
                    "SupportAI: It looks like I'm having trouble answering "
                    "your questions. Would you like to escalate to human "
                    "support? (type 'escalate')"
                )

        print()  # blank line between turns


# ---------------------------------------------------------------------------
# 4. End-to-End Demonstration (non-interactive, scripted)
# ---------------------------------------------------------------------------
def run_demo(agent):
    """
    Scripted walkthrough of the 4 required scenarios, so grading doesn't
    depend on manual typing. Prints each step the same way the live chat
    interface would.
    """
    print_banner()

    scenarios = [
        ("How do I change my password?", "handle"),          # 1. clear FAQ question
        ("I forgot my login credentials", "handle"),          # 2. paraphrased question
        ("What are your office hours in Tokyo?", "handle"),   # 3. outside FAQ scope
        ("escalate", "escalate"),                              # 4. escalation request
    ]

    for user_text, action in scenarios:
        print(f"You: {user_text}")

        if action == "escalate":
            print("SupportAI:")
            print(agent.escalate())
        else:
            response = agent.handle_message(user_text)
            last_turn = agent.history[-1]
            if last_turn.faq_id:
                print(
                    f"SupportAI [confidence: {last_turn.confidence:.2f}, "
                    f"faq: {last_turn.faq_id}]:"
                )
            else:
                print("SupportAI [confidence: 0.00]:")
            print(response)

        print()

    print("You: quit")
    print("Thank you for using SupportAI. Goodbye!")
    print()

    # Show the full summary at the end too
    print(agent.get_conversation_summary())


if __name__ == "__main__":

    load_env_file()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "ERROR: Please set the GROQ_API_KEY environment variable "
            "before running this script."
        )

    llm_client = LLMClient(api_key=api_key)
    agent = SupportAgent(faqs, llm_client, confidence_threshold=0.15)

    # Run the scripted 4-scenario demo (set RUN_INTERACTIVE=1 to chat live instead)
    if os.environ.get("RUN_INTERACTIVE") == "1":
        run_chat(agent)
    else:
        run_demo(agent)
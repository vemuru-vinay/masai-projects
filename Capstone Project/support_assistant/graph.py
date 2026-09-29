import os
import json
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

from ingestion import retrieve_top_chunks
from prompt_template import build_prompt
from schemas import AskResponse

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours",
]

DIRECT_ANSWER_CANNED_RESPONSE = "I can only answer questions about Zepto policies right now."


class GraphState(TypedDict):
    query: str
    intent: Optional[str]
    retrieved_chunks: Optional[List[dict]]
    answer: Optional[str]
    sources: Optional[List[str]]
    confidence: Optional[float]


def is_mock_mode():
    return os.environ.get("MOCK_LLM", "1") == "1"


def classify_intent(state: GraphState) -> GraphState:
    query_lower = state["query"].lower()

    if is_mock_mode():
        matched = any(keyword in query_lower for keyword in POLICY_KEYWORDS)
        intent = "policy_question" if matched else "general_question"
    else:
        intent = classify_intent_with_llm(state["query"])

    state["intent"] = intent
    return state


def classify_intent_with_llm(query):
    prompt = (
        "Classify the following customer query as exactly one of: "
        "policy_question or general_question. "
        f"Query: {query}\nRespond with only the label."
    )
    label = call_real_llm(prompt).strip().lower()
    if "policy" in label:
        return "policy_question"
    return "general_question"


def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]
    retrieved_chunks = retrieve_top_chunks(query, n_results=3)
    state["retrieved_chunks"] = retrieved_chunks

    if is_mock_mode():
        top_chunk = retrieved_chunks[0]
        top_chunk_snippet = top_chunk["text"][:200]
        state["answer"] = f"Based on the retrieved context: {top_chunk_snippet}"
        state["sources"] = [chunk["chunk_id"] for chunk in retrieved_chunks]
        state["confidence"] = 1.0
    else:
        context_text = "\n".join(
            f"[{chunk['chunk_id']}]: {chunk['text']}" for chunk in retrieved_chunks
        )
        prompt = build_prompt(query, context_text)
        parsed = call_real_llm_with_validation(prompt, retrieved_chunks)
        state["answer"] = parsed["answer"]
        state["sources"] = parsed["sources"]
        state["confidence"] = parsed["confidence"]

    return state


def direct_answer(state: GraphState) -> GraphState:
    if is_mock_mode():
        state["answer"] = DIRECT_ANSWER_CANNED_RESPONSE
        state["sources"] = []
        state["confidence"] = 1.0
    else:
        prompt = build_prompt(state["query"], "No retrieval performed for general questions.")
        parsed = call_real_llm_with_validation(prompt, [])
        state["answer"] = parsed["answer"]
        state["sources"] = parsed["sources"]
        state["confidence"] = parsed["confidence"]

    return state


def route_from_classification(state: GraphState) -> str:
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"


def call_real_llm(prompt):
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


def call_real_llm_with_validation(prompt, retrieved_chunks, max_retries=2):
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            raw_output = call_real_llm(prompt)
            parsed_json = json.loads(raw_output)
            validated = AskResponse(**parsed_json)
            return validated.model_dump()
        except Exception as error:
            last_error = error
            prompt = (
                prompt
                + f"\n\nYour previous output was invalid JSON or failed schema validation "
                  f"({error}). Return ONLY a valid JSON object matching the required schema."
            )

    return {
        "answer": f"Error: could not produce a valid response after {max_retries + 1} attempts ({last_error}).",
        "sources": [chunk["chunk_id"] for chunk in retrieved_chunks],
        "confidence": 0.0,
    }


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_from_classification,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


def run_query(query):
    app = build_graph()
    initial_state: GraphState = {
        "query": query,
        "intent": None,
        "retrieved_chunks": None,
        "answer": None,
        "sources": None,
        "confidence": None,
    }
    final_state = app.invoke(initial_state)
    return AskResponse(
        answer=final_state["answer"],
        sources=final_state["sources"],
        confidence=final_state["confidence"],
    )


if __name__ == "__main__":
    print(run_query("What is the delivery fee for small orders?"))
    print(run_query("What's the capital of France?"))

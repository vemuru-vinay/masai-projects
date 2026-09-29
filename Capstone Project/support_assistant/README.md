# Module 3 — Support Assistant

## What this does
A small RAG service for Zepto customer support: 8 policy documents are embedded and stored in ChromaDB, a LangGraph flow classifies each incoming query and either retrieves grounded context or answers directly, and a FastAPI `/ask` endpoint returns a schema-validated JSON response. The entire graded path runs offline with `MOCK_LLM=1` (the default) — no API key, no signup, no network call to any LLM provider.

## Files
- `docs/doc_01.txt` … `docs/doc_08.txt` — the 8 corpus documents, copied verbatim from the assignment.
- `ingestion.py` — loads the docs, chunks them (one chunk per document, since each is already a single coherent policy statement under 700 characters), embeds each chunk with `sentence-transformers/all-MiniLM-L6-v2`, and stores the embeddings in a persistent ChromaDB collection (`zepto_policies`). Also exposes `retrieve_top_chunks(query, n_results=3)` used at query time.
- `prompt_template.py` — the structured prompt template (role–context–task–format–length skeleton, a negative constraint, and a few-shot example), used only by the optional `MOCK_LLM=0` path.
- `graph.py` — the LangGraph `StateGraph`: a `TypedDict` state and the 3 required nodes (`classify_intent`, `retrieve_and_answer`, `direct_answer`), wired with a conditional edge from `classify_intent`.
- `schemas.py` — the Pydantic `AskRequest` / `AskResponse` models (`answer: str`, `sources: List[str]`, `confidence: float`).
- `main.py` — the FastAPI app exposing `POST /ask`.
- `Dockerfile`, `requirements.txt` — for local container build/run.

## How to run
```
pip install -r requirements.txt
python ingestion.py
uvicorn main:app --host 0.0.0.0 --port 7860
```
`ingestion.py` only needs to be run once — it builds the persistent ChromaDB store in `chroma_store/`. After that, `main.py` reuses it on every request. The first run of `ingestion.py` (or the first API call, if you skip that step) downloads the `all-MiniLM-L6-v2` weights from Hugging Face — this requires internet access on that one occasion; the model is then cached locally and no further network access is needed for the graded `MOCK_LLM=1` path.

### Example calls
With the server running (`MOCK_LLM` left at its default), test with:
```
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d "{\"query\": \"What is the delivery fee for small orders?\"}"
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d "{\"query\": \"What is the capital of France?\"}"
```
The first should route through `retrieve_and_answer` (it contains the keyword "delivery") and return a `"Based on the retrieved context: ..."` answer with populated `sources`. The second should route through `direct_answer` and return the fixed canned string with empty `sources`.

## Json Responses
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": [
    "doc_01_chunk_0",
    "doc_05_chunk_0",
    "doc_02_chunk_0"
  ],
  "confidence": 1
}

{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1
}

## Docker
```
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```
The container serves `/ask` on port 7860 with `MOCK_LLM=1` by default (set in the Dockerfile), so it needs no API key to run the graded path.

## Architecture: ingestion → embedding → retrieval → generation

**Ingestion.** `ingestion.load_documents()` reads all 8 `.txt` files from `docs/`. `ingestion.chunk_document()` then splits each into chunks — in practice each document stays as a single chunk, since none exceeds the 700-character threshold used here; this keeps each policy statement intact rather than splitting it mid-sentence.

**Embedding.** `ingestion.embed_texts()` calls a `sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")` instance to turn each chunk's text into a vector, entirely locally with no API call. `ingestion.build_chroma_collection()` stores these vectors, their raw text, and a `source_doc` metadata tag in a persistent ChromaDB collection named `zepto_policies` (on disk under `chroma_store/`).

**Routing.** A query arrives at the FastAPI `/ask` endpoint (`main.py`), which invokes the compiled LangGraph app (`graph.build_graph()`). The graph's entry node, `classify_intent`, lowercases the query and checks it against a fixed keyword list (`delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`). A conditional edge (`route_from_classification`) then sends the state to either `retrieve_and_answer` or `direct_answer` based on that classification. This routing decision itself does not depend on `MOCK_LLM` — only what happens *inside* the chosen node does.

**Retrieval.** For `policy_question` queries, `retrieve_and_answer` calls `ingestion.retrieve_top_chunks()`, which embeds the query with the same MiniLM model and asks ChromaDB for the top-3 chunks by cosine similarity. This retrieval step always runs for real, in both `MOCK_LLM` states, since it needs no LLM and no external API.

**Generation.** This is the only stage that branches on `MOCK_LLM`:
- **Default (`MOCK_LLM=1` or unset) — the graded baseline:** `retrieve_and_answer` builds the answer directly in code as `f"Based on the retrieved context: {top_chunk_snippet}"` (first ~200 characters of the top chunk), with `sources` set to the retrieved chunk IDs and `confidence` fixed at `1.0`. `direct_answer` returns a fixed canned string with empty `sources`. No LLM is called in either branch.
- **Optional (`MOCK_LLM=0`):** `retrieve_and_answer` instead builds the structured prompt from `prompt_template.py`, injects the retrieved chunks as context, and calls a real LLM (Groq's free tier, `llama-3.1-8b-instant`, via `call_real_llm`). The raw output is parsed as JSON and validated against the `AskResponse` schema; if validation fails, `call_real_llm_with_validation` retries up to 2 more times with a corrective instruction appended to the prompt before falling back to a clearly marked error response. `direct_answer` follows the same real-LLM-plus-validation path but without retrieval.

## Notes on the offline/mock design
Every acceptance criterion in this module is satisfiable with `MOCK_LLM` left at its default — the keyword classifier, the canned answer templates, and the deterministic schema population all require no network access beyond the one-time model download. The `MOCK_LLM=0` branch is fully implemented in `graph.py` (including the JSON-validation retry loop) but is an optional, ungraded extension per the assignment — it requires a `GROQ_API_KEY` environment variable to run.

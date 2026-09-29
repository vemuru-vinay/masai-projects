import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts):
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True).tolist()


def load_documents():
    paths = sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt")))
    documents = []
    for path in paths:
        doc_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        documents.append({"id": doc_id, "text": text})
    return documents


def chunk_document(doc_id, text, max_chars=700):
    if len(text) <= max_chars:
        return [{"chunk_id": f"{doc_id}_chunk_0", "text": text}]
    chunks = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk_text = text[start:end]
        chunks.append({"chunk_id": f"{doc_id}_chunk_{index}", "text": chunk_text})
        start = end
        index += 1
    return chunks


def build_chroma_collection(reset=False):
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() > 0 and not reset:
        return collection

    documents = load_documents()
    all_chunk_ids = []
    all_chunk_texts = []
    all_metadatas = []

    for doc in documents:
        chunks = chunk_document(doc["id"], doc["text"])
        for chunk in chunks:
            all_chunk_ids.append(chunk["chunk_id"])
            all_chunk_texts.append(chunk["text"])
            all_metadatas.append({"source_doc": doc["id"]})

    embeddings = embed_texts(all_chunk_texts)

    collection.add(
        ids=all_chunk_ids,
        embeddings=embeddings,
        documents=all_chunk_texts,
        metadatas=all_metadatas,
    )

    return collection


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def retrieve_top_chunks(query, n_results=3):
    collection = get_collection()
    query_embedding = embed_texts([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    chunks = []
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(ids)):
        chunks.append({
            "chunk_id": ids[i],
            "text": documents[i],
            "source_doc": metadatas[i]["source_doc"],
            "distance": distances[i],
        })

    return chunks


if __name__ == "__main__":
    collection = build_chroma_collection(reset=True)
    print(f"Collection '{COLLECTION_NAME}' built with {collection.count()} chunks")
    sample = retrieve_top_chunks("What is the delivery fee for small orders?")
    for chunk in sample:
        print(chunk["chunk_id"], chunk["source_doc"], chunk["distance"])

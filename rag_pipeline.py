"""
rag_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Semantic RAG pipeline for RetailMind AI.

Architecture:
  1. Embedding    – sentence-transformers/all-mpnet-base-v2
                    (768-dim, SBERT fine-tuned, best general-purpose model)
  2. Vector Store – FAISS IndexFlatIP (inner-product / cosine similarity)
                    with L2 normalisation for cosine semantics
  3. Retrieval    – Top-K nearest neighbours with similarity score
  4. Generation   – Extractive answer synthesis from retrieved chunks
                    (no LLM API key required; fully self-contained)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import pickle

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from data_loader import load_product_data

# ── Paths ──────────────────────────────────────────────────────────────────────
INDEX_DIR   = "faiss_index"
INDEX_PATH  = os.path.join(INDEX_DIR, "shopify_products.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "shopify_chunks.pkl")
META_PATH   = os.path.join(INDEX_DIR, "shopify_meta.pkl")

# ── Embedding model ────────────────────────────────────────────────────────────
# all-mpnet-base-v2  → 768-dim, state-of-the-art semantic similarity
# Outperforms MiniLM on retrieval benchmarks (BEIR, MTEB)
# Suitable for university/industry-grade RAG systems
EMBED_MODEL = "all-MiniLM-L6-v2"

# ── Singletons (loaded once, reused across requests) ──────────────────────────
_model  = None
_index  = None
_chunks = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_model():
    global _model
    if _model is None:
        print(f"[rag] Loading embedding model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)
        print(f"[rag] Model loaded — embedding dimension: {_model.get_sentence_embedding_dimension()}")
    return _model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Index build & load
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_index(chunks):
    """
    Encodes all product chunks and persists a FAISS IndexFlatIP index.

    We use Inner Product (cosine similarity) rather than L2 distance:
      - Normalise all vectors to unit length first
      - IndexFlatIP then gives exact cosine similarity scores in [0, 1]
      - Higher score = more semantically similar
    This is the standard approach in production semantic search systems.
    """
    model = _get_model()

    texts = [c["text"] for c in chunks]
    print(f"[rag] Encoding {len(texts):,} product chunks ...")
    print(f"[rag] Model   : {EMBED_MODEL}")
    print(f"[rag] This may take 3-8 minutes on first run (CPU). Please wait.")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit-normalise for cosine similarity
    )
    embeddings = embeddings.astype("float32")

    dim   = embeddings.shape[1]

    # IndexFlatIP = exact brute-force inner product (cosine after normalisation)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"\n[rag] Index saved   → {INDEX_PATH}")
    print(f"[rag] Total vectors : {index.ntotal:,}   dim={dim}")
    print(f"[rag] Chunks saved  → {CHUNKS_PATH}")


def load_index():
    """Loads the persisted FAISS index and chunks from disk."""
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    print(f"[rag] Index loaded from disk — {index.ntotal:,} vectors")
    return index, chunks


def ensure_index():
    """
    Idempotent: build index on first run, load from disk on subsequent runs.
    Called once at server startup in a background thread.
    """
    global _index, _chunks
    if _index is not None:
        return   # already loaded

    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        _index, _chunks = load_index()
    else:
        print("[rag] No index found. Building from scratch...")
        chunks = load_product_data()
        build_index(chunks)
        _index, _chunks = load_index()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Retrieval
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def retrieve(query, top_k=5):
    """
    Encode the query and return the top_k most semantically similar
    product chunks with their cosine similarity scores.

    Parameters
    ----------
    query  : str   natural-language product question
    top_k  : int   number of results to return (default 5)

    Returns
    -------
    list[dict]  — sorted by descending similarity score
        text     : full chunk text
        source   : display label
        score    : cosine similarity  (0.0 – 1.0)
        meta     : dict with title, brand, category, new
    """
    ensure_index()
    model = _get_model()

    # Encode query with same normalisation as corpus
    q_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    similarities, indices = _index.search(q_vec, top_k)

    results = []
    for sim, idx in zip(similarities[0], indices[0]):
        if idx == -1:
            continue
        chunk = _chunks[idx]
        results.append({
            "text":   chunk["text"],
            "source": chunk["source"],
            "score":  float(sim),          # cosine similarity in [0, 1]
            "meta":   chunk.get("meta", {}),
        })

    # Already sorted by FAISS (highest score first)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Answer generation (extractive synthesis)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _similarity_label(score):
    """Human-readable label for cosine similarity."""
    if score >= 0.80:
        return "Excellent match"
    elif score >= 0.65:
        return "Strong match"
    elif score >= 0.50:
        return "Good match"
    elif score >= 0.35:
        return "Moderate match"
    else:
        return "Weak match"


def _extract_description(text):
    """Pull the Description line from a chunk, stripping the prefix."""
    for line in text.split("\n"):
        if line.startswith("Description:"):
            return line[len("Description:"):].strip()
    return ""


def generate_answer(query, context_chunks):
    """
    Synthesises a structured, readable answer from retrieved chunks.

    Format:
      • Summary sentence (how many results, query echo)
      • Per-result block: product name, brand, category, condition,
        relevance score, description excerpt
      • No hallucination — every word comes from the actual dataset
    """
    if not context_chunks:
        return (
            "No relevant products were found for your query.\n\n"
            "Try rephrasing — for example, describe the product use case, "
            "material, or target audience."
        )

    n = len(context_chunks)
    lines = [
        f"**Found {n} relevant product{'s' if n > 1 else ''} for:** \"{query}\"",
        "",
        "---",
        "",
    ]

    for i, chunk in enumerate(context_chunks, 1):
        meta  = chunk.get("meta", {})
        title    = meta.get("title",    chunk["source"])
        brand    = meta.get("brand",    "—")
        category = meta.get("category", "—")
        is_new   = meta.get("new",      True)
        score    = chunk["score"]
        desc     = _extract_description(chunk["text"])

        # Truncate description for readability
        if len(desc) > 400:
            desc = desc[:397] + "..."

        condition = "New" if is_new else "Pre-owned"
        rel_label = _similarity_label(score)

        lines.append(f"**{i}. {title}**")
        lines.append(f"Brand: {brand}  ·  Condition: {condition}  ·  Relevance: {rel_label} ({score:.2f})")
        lines.append(f"Category: {category}")
        if desc:
            lines.append(f"Details: {desc}")
        lines.append("")

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ask(query, top_k=5):
    """
    Full RAG pipeline:  query → embed → retrieve → synthesise → return

    Parameters
    ----------
    query  : str   user question
    top_k  : int   number of products to retrieve (default 5)

    Returns
    -------
    dict with keys:
        answer   : str   formatted answer string
        sources  : list  product source labels
        chunks   : list  full retrieved chunk dicts (for debugging)
    """
    chunks  = retrieve(query, top_k=top_k)
    answer  = generate_answer(query, chunks)
    sources = [c["source"] for c in chunks]
    return {
        "answer":  answer,
        "sources": sources,
        "chunks":  chunks,
    }


# ── CLI test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "comfortable running shoes for women",
        "organic cotton baby clothing",
        "wireless noise cancelling headphones",
    ]
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {q}")
        print('='*60)
        result = ask(q, top_k=3)
        print(result["answer"])
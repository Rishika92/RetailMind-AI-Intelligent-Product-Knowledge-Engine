"""
rag_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Semantic RAG pipeline for RetailMind AI.

INDEXING STRATEGY — HNSW (Hierarchical Navigable Small World)
─────────────────────────────────────────────────────────────
We use hnswlib instead of FAISS flat search for three reasons:

1. SCALABILITY  — HNSW scales to millions of vectors with O(log n)
   query time. FAISS flat is O(n) — fine at 1k chunks, slow at 1M.

2. ACCURACY     — HNSW is an approximate nearest neighbour (ANN)
   algorithm that consistently achieves >99% recall vs exact search
   at 5–10× the speed at large scales (BEIR / ANN benchmarks).

3. INDUSTRY STANDARD — Used in production by Pinecone, Weaviate,
   Qdrant, ElasticSearch, and OpenSearch for vector retrieval.
   Understanding HNSW is an expected university-level competency.

HNSW KEY PARAMETERS:
  M          = 32   (max edges per node; higher → better recall, more RAM)
  ef_construction= 200  (build-time search depth; higher → better index quality)
  ef_search  = 100  (query-time search depth; higher → better recall)
  space      = 'cosine'  (normalised inner-product similarity)

EMBEDDING MODEL:
  all-MiniLM-L6-v2  (384-dim, 22 MB, fast CPU inference)
  Fine-tuned for semantic similarity on 1B+ sentence pairs (SBERT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import pickle
import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer
from data_loader import load_product_data


# ── Paths ──────────────────────────────────────────────────────────────
INDEX_DIR    = "faiss_index"          # kept same folder name for compat
INDEX_PATH   = os.path.join(INDEX_DIR, "hnsw_support.bin")
CHUNKS_PATH  = os.path.join(INDEX_DIR, "support_chunks.pkl")

# ── Model ──────────────────────────────────────────────────────────────
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM    = 384

# ── HNSW hyper-parameters ──────────────────────────────────────────────
HNSW_M              = 32    # graph connectivity (16–64 typical range)
HNSW_EF_CONSTRUCTION= 200   # index build quality (100–400)
HNSW_EF_SEARCH      = 100   # query-time recall depth (50–200)
HNSW_SPACE          = "cosine"

# ── Singletons ─────────────────────────────────────────────────────────
_model  = None
_index  = None
_chunks = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Embedding model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_model():
    global _model
    if _model is None:
        print(f"[rag] Loading embedding model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)
        dim = _model.get_sentence_embedding_dimension()
        print(f"[rag] Model ready — embedding dim: {dim}")
    return _model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HNSW index build & persistence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_index(chunks):
    """
    Encodes all chunks and builds an HNSW index.

    HNSW builds a multi-layer proximity graph during indexing:
      - Lower layers  → coarse navigation (few high-degree hub nodes)
      - Upper layers  → fine-grained neighbourhood connections
    At query time, the algorithm greedily navigates the graph from a
    random entry point, finding approximate nearest neighbours in
    O(log n) hops without scanning the full corpus.

    The index is saved as a binary file for instant reload.
    """
    model  = _get_model()
    texts  = [c["text"] for c in chunks]
    n      = len(texts)

    print(f"\n[rag] ── Building HNSW Index ─────────────────────────────")
    print(f"[rag]   Chunks       : {n:,}")
    print(f"[rag]   Embedding dim: {EMBED_DIM}")
    print(f"[rag]   M            : {HNSW_M}  (graph connectivity)")
    print(f"[rag]   ef_construct : {HNSW_EF_CONSTRUCTION}  (build quality)")
    print(f"[rag]   Space        : {HNSW_SPACE}  (cosine similarity)")
    print(f"[rag] Encoding {n:,} chunks with {EMBED_MODEL}...")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # required for cosine space in hnswlib
    ).astype("float32")

    # Initialise HNSW index
    index = hnswlib.Index(space=HNSW_SPACE, dim=EMBED_DIM)
    index.init_index(
        max_elements=n,
        ef_construction=HNSW_EF_CONSTRUCTION,
        M=HNSW_M,
    )

    # Add all vectors — hnswlib uses integer IDs (0..n-1)
    index.add_items(embeddings, list(range(n)))

    # Set query-time ef (higher = better recall, slower query)
    index.set_ef(HNSW_EF_SEARCH)

    # Persist
    os.makedirs(INDEX_DIR, exist_ok=True)
    index.save_index(INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"[rag] ✓ HNSW index saved  → {INDEX_PATH}")
    print(f"[rag] ✓ Chunks saved      → {CHUNKS_PATH}")
    print(f"[rag] ✓ Index size        : {index.get_current_count():,} vectors")
    print(f"[rag] ─────────────────────────────────────────────────────\n")

    return index


def load_index(n_chunks):
    """Loads the persisted HNSW index from disk."""
    index = hnswlib.Index(space=HNSW_SPACE, dim=EMBED_DIM)
    index.load_index(INDEX_PATH, max_elements=n_chunks)
    index.set_ef(HNSW_EF_SEARCH)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    print(f"[rag] HNSW index loaded from disk — {index.get_current_count():,} vectors")
    return index, chunks


def ensure_index():
    """
    Idempotent startup routine:
      - If index files exist on disk → load instantly
      - Otherwise → download dataset, embed, build & save HNSW index
    Called once in a background thread when Flask starts.
    """
    global _index, _chunks
    if _index is not None:
        return

    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        # Load chunks first to get count for hnswlib max_elements
        with open(CHUNKS_PATH, "rb") as f:
            _chunks = pickle.load(f)
        index = hnswlib.Index(space=HNSW_SPACE, dim=EMBED_DIM)
        index.load_index(INDEX_PATH, max_elements=len(_chunks))
        index.set_ef(HNSW_EF_SEARCH)
        _index = index
        print(f"[rag] HNSW index loaded — {_index.get_current_count():,} vectors")
    else:
        print("[rag] No index found — building from scratch...")
        _chunks = load_product_data()
        _index  = build_index(_chunks)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Semantic retrieval
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def retrieve(query, top_k=5):
    """
    Encode the query, run HNSW approximate nearest-neighbour search,
    and return the top_k most semantically similar support documents.

    HNSW returns distances in cosine space (0 = identical, 2 = opposite).
    We convert to similarity: sim = 1 - (distance / 2)  → [0, 1].

    Parameters
    ----------
    query  : str   natural-language customer question
    top_k  : int   number of results (default 5)

    Returns
    -------
    list[dict] — sorted by descending similarity
    """
    ensure_index()
    model = _get_model()

    q_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # knn_query returns (labels, distances)
    labels, distances = _index.knn_query(q_vec, k=min(top_k, _index.get_current_count()))

    results = []
    for label, dist in zip(labels[0], distances[0]):
        # Convert cosine distance → similarity score [0,1]
        similarity = float(1.0 - dist / 2.0)
        chunk = _chunks[label]
        results.append({
            "text":   chunk["text"],
            "source": chunk["source"],
            "score":  round(similarity, 4),
            "meta":   chunk.get("meta", {}),
        })

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Answer generation (extractive synthesis)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _relevance_label(score):
    if score >= 0.80: return "Excellent match"
    if score >= 0.65: return "Strong match"
    if score >= 0.50: return "Good match"
    if score >= 0.35: return "Moderate match"
    return "Weak match"


def generate_answer(query, context_chunks):
    """
    Synthesises a structured answer from the retrieved support documents.
    Every word comes directly from the indexed dataset — no hallucination.

    Format:
      Header line → per-result block (area, category, product, answer)
    """
    if not context_chunks:
        return (
            "No relevant support documents were found for your query.\n\n"
            "Try rephrasing — for example, mention the product category, "
            "the specific issue (warranty, return, compatibility), or "
            "describe the problem in more detail."
        )

    n = len(context_chunks)
    lines = [
        f"**Found {n} relevant support document{'s' if n > 1 else ''} for:** \"{query}\"",
        "",
        "---",
        "",
    ]

    for i, chunk in enumerate(context_chunks, 1):
        meta       = chunk.get("meta", {})
        issue_area = meta.get("issue_area", "—")
        category   = meta.get("category",   "—")
        issue      = meta.get("issue",       "—")
        product    = meta.get("product",     "—")
        complexity = meta.get("complexity",  "—")
        answer     = meta.get("answer",      "")
        score      = chunk["score"]
        rel_label  = _relevance_label(score)

        # Truncate very long answers for readability
        if len(answer) > 450:
            answer = answer[:447] + "..."

        lines.append(f"**{i}. {issue}**")
        lines.append(f"Area: {issue_area}  ·  Category: {category}  ·  Product: {product}")
        lines.append(f"Complexity: {complexity}  ·  Relevance: {rel_label} ({score:.2f})")
        lines.append(f"Answer: {answer}")
        lines.append("")

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ask(query, top_k=5):
    """
    Full RAG pipeline: query → embed → HNSW retrieve → synthesise → return

    Returns dict with keys: answer, sources, chunks
    """
    chunks  = retrieve(query, top_k=top_k)
    answer  = generate_answer(query, chunks)
    sources = [c["source"] for c in chunks]
    return {
        "answer":  answer,
        "sources": sources,
        "chunks":  chunks,
    }


# ── CLI smoke test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What is the return policy for electronics?",
        "How do I claim warranty for a defective product?",
        "Can I cancel my order after it has been shipped?",
    ]
    for q in test_queries:
        print(f"\n{'='*65}")
        print(f"QUERY: {q}")
        print("=" * 65)
        result = ask(q, top_k=3)
        print(result["answer"])
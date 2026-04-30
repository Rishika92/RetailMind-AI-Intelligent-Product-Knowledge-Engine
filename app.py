"""
app.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Flask backend for RetailMind AI — Semantic Product Knowledge Engine.

Problem Statement:
  A mid-sized e-commerce company maintains thousands of product
  descriptions, specifications, and FAQs. Customer support agents
  frequently need to search through multiple files to answer queries
  about warranty coverage, compatibility details, or return conditions.
  The current keyword-based search fails due to inconsistent wording.
  This system provides semantic search + RAG over a pre-indexed
  customer-support knowledge base using HNSW vector indexing.

Routes
──────
GET  /              → Chat UI
POST /ask           → { query, top_k? } → { answer, sources }
GET  /index-status  → Index readiness + stats
GET  /health        → Liveness probe
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import threading
import time
from flask import Flask, render_template, request, jsonify
from rag_pipeline import ask, ensure_index, _chunks

app = Flask(__name__)

# ── Index loading state ─────────────────────────────────────────────────
_index_ready  = False
_index_error  = None
_index_stats  = {}
_load_start   = time.time()


def _preload_index():
    global _index_ready, _index_error, _index_stats
    try:
        print("\n[app] ── Building HNSW index in background ──")
        ensure_index()
        _index_stats = {
            "documents":  len(_chunks) if _chunks else 0,
            "model":      "sentence-transformers/all-MiniLM-L6-v2",
            "index_type": "HNSW (hnswlib) — M=32, ef=200",
            "dataset":    "rjac/e-commerce-customer-support-qa (MIT)",
            "topics":     "Returns · Warranty · Shipping · Payments · Account",
            "elapsed_s":  round(time.time() - _load_start, 1),
        }
        _index_ready = True
        print(f"[app] Index ready ✓  ({_index_stats['documents']:,} documents indexed)")
    except Exception as exc:
        _index_error = str(exc)
        print(f"[app] Index build failed: {exc}")


# Start background loader immediately
threading.Thread(target=_preload_index, daemon=True).start()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Routes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "index_ready": _index_ready})


@app.route("/index-status")
def index_status():
    if _index_error:
        return jsonify({"ready": False, "error": _index_error}), 500
    elapsed = round(time.time() - _load_start, 1)
    return jsonify({
        "ready":   _index_ready,
        "stats":   _index_stats,
        "elapsed": elapsed,
    })


@app.route("/ask", methods=["POST"])
def ask_route():
    # Guard: index not ready
    if not _index_ready:
        if _index_error:
            return jsonify({"error": f"Index failed: {_index_error}"}), 500
        elapsed = round(time.time() - _load_start, 1)
        return jsonify({
            "error": (
                f"Knowledge base is still loading ({elapsed}s elapsed). "
                "Please wait a moment and try again."
            )
        }), 503

    # Parse request
    data  = request.get_json(force=True, silent=True) or {}
    query = str(data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400
    if len(query) > 600:
        return jsonify({"error": "Query too long (max 600 characters)."}), 400

    top_k = int(data.get("top_k", 5))
    top_k = max(1, min(top_k, 10))

    try:
        result = ask(query, top_k=top_k)
        return jsonify({
            "answer":  result["answer"],
            "sources": result["sources"],
        })
    except Exception as exc:
        app.logger.exception("Error in /ask")
        return jsonify({"error": "Internal error. Please try again."}), 500


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  RetailMind AI  —  Semantic Product Knowledge Engine")
    print("  Problem: Warranty · Returns · Compatibility · FAQs")
    print("  Dataset: rjac/e-commerce-customer-support-qa (1k docs)")
    print("  Index  : HNSW via hnswlib (M=32, cosine similarity)")
    print("  Model  : sentence-transformers/all-MiniLM-L6-v2 (384-dim)")
    print("  URL    : http://127.0.0.1:5000")
    print("=" * 65 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
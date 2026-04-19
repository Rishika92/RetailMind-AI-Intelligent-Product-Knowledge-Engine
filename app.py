"""
app.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Flask backend for RetailMind AI — Semantic Product Knowledge Engine.

Routes
------
GET  /              → Chat UI (index.html)
POST /ask           → Accepts {query} JSON, returns RAG answer
GET  /index-status  → Returns index readiness + stats
GET  /health        → Liveness probe

Startup behaviour
-----------------
On first request the FAISS index is built in a background daemon thread
so that Flask starts immediately. The /index-status endpoint is polled
by the frontend every 2.5 s and turns green when the index is ready.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import threading
import time

from flask import Flask, render_template, request, jsonify
from rag_pipeline import ask, ensure_index, _chunks

app = Flask(__name__)

# ── Index loading state ────────────────────────────────────────────────────────
_index_ready  = False
_index_error  = None
_index_stats  = {}       # populated once index is built
_load_start   = time.time()


def _preload_index():
    global _index_ready, _index_error, _index_stats
    try:
        print("\n[app] ── Pre-loading FAISS index in background ──")
        ensure_index()
        _index_stats = {
            "products": len(_chunks) if _chunks else 0,
            "model":    "sentence-transformers/all-mpnet-base-v2",
            "dataset":  "Shopify/product-catalogue (Apache-2.0)",
            "elapsed_s": round(time.time() - _load_start, 1),
        }
        _index_ready = True
        print(f"[app] Index ready ✓  ({_index_stats['products']:,} products indexed)")
    except Exception as exc:
        _index_error = str(exc)
        print(f"[app] Index build failed: {exc}")


# Launch background loader immediately at import time
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
    # ── Guard: index not ready ────────────────────────────────────────
    if not _index_ready:
        if _index_error:
            return jsonify({"error": f"Index failed: {_index_error}"}), 500
        elapsed = round(time.time() - _load_start, 1)
        return jsonify({
            "error": (
                f"Index is still loading ({elapsed}s elapsed). "
                "Please wait a moment and try again."
            )
        }), 503

    # ── Parse request ─────────────────────────────────────────────────
    data  = request.get_json(force=True, silent=True) or {}
    query = str(data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    if len(query) > 600:
        return jsonify({"error": "Query too long (max 600 characters)."}), 400

    top_k = int(data.get("top_k", 5))
    top_k = max(1, min(top_k, 10))   # clamp to [1, 10]

    # ── Run RAG pipeline ─────────────────────────────────────────────
    try:
        result = ask(query, top_k=top_k)
        return jsonify({
            "answer":  result["answer"],
            "sources": result["sources"],
        })
    except Exception as exc:
        app.logger.exception("Error in /ask route")
        return jsonify({"error": "An internal error occurred. Please try again."}), 500


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  RetailMind AI  —  Semantic Product Knowledge Engine")
    print("  Dataset  : Shopify/product-catalogue  (48k products)")
    print("  Embedding: sentence-transformers/all-mpnet-base-v2")
    print("  Index    : FAISS IndexFlatIP (cosine similarity)")
    print("  URL      : http://127.0.0.1:5000")
    print("=" * 62 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
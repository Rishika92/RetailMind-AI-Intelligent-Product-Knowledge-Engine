# RetailMind AI — Semantic Product Knowledge Engine

A semantic search + RAG system for e-commerce customer support.
Built with Flask, HNSW (hnswlib), Sentence-Transformers, and a real customer support dataset.

---

## Problem Statement

A mid-sized e-commerce company stores thousands of product descriptions, specifications, and FAQs across multiple files. Customer support agents struggle to find accurate answers about warranty coverage, return conditions, and compatibility details. The existing keyword-based search fails due to inconsistent wording. This system solves that using semantic search — understanding the *meaning* of a query, not just matching words.

---

## What It Does

- Accepts natural language queries from support agents
- Semantically searches a pre-indexed knowledge base of 1,000 real support documents
- Returns the top 5 most relevant answers with relevance scores
- Covers: Returns, Warranty, Order Cancellation, Shipping, Payments, Compatibility

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Flask 3.0 |
| Vector Index | HNSW via hnswlib (M=32, cosine similarity) |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| Dataset | rjac/e-commerce-customer-support-qa (MIT, 1k docs) |
| Frontend | HTML + CSS + Vanilla JavaScript |

---

## Project Structure

```
retailmind-ai/
├── app.py               ← Flask server and API routes
├── data_loader.py       ← Loads HuggingFace dataset and builds chunks
├── rag_pipeline.py      ← HNSW index, embedding, retrieval, answer generation
├── requirements.txt     ← All dependencies
├── templates/
│   └── index.html       ← Chat UI
├── static/
│   └── style.css        ← Styling
└── faiss_index/         ← Auto-created on first run (stores HNSW index)
```

---

## Setup & Run

**Step 1 — Create virtual environment**
```bash
python -m venv venv
```

**Step 2 — Activate it**
```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 — Run the app**
```bash
python app.py
```

**Step 5 — Open in browser**
```
http://127.0.0.1:5000
```

---

## First Run Notes

- On first run, the app downloads the dataset and builds the HNSW index in the background
- This takes about **1–3 minutes** on CPU
- The status indicator in the UI turns **green** when the index is ready
- On all future runs, the index loads from disk in under 3 seconds

---

## How Semantic Search Works

Unlike keyword search, this system converts both documents and queries into 384-dimensional vectors using a fine-tuned SBERT model. The HNSW index finds documents with the closest meaning in vector space — not just matching words.

```
User query  →  384-dim vector  →  HNSW graph search  →  Top 5 similar docs  →  Answer
```

This means a query like *"item stopped working after a week"* correctly retrieves warranty and defective product policies — even though none of those exact words appear in the query.

---

## Sample Questions to Try

```
What is the return policy for electronics?
How do I claim warranty for a defective product?
Can I cancel my order after it has shipped?
My product is not compatible with my device
How long does a refund take to process?
Product damaged during delivery, what to do?
Payment failed but amount was deducted
```

---

## Dataset

**rjac/e-commerce-customer-support-qa**
- 1,000 real customer support conversations
- License: MIT
- Source: https://huggingface.co/datasets/rjac/e-commerce-customer-support-qa
- Topics: Returns, Warranty, Shipping, Payments, Account, Cancellations

---

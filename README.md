# RetailMind AI — Semantic Product Knowledge Engine

> A university & industry-grade **Retrieval-Augmented Generation (RAG)** system for intelligent product search, built with Flask, FAISS, Sentence-Transformers, and the Shopify product-catalogue dataset.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Dataset](#dataset)
5. [Project Structure](#project-structure)
6. [Setup & Installation](#setup--installation)
7. [Running the Application](#running-the-application)


---

## Project Overview

RetailMind AI solves a real problem faced by e-commerce companies: **customer support agents must search through thousands of product documents to answer detailed questions**. Traditional keyword search fails when users phrase queries differently from how products are described.

This system uses **semantic search** — understanding the *meaning* of a query — combined with **RAG (Retrieval-Augmented Generation)** to retrieve and synthesise accurate answers from a product knowledge base.

### Key Capabilities

- Natural language product search across 48,300 real Shopify products
- Semantic similarity matching (not just keyword matching)
- Cosine similarity scoring with human-readable relevance labels
- Brand, category, and condition metadata surfaced per result
- Real-time chat interface with typing indicators and status polling
- Persistent FAISS index — built once, reused on every restart

---

## Architecture

```
User Query (natural language)
        │
        ▼
┌─────────────────────────────┐
│   Flask Web Server (app.py) │  ← Serves UI + REST API
└──────────────┬──────────────┘
               │  POST /ask
               ▼
┌─────────────────────────────┐
│  Embedding Layer            │
│  all-mpnet-base-v2 (768-dim)│  ← Encodes query → float32 vector
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  FAISS IndexFlatIP          │  ← Cosine similarity search
│  (5,000 product vectors)    │     Returns top-K nearest neighbours
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Answer Synthesis           │  ← Extractive: no hallucination,
│  (rag_pipeline.py)          │     every word from real data
└──────────────┬──────────────┘
               │
               ▼
        JSON Response
   { answer, sources, chunks }
        │
        ▼
  Chat UI (index.html + style.css)
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Backend** | Flask 3.0 | REST API + HTML serving |
| **Embedding Model** | `sentence-transformers/all-mpnet-base-v2` | 768-dim semantic vectors |
| **Vector Database** | FAISS `IndexFlatIP` | Exact cosine similarity search |
| **Dataset** | `Shopify/product-catalogue` | 48,300 real products |
| **Data Loading** | HuggingFace `datasets` library | Streaming parquet load |
| **Frontend** | HTML + CSS + Vanilla JS | Chat interface, no framework |

---

## Dataset

**Shopify/product-catalogue** — published by Shopify Inc.

| Property | Detail |
|---|---|
| Source | [huggingface.co/datasets/Shopify/product-catalogue](https://huggingface.co/datasets/Shopify/product-catalogue) |
| Publisher | Shopify Inc. |
| License | Apache-2.0 |
| Total records | 48,300 products |
| Records indexed | 5,000 (configurable via `MAX_PRODUCTS` in `data_loader.py`) |
| Fields used | `product_title`, `product_description`, `ground_truth_brand`, `ground_truth_category`, `ground_truth_is_secondhand` |

### Category Coverage

The dataset spans a wide taxonomy including:
- Apparel & Accessories (clothing, shoes, bags, jewellery)
- Health & Beauty (skincare, haircare, cosmetics)
- Sporting Goods (fitness, outdoor, yoga)
- Home & Garden (décor, furniture, kitchen)
- Toys & Games (children, board games)
- Electronics (audio, accessories)
- Food & Beverages
- Baby & Toddler

---

## Project Structure

```
retailmind-ai/
│
├── app.py                  ← Flask server: routes, background index loader
├── data_loader.py          ← HuggingFace dataset loader + chunk builder
├── rag_pipeline.py         ← FAISS index, embedding, retrieval, answer gen
├── requirements.txt        ← All Python dependencies with pinned versions
├── README.md               ← This file
├── .gitignore
│
├── templates/
│   └── index.html          ← Chat UI: suggestion chips, live status, JS
│
├── static/
│   └── style.css           ← Full styling: warm editorial design system
│
└── faiss_index/            ← Auto-created on first run
    ├── shopify_products.index   ← FAISS binary index (5k vectors × 768-dim)
    └── shopify_chunks.pkl       ← Pickled product chunk metadata
```

---

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- pip
- ~3 GB free disk space (model + dataset + index)
- Internet connection (for first run only)

### Step 1 — Clone or Download

```bash
# If using git
git clone <your-repo-url>
cd retailmind-ai

# Or simply open the folder in VS Code
```

### Step 2 — Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏱️ This downloads ~1.8 GB total (PyTorch + Transformers + FAISS). Run once, then it's cached locally.

### Step 4 — Run the Application

```bash
python app.py
```

### Step 5 — Open in Browser

```
http://127.0.0.1:5000
```

---

## Running the Application

### First Run Behaviour

When you run `python app.py` for the first time:

1. Flask starts instantly at `http://127.0.0.1:5000`
2. A background thread begins downloading the Shopify dataset (~250 MB)
3. `all-mpnet-base-v2` encodes 5,000 product chunks into 768-dim vectors
4. FAISS index is built and saved to `faiss_index/`
5. The status pill in the UI turns **green** — you are ready to query

**First-run time estimate:** 5–10 minutes on CPU (encoding is the bottleneck)

### Subsequent Runs

The FAISS index loads from disk in under 3 seconds — no re-encoding needed.

### Increasing the Index Size

In `data_loader.py`, change:

```python
MAX_PRODUCTS = 5_000   # change to 20_000 for a larger corpus
```

Then delete `faiss_index/` and restart to rebuild.

---


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
8. [Test Questions](#test-questions)
9. [API Reference](#api-reference)
10. [How RAG Works in This Project](#how-rag-works-in-this-project)
11. [Performance Notes](#performance-notes)
12. [Troubleshooting](#troubleshooting)

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
│  all-MiniLM-L6-v2 (384-dim) │  ← Encodes query → float32 vector
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
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim semantic vectors |
| **Vector Database** | FAISS `IndexFlatIP` | Exact cosine similarity search |
| **Dataset** | `Shopify/product-catalogue` | 48,300 real products |
| **Data Loading** | HuggingFace `datasets` library | Streaming parquet load |
| **Frontend** | HTML + CSS + Vanilla JS | Chat interface, no framework |
| **Fonts** | DM Serif Display + DM Sans | Editorial typography |

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
3. `all-MiniLM-L6-v2` encodes 5,000 product chunks into 384-dim vectors
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

## Test Questions

Use these questions to thoroughly test the system across different product categories, query styles, and difficulty levels.

---

### Category 1 — Direct Product Queries

Simple, specific product lookups.

| # | Question | Expected Behaviour |
|---|---|---|
| 1 | `comfortable running shoes for women` | Returns athletic footwear with descriptions mentioning comfort, support |
| 2 | `waterproof hiking backpack` | Returns outdoor gear with weather-resistance mentions |
| 3 | `wireless noise cancelling headphones` | Returns audio products with noise-isolation features |
| 4 | `stainless steel water bottle` | Returns drinkware, likely with capacity/material details |
| 5 | `yoga mat non-slip eco-friendly` | Returns fitness/wellness products |

---

### Category 2 — Semantic / Paraphrase Queries

These test whether the system understands meaning, not just keywords.

| # | Question | Why It's Interesting |
|---|---|---|
| 6 | `something to keep my drink cold` | No keyword match — tests semantic understanding of "insulated bottle" |
| 7 | `footwear for rainy weather` | Should return waterproof boots/shoes without the word "boot" |
| 8 | `a bag to carry a laptop to work` | Should return laptop bags / briefcases |
| 9 | `clothes for a newborn baby` | Tests category inference for infant apparel |
| 10 | `something relaxing for a bath` | Should return bath bombs, oils, soaks |

---

### Category 3 — Brand & Quality Queries

| # | Question | What to Check |
|---|---|---|
| 11 | `premium leather handbag for office` | Returns luxury/leather items with brand tags |
| 12 | `affordable gym clothes for men` | Returns men's activewear |
| 13 | `organic skincare for sensitive skin` | Returns beauty/personal care with natural ingredient mentions |
| 14 | `sustainable clothing made from recycled materials` | Tests eco/sustainability vocabulary |
| 15 | `best gift for a 5 year old child` | Returns toys, games, children's products |

---

### Category 4 — Comparison & Feature Queries

| # | Question | What to Check |
|---|---|---|
| 16 | `difference between gel and foam mattress` | Returns sleep/home products |
| 17 | `face cream with SPF and anti-aging` | Multi-attribute query — beauty |
| 18 | `lightweight jacket for travel` | Returns outerwear with portability focus |
| 19 | `protein powder for muscle gain` | Returns health/nutrition supplements |
| 20 | `standing desk or ergonomic chair for home office` | Returns furniture / office products |

---

### Category 5 — Edge Cases & Stress Tests

Use these to evaluate robustness.

| # | Question | Purpose |
|---|---|---|
| 21 | `blue` | Single-word vague query — should still return something |
| 22 | `give me something nice` | Ambiguous — tests graceful handling |
| 23 | `I need a present for my wife's birthday she likes flowers` | Long, conversational query |
| 24 | `anti-aging serum with vitamin C hyaluronic acid and retinol` | Multi-ingredient beauty query |
| 25 | `xyz123 purple flying widget` | Nonsense query — should return low-confidence results gracefully |

---

### Category 6 — Category Browsing Queries

| # | Question |
|---|---|
| 26 | `show me all types of women's dresses` |
| 27 | `kitchen tools for baking` |
| 28 | `outdoor furniture for a small balcony` |
| 29 | `educational toys for toddlers` |
| 30 | `men's formal wear for a wedding` |

---

## API Reference

### `POST /ask`

Accepts a product query and returns a RAG-generated answer.

**Request body:**
```json
{
  "query": "comfortable running shoes for women",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "Found 5 relevant products for: \"comfortable running shoes for women\"\n\n...",
  "sources": [
    "Nike Air Zoom Pegasus by Nike",
    "Women's CloudRun Trainer by Adidas",
    ...
  ]
}
```

**Parameters:**
- `query` (string, required) — natural language product question, max 600 chars
- `top_k` (int, optional) — number of results to return, default `5`, max `10`

---

### `GET /index-status`

Returns current index loading state.

**Response (loading):**
```json
{ "ready": false, "elapsed": 45.2 }
```

**Response (ready):**
```json
{
  "ready": true,
  "stats": {
    "products": 4876,
    "model": "sentence-transformers/all-mpnet-base-v2",
    "dataset": "Shopify/product-catalogue (Apache-2.0)",
    "elapsed_s": 312.5
  }
}
```

---

### `GET /health`

Liveness probe.

```json
{ "status": "ok", "index_ready": true }
```

---

## How RAG Works in This Project

### 1. Indexing Phase (first run)

```
Shopify Dataset (48k products)
        │
        │  data_loader.py
        ▼
Rich text chunk per product:
  "Product: Strap Top
   Brand: Zara
   Category: Apparel > Clothing > Tops
   Condition: New
   Description: Jersey top with narrow shoulder straps..."
        │
        │  SentenceTransformer.encode()
        ▼
768-dimensional float32 vector
        │
        │  L2-normalised for cosine similarity
        ▼
FAISS IndexFlatIP  →  saved to disk
```

### 2. Query Phase (every request)

```
User query: "lightweight summer dress"
        │
        │  SentenceTransformer.encode()  (same model, normalised)
        ▼
Query vector  [0.023, -0.041, 0.187, ...]
        │
        │  FAISS.search(query_vec, top_k=5)
        ▼
Top-5 product indices + cosine similarity scores
        │
        │  generate_answer()
        ▼
Structured response with title, brand, category,
relevance label, description excerpt
```

### Why Cosine Similarity?

Cosine similarity measures the **angle** between vectors, not their magnitude. This means a short query like "summer dress" and a long product description both get a fair comparison — what matters is semantic direction, not text length.

---

## Performance Notes

| Metric | Value |
|---|---|
| Index build time (5k products, CPU) | ~5–10 minutes |
| Index build time (5k products, GPU) | ~1–2 minutes |
| Query latency (after index loaded) | < 100ms |
| Index file size (5k products) | ~15 MB |
| RAM required | ~2.5 GB (model + index) |

### To use GPU acceleration (if available):

```python
# In rag_pipeline.py, the SentenceTransformer automatically uses CUDA if available.
# No code change needed — just ensure torch with CUDA is installed:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'faiss'`**
```bash
pip install faiss-cpu
```

**`OSError: [Errno 28] No space left on device`**
The model + dataset needs ~3 GB. Free up space or change `MAX_PRODUCTS` to a smaller value.

**`Server returns 503 on /ask`**
The index is still building. Watch the status pill in the UI — wait for it to turn green.

**`Slow first query after index loads`**
The first inference call warms up PyTorch. Subsequent queries are fast.

**`Out of memory error`**
Reduce `MAX_PRODUCTS` in `data_loader.py` or reduce `batch_size` in `rag_pipeline.py` from 64 to 32.

**`Dataset download fails`**
Check your internet connection. The Shopify dataset is ~250 MB and requires access to `huggingface.co`.

---

## Academic / Project Citations

If using this project in a university submission, you may cite:

```
Dataset:
Shopify Inc. (2024). Shopify Product Catalogue.
HuggingFace Datasets. https://huggingface.co/datasets/Shopify/product-catalogue
License: Apache-2.0

Embedding Model:
Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings
using Siamese BERT-Networks. EMNLP 2019.
https://arxiv.org/abs/1908.10084

Vector Search:
Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search
with GPUs. IEEE Transactions on Big Data.
https://arxiv.org/abs/1702.08734
```

---

*Built with Flask · FAISS · SentenceTransformers · Shopify Dataset*

"""
data_loader.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset  : rjac/e-commerce-customer-support-qa  (MIT License)
Source   : https://huggingface.co/datasets/rjac/e-commerce-customer-support-qa
Size     : 1,000 real customer-support conversations
Publisher: rjac / HuggingFace Community

WHY THIS DATASET?
─────────────────
Directly matches the problem statement:
  ✔ Warranty coverage queries
  ✔ Return & cancellation conditions
  ✔ Compatibility / product specification questions
  ✔ Shipping & order tracking issues
  ✔ Payment & account management FAQs
  ✔ Covers Appliances, Electronics, Clothing categories

COLUMNS USED:
  issue_area             – top-level topic  (Returns, Warranty, etc.)
  issue_category         – mid-level topic
  issue_sub_category     – specific issue
  product_category       – Electronics / Appliances / Clothing
  product_sub_category   – specific product (OTG, Monitor, etc.)
  qa                     – JSON with customer_summary_question +
                           agent_summary_solution  ← primary RAG content
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
from datasets import load_dataset


def _parse_qa(qa_raw):
    """
    Extract question + solution from the nested qa JSON field.
    Returns (question_str, solution_str) or ("", "") on failure.
    """
    try:
        if isinstance(qa_raw, str):
            qa_obj = json.loads(qa_raw)
        else:
            qa_obj = qa_raw

        knowledge = qa_obj.get("knowledge", [])
        if not knowledge:
            return "", ""

        first = knowledge[0]
        question = (first.get("customer_summary_question") or "").strip()
        solution = (first.get("agent_summary_solution") or "").strip()
        return question, solution
    except Exception:
        return "", ""


def _clean(text, max_chars=500):
    if not text:
        return ""
    return " ".join(str(text).split())[:max_chars]


def load_product_data():
    """
    Loads the e-commerce customer-support Q&A dataset and builds
    rich text chunks for HNSW indexing.

    Each chunk encodes:
      - Issue area & category (semantic topic signal)
      - Product type (domain signal)
      - Customer question (what the user asks)
      - Agent solution (the authoritative answer)

    Returns
    -------
    list[dict]  — each dict has:
        id      (int)   row index
        text    (str)   rich chunk text for embedding
        source  (str)   display label shown in UI
        meta    (dict)  structured metadata for result cards
    """
    print("=" * 65)
    print("[data_loader] Dataset : rjac/e-commerce-customer-support-qa")
    print("[data_loader] License : MIT  |  Size: 1,000 support conversations")
    print("[data_loader] Topics  : Returns · Warranty · Shipping · Payments")
    print("[data_loader] Loading from Hugging Face Hub ...")
    print("=" * 65)

    dataset = load_dataset(
        "rjac/e-commerce-customer-support-qa",
        split="train",
    )

    print(f"[data_loader] Loaded {len(dataset):,} records")

    chunks = []
    seen = set()

    for i, record in enumerate(dataset):
        issue_area    = _clean(record.get("issue_area"))
        issue_cat     = _clean(record.get("issue_category"))
        issue_sub     = _clean(record.get("issue_sub_category"))
        prod_cat      = _clean(record.get("product_category"))
        prod_sub      = _clean(record.get("product_sub_category"))
        complexity    = _clean(record.get("issue_complexity"))

        question, solution = _parse_qa(record.get("qa"))

        # Skip if no useful Q&A content
        if not question or not solution:
            continue

        # Deduplicate by question text
        q_key = question.lower()[:120]
        if q_key in seen:
            continue
        seen.add(q_key)

        # ── Build rich text chunk ─────────────────────────────────────
        # Structured so that both the problem (question) and the
        # authoritative answer (solution) are embedded together.
        # This allows semantic retrieval on EITHER query phrasing OR
        # solution keywords (warranty period, return window, etc.)
        parts = []

        if issue_area:
            parts.append(f"Support Area: {issue_area}")
        if issue_cat:
            parts.append(f"Category: {issue_cat}")
        if issue_sub:
            parts.append(f"Issue: {issue_sub}")
        if prod_cat and prod_sub:
            parts.append(f"Product: {prod_sub} ({prod_cat})")
        elif prod_cat:
            parts.append(f"Product Category: {prod_cat}")

        parts.append(f"Customer Question: {question}")
        parts.append(f"Support Answer: {solution}")

        full_text = "\n".join(parts)

        # Source label shown in the UI result cards
        source = f"{issue_sub or issue_cat} — {prod_sub or prod_cat}"
        if len(source) > 80:
            source = source[:77] + "..."

        chunks.append({
            "id":     i,
            "text":   full_text,
            "source": source,
            "meta": {
                "issue_area":  issue_area  or "General",
                "category":    issue_cat   or "—",
                "issue":       issue_sub   or issue_cat or "—",
                "product":     prod_sub    or prod_cat or "—",
                "complexity":  complexity  or "—",
                "question":    question,
                "answer":      solution,
            },
        })

    print(f"[data_loader] ✓ Prepared {len(chunks):,} unique Q&A chunks for indexing.")
    return chunks


if __name__ == "__main__":
    chunks = load_product_data()
    print("\n" + "─" * 65)
    print("SAMPLE CHUNK #1")
    print("─" * 65)
    print(chunks[0]["text"])
    print("\n" + "─" * 65)
    print("SAMPLE CHUNK #50")
    print("─" * 65)
    print(chunks[min(50, len(chunks) - 1)]["text"])
    print(f"\nTotal indexable chunks: {len(chunks):,}")
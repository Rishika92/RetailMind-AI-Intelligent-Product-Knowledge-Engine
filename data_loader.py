"""
data_loader.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dataset  : Shopify/product-catalogue  (Apache-2.0)
Source   : https://huggingface.co/datasets/Shopify/product-catalogue
Size     : 48,300 real e-commerce products  (~250 MB parquet)
Publisher: Shopify Inc. — industry-grade, used in AI/ML research
Columns used:
    product_title          – product name
    product_description    – full marketing description
    ground_truth_brand     – brand name
    ground_truth_category  – full taxonomy path  e.g.
                             "Apparel & Accessories > Clothing > Tops"
    ground_truth_is_secondhand – bool flag
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from datasets import load_dataset


# Maximum products to index.  5_000 gives a rich corpus while staying fast.
# Raise to 20_000+ for a full production-grade index (takes ~10 min on CPU).
MAX_PRODUCTS = 5_000


def _clean(text, max_chars=800):
    """Strip whitespace and truncate to max_chars."""
    if not text:
        return ""
    cleaned = " ".join(str(text).split())
    return cleaned[:max_chars]


def load_product_data():
    """
    Loads the Shopify product-catalogue dataset from Hugging Face,
    builds rich text chunks combining title + brand + category path +
    description, and returns them for FAISS indexing.

    Returns
    -------
    list[dict]  — each dict has keys:
        id       (int)   row index
        text     (str)   rich combined text for embedding
        source   (str)   short human-readable label shown in the UI
        meta     (dict)  extra fields (brand, category, etc.)
    """
    print("=" * 60)
    print("[data_loader] Dataset : Shopify/product-catalogue")
    print("[data_loader] License : Apache-2.0  |  Publisher: Shopify Inc.")
    print("[data_loader] Loading from Hugging Face Hub...")
    print("=" * 60)

    dataset = load_dataset(
        "Shopify/product-catalogue",
        split="train",
    )

    print(f"[data_loader] Full dataset size : {len(dataset):,} products")
    print(f"[data_loader] Indexing first    : {MAX_PRODUCTS:,} products")

    chunks = []
    seen_titles = set()

    for i, record in enumerate(dataset):
        if len(chunks) >= MAX_PRODUCTS:
            break

        title      = _clean(record.get("product_title"))
        desc       = _clean(record.get("product_description"), max_chars=600)
        brand      = _clean(record.get("ground_truth_brand"))
        category   = _clean(record.get("ground_truth_category"))
        secondhand = record.get("ground_truth_is_secondhand") or False

        # Must have at least a title + description to be useful
        if not title or not desc:
            continue

        # Deduplicate by title to avoid near-identical embeddings
        title_key = title.lower()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        # ── Build the rich text chunk ──────────────────────────────────────
        # This chunk is what gets embedded and searched.
        # We include every semantic signal: title, brand, taxonomy path,
        # condition flag, and full description.
        parts = []

        parts.append(f"Product: {title}")

        if brand:
            parts.append(f"Brand: {brand}")

        if category:
            # Taxonomy path e.g. "Apparel & Accessories > Clothing > Tops"
            parts.append(f"Category: {category}")

        if secondhand:
            parts.append("Condition: Pre-owned / Second-hand")
        else:
            parts.append("Condition: New")

        if desc:
            parts.append(f"Description: {desc}")

        full_text = "\n".join(parts)

        # ── Short label shown in the UI ────────────────────────────────────
        brand_tag = f" by {brand}" if brand else ""
        source    = f"{title}{brand_tag}"
        if len(source) > 80:
            source = source[:77] + "..."

        chunks.append({
            "id":     i,
            "text":   full_text,
            "source": source,
            "meta": {
                "title":    title,
                "brand":    brand or "Unknown",
                "category": category or "Uncategorized",
                "new":      not secondhand,
            },
        })

    print(f"[data_loader] Prepared {len(chunks):,} product chunks for indexing.")
    return chunks


if __name__ == "__main__":
    chunks = load_product_data()

    print("\n" + "-" * 60)
    print("SAMPLE CHUNK #1")
    print("-" * 60)
    print(chunks[0]["text"])

    print("\n" + "-" * 60)
    print("SAMPLE CHUNK #500")
    print("-" * 60)
    print(chunks[min(500, len(chunks) - 1)]["text"])

    print(f"\nTotal indexable chunks: {len(chunks):,}")
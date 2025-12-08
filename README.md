Unsupervised Text Baseline – 3D Asset Retrieval

This code implements the unsupervised text-based baseline for the 3D asset retrieval project:

- We use a pretrained text encoder ( all-MiniLM-L6-v2 from `sentence-transformers`) to embed all captions.
- We L2-normalize the embeddings so cosine similarity ≈ inner product.
- We build a FAISS inner product index over the caption embeddings.
- At query time, we embed the query text and retrieve the most similar assets by caption.
- We evaluate using Recall@K (R@1, R@5, R@10) and MRR and simple robustness tests.

This is meant to be the unsupervised baseline: pretrained text encoder + embedding similarity, no fine-tuning.

---

## File structure

- `build_index.py`

  - Loads metadata (`final_metadata_10k.csv`) and optional UID filter (`available_uids.txt`).
  - Encodes captions with MiniLM into embeddings.
  - L2-normalizes embeddings and saves them to `data/index/text_embeddings_norm.npy`.
  - Builds a FAISS index and saves it to `data/index/text_index.faiss`.
  - Saves metadata (`uid`, `caption`, `image_path`) to `data/index/text_index_metadata.csv`.
  - Runs a small demo search to sanity check the index.

- `eval_text_baseline.py`

  - Loads the FAISS index, metadata, and the same MiniLM model.
  - Exposes a `search(query, top_k)` helper for text retrieval.
  - Evaluates the baseline by:
    - Sampling random captions as queries.
    - Computing R@1, R@5, R@10, and MRR.
  - Runs simple robustness tests (synonyms, extra adjectives, word reordering).

- `scripts/get_available_uids.py` (if present)
  - Scans `data/raw/all_asset_zips/` for `.zip` files.
  - Writes one UID per line into `data/available_uids.txt`.
  - Used to filter the caption metadata to assets we actually have locally.

You’ll also need the data files (already provided in the project):

- `data/available_uids.txt`
- `data/processed/final_metadata_10k.csv`

---

## Environment setup

From the repository root:

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate   # On macOS / Linux
# .venv\Scripts\activate    # On Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt
```

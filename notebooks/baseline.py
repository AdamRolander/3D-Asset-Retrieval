"""
build_index.py

Builds a text-based retrieval index for the Cap3D captions.

  1. Loads the Cap3D metadata CSV (final_metadata_10k.csv).
  2. Loads the list of available UIDs (available_uids.txt) and filters rows.
  3. Encodes all captions using a MiniLM text encoder.
  4. L2-normalizes the embeddings and saves them to disk.
  5. Builds a FAISS inner-product index over the normalized embeddings.
  6. Saves metadata so index results can be mapped back to (uid, caption, image_path).
  7. Runs a small demo query to sanity-check the index.
"""

import os

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm



# Path to text file containing one UID per line (matching asset zip names)
AVAILABLE_UIDS_PATH = "data/available_uids.txt"

# Path to CSV with columns: uid, caption, image_path
CAPTIONS_CSV_PATH = "data/processed/final_metadata_10k.csv"


#loading data 


def load_available_uids(path: str) -> set:
    """
    Load the list of available UIDs from a text file.
    path : str
        Path to 'available_uids.txt', one UID per line.
    Returns
    set[str]
        Set of UID strings. Empty set if the file does not exist.
    """
    if not os.path.exists(path):
        print(f"[WARN] available_uids file not found at {path}.")
        print("       Proceeding without UID-based filtering.")
        return set()

    with open(path, "r") as f:
        uids = {line.strip() for line in f if line.strip()}

    print(f"[INFO] Loaded {len(uids)} available UIDs from {path}")
    return uids


def load_and_filter_captions(captions_path: str, available_uids: set) -> pd.DataFrame:
    """
    Load the captions CSV and optionally filter rows to a set of UIDs.

    Parameters
    captions_path : str
        Path to the CSV file with columns: uid, caption, image_path.
    available_uids : set[str]
        Set of allowed UIDs. If empty, no filtering is applied.
    Output: 
    pandas.DataFrame
        DataFrame with at least ['uid', 'caption', 'image_path'] columns,
        filtered to available_uids (if provided) and with missing captions removed.
    """
    if not os.path.exists(captions_path):
        raise FileNotFoundError(f"Captions CSV not found at {captions_path}")

    print(f"[INFO] Loading captions from {captions_path} ...")
    df = pd.read_csv(captions_path)

    print(f"[INFO] Original rows in CSV: {len(df)}")
    print(f"[INFO] Columns: {df.columns.tolist()}")

    expected_cols = {"uid", "caption", "image_path"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Expected columns {expected_cols}, but got {df.columns.tolist()}")

    # Filter by available_uids if provided
    if available_uids:
        before = len(df)
        df = df[df["uid"].isin(available_uids)].reset_index(drop=True)
        print(f"[INFO] Rows after filtering by available_uids: {before} -> {len(df)}")
    else:
        print("[INFO] No available_uids provided, skipping UID filtering.")

    # Drop rows with missing captions
    df = df.dropna(subset=["caption"]).reset_index(drop=True)

    print("\n[INFO] Preview of filtered data:")
    print(df.head(5))

    return df

# embedding and normalization

def build_embeddings(df: pd.DataFrame, batch_size: int = 64) -> np.ndarray:
    """
    Encode all captions into MiniLM embeddings.
    Input: 
    df : pandas.DataFrame
        DataFrame with at least a 'caption' column.
    batch_size : int, optional
        Number of captions to encode at once (controls speed vs. memory).

    Output: 
    numpy.ndarray
        Embedding matrix of shape (N, D),
        where N = number of rows and D = embedding dimension.
    """
    print("\n[INFO] Loading MiniLM model (all-MiniLM-L6-v2)")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    captions = df["caption"].tolist()
    all_embeddings = []

    print(f"[INFO] Encoding {len(captions)} captions in batches of {batch_size}...")
    for i in tqdm(range(0, len(captions), batch_size)):
        batch = captions[i : i + batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings)
    print(f"[INFO] Embedding matrix shape = {embeddings.shape}")
    return embeddings


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalize each embedding vector so that cosine similarity
    is equivalent to inner product.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Array of shape (N, D) with raw embeddings.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, D) where each row has L2 norm ≈ 1.
    """
    print("\n[INFO] Normalizing embeddings (L2)...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    embeddings_norm = embeddings / norms
    return embeddings_norm


# ---------------------------------------------------------------------------
# FAISS index building and metadata
# ---------------------------------------------------------------------------

def build_and_save_faiss_index(embeddings_norm: np.ndarray, output_path: str) -> faiss.IndexFlatIP:
    """
    Build a FAISS inner-product index over L2-normalized embeddings and save it.

    Parameters
    ----------
    embeddings_norm : numpy.ndarray
        Array of shape (N, D) with L2-normalized embeddings (float or float32).
    output_path : str
        File path where the FAISS index (.faiss) will be written.

    Returns
    -------
    faiss.IndexFlatIP
        The constructed FAISS index object.
    """
    n, d = embeddings_norm.shape
    print(f"\n[INFO] Building FAISS IndexFlatIP with {n} vectors of dimension {d}...")

    embeddings_f32 = embeddings_norm.astype("float32")

    index = faiss.IndexFlatIP(d)
    index.add(embeddings_f32)

    print("[INFO] Index built. Saving to:", output_path)
    faiss.write_index(index, output_path)
    print("[INFO] FAISS index saved.")

    return index


def save_metadata(df: pd.DataFrame, output_path: str) -> None:
    """
    Save (uid, caption, image_path) metadata so FAISS results can be interpreted.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns ['uid', 'caption', 'image_path'].
    output_path : str
        Path to a .csv file where metadata will be stored.
    """
    cols_to_save = ["uid", "caption", "image_path"]
    meta_df = df[cols_to_save].copy()
    meta_df.to_csv(output_path, index=False)
    print(f"[INFO] Saved metadata ({len(meta_df)} rows) to {output_path}")


# ---------------------------------------------------------------------------
# Small demo: load index and run a couple of text queries
# ---------------------------------------------------------------------------

def demo_search_from_disk(top_k: int = 5) -> None:
    """
    Load the FAISS index and metadata from disk and run a few example queries.

    This function:
      - encodes query text with the same MiniLM model,
      - L2-normalizes the query embedding,
      - runs FAISS inner-product search,
      - prints the top-k (uid, caption) matches.

    Parameters
    ----------
    top_k : int, optional
        Number of results to print per query.
    """
    faiss_index_path = "data/index/text_index.faiss"
    metadata_path = "data/index/text_index_metadata.csv"

    if not os.path.exists(faiss_index_path):
        print(f"[ERROR] FAISS index not found at {faiss_index_path}")
        return
    if not os.path.exists(metadata_path):
        print(f"[ERROR] Metadata file not found at {metadata_path}")
        return

    print("\n[INFO] Loading FAISS index from disk...")
    index = faiss.read_index(faiss_index_path)

    print("[INFO] Loading metadata from disk...")
    meta_df = pd.read_csv(metadata_path)

    print("[INFO] Loading MiniLM model for query encoding...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    queries = [
        "white sofa with wooden legs",
        "airplane",
        "robot",
        "dump truck",
    ]

    for q in queries:
        print("\n" + "=" * 80)
        print(f"[QUERY] {q}")

        q_emb = model.encode([q], show_progress_bar=False)
        norms = np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12
        q_emb_norm = (q_emb / norms).astype("float32")

        scores, indices = index.search(q_emb_norm, top_k)
        scores = scores[0]
        indices = indices[0]

        print(f"[INFO] Top-{top_k} results:")
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            row = meta_df.iloc[idx]
            uid = row["uid"]
            caption = row["caption"]
            print(f"  {rank}. uid={uid}  score={score:.4f}")
            print(f"     caption: {caption}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Full pipeline:
      1. Load available UIDs.
      2. Load and filter caption metadata.
      3. Build text embeddings for all captions.
      4. Normalize and save embeddings.
      5. Build and save FAISS index.
      6. Save metadata for later lookup.
      7. Run a small demo query on the saved index.
    """
    # 1. Load available UIDs
    available_uids = load_available_uids(AVAILABLE_UIDS_PATH)

    # 2. Load and filter captions
    df = load_and_filter_captions(CAPTIONS_CSV_PATH, available_uids)
    print(f"\n[SUMMARY] Final number of caption rows: {len(df)}")

    os.makedirs("data/index", exist_ok=True)

    # 3. Build embeddings
    embeddings = build_embeddings(df)

    # 4. Normalize and save embeddings
    embeddings_norm = normalize_embeddings(embeddings)
    embeddings_path = "data/index/text_embeddings_norm.npy"
    np.save(embeddings_path, embeddings_norm)
    print(f"[INFO] Saved normalized embeddings to {embeddings_path}")

    # 5. Build and save FAISS index
    faiss_index_path = "data/index/text_index.faiss"
    build_and_save_faiss_index(embeddings_norm, faiss_index_path)

    # 6. Save metadata
    metadata_path = "data/index/text_index_metadata.csv"
    save_metadata(df, metadata_path)

    print("\n[SUMMARY] Index + embeddings + metadata saved.")

    # 7. Run a quick demo search
    demo_search_from_disk(top_k=5)


if __name__ == "__main__":
    main()

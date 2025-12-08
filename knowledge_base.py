# knowledge_base.py
import os
import io
import json
import pickle
from typing import List, Dict, Any, Tuple
from datetime import datetime

import numpy as np
import faiss
import openai
from googleapiclient.discovery import build
from google.oauth2 import service_account
from PyPDF2 import PdfReader

# =========================
# CONFIG
# =========================
SERVICE_ACCOUNT_FILE = "credentials.json"   # path to your service-account JSON
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_ID = "13J-DiERhtS1VWgF2GtZ1wnMfbUzkq6-G"          # <---- SET THIS
EMBED_MODEL = "text-embedding-3-small"

# Output files (kept the same names your app expects for index/metadata)
FAISS_PATH = "faiss_index.index"
METADATA_PATH = "metadata.pkl"
VECTORS_PATH = "embeddings.npy"             # stores the embeddings array
STATE_PATH = "kb_state.json"                # stores doc -> modifiedTime mapping

# Chunking params (tweak if needed)
CHUNK_CHARS = 1800        # ~ approx. 500-700 tokens depending on content
CHUNK_OVERLAP = 300       # overlap between consecutive chunks


# =========================
# DRIVE + PDF HELPERS
# =========================
def _drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)

def list_pdfs(folder_id: str) -> List[Dict[str, Any]]:
    """List PDFs in the folder with id, name, modifiedTime."""
    svc = _drive_service()
    results = svc.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed = false",
        fields="files(id, name, modifiedTime)",
        pageSize=1000
    ).execute()
    return results.get("files", [])

def download_pdf_bytes(file_id: str) -> bytes:
    svc = _drive_service()
    return svc.files().get_media(fileId=file_id).execute()

def pdf_to_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


# =========================
# CHUNKING
# =========================
def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Simple character-based chunking with overlap.
    This is robust for legal PDFs without needing a tokenizer.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


# =========================
# PERSISTENCE
# =========================
def load_state() -> Dict[str, Any]:
    """doc_id -> modifiedTime mapping, and any other small state you may add later."""
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"docs": {}}  # {"docs": {doc_id: {"modifiedTime": "..."} } }

def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def load_embeddings_and_metadata() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    embeddings: np.ndarray shape (N, D)
    metadata:   list of dicts aligned index-by-index with embeddings
    """
    if os.path.exists(VECTORS_PATH) and os.path.exists(METADATA_PATH):
        embeddings = np.load(VECTORS_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        return embeddings, metadata
    # empty
    return np.zeros((0, 0), dtype="float32"), []

def save_embeddings_and_metadata(embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
    np.save(VECTORS_PATH, embeddings.astype("float32"))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

def write_faiss_index(embeddings: np.ndarray) -> None:
    """
    Builds a fresh FAISS index from the provided embeddings.
    (We rebuild the FAISS file each time, but we *reuse* embeddings for unchanged docs,
     so we still avoid re-embedding.)
    """
    if embeddings.size == 0:
        # Create an empty index with default dim if needed
        dim = 1536  # text-embedding-3-small output
        index = faiss.IndexFlatL2(dim)
        faiss.write_index(index, FAISS_PATH)
        return

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, FAISS_PATH)


# =========================
# EMBEDDING
# =========================
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embeds a list of strings. Returns np.ndarray (len(texts), dim).
    Uses OpenAI embeddings batch API pattern for efficiency.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    # OpenAI Python SDK v1 style
    resp = openai.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype="float32")


# =========================
# CORE: PARTIAL UPDATE PIPELINE
# =========================
def rebuild_knowledge_base(folder_id: str = FOLDER_ID) -> None:
    """
    - Checks Google Drive PDFs and their modifiedTime
    - Only (re)embeds docs that are new or changed
    - Rebuilds the FAISS file from the combined embeddings (old + new)
    """
    # 1) Load prior state + data
    state = load_state()
    prior_map = state.get("docs", {})  # doc_id -> {"modifiedTime": "..."}
    old_embeddings, old_metadata = load_embeddings_and_metadata()

    # consistency: old_embeddings matches old_metadata length
    assert old_embeddings.shape[0] == len(old_metadata), "Embeddings/metadata size mismatch."

    # 2) List current PDFs
    files = list_pdfs(folder_id)

    # 3) Prepare new containers
    # We'll rebuild the *combined* arrays from scratch using:
    # - Unchanged doc chunks (reused from old arrays)
    # - Newly embedded chunks for new/changed docs
    combined_embeddings: List[np.ndarray] = []
    combined_metadata: List[Dict[str, Any]] = []

    # Build helper: map (doc_id) -> list of old rows
    # We stored 'doc_id' and 'chunk_idx' in metadata so we can filter unchanged chunks.
    from collections import defaultdict
    old_rows_by_doc: Dict[str, List[int]] = defaultdict(list)
    for i, m in enumerate(old_metadata):
        old_rows_by_doc[m["doc_id"]].append(i)

    # 4) Iterate current files and decide per-doc action
    docs_updated_in_state = {}

    for f in files:
        doc_id = f["id"]
        name = f["name"]
        modified = f["modifiedTime"]

        previously_known = doc_id in prior_map
        is_changed = (not previously_known) or (modified > prior_map[doc_id]["modifiedTime"])

        if not is_changed:
            # Reuse all old chunks for this doc
            for i in old_rows_by_doc.get(doc_id, []):
                combined_embeddings.append(old_embeddings[i:i+1])  # keep shape
                combined_metadata.append(old_metadata[i])
            docs_updated_in_state[doc_id] = {"modifiedTime": modified}
            continue

        # New or changed: download + extract + chunk + embed
        pdf_bytes = download_pdf_bytes(doc_id)
        text = pdf_to_text(pdf_bytes)
        if not text.strip():
            # If empty, skip and don't carry over old chunks
            docs_updated_in_state[doc_id] = {"modifiedTime": modified}
            continue

        chunks = chunk_text(text, CHUNK_CHARS, CHUNK_OVERLAP)
        # embed in batches (API can handle a list)
        chunk_embeddings = embed_texts(chunks)
        # Add to combined arrays
        for idx, (chunk_text_val) in enumerate(chunks):
            combined_embeddings.append(chunk_embeddings[idx:idx+1])
            combined_metadata.append({
                "doc_id": doc_id,
                "doc_title": name,
                "chunk_idx": idx,
                "content": chunk_text_val,
                "modifiedTime": modified,
            })
        docs_updated_in_state[doc_id] = {"modifiedTime": modified}

    # 5) Persist combined arrays (stack rows)
    if combined_embeddings:
        new_embeddings = np.vstack(combined_embeddings).astype("float32")
    else:
        new_embeddings = np.zeros((0, 1536), dtype="float32")  # default dim
    save_embeddings_and_metadata(new_embeddings, combined_metadata)

    # 6) Write FAISS index file from the combined embeddings
    write_faiss_index(new_embeddings)

    # 7) Save updated state (only docs seen in this run are kept)
    state["docs"] = docs_updated_in_state
    save_state(state)


# =========================
# LOAD FOR YOUR APP
# =========================
def load_index_and_metadata_for_app():
    """
    Call this from your Streamlit app to load FAISS + metadata into memory for search.
    This function does *not* rebuild — it only reads the files produced by the scheduler.
    """
    # 1) Load FAISS index
    if os.path.exists(FAISS_PATH):
        index = faiss.read_index(FAISS_PATH)
    else:
        # create empty index to avoid crashes on first boot
        index = faiss.IndexFlatL2(1536)

    # 2) Load metadata
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
    else:
        metadata = []

    return index, metadata

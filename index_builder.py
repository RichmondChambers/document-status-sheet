import os
import io
import json
import pickle
import time
import datetime
from typing import List, Dict

import numpy as np
import faiss
import docx
import PyPDF2

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError

import streamlit as st

# 🔹 Folder ID of your UK-Immigration-Knowledge folder in Drive
DRIVE_FOLDER_ID = "13J-DiERhtS1VWgF2GtZ1wnMfbUzkq6-G"

# 🔹 Local files the app uses
INDEX_FILE = "faiss_index.index"
METADATA_FILE = "metadata.pkl"
STATE_FILE = "drive_index_state.json"  # to detect changes + store timestamps

# ✅ Check Drive at most once per day (global cooldown)
CHECK_COOLDOWN_SECONDS = 24 * 60 * 60  # 1 day


def get_openai_client() -> OpenAI:
    """
    Create an OpenAI client using Streamlit secrets.
    Keeps index_builder aligned with app.py SDK usage.
    """
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def get_drive_service():
    """
    Build an authenticated Google Drive API client using the service account
    stored in st.secrets["gcp_service_account"].
    """
    creds_info = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    service = build("drive", "v3", credentials=credentials)
    return service


def list_files_recursive(folder_id: str, service) -> List[Dict]:
    """
    Recursively list all non-folder files under a Drive folder (including sub-folders).
    """
    files: List[Dict] = []

    page_token = None
    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
            pageToken=page_token,
        ).execute()

        for f in response.get("files", []):
            mime_type = f.get("mimeType", "")
            if mime_type == "application/vnd.google-apps.folder":
                files.extend(list_files_recursive(f["id"], service))
            else:
                files.append(f)

        page_token = response.get("nextPageToken", None)
        if page_token is None:
            break

    return files


def list_drive_files() -> List[Dict]:
    """
    Return a list of all files (id, name, mimeType, modifiedTime)
    under the main knowledge folder (including sub-folders).
    """
    service = get_drive_service()
    return list_files_recursive(DRIVE_FOLDER_ID, service)


def load_previous_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def have_files_changed(current_files, previous_state):
    """
    Compare current Drive files with previous state to see if anything is new or modified.
    """
    current_state = {f["id"]: f["modifiedTime"] for f in current_files}
    if current_state != previous_state.get("files", {}):
        return True, current_state
    return False, current_state


def download_file_bytes(service, file):
    """
    Download the raw bytes of a file from Google Drive.
    Handles both normal files (pdf/docx/txt) and Google Docs (exported as DOCX).
    """
    file_id = file["id"]
    mime_type = file.get("mimeType", "")

    if mime_type == "application/vnd.google-apps.document":
        request = service.files().export_media(
            fileId=file_id,
            mimeType="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        effective_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        request = service.files().get_media(fileId=file_id)
        effective_mime = mime_type

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return fh.read(), effective_mime


def extract_text_from_bytes(file_bytes: bytes, mime_type: str, file_name: str) -> str:
    """
    Convert downloaded bytes into plain text, depending on MIME type / extension.
    Supports DOCX, PDF, TXT, MD. Extend if needed.
    """
    name_lower = file_name.lower()

    if (
        mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or name_lower.endswith(".docx")
    ):
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)

    if mime_type == "application/pdf" or name_lower.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages)

    if mime_type.startswith("text/") or name_lower.endswith(".txt") or name_lower.endswith(".md"):
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="ignore")

    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return ""


def split_into_chunks(text: str, max_chars: int = 1500, overlap: int = 200):
    """
    Simple character-based chunking with overlap.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def embed_texts(
    texts,
    model="text-embedding-3-small",
    batch_size=16,
    max_retries=6
) -> np.ndarray:
    """
    Get embeddings for a list of texts using OpenAI embeddings API,
    retrying only transient failures and failing fast on real config errors.
    """
    client = get_openai_client()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        attempts = 0
        while True:
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=model,
                    timeout=30,
                )
                break  # success

            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                attempts += 1
                if attempts > max_retries:
                    raise RuntimeError(
                        f"OpenAI embeddings failed after {max_retries} retries. Last error: {e}"
                    )
                wait_seconds = 5 * attempts
                print(
                    f"[index_builder] Transient error, retry {attempts}/{max_retries} in {wait_seconds}s: {e}"
                )
                time.sleep(wait_seconds)

            except Exception as e:
                # Non-transient errors should stop the rebuild immediately
                raise RuntimeError(
                    f"OpenAI embeddings failed with a non-retryable error: {e}"
                )

        for item in response.data:
            all_embeddings.append(item.embedding)

    return np.array(all_embeddings, dtype=np.float32)


def rebuild_index_from_drive(files: List[Dict]):
    """
    Download files, extract text, chunk, embed, and rebuild FAISS + metadata.
    """
    service = get_drive_service()
    all_chunks = []
    metadata = []

    for file in files:
        file_id = file["id"]
        file_name = file.get("name", "unnamed")
        mime_type = file.get("mimeType", "")

        if mime_type == "application/vnd.google-apps.folder":
            continue

        file_bytes, effective_mime = download_file_bytes(service, file)

        text = extract_text_from_bytes(file_bytes, effective_mime, file_name)
        if not text.strip():
            continue

        chunks = split_into_chunks(text)

        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append(
                {
                    "content": chunk,
                    "file_id": file_id,
                    "file_name": file_name,
                    "chunk_index": idx,
                }
            )

    if not all_chunks:
        dim = 1536  # embedding dim for text-embedding-3-small
        index = faiss.IndexFlatL2(dim)
        faiss.write_index(index, INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump([], f)
        return

    embeddings = embed_texts(all_chunks, model="text-embedding-3-small")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)


def sync_drive_and_rebuild_index_if_needed():
    """
    Check Drive for new/updated files; if changes are detected,
    rebuild FAISS index and metadata from scratch.

    ✅ Debounced: will not check Drive more than once per day globally.

    Returns:
        True  -> rebuilt now
        False -> not rebuilt
    """
    previous_state = load_previous_state()

    # ---- DAILY COOLDOWN ----
    last_checked = previous_state.get("last_checked")
    if last_checked:
        try:
            last_checked_dt = datetime.datetime.fromisoformat(last_checked.replace("Z", ""))
            age_seconds = (datetime.datetime.utcnow() - last_checked_dt).total_seconds()
            if age_seconds < CHECK_COOLDOWN_SECONDS:
                return False
        except Exception:
            pass
    # ------------------------

    files = list_drive_files()
    changed, current_state = have_files_changed(files, previous_state)

    if changed or not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        rebuild_index_from_drive(files)

        save_state(
            {
                "files": current_state,
                "last_rebuilt": datetime.datetime.utcnow().isoformat() + "Z",
                "last_checked": datetime.datetime.utcnow().isoformat() + "Z",
            }
        )
        return True

    save_state(
        {
            "files": previous_state.get("files", {}),
            "last_rebuilt": previous_state.get("last_rebuilt"),
            "last_checked": datetime.datetime.utcnow().isoformat() + "Z",
        }
    )
    return False

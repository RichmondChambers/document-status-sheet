import os
import io
import re
import json
import pickle
import time
import datetime
from typing import List, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import faiss
import docx
import PyPDF2

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError

import streamlit as st

# ============================================================
# ONLY THESE TWO FOLDERS WILL BE INDEXED
# ============================================================

# ✅ Immigration Rules folder (authoritative)
IMMIGRATION_RULES_FOLDER_ID = "1BGrO2uRTO0axcLE7pwGpv_1EONb25vYe"

# ✅ Internal Precedents folder (style / best practice)
INTERNAL_PRECEDENTS_FOLDER_ID = "1VY2dCt5KgAkgofn0eSDrZWI5zhXQMwaT"

# 🔹 Local files the app uses
INDEX_FILE = "faiss_index.index"
METADATA_FILE = "metadata.pkl"
STATE_FILE = "drive_index_state.json"

DEFAULT_COOLDOWN_SECONDS = 24 * 60 * 60  # 1 day
DEFAULT_OVERNIGHT_START_HOUR_UK = 1  # 01:00 UK time
DEFAULT_OVERNIGHT_END_HOUR_UK = 6    # 06:00 UK time


def get_openai_client() -> OpenAI:
    """Create an OpenAI client using Streamlit secrets."""
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def get_secret_value(key: str, default):
    """Safely fetch a Streamlit secret, falling back to a default when missing."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def get_drive_sync_cooldown_seconds() -> float:
    """Return configurable cooldown in seconds (defaults to 24h)."""
    hours = get_secret_value("drive_sync_cooldown_hours", DEFAULT_COOLDOWN_SECONDS / 3600)
    try:
        return max(0.0, float(hours)) * 3600
    except (TypeError, ValueError):
        return DEFAULT_COOLDOWN_SECONDS


def get_uk_overnight_window_hours() -> Tuple[int, int]:
    """Return (start_hour, end_hour) for the UK overnight window (0-23 inclusive)."""
    start = get_secret_value("drive_sync_overnight_start_hour", DEFAULT_OVERNIGHT_START_HOUR_UK)
    end = get_secret_value("drive_sync_overnight_end_hour", DEFAULT_OVERNIGHT_END_HOUR_UK)

    try:
        start_int = int(start)
        end_int = int(end)
        start_int = max(0, min(23, start_int))
        end_int = max(0, min(23, end_int))
        return start_int, end_int
    except (TypeError, ValueError):
        return DEFAULT_OVERNIGHT_START_HOUR_UK, DEFAULT_OVERNIGHT_END_HOUR_UK


def is_within_uk_overnight_window(now_utc: Optional[datetime.datetime] = None) -> bool:
    """Determine if the given (or current) UTC time falls within the UK overnight window."""
    if now_utc is None:
        now_utc = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)

    uk_tz = ZoneInfo("Europe/London")
    now_uk = now_utc.astimezone(uk_tz)

    start_hour, end_hour = get_uk_overnight_window_hours()

    if start_hour == end_hour:
        return True  # full-day window

    if start_hour < end_hour:
        return start_hour <= now_uk.hour < end_hour

    # Window crosses midnight (e.g., 22 -> 5)
    return now_uk.hour >= start_hour or now_uk.hour < end_hour


def get_drive_service():
    """Build an authenticated Google Drive API client using a service account."""
    creds_info = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    service = build("drive", "v3", credentials=credentials)
    return service


# =========================
# 1) List files WITH path
# =========================
def list_files_recursive(folder_id: str, service, path_prefix: str = "") -> List[Dict]:
    """
    Recursively list all non-folder files under a Drive folder (including sub-folders),
    preserving a human-readable folder path.

    Each file dict includes:
      - id, name, mimeType, modifiedTime, parents
      - path (like "Immigration Rules/Appendix FM.pdf")
    """
    files: List[Dict] = []
    page_token = None

    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType, modifiedTime, parents)",
            pageToken=page_token,
        ).execute()

        for f in response.get("files", []):
            mime_type = f.get("mimeType", "")
            name = f.get("name", "")
            if mime_type == "application/vnd.google-apps.folder":
                child_prefix = f"{path_prefix}/{name}" if path_prefix else name
                files.extend(list_files_recursive(f["id"], service, child_prefix))
            else:
                f["path"] = f"{path_prefix}/{name}" if path_prefix else name
                files.append(f)

        page_token = response.get("nextPageToken")
        if page_token is None:
            break

    return files


def list_drive_files_only_target_folders() -> List[Tuple[Dict, str]]:
    """
    Return a list of (file_dict, source_type) for ONLY:
      - Immigration Rules folder -> "rule"
      - Internal Precedents folder -> "precedent"
    """
    service = get_drive_service()

    rule_files = list_files_recursive(
        IMMIGRATION_RULES_FOLDER_ID, service, "Immigration Rules"
    )
    precedent_files = list_files_recursive(
        INTERNAL_PRECEDENTS_FOLDER_ID, service, "Internal Precedents"
    )

    tagged: List[Tuple[Dict, str]] = []
    for f in rule_files:
        tagged.append((f, "rule"))
    for f in precedent_files:
        tagged.append((f, "precedent"))

    return tagged


def load_previous_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def have_files_changed(tagged_files: List[Tuple[Dict, str]], previous_state):
    """
    Compare current Drive files (only the two target folders)
    with previous state to see if anything is new or modified.
    """
    current_state = {f["id"]: f["modifiedTime"] for (f, _) in tagged_files}
    if current_state != previous_state.get("files", {}):
        return True, current_state
    return False, current_state


def download_file_bytes(service, file):
    """
    Download raw bytes of a file from Google Drive.
    Handles normal files and Google Docs (exported as DOCX).
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
    """Convert downloaded bytes into plain text."""
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


def normalise_rule_text(text: str) -> str:
    """Light normalisation to improve paragraph-ref matching."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text: str, max_chars: int = 1800, overlap: int = 200):
    """
    Paragraph-aware chunking:
    - splits on blank lines
    - rebuilds into chunks up to max_chars
    - adds overlap by prefixing with tail of previous chunk
    """
    text = text.strip()
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""

    for p in paras:
        if len(current) + len(p) + 2 <= max_chars:
            current = f"{current}\n\n{p}".strip()
        else:
            if current:
                chunks.append(current)
            current = p

    if current:
        chunks.append(current)

    if overlap and len(chunks) > 1:
        overlapped = []
        for i, ch in enumerate(chunks):
            if i == 0:
                overlapped.append(ch)
                continue
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + "\n\n" + ch)
        chunks = overlapped

    return chunks


def embed_texts(
    texts,
    model="text-embedding-3-small",
    batch_size=16,
    max_retries=6
) -> np.ndarray:
    """Batch embeddings with retry on transient failures."""
    client = get_openai_client()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        attempts = 0

        while True:
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=model,
                    timeout=30,
                )
                break
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                attempts += 1
                if attempts > max_retries:
                    raise RuntimeError(
                        f"OpenAI embeddings failed after {max_retries} retries. Last error: {e}"
                    )
                wait_seconds = 5 * attempts
                print(f"[index_builder] Transient error, retry {attempts}/{max_retries} in {wait_seconds}s: {e}")
                time.sleep(wait_seconds)
            except Exception as e:
                raise RuntimeError(f"OpenAI embeddings failed with a non-retryable error: {e}")

        for item in response.data:
            all_embeddings.append(item.embedding)

    return np.array(all_embeddings, dtype=np.float32)


def infer_appendix_or_part(file_name: str) -> Optional[str]:
    """Heuristic appendix/part label from filename."""
    name = file_name.lower()

    m = re.match(r"(appendix\s+[a-z0-9\- ]+)", name)
    if m:
        return m.group(1).title()

    m = re.match(r"(part\s+\d+)", name)
    if m:
        return m.group(1).title()

    return None


def extract_paragraph_refs(chunk: str) -> Optional[List[str]]:
    """
    Extract Immigration Rules paragraph references from a chunk.

    Returns list[str] or None.
    Handles:
      - Appendix FM / FM-SE codes: E-ECP.2.1., R-LTRP.1.1., GEN.3.1., EX.1., etc.
      - Skilled Worker shorthand: SW 1.1, SW 16.2A, etc.
      - Appendix codes with hyphens: E-LTRP.2.2., D-ECPT.1.1., etc.
      - Plain numeric paras like 9.1.1 / 9.2A when context indicates rules.
      - Range expressions capturing both ends.
    """
    text = chunk
    refs = set()

    # 1) FM / FM-SE / appendix dot codes ending in '.'
    fm_like = re.findall(
        r"\b([A-Z]{1,2}(?:-[A-Z]{2,6})?\.\d+(?:\.\d+)*[A-Z]?\.)\b",
        text
    )
    refs.update(fm_like)

    # 2) GEN / EX / suitability etc explicit blocks
    gen_ex = re.findall(
        r"\b((?:GEN|EX|S-EC|S-LTR|S-ILR|EC-PPT|ECP|LTRP)\.\d+(?:\.\d+)*[A-Z]?\.)\b",
        text
    )
    refs.update(gen_ex)

    # 3) Shorthand blocks (Skilled Worker etc)
    shorthand = re.findall(
        r"\b([A-Z]{1,4})\s(\d+(?:\.\d+)*[A-Z]?)\b",
        text
    )
    allowed_blocks = {"SW", "V", "EL", "HCCW", "GBM", "T5", "T2", "PBS", "FIN"}
    for block, num in shorthand:
        if block in allowed_blocks:
            refs.add(f"{block} {num}")

    # 4) Numeric paras in context (Part 9 etc)
    numeric_context = re.findall(
        r"(?:paragraph|para|part|rule|under)\s+(\d+(?:\.\d+){1,3}[A-Z]?)\b",
        text,
        flags=re.IGNORECASE
    )
    refs.update(numeric_context)

    # 5) Ranges like “E-ECP.2.1. to E-ECP.2.3.”
    ranges = re.findall(
        r"\b([A-Z]{1,2}(?:-[A-Z]{2,6})?\.\d+(?:\.\d+)*[A-Z]?\.)\s*(?:to|\-|–)\s*([A-Z]{1,2}(?:-[A-Z]{2,6})?\.\d+(?:\.\d+)*[A-Z]?\.)\b",
        text
    )
    for start, end in ranges:
        refs.add(start)
        refs.add(end)

    if not refs:
        return None

    return sorted(refs)


def rebuild_index_from_drive(tagged_files: List[Tuple[Dict, str]]):
    """
    Download files, extract text, chunk, embed, and rebuild FAISS + metadata
    from ONLY the two target folders.
    """
    service = get_drive_service()
    all_chunks = []
    metadata = []

    for file, source_type in tagged_files:
        file_id = file["id"]
        file_name = file.get("name", "unnamed")
        mime_type = file.get("mimeType", "")
        file_path = file.get("path", file_name)

        file_bytes, effective_mime = download_file_bytes(service, file)

        text = extract_text_from_bytes(file_bytes, effective_mime, file_name)
        text = normalise_rule_text(text)
        if not text.strip():
            continue

        chunks = split_into_chunks(text)
        appendix_or_part = infer_appendix_or_part(file_name)

        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append(
                {
                    "content": chunk,
                    "file_id": file_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "chunk_index": idx,

                    # ✅ Used by app.py
                    "type": source_type,  # "rule" or "precedent"
                    "source": file_name,
                    "appendix_or_part": appendix_or_part,
                    "paragraph_refs": extract_paragraph_refs(chunk) if source_type == "rule" else None,
                }
            )

    if not all_chunks:
        dim = 1536
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


def sync_drive_and_rebuild_index_if_needed(
    *,
    bypass_cooldown: bool = False,
    respect_overnight_window: bool = True,
):
    """
    Check ONLY the two target folders for new/updated files; rebuild if changed.

    Controls:
      - Cooldown: by default uses a configurable cooldown to avoid repeated Drive scans.
      - Overnight window: optionally skip Drive checks outside a UK overnight window to
        keep daytime sessions snappy. Missing artifacts always trigger a rebuild.
      - bypass_cooldown: set True for scheduled off-peak runs that should always check.
    """
    previous_state = load_previous_state()

    artifacts_missing = not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE)

    # ---- UK OVERNIGHT WINDOW GUARD (opt-out) ----
    if respect_overnight_window and not artifacts_missing:
        if not is_within_uk_overnight_window():
            return False
    # --------------------------------------------

    # ---- DAILY COOLDOWN ----
    last_checked = previous_state.get("last_checked")
    cooldown_seconds = get_drive_sync_cooldown_seconds()
    if last_checked and not bypass_cooldown:
        try:
            last_checked_dt = datetime.datetime.fromisoformat(last_checked.replace("Z", ""))
            age_seconds = (datetime.datetime.utcnow() - last_checked_dt).total_seconds()
            if age_seconds < cooldown_seconds:
                return False
        except Exception:
            pass
    # ------------------------

    tagged_files = list_drive_files_only_target_folders()
    changed, current_state = have_files_changed(tagged_files, previous_state)

    if changed or not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        rebuild_index_from_drive(tagged_files)

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

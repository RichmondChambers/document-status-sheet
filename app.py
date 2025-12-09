import streamlit as st
import faiss
import pickle
import numpy as np
import re
import json
import requests
import jwt  # PyJWT
import streamlit.components.v1 as components
from openai import OpenAI
from pathlib import Path
import base64
import pandas as pd
import io

from index_builder import sync_drive_and_rebuild_index_if_needed, INDEX_FILE, METADATA_FILE


# =========================
# 0. Google SSO
# =========================
def google_login():
    """
    Require the user to sign in with a Google account and restrict access
    to @richmondchambers.com email addresses.
    """
    if "user_email" in st.session_state:
        return st.session_state["user_email"]

    params = st.query_params
    if "code" in params:
        code = params["code"]

        token_response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": st.secrets["GOOGLE_CLIENT_ID"],
                "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
                "redirect_uri": st.secrets["GOOGLE_REDIRECT_URI"],
                "grant_type": "authorization_code",
            },
            timeout=15
        )

        if token_response.status_code != 200:
            st.error("Authentication with Google failed. Please refresh and try again.")
            st.stop()

        token_data = token_response.json()
        id_token = token_data.get("id_token")

        if not id_token:
            st.error("No ID token received from Google.")
            st.stop()

        try:
            # NOTE: signature verification skipped for simplicity
            claims = jwt.decode(id_token, options={"verify_signature": False})
        except Exception:
            st.error("Could not decode ID token.")
            st.stop()

        email = claims.get("email", "")
        hosted_domain = claims.get("hd", "")

        if email.endswith("@richmondchambers.com") or hosted_domain == "richmondchambers.com":
            st.session_state["user_email"] = email
            return email
        else:
            st.error("Access is restricted to employees of Richmond Chambers.")
            st.stop()

    auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        "?response_type=code"
        f"&client_id={st.secrets['GOOGLE_CLIENT_ID']}"
        f"&redirect_uri={st.secrets['GOOGLE_REDIRECT_URI']}"
        "&scope=openid%20email"
        "&prompt=select_account"
        "&access_type=offline"
    )

    st.markdown("### Richmond Chambers – Internal Tool")
    st.write("Please sign in with a Richmond Chambers Google Workspace account to access this app.")
    st.markdown(f"[Sign in with Google]({auth_url})")
    st.stop()


# =========================
# 1. Keys + Auth (OpenAI client)
# =========================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
user_email = google_login()


# =========================
# 2. FAISS + metadata
# =========================
@st.cache_resource
def load_index_and_metadata():
    """
    Ensure FAISS index is up to date, then load index, metadata,
    and read last rebuilt timestamp for UI display.
    """
    sync_drive_and_rebuild_index_if_needed()

    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)

    try:
        with open("drive_index_state.json", "r") as f:
            state = json.load(f)
            last_rebuilt = state.get("last_rebuilt", "Unknown")
    except Exception:
        last_rebuilt = "Unknown"

    return index, metadata, last_rebuilt


index, metadata, last_rebuilt = load_index_and_metadata()


# =========================
# 3. File extraction
# =========================
def extract_text_from_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception:
            return ""

    if name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(uploaded_file)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""

    try:
        return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


# =========================
# 4. Embeddings + retrieval
# =========================
def get_embedding(text, model="text-embedding-3-small"):
    result = client.embeddings.create(input=[text], model=model)
    return result.data[0].embedding


def search_index(query, k=8, source_type=None):
    """
    Search FAISS for top-k chunks.
    source_type filters by metadata[i]["type"] in {"rule","precedent"}.
    """
    if not metadata:
        return []

    query_embedding = get_embedding(query)
    distances, indices_ = index.search(
        np.array([query_embedding], dtype=np.float32),
        k * 3
    )

    results = []
    for i in indices_[0]:
        if i < len(metadata):
            item = metadata[i]
            if source_type and item.get("type") != source_type:
                continue
            results.append(item)
        if len(results) >= k:
            break
    return results


# =========================
# 5. Rule update date
# =========================
def fetch_latest_rule_update_date():
    """
    Light scrape of GOV.UK updates page.
    If it fails, fallback to today's date (UTC).
    """
    try:
        r = requests.get("https://www.gov.uk/guidance/immigration-rules/updates", timeout=10)
        if r.status_code != 200:
            raise Exception("Bad status")

        m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", r.text)
        if not m:
            raise Exception("No date found")
        return m.group(1)
    except Exception:
        return str(np.datetime64("today"))


# =========================
# 6. System prompt
# =========================
BASE_SYSTEM_PROMPT = """
You are a UK immigration lawyer specialising in document-checklist guidance for visa/immigration applications under the UK's Immigration Rules.

CRITICAL GROUNDING REQUIREMENT:
- You MUST use the lookup_rule tool to obtain the exact rule text supporting EVERY mandatory or recommended document requirement AND every discretion/judgment point.
- Do not invent citations. If you cannot find supporting text using lookup_rule, say so and limit guidance accordingly.

Hard rules:
- Base guidance ONLY on the Immigration Rules + official Home Office guidance. Avoid speculation.
- Quote/cite the exact relevant paragraph(s).
- Always reflect the April 2024 Appendix FM financial requirement (£29,000).
- Do NOT say two passport-sized photos are required.

Output rules (SPREADSHEET-READY TSV, CLIENT + LAWYER REVIEW):
- You MUST output ONLY tab-separated values (TSV). No Markdown, no bullets, no numbering, no pipes "|".
- Produce EXACTLY 4 columns per row:
  Column A: Document
  Column B: Evidential Requirements (what must be shown)
  Column C: Client Notes (clear, client-ready explanation; no legalese)
  Column D: Rule Authority (pinpoint paragraph reference + a SHORT supporting quotation)
- First row MUST be the header exactly:
  Document<TAB>Evidential Requirements<TAB>Client Notes<TAB>Rule Authority
- Do NOT include any extra text before or after the TSV.
- Put section titles as a standalone row in Column A only, like:
  === Section: Financial Requirement ===<TAB><TAB><TAB>
- NEVER start any cell with "=", "+", "-", or "@". If unavoidable, prefix that cell with "'".
- Client Notes must be concise, readable, and suitable to send to a client.
- Rule Authority must include:
  (i) the rule/appendix + paragraph reference, and
  (ii) a supporting quote no longer than ~25 words.
- If a filter is requested:
  * output ONLY that subset;
  * begin with a section row:
    === Filtered Checklist: {filter_label} ===<TAB><TAB><TAB>
- If you cannot find supporting rule text using lookup_rule, state that briefly in Rule Authority and keep the document as "Recommended (not mandatory)" where appropriate.

Link Appendix references to GOV.UK unless user asks otherwise.

Country-specificity:
- If nationality or location is provided, add relevant country-specific evidence notes (TB test, approved English tests, apostille/translation norms), citing GOV.UK guidance.

Legal Authority Summary:
- Do NOT include a separate summary section. All authority must be in Column D only.
"""


# =========================
# 7. Tool implementation: lookup_rule
# =========================
def lookup_rule_tool(appendix_or_part, paragraph_ref=None, query=None):
    """
    Backend tool the model calls.
    We search RULE chunks only.
    """
    q = " ".join([s for s in [appendix_or_part, paragraph_ref, query] if s])
    hits = search_index(q, k=3, source_type="rule")

    inferred_ref = None
    if hits:
        refs = hits[0].get("paragraph_refs") or []
        inferred_ref = refs[0] if refs else None

    return {
        "appendix_or_part": appendix_or_part,
        "paragraph_ref": paragraph_ref or inferred_ref,
        "passages": [
            {
                "text": h.get("content", ""),
                "source": h.get("source", ""),
                "paragraph_ref": (h.get("paragraph_refs") or [""])[0],
            }
            for h in hits
        ],
    }


# =========================
# 8. Model call with tool loop (HARDENED)
# =========================
def generate_checklist(route_text, facts_text, extra_route_facts_text=None, filter_mode=None, filter_label=None):
    rule_date = fetch_latest_rule_update_date()
    system_prompt = BASE_SYSTEM_PROMPT.replace("{RULE_UPDATE_DATE}", rule_date)

    enquiry_text = f"ROUTE:\n{route_text.strip()}\n\nFACTS:\n{facts_text.strip()}"

    rule_chunks = search_index(enquiry_text, k=10, source_type="rule")
    precedent_chunks = search_index(enquiry_text, k=4, source_type="precedent")

    grounding_context = "AUTHORITATIVE IMMIGRATION RULES EXTRACTS:\n"
    grounding_context += "\n\n".join(
        [
            f"[R{i+1}] ({(rc.get('appendix_or_part') or '')}"
            f"{' ' + (rc.get('paragraph_refs') or [''])[0] if rc.get('paragraph_refs') else ''}"
            f" — {rc.get('source','')})\n{rc.get('content','')}"
            for i, rc in enumerate(rule_chunks)
        ]
    )

    grounding_context += "\n\nINTERNAL PRECEDENT EXTRACTS (style only, not authority):\n"
    grounding_context += "\n\n".join(
        [
            f"[P{i+1}] ({pc.get('source','')})\n{pc.get('content','')}"
            for i, pc in enumerate(precedent_chunks)
        ]
    )

    if extra_route_facts_text and extra_route_facts_text.strip():
        grounding_context += "\n\nUPLOADED ROUTE/FACTS DOCUMENT:\n"
        grounding_context += extra_route_facts_text.strip()

    user_instruction = enquiry_text
    if filter_mode:
        user_instruction += (
            f"\n\nFILTER LABEL: {filter_label}\n"
            f"FILTER REQUEST: {filter_mode}"
        )

    tools = [
        {
            "type": "function",
            "name": "lookup_rule",
            "description": "Return exact Immigration Rules text for a given appendix/part and paragraph reference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appendix_or_part": {"type": "string"},
                    "paragraph_ref": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["appendix_or_part"],
            },
        }
    ]

    # ---- First call (allow tools) ----
    resp = client.responses.create(
        model="gpt-5.1",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": grounding_context},
            {"role": "user", "content": user_instruction},
        ],
        tools=tools,
        tool_choice="auto",
        temperature=0.2,
    )

    # Collect any tool calls
    pending_tool_calls = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) in ("tool_call", "function_call") and getattr(item, "name", None) == "lookup_rule":
            pending_tool_calls.append(item)

    # If no tools called, return text directly
    if not pending_tool_calls:
        return (getattr(resp, "output_text", "") or "").strip()

    # ---- Resolve tools ourselves ----
    resolved_rules = []
    for tc in pending_tool_calls:
        raw_args = getattr(tc, "arguments", None) or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except Exception:
            args = {}

        tool_result = lookup_rule_tool(
            appendix_or_part=args.get("appendix_or_part", ""),
            paragraph_ref=args.get("paragraph_ref"),
            query=args.get("query"),
        )
        resolved_rules.append(tool_result)

    # Add resolved rule text into context
    resolved_context = grounding_context + "\n\nLOOKUP_RULE RESULTS (authoritative):\n"
    for i, tr in enumerate(resolved_rules):
        resolved_context += f"\n[LR{i+1}] {tr.get('appendix_or_part','')} {tr.get('paragraph_ref','')}\n"
        for p in tr.get("passages", []):
            resolved_context += f"- ({p.get('source','')} {p.get('paragraph_ref','')}) {p.get('text','')}\n"

    # ---- Second full call (NO tools, NO resume) ----
    followup = client.responses.create(
        model="gpt-5.1",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": resolved_context},
            {"role": "user", "content": user_instruction},
        ],
        temperature=0.2,
    )

    return (getattr(followup, "output_text", "") or "").strip()


# =========================
# 9. Streamlit UI helpers
# =========================
def render_logo():
    """
    Load logo.png from the repo root (same folder as this app.py),
    embed as base64, and center it perfectly with HTML/CSS.
    """
    logo_path = Path(__file__).parent / "logo.png"
    if logo_path.exists():
        b64 = base64.b64encode(logo_path.read_bytes()).decode()
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; padding-bottom: 10px;">
                <img src="data:image/png;base64,{b64}" width="150">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f"Logo not found at {logo_path}")


def sanitize_for_sheets(tsv_text: str) -> str:
    """
    Safety net to:
    - strip any stray markdown pipes
    - prevent Sheets formula parsing
    - force exactly 4 columns per row (pad/merge)
    """
    lines = []
    for line in tsv_text.splitlines():
        line = line.replace("|", "")
        cols = line.split("\t")

        if len(cols) < 4:
            cols = cols + [""] * (4 - len(cols))
        elif len(cols) > 4:
            cols = cols[:3] + [" ".join(cols[3:])]

        safe_cols = []
        for c in cols:
            c = c.strip()
            if c.startswith(("=", "+", "-", "@")):
                c = "'" + c
            safe_cols.append(c)

        lines.append("\t".join(safe_cols))

    return "\n".join(lines).strip()


def add_numbering_column(tsv_text: str) -> str:
    """
    Add a leftmost numbering column ("No.").

    Rules:
    - Header row becomes: No. | Document | Evidential Requirements | Client Notes | Rule Authority
    - Section/heading rows (Document cell starts with "===") are NOT numbered.
    - Blank rows are not numbered.
    - Only real document rows get consecutive numbers.
    """
    lines = tsv_text.splitlines()
    if not lines:
        return tsv_text

    out = []
    counter = 0

    for i, line in enumerate(lines):
        cols = line.split("\t")

        # Ensure at least 4 cols before numbering
        if len(cols) < 4:
            cols = cols + [""] * (4 - len(cols))
        elif len(cols) > 4:
            cols = cols[:3] + [" ".join(cols[3:])]

        doc_cell = (cols[0] or "").strip()

        if i == 0:
            out.append("\t".join(["No."] + cols))
            continue

        is_heading = doc_cell.startswith("===")
        is_blank = doc_cell == ""

        if is_heading or is_blank:
            out.append("\t".join([""] + cols))
        else:
            counter += 1
            out.append("\t".join([str(counter)] + cols))

    return "\n".join(out).strip()


def strip_authority_column(tsv_text: str) -> str:
    """
    Convert TSV into client 3-column version by removing Rule Authority.

    Handles BOTH:
    - 4-col input: Document | Evidential | Client Notes | Rule Authority
    - 5-col input (numbered): No. | Document | Evidential | Client Notes | Rule Authority
    """
    out_lines = []
    for i, line in enumerate(tsv_text.splitlines()):
        cols = line.split("\t")

        numbered = (i == 0 and len(cols) >= 1 and cols[0].strip().lower() in ("no.", "no", "#"))

        if numbered:
            if len(cols) < 5:
                cols = cols + [""] * (5 - len(cols))
            cols4 = cols[:4]  # No. + first 3 content cols
            if i == 0:
                cols4 = ["No.", "Document", "Evidential Requirements", "Client Notes"]
            out_lines.append("\t".join(c.strip() for c in cols4))
        else:
            if len(cols) < 4:
                cols = cols + [""] * (4 - len(cols))
            cols3 = cols[:3]
            if i == 0:
                cols3 = ["Document", "Evidential Requirements", "Client Notes"]
            out_lines.append("\t".join(c.strip() for c in cols3))

    return "\n".join(out_lines).strip()


def tsv_to_dataframe(tsv_text: str) -> pd.DataFrame:
    """
    Convert TSV text into a DataFrame.
    Assumes first row is header.
    """
    if not tsv_text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(tsv_text), sep="\t", dtype=str).fillna("")


def dataframe_to_formatted_xlsx_bytes(df: pd.DataFrame, sheet_name="Status Sheet") -> bytes:
    """
    Export DataFrame to a formatted Excel file (bytes) using openpyxl:
    - wrap text
    - centre align
    - bold header
    - freeze row 1
    - set column widths
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, Border, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        raise ImportError(
            "openpyxl is required for formatted Excel export. "
            "Add 'openpyxl' to requirements.txt."
        )

    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name

    thin = Side(style="thin", color="000000")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # Write header
    ws.append(list(df.columns))
    for col_idx in range(1, len(df.columns) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = Font(bold=True)
        cell.alignment = header_alignment
        cell.border = border

    # Write body
    for row_vals in df.itertuples(index=False):
        ws.append(list(row_vals))

    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
        for cell in row:
            cell.alignment = cell_alignment
            cell.border = border

    # Freeze header row
    ws.freeze_panes = "A2"

    # Column widths
    for col_idx, col_name in enumerate(df.columns, start=1):
        values = [str(col_name)] + [str(v) for v in df.iloc[:, col_idx - 1].values]
        max_len = max(len(v) for v in values)
        width = min(max(max_len * 0.9, 10), 60)
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer.read()


# =========================
# Filter options (label -> precise model instruction)
# =========================
FILTER_OPTIONS = {
    "Full checklist": None,

    "Mandatory documents only": (
        "Return ONLY documents that are strictly mandatory under the Immigration Rules or Home Office guidance "
        "(i.e., specified evidence or items without which the application is likely to be refused/invalid). "
        "Exclude purely recommended/supporting documents unless they are needed to satisfy a mandatory requirement."
    ),

    "Identity / nationality evidence only": (
        "Return ONLY identity and nationality evidence. Include passports/travel documents, BRP/eVisa status proof, "
        "national ID cards, biometrics/photo requirements where relevant, name-change evidence, and any identity-linked "
        "civil documents. Exclude financial, relationship, accommodation, or other categories."
    ),

    "Immigration history / status evidence only": (
        "Return ONLY evidence relating to the applicant’s UK immigration history and current/previous status. "
        "Include current grant/leave evidence, visa vignettes, entry/exit stamps, prior approvals/refusals/curtailments, "
        "overstay or breach explanations, conditions of leave, and any required history disclosures."
    ),

    "Application forms / administrative evidence only": (
        "Return ONLY administrative/process documents needed to lodge the application. "
        "Include online application form confirmation, IHS/payment receipts, appointment/biometrics confirmation, "
        "consent forms, document checklists, and any required declarations. "
        "Include translation/certification requirements ONLY insofar as they relate to admin validity."
    ),

    "Financial requirement evidence only": (
        "Return ONLY evidence proving the financial requirement for the route. "
        "Include specified evidence per the Rules/guidance: payslips, bank statements, employer letters, tax returns, "
        "audited/unaudited accounts, dividend vouchers, savings evidence, pension evidence, benefits letters, "
        "and evidence of source/ownership of funds where required."
    ),

    "Accommodation evidence only": (
        "Return ONLY evidence proving adequate accommodation. "
        "Include tenancy/mortgage documents, landlord/freeholder consent, property ownership proof, "
        "property inspection/overcrowding reports where relevant, utilities/council tax if used to prove residence, "
        "and sponsor residence proof tied to accommodation adequacy."
    ),

    "English language evidence only": (
        "Return ONLY evidence proving English language requirement or exemption. "
        "Include approved SELT certificate, degree taught in English plus ECCTIS/UK NARIC comparability if required, "
        "nationality-based exemptions, age/medical exemptions with supporting proof."
    ),

    "Relationship / family evidence only": (
        "Return ONLY evidence proving relationship/family link. "
        "Include marriage/civil partnership certificates, divorce/death certificates, evidence of durable partnership, "
        "cohabitation evidence, communication/interaction evidence, family composition proof. "
        "Exclude genuineness-focused narrative unless it is part of proving the relationship."
    ),

    "Genuine relationship / genuine intention evidence only": (
        "Return ONLY evidence aimed at demonstrating a genuine relationship or genuine intention to live together/continue "
        "the relationship (where required). "
        "Include relationship timeline, communications, visits, shared life evidence, joint commitments, "
        "statements addressing genuineness. Exclude civil status documents unless needed to support genuineness."
    ),

    "Employment / role evidence only": (
        "Return ONLY evidence of a job/role where employment is a core requirement. "
        "Include CoS, job offer/contract, SOC code fit evidence, salary breakdown, start date confirmation, "
        "sponsor licence/status proof, employer letters describing duties. "
        "Exclude general financial evidence unless required to prove salary for the role."
    ),

    "Qualifications / skills evidence only": (
        "Return ONLY evidence of qualifications/skills relevant to eligibility. "
        "Include degree certificates/transcripts, professional qualifications, registrations/licences, "
        "ATAS if applicable, evidence of equivalency or recognition where required."
    ),

    "Maintenance / funds evidence only (non-salary routes)": (
        "Return ONLY evidence of maintenance/funds where the route requires proof of funds rather than salary "
        "(e.g., Student/Visitor/PBS dependants). "
        "Include bank statements showing required level and holding period, proof of ownership/control, "
        "sponsor undertaking where permitted, financial consent letters if jointly held."
    ),

    "Sponsor evidence only": (
        "Return ONLY evidence relating to the sponsor. "
        "Include sponsor identity/status (passport/BRP/eVisa/ILR/citizenship), immigration permission, "
        "residence in the UK if required, sponsor employment/financial documents only where sponsor-led financial "
        "requirements apply."
    ),

    "Dependants’ evidence only": (
        "Return ONLY evidence needed for dependants (partner/children/other dependants). "
        "Include relationship to main applicant, dependency evidence, age evidence, living arrangements, "
        "and route-specific dependant requirements."
    ),

    "Children / parental responsibility evidence only": (
        "Return ONLY evidence concerning children and parental responsibility. "
        "Include full birth certificates, adoption/custody orders, parental consent letters, "
        "sole responsibility evidence, evidence of child’s residence, school/medical records where used to show care."
    ),

    "Study / CAS evidence only": (
        "Return ONLY study-related evidence for Student routes. "
        "Include CAS statement/number, offer letters, course details, tuition payment evidence if relevant, "
        "academic progression evidence, ATAS where required."
    ),

    "Business / investment evidence only": (
        "Return ONLY business/investment-related evidence for entrepreneurship/investor/GBM/self-sponsorship style routes. "
        "Include business plans, company registration/ownership, corporate structures, share certificates, "
        "investment source and availability evidence, contracts/invoices, overseas business link evidence for GBM routes."
    ),

    "TB / medical evidence only": (
        "Return ONLY TB/medical evidence. "
        "Include TB test certificates from approved clinics where required, "
        "medical evidence supporting exemptions or compassionate/discretionary factors where relevant."
    ),

    "Criminality / character evidence only": (
        "Return ONLY evidence relating to criminality/character. "
        "Include police certificates where required, court records, sentencing details, rehabilitation evidence, "
        "and explanatory statements addressing character/suitability."
    ),

    "Suitability / refusal-risk evidence only": (
        "Return ONLY evidence aimed at addressing suitability or refusal risks. "
        "Include explanations and proof relating to deception concerns, overstays/breaches, sham/genuineness doubts, "
        "credibility gaps, inconsistencies, previous refusals, and mitigation evidence."
    ),

    "Exceptional circumstances / discretion evidence only": (
        "Return ONLY evidence supporting exceptional circumstances or discretionary grants. "
        "Include Article 8/private and family life factors, compelling compassionate evidence, hardship, "
        "best-interests-of-child materials, medical dependency evidence, "
        "and any other materials supporting discretion outside strict rule satisfaction."
    ),

    "Translations / format / copy certification evidence only": (
        "Return ONLY evidence about translations, formatting, and certification. "
        "Include requirements for certified translations, translator credentials, "
        "copy certification wording, legibility/completeness requirements (e.g., all pages of statements), "
        "and document date/validity formatting rules."
    ),

    "Country-specific evidence only": (
        "Return ONLY country-specific evidence requirements triggered by nationality/location. "
        "Include TB test triggers, local civil document formats, apostille/legalisation norms, "
        "approved test availability by country, and any region-specific Home Office requirements."
    ),
}


# =========================
# 10. Streamlit UI
# =========================
render_logo()

st.markdown(
    "<h1 style='text-align: center; font-size: 2.6rem;'>Document Status Sheet Generator</h1>",
    unsafe_allow_html=True
)

st.markdown(
    f"<p style='color: grey; text-align: center; font-size: 0.9rem;'>"
    f"Immigration Rules index last rebuilt from Drive on: <b>{last_rebuilt}</b></p>",
    unsafe_allow_html=True
)

st.markdown(
    "Provide the immigration route and the case facts in separate fields. "
    "The app will generate a rule-based document status sheet "
    "in 5 columns suitable for Google Sheets, including numbering, client-ready notes, and rule authority."
)

uploaded_doc = st.file_uploader(
    "Optional: upload a document describing the route and facts.",
    type=["pdf", "txt", "docx"],
    help="E.g., case summary, client instructions, or notes setting out the route and facts."
)

filter_label = st.selectbox(
    "Checklist filter (optional)",
    list(FILTER_OPTIONS.keys()),
    index=0
)
filter_instruction = FILTER_OPTIONS[filter_label]

with st.form("checklist_form"):
    route = st.text_area(
        "Route",
        height=140,
        placeholder="Example:\nSpouse visa extension under Appendix FM."
    )
    facts = st.text_area(
        "Facts (include applicant nationality/location if relevant)",
        height=260,
        placeholder=(
            "Example:\n"
            "Sponsor is British citizen. Applicant is Swiss, applying from London. "
            "Relationship married 3 years, cohabiting. Salaried income £35,000. "
            "One child British. Need a rule-based document status sheet."
        )
    )
    submit = st.form_submit_button("Generate Status Sheet")


if submit and (route.strip() or facts.strip()):
    with st.spinner("Retrieving Rules, checking precedents, and generating status sheet..."):
        extra_text = None
        if uploaded_doc is not None:
            extra_text = extract_text_from_uploaded_file(uploaded_doc)

        reply = generate_checklist(
            route_text=route,
            facts_text=facts,
            extra_route_facts_text=extra_text,
            filter_mode=filter_instruction,
            filter_label=filter_label
        )

        reply = sanitize_for_sheets(reply)
        reply = add_numbering_column(reply)

        st.session_state["internal_tsv"] = reply
        st.session_state.pop("client_tsv", None)  # reset client version on new run

        st.success("Status sheet generated.")

        st.subheader("Status Sheet Output")
        tab_internal, tab_client = st.tabs(["Internal review (5 columns)", "Client version (4 columns)"])

        with tab_internal:
            st.write("Includes Rule Authority column for lawyer review.")
            internal_tsv = st.session_state.get("internal_tsv", reply)
            st.code(internal_tsv, language="text")

            # Formatted XLSX download (internal)
            internal_df = tsv_to_dataframe(internal_tsv)
            if not internal_df.empty:
                try:
                    internal_xlsx = dataframe_to_formatted_xlsx_bytes(internal_df, sheet_name="Internal Review")
                    st.download_button(
                        "Download INTERNAL formatted Excel (.xlsx)",
                        data=internal_xlsx,
                        file_name="document_status_sheet_internal.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.warning(f"Formatted Excel export unavailable: {e}")

        with tab_client:
            st.write("Click to generate a client-ready version (no authority column).")
            make_client = st.button("Generate client version (strip Rule Authority)")

            if make_client:
                internal_tsv = st.session_state.get("internal_tsv", "")
                client_tsv = strip_authority_column(internal_tsv) if internal_tsv else ""
                st.session_state["client_tsv"] = client_tsv

            client_tsv = st.session_state.get("client_tsv", "")
            if client_tsv:
                st.code(client_tsv, language="text")
                st.download_button(
                    "Download client TSV",
                    data=client_tsv,
                    file_name="document_status_sheet_client.tsv",
                    mime="text/tab-separated-values"
                )

                # Formatted XLSX download (client)
                client_df = tsv_to_dataframe(client_tsv)
                if not client_df.empty:
                    try:
                        client_xlsx = dataframe_to_formatted_xlsx_bytes(client_df, sheet_name="Client Version")
                        st.download_button(
                            "Download CLIENT formatted Excel (.xlsx)",
                            data=client_xlsx,
                            file_name="document_status_sheet_client.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.warning(f"Formatted Excel export unavailable: {e}")
            else:
                st.info("Client version will appear here after you click the button.")

        st.markdown(
            """
            ---  
            **Professional Responsibility Statement**

            AI-generated content must not be relied upon without human review. Where such
            content is used, the barrister is responsible for verifying and ensuring the accuracy
            and legal soundness of that content. AI tools are used solely to support drafting and
            research; they do not replace the barrister’s independent judgment, analysis, or duty
            of care.
            """,
            unsafe_allow_html=False,
        )

        # Copy button (copies internal TSV by default)
        components.html(
            f"""
            <style>
            .copy-button {{
                margin-top: 10px;
                padding: 8px 16px;
                background-color: #2e2e2e;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }}
            .copy-button:hover {{ background-color: #4a4a4a; }}
            </style>

            <button class="copy-button" onclick="copyToClipboard()">📋 Copy internal TSV to Clipboard</button>

            <script>
            async function copyToClipboard() {{
                const plainText = `{reply.replace("`", "\\`")}`;
                const blobText = new Blob([plainText], {{ type: 'text/plain' }});
                const clipboardItem = new ClipboardItem({{
                    'text/plain': blobText
                }});
                await navigator.clipboard.write([clipboardItem]);
                alert("Copied! Paste into Google Sheets.");
            }}
            </script>
            """,
            height=110,
            scrolling=False
        )

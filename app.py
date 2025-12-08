import streamlit as st
import openai
import faiss
import pickle
import numpy as np
import re
import json
import requests
import jwt  # from PyJWT
import streamlit.components.v1 as components
from markdown_it import MarkdownIt
from index_builder import sync_drive_and_rebuild_index_if_needed, INDEX_FILE, METADATA_FILE


# =========================
# 0. Google SSO (unchanged)
# =========================
def google_login():
    """
    Require the user to sign in with a Google account and restrict access
    to @richmondchambers.com email addresses.
    """
    if "user_email" in st.session_state:
        return st.session_state["user_email"]

    params = st.experimental_get_query_params()
    if "code" in params:
        code = params["code"][0]

        token_response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": st.secrets["GOOGLE_CLIENT_ID"],
                "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
                "redirect_uri": st.secrets["GOOGLE_REDIRECT_URI"],
                "grant_type": "authorization_code",
            },
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
# 1. Keys + Auth
# =========================
openai.api_key = st.secrets["OPENAI_API_KEY"]
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
# 3. File extraction (unchanged)
# =========================
def extract_text_from_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    elif name.endswith(".pdf"):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception:
            return ""

    elif name.endswith(".docx"):
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
    result = openai.embeddings.create(input=[text], model=model)
    return result.data[0].embedding


def search_index(query, k=8, source_type=None):
    """
    Search FAISS for top-k chunks.

    If source_type is provided ("rule" or "precedent"),
    filter results by metadata[i].get("type").

    IMPORTANT: Your index_builder should set metadata items like:
      {
        "content": "...chunk text...",
        "source": "Appendix FM.pdf",
        "type": "rule",  # or "precedent"
        "appendix_or_part": "Appendix FM",
        "paragraph_ref": "E-ECP.2.1."  # optional if you can extract it
      }
    """
    query_embedding = get_embedding(query)
    distances, indices_ = index.search(np.array([query_embedding], dtype=np.float32), k*3)

    results = []
    for i in indices_[0]:
        if i < len(metadata):
            item = metadata[i]
            if source_type:
                # safe fallback if old metadata doesn't have type
                t = item.get("type")
                if t != source_type:
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
    If it fails, fallback to today's date.
    """
    try:
        r = requests.get("https://www.gov.uk/guidance/immigration-rules/updates", timeout=10)
        if r.status_code != 200:
            raise Exception("Bad status")

        # crude but reliable: look for first YYYY-MM-DD in page header area
        m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", r.text)
        if not m:
            raise Exception("No date found")

        return m.group(1)
    except Exception:
        return str(np.datetime64("today"))


# =========================
# 6. System prompt for your Custom GPT
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

Output rules:
- Start every response with:
  “The guidance below is based on the Immigration Rules as updated on {RULE_UPDATE_DATE}, per the Home Office updates page.”
- Use clear headings and legal-professional tone suitable for case notes.
- Present each checklist as THREE columns:
  Column A: Document
  Column B: Evidential Requirements
  Column C: Notes (use for discretion/judgment calls; quote/cite rule).
- Organise with Immigration-Rule-based sub-headings.
- If multiple applicants: separate checklists with:
  “==== Main Applicant ====”, “==== Dependent Partner ====”, etc.
- If the user requests a filter, return only that subset and label it
  “📄 Filtered Checklist: …”.

Link Appendix references to GOV.UK unless user asks otherwise.

Country-specificity:
- If nationality or location is provided, add relevant country-specific evidence notes (TB test, approved English tests, apostille/translation norms), citing GOV.UK guidance.

Bundle review:
- If user provides/uploads a draft bundle, compare it against rule-based requirements; flag missing mandatory docs and insufficiencies, each with rule citation.

Legal Authority Summary:
- If any rules cited, end with “Legal Authority Summary” listing each cited rule with GOV.UK hyperlink + one-line scope.
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

    return {
        "appendix_or_part": appendix_or_part,
        "paragraph_ref": paragraph_ref or hits[0].get("paragraph_ref") if hits else None,
        "passages": [
            {
                "text": h.get("content", ""),
                "source": h.get("source", ""),
                "paragraph_ref": h.get("paragraph_ref", "")
            }
            for h in hits
        ]
    }


# =========================
# 8. Model call with tool loop
# =========================
def generate_checklist(enquiry_text, extra_bundle_text=None, filter_mode=None):
    """
    Single-call checklist generation, with forced lookup_rule tool usage.
    """
    rule_date = fetch_latest_rule_update_date()
    system_prompt = BASE_SYSTEM_PROMPT.replace("{RULE_UPDATE_DATE}", rule_date)

    # Retrieve rules + precedents
    rule_chunks = search_index(enquiry_text, k=10, source_type="rule")
    precedent_chunks = search_index(enquiry_text, k=4, source_type="precedent")

    grounding_context = "AUTHORITATIVE IMMIGRATION RULES EXTRACTS:\n"
    grounding_context += "\n\n".join(
        [f"[R{i+1}] ({rc.get('appendix_or_part','')}{' '+rc.get('paragraph_ref','') if rc.get('paragraph_ref') else ''} — {rc.get('source','')})\n{rc.get('content','')}"
         for i, rc in enumerate(rule_chunks)]
    )

    grounding_context += "\n\nINTERNAL PRECEDENT EXTRACTS (style only, not authority):\n"
    grounding_context += "\n\n".join(
        [f"[P{i+1}] ({pc.get('source','')})\n{pc.get('content','')}"
         for i, pc in enumerate(precedent_chunks)]
    )

    # Add uploaded bundle if present
    if extra_bundle_text and extra_bundle_text.strip():
        grounding_context += "\n\nUPLOADED DRAFT BUNDLE / USER DOCUMENT:\n"
        grounding_context += extra_bundle_text.strip()

    # If filter requested
    user_instruction = enquiry_text
    if filter_mode and filter_mode != "Full checklist":
        user_instruction += f"\n\nFILTER REQUEST: {filter_mode}"

    # Call Responses API (Python SDK style)
    resp = openai.responses.create(
        model="gpt-5.1",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": grounding_context},
            {"role": "user", "content": user_instruction},
        ],
        tools=[
            {
                "type": "function",
                "name": "lookup_rule",
                "description": "Return exact Immigration Rules text for a given appendix/part and paragraph reference. Use to support every requirement/discretion item.",
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
        ],
        tool_choice="auto",
        temperature=0.2,
    )

    # Tool handling loop
    output_text = ""
    pending_tool_calls = []

    for item in resp.output:
        if item.type == "tool_call" and item.name == "lookup_rule":
            pending_tool_calls.append(item)
        elif item.type == "output_text":
            output_text += item.text

    # If tools were called, resolve them and do a follow-up call
    if pending_tool_calls:
        tool_messages = []
        for tc in pending_tool_calls:
            args = tc.arguments or {}
            tool_result = lookup_rule_tool(
                appendix_or_part=args.get("appendix_or_part", ""),
                paragraph_ref=args.get("paragraph_ref"),
                query=args.get("query"),
            )
            tool_messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(tool_result)}
            )

        followup = openai.responses.create(
            model="gpt-5.1",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": grounding_context},
                {"role": "user", "content": user_instruction},
                # tell model we're giving tool results
                {"role": "assistant", "content": "Tool results provided. Continue and produce final checklist."},
                *tool_messages
            ],
            temperature=0.2,
        )
        output_text = followup.output_text

    return output_text


# =========================
# 9. Streamlit UI
# =========================
st.markdown(
    "<h1 style='text-align: center; font-size: 2.6rem;'>Document Checklist Generator</h1>",
    unsafe_allow_html=True
)

st.markdown(
    f"<p style='color: grey; text-align: center; font-size: 0.9rem;'>Immigration Rules index last rebuilt from Drive on: <b>{last_rebuilt}</b></p>",
    unsafe_allow_html=True
)

st.markdown(
    "Provide the route + facts. The app will generate a rule-based document checklist "
    "in 3 columns suitable for Google Sheets, with exact rule quotations."
)

uploaded_bundle = st.file_uploader(
    "Optional: upload a draft bundle or indexed list to review against the Rules.",
    type=["pdf", "txt", "docx"],
    help="E.g., draft document index, refusal letter, or current bundle."
)

filter_mode = st.selectbox(
    "Checklist filter (optional)",
    [
        "Full checklist",
        "Mandatory documents only",
        "Financial evidence only",
        "Identity / nationality documents only",
        "Relationship evidence only",
        "Immigration history only",
    ],
    index=0
)

with st.form("checklist_form"):
    enquiry = st.text_area(
        "Route + Facts (include applicant nationality/location if relevant)",
        height=260,
        placeholder=(
            "Example:\n"
            "Spouse visa extension (Appendix FM). Sponsor is British citizen. "
            "Applicant is Swiss, applying from London. Relationship married 3 years, "
            "cohabiting. Salaried income £35,000. One child British. "
            "Need a rule-based checklist."
        )
    )
    submit = st.form_submit_button("Generate Checklist")


if submit and enquiry:
    with st.spinner("Retrieving Rules, checking precedents, and generating checklist..."):
        extra_text = None
        if uploaded_bundle is not None:
            extra_text = extract_text_from_uploaded_file(uploaded_bundle)

        reply = generate_checklist(
            enquiry_text=enquiry,
            extra_bundle_text=extra_text,
            filter_mode=filter_mode
        )

        st.success("Checklist generated.")

        st.subheader("Checklist Output (copy into Google Sheets)")
        st.text_area("Checklist", value=reply, height=650)

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

        # Copy button (same as before)
        md = MarkdownIt()
        html_reply = md.render(reply)

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

            <button class="copy-button" onclick="copyToClipboard()">📋 Copy to Clipboard</button>

            <script>
            async function copyToClipboard() {{
                const htmlContent = `{html_reply.replace("`", "\\`")}`;
                const plainText = `{reply.replace("`", "\\`")}`;

                const blobHtml = new Blob([htmlContent], {{ type: 'text/html' }});
                const blobText = new Blob([plainText], {{ type: 'text/plain' }});

                const clipboardItem = new ClipboardItem({{
                    'text/html': blobHtml,
                    'text/plain': blobText
                }});

                await navigator.clipboard.write([clipboardItem]);
                alert("Copied! Paste into Gmail/Docs/Sheets.");
            }}
            </script>
            """,
            height=110,
            scrolling=False
        )

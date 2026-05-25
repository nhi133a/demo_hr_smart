import os
import hashlib

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import certifi
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient

from jd_matcher import match_jd_to_cvs
from jd_store import SAMPLE_JDS, count_indexed_jds, ingest_jd_text, ingest_jds, list_indexed_jds
from mongo_utils import (
    count_documents,
    delete_documents_by_source,
    get_candidate_name,
    get_candidate_profile,
    get_chunks_by_source_for_matching,
    get_distinct_sources,
)
from rag_core import answer_question, process_multiple_pdfs

load_dotenv()

_mongo_client = None


def get_mongo_client():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(os.getenv("MONGO_URI"), tlsCAFile=certifi.where())
    return _mongo_client


client = get_mongo_client()

st.set_page_config(
    page_title="SmartHire - CV RAG Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_ui_theme():
    st.markdown(
        """
        <style>
            :root {
                --surface: #ffffff;
                --surface-soft: #f4f7fb;
                --ink: #17304d;
                --muted: #5d7188;
                --line: #d7e1ed;
                --navy: #173b63;
                --teal: #0f8b8d;
                --coral: #ef6f6c;
            }

            [data-testid="stAppViewContainer"] {
                background:
                    linear-gradient(135deg, rgba(15, 139, 141, 0.08), transparent 30%),
                    linear-gradient(180deg, #f7fbff 0%, #eef4fb 100%);
                color: var(--ink);
            }

            [data-testid="stHeader"] {
                background: rgba(247, 251, 255, 0.88);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #102d4a 0%, #153d62 100%);
            }

            [data-testid="stSidebar"] * {
                color: #eef6ff;
            }

            [data-testid="stSidebar"] [data-baseweb="select"] > div,
            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
                background: rgba(255, 255, 255, 0.12);
                border-color: rgba(255, 255, 255, 0.24);
            }

            .block-container {
                max-width: 1480px;
                padding-top: 1.35rem;
                padding-bottom: 2.5rem;
            }

            h1, h2, h3 {
                color: var(--ink);
                letter-spacing: 0;
            }

            .hero-panel {
                display: flex;
                align-items: flex-end;
                justify-content: space-between;
                gap: 1.5rem;
                padding: 1.65rem 1.8rem;
                margin-bottom: 1rem;
                border: 1px solid rgba(23, 59, 99, 0.12);
                border-radius: 8px;
                background: linear-gradient(120deg, rgba(23, 59, 99, 0.96), rgba(15, 139, 141, 0.88));
                box-shadow: 0 18px 48px rgba(23, 48, 77, 0.14);
            }

            .hero-eyebrow {
                color: #bde9e7;
                font-size: 0.82rem;
                font-weight: 700;
                margin-bottom: 0.35rem;
                text-transform: uppercase;
            }

            .hero-panel h1 {
                color: #ffffff;
                font-size: clamp(1.9rem, 2.5vw, 3rem);
                line-height: 1.08;
                margin: 0;
            }

            .hero-panel p {
                color: rgba(255, 255, 255, 0.86);
                font-size: 1rem;
                margin: 0.65rem 0 0;
                max-width: 760px;
            }

            .hero-badge {
                min-width: 190px;
                padding: 0.85rem 1rem;
                border: 1px solid rgba(255, 255, 255, 0.22);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.13);
                color: #ffffff;
            }

            .hero-badge strong {
                display: block;
                font-size: 1.25rem;
            }

            .hero-badge span {
                color: rgba(255, 255, 255, 0.78);
                font-size: 0.84rem;
            }

            .section-label {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                margin: 0.4rem 0 0.25rem;
                color: var(--teal);
                font-size: 0.82rem;
                font-weight: 700;
                text-transform: uppercase;
            }

            .section-label::before {
                width: 0.55rem;
                height: 0.55rem;
                border-radius: 2px;
                background: var(--coral);
                content: "";
            }

            .panel-note {
                margin: 0.2rem 0 0.95rem;
                color: var(--muted);
            }

            .sidebar-brand {
                padding: 0.35rem 0 1rem;
            }

            .sidebar-brand strong {
                display: block;
                color: #ffffff;
                font-size: 1.35rem;
            }

            .sidebar-brand span {
                color: rgba(238, 246, 255, 0.72);
                font-size: 0.88rem;
            }

            div[data-testid="stMetric"] {
                padding: 0.95rem 1rem;
                border: 1px solid var(--line);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.88);
                box-shadow: 0 10px 28px rgba(23, 48, 77, 0.08);
            }

            div[data-testid="stExpander"],
            div[data-testid="stChatMessage"] {
                border: 1px solid var(--line);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.84);
            }

            .stButton > button {
                min-height: 2.55rem;
                border: 0;
                border-radius: 8px;
                background: linear-gradient(90deg, var(--navy), var(--teal));
                color: #ffffff;
                font-weight: 700;
                box-shadow: 0 10px 22px rgba(15, 139, 141, 0.18);
            }

            .stButton > button:hover {
                border: 0;
                color: #ffffff;
                filter: brightness(1.05);
            }

            [data-testid="stDataFrame"],
            [data-testid="stTable"] {
                border: 1px solid var(--line);
                border-radius: 8px;
                overflow: hidden;
                background: rgba(255, 255, 255, 0.9);
            }

            @media (max-width: 900px) {
                .block-container {
                    padding-top: 0.9rem;
                }

                .hero-panel {
                    align-items: flex-start;
                    flex-direction: column;
                    padding: 1.25rem;
                }

                .hero-badge {
                    min-width: 0;
                    width: 100%;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_label(label, note=None):
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="panel-note">{note}</div>', unsafe_allow_html=True)


def _skill_rows_for_audit(schema):
    rows = []
    for skill_group, label in (
        ("required_skills", "Explicit"),
        ("preferred_skills", "Inferred"),
    ):
        for skill in schema.get(skill_group, []) if isinstance(schema, dict) else []:
            if isinstance(skill, dict):
                rows.append(
                    {
                        "Skill": skill.get("name", ""),
                        "Group": label,
                        "Type": skill.get("type", ""),
                        "Confidence": skill.get("confidence", ""),
                        "Years": skill.get("years", ""),
                        "Evidence": skill.get("evidence", ""),
                    }
                )
            elif str(skill or "").strip():
                rows.append({"Skill": str(skill), "Group": label})
    return rows


def _display_dataframe(rows, **kwargs):
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    for col in df.columns:
        df[col] = df[col].apply(
            lambda value: ", ".join(map(str, value))
            if isinstance(value, list)
            else str(value)
            if value is not None
            else ""
        )
    st.dataframe(df, **kwargs)


def _chunk_rows_for_audit(chunks):
    rows = []
    for chunk in chunks:
        text = str(chunk.get("text") or chunk.get("embedding_text") or "")
        rows.append(
            {
                "Chunk": chunk.get("chunk_index", len(rows)),
                "Section": chunk.get("section", "unknown"),
                "Characters": len(text),
                "Preview": " ".join(text.split())[:240],
                "Text": text,
            }
        )
    return rows


apply_ui_theme()


fields = {
    "Name": "What is the candidate's name?",
    "Title": "What is the candidate's current title?",
    "Certifications": "Which certifications does the candidate hold?",
    "Passion": "Provide the candidate's personal summary/passion statement.",
    "Education": "What is the candidate's academic background?",
    "Experience": "What professional experiences does the candidate have?",
    "Skills & Tools": "List key skills and tools mentioned.",
    "Languages": "Which languages does the candidate speak?",
    "Contact": "How can we contact the candidate?",
    "Location": "Where is the candidate based?",
}


for key in fields:
    if key not in st.session_state:
        st.session_state[key] = ""

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "jd_matches" not in st.session_state:
    st.session_state["jd_matches"] = []
if "cv_matches" not in st.session_state:
    st.session_state["cv_matches"] = []
if "last_active_cv" not in st.session_state:
    st.session_state["last_active_cv"] = None
if "processed_upload_token" not in st.session_state:
    st.session_state["processed_upload_token"] = None


def _reset_cv_state():
    for key in fields:
        st.session_state[key] = ""
    st.session_state["jd_matches"] = []
    st.session_state["cv_matches"] = []
    st.session_state["chat_history"] = []


st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <strong>SmartHire</strong>
        <span>CV library, JD matching and RAG review</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.header("CV Library")


@st.cache_data(ttl=10)
def get_cv_list():
    return get_distinct_sources() or []


sources = get_cv_list()

if sources:
    to_delete = st.sidebar.selectbox("Delete a CV", sources)
    if st.sidebar.button("Delete CV"):
        deleted = delete_documents_by_source(to_delete)
        st.sidebar.success(f"Removed {deleted} chunks from {to_delete}")
        if to_delete == st.session_state["last_active_cv"]:
            st.session_state["last_active_cv"] = None
            _reset_cv_state()
        get_cv_list.clear()
        st.rerun()
else:
    st.sidebar.info("No CVs indexed yet.")

indexed_chunk_count = count_documents()
st.sidebar.markdown(f"**Total indexed chunks:** `{indexed_chunk_count}`")

uploaded_files = st.sidebar.file_uploader(
    "Upload CVs (PDF)",
    type="pdf",
    accept_multiple_files=True,
    key="cv_uploader",
)

if not uploaded_files:
    st.session_state["processed_upload_token"] = None
else:
    upload_token = "|".join(
        f"{file.name}:{hashlib.sha256(file.getvalue()).hexdigest()}"
        for file in uploaded_files
    )
    if st.session_state["processed_upload_token"] != upload_token:
        for file in uploaded_files:
            file.seek(0)
        with st.spinner(f"Indexing {len(uploaded_files)} CV(s)..."):
            total_chunks, uploaded_sources = process_multiple_pdfs(uploaded_files)
        st.session_state["processed_upload_token"] = upload_token
        st.session_state["uploaded_sources_current_session"] = uploaded_sources
        st.sidebar.success(f"Indexed {total_chunks} chunks from {len(uploaded_files)} CV(s)")
        get_cv_list.clear()
        st.rerun()


sources = get_cv_list()

active_cv = None
if sources:
    active_cv = st.sidebar.selectbox("Select CV to analyze", sources)
    st.sidebar.info(f"Analyzing: **{active_cv}**")

    if st.session_state["last_active_cv"] != active_cv:
        st.session_state["last_active_cv"] = active_cv
        _reset_cv_state()
        st.rerun()
else:
    st.sidebar.info("Upload and index a CV to begin")


st.sidebar.markdown("---")
st.sidebar.header("JD Management")

try:
    jd_count = count_indexed_jds()
    st.sidebar.markdown(f"**Total JD chunks:** `{jd_count}`")
except Exception:
    jd_count = 0
    st.sidebar.markdown("**Total JD chunks:** `0 (not indexed)`")

with st.sidebar.expander(f"View JDs ({len(SAMPLE_JDS)})"):
    for jd in SAMPLE_JDS:
        st.markdown(f"- **{jd['title']}** `{jd['id']}`")

if st.sidebar.button("Index JDs"):
    with st.spinner("Indexing JDs into MongoDB..."):
        try:
            indexed = ingest_jds()
            st.sidebar.success(f"Indexed {indexed} JDs successfully!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error: {e}")


def generate_full_profile():
    if not active_cv:
        st.warning("Please select a CV first.")
        return

    # Tiết kiệm LLM: chỉ lấy Name (candidate_name) từ Mongo, không gọi answer_question cho các field khác.
    candidate_name = get_candidate_name(active_cv)
    st.session_state["Name"] = candidate_name or ""

    # Các field còn lại giữ mặc định (""), UI sẽ hiển thị "-"
    for label in fields.keys():
        if label == "Name":
            continue
        st.session_state[label] = ""

active_cv_label = active_cv or "No CV selected"

st.markdown(
    f"""
    <section class="hero-panel">
        <div>
            <div class="hero-eyebrow">Recruitment Intelligence Workspace</div>
            <h1>CV RAG Analyzer</h1>
            <p>Screen resumes, compare job descriptions and keep candidate evidence close while reviewing hiring fit.</p>
        </div>
        <div class="hero-badge">
            <strong>{len(sources)} CV(s)</strong>
            <span>Active review: {active_cv_label}</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

summary_col_1, summary_col_2, summary_col_3 = st.columns(3)
summary_col_1.metric("Indexed CVs", len(sources))
summary_col_2.metric("CV Chunks", indexed_chunk_count)
summary_col_3.metric("JD Chunks", jd_count)

col_profile, col_chat = st.columns([1.7, 1], gap="large")

with col_profile:
    section_label("Candidate Workspace", "Structured profile review and job matching in one place.")
    st.header("Candidate Profile Snapshot")

    if active_cv:
        st.caption(f"CV being analyzed: **{active_cv}**")
    else:
        st.caption("No CV selected")

    st.button("Generate Full Profile", on_click=generate_full_profile, use_container_width=True)

    # Thẩm mỹ UI: chỉ hiển thị những field có giá trị thực sự (không phải "-"/rỗng)
    non_empty = {}
    for label in fields:
        val = (st.session_state.get(label) or "").strip()
        if not val or val == "-":
            continue
        non_empty[label] = val

    if non_empty:
        df = pd.DataFrame.from_dict(non_empty, orient="index", columns=["Value"])
        _display_dataframe(df, use_container_width=True)
    else:
        st.caption("Chưa có dữ liệu profile (đang tiết kiệm LLM: chỉ lấy Name).")

    with st.expander("CV Extraction Audit", expanded=False):
        if not active_cv:
            st.info("Select an indexed CV to review its extracted schema and raw chunks.")
        else:
            candidate_profile = get_candidate_profile(active_cv)
            cv_schema = candidate_profile.get("cv_schema") if isinstance(candidate_profile, dict) else {}
            audit_chunks = get_chunks_by_source_for_matching(active_cv)
            skill_rows = _skill_rows_for_audit(cv_schema or {})
            chunk_rows = _chunk_rows_for_audit(audit_chunks)

            audit_col_1, audit_col_2, audit_col_3 = st.columns(3)
            audit_col_1.metric("Extracted Skills", len(skill_rows))
            audit_col_2.metric("Stored CV Chunks", len(chunk_rows))
            audit_col_3.metric("Raw Text Characters", sum(row["Characters"] for row in chunk_rows))

            skills_tab, chunks_tab, schema_tab = st.tabs(["Extracted Skills", "Raw CV Chunks", "Schema"])

            with skills_tab:
                if skill_rows:
                    _display_dataframe(skill_rows, use_container_width=True, hide_index=True)
                else:
                    st.warning("No skill rows were found in the stored CV schema.")

                experience_rows = cv_schema.get("experience", []) if isinstance(cv_schema, dict) else []
                projects = cv_schema.get("projects", []) if isinstance(cv_schema, dict) else []
                schema_meta_1, schema_meta_2, schema_meta_3 = st.columns(3)
                schema_meta_1.metric("Experience Months", cv_schema.get("experience_months", 0) if cv_schema else 0)
                schema_meta_2.metric("Experience Rows", len(experience_rows))
                schema_meta_3.metric("Projects", len(projects))

                if experience_rows:
                    st.markdown("**Extracted Experience**")
                    _display_dataframe(experience_rows, use_container_width=True, hide_index=True)

            with chunks_tab:
                lookup = st.text_input(
                    "Find text in stored CV chunks",
                    placeholder="Example: Docker, REST API, SQL",
                    key="cv_audit_lookup",
                ).strip()
                visible_chunks = chunk_rows
                if lookup:
                    visible_chunks = [
                        row for row in chunk_rows
                        if lookup.lower() in row["Text"].lower()
                    ]
                    if visible_chunks:
                        st.success(f"Found `{lookup}` in {len(visible_chunks)} stored chunk(s).")
                    else:
                        st.warning(f"`{lookup}` was not found in the stored CV chunk text.")

                if visible_chunks:
                    chunk_table = pd.DataFrame(
                        [
                            {key: row[key] for key in ("Chunk", "Section", "Characters", "Preview")}
                            for row in visible_chunks
                        ]
                    )
                    _display_dataframe(chunk_table, use_container_width=True, hide_index=True)
                    chunk_options = {
                        f"Chunk {row['Chunk']} - {row['Section']} ({row['Characters']} chars)": row
                        for row in visible_chunks
                    }
                    selected_chunk_label = st.selectbox(
                        "Raw chunk text",
                        list(chunk_options.keys()),
                        key="cv_audit_chunk",
                    )
                    selected_chunk = chunk_options[selected_chunk_label]
                    st.text_area(
                        "Stored text",
                        selected_chunk["Text"],
                        height=260,
                        disabled=True,
                        key="cv_audit_chunk_text",
                    )
                else:
                    st.warning("No stored raw chunks are available for this CV.")

            with schema_tab:
                if cv_schema:
                    st.json(cv_schema, expanded=False)
                else:
                    st.warning("No stored CV schema is available for this CV.")

    st.markdown("---")
    section_label("Matching", "Tune the search scope before comparing candidates against a JD.")
    st.header("JD → Candidate CV Matching")

    st.caption("HR dán/paste JD (text) hoặc chọn JD có sẵn trong hệ thống. Hệ thống sẽ trả về top K CV phù hợp nhất (must-have/gap).")

    match_controls_1, match_controls_2 = st.columns([1, 1.4])
    with match_controls_1:
        top_k_cvs = st.number_input("Number of CVs to return", min_value=1, max_value=20, value=5, step=1)

    with match_controls_2:
        jd_mode = st.radio(
            "JD input mode",
            options=["Paste JD text", "Use indexed/sample JD"],
            index=1,
            horizontal=True,
        )

    jd_text = ""
    jd_id_for_call = None

    match_scope = st.radio(
        "Match CV scope",
        options=["All indexed CVs", "Only the CVs uploaded in this session"],
        index=1,
        horizontal=True,
    )

    uploaded_sources = None
    if match_scope == "Only the CVs uploaded in this session":
        uploaded_sources = st.session_state.get("uploaded_sources_current_session") or []
        if not uploaded_sources:
            st.warning("No CVs were uploaded in this session yet. Upload CVs first.")

    if jd_mode == "Paste JD text":
        jd_title = st.text_input("JD title (optional)", placeholder="Example: Backend Developer Intern")
        jd_text = st.text_area("Paste Job Description", height=220, placeholder="Dán nội dung JD vào đây...")
        if st.button("Find Best Matching CVs (from pasted JD)", use_container_width=True):
            if not jd_text.strip():
                st.error("JD text is empty.")
            else:
                pasted_jd = None
                with st.spinner("Chunking, embedding and storing pasted JD..."):
                    try:
                        pasted_jd = ingest_jd_text(jd_text, title=jd_title)
                        st.sidebar.success(
                            f"Stored JD `{pasted_jd['jd_id']}` with {pasted_jd['chunk_count']} chunks"
                        )
                    except Exception as e:
                        st.error(f"JD indexing error: {e}")

                if pasted_jd:
                    with st.spinner("Matching pasted JD against indexed CVs..."):
                        try:
                            st.session_state["cv_matches"] = match_jd_to_cvs(
                                pasted_jd["jd_id"],
                                top_k=int(top_k_cvs),
                                source_whitelist=uploaded_sources,
                            )
                        except Exception as e:
                            st.error(f"Matching error: {e}")
    else:
        if jd_count == 0:
            st.warning("No JDs indexed yet. Click **Index JDs** in the sidebar first.")
        else:
            try:
                indexed_jds = list_indexed_jds()
            except Exception:
                indexed_jds = []

            if indexed_jds:
                jd_options = {
                    f"{jd['title']} ({jd['jd_id']})": jd["jd_id"]
                    for jd in indexed_jds
                }
                selected_jd_label = st.selectbox("Select JD to match candidates", list(jd_options.keys()))

                jd_id_for_call = jd_options[selected_jd_label]

                if st.button("Find Best Matching CVs", use_container_width=True):
                    with st.spinner("Matching JD against indexed CVs..."):
                        try:
                            st.session_state["cv_matches"] = match_jd_to_cvs(
                                jd_id_for_call,
                                top_k=int(top_k_cvs),
                                source_whitelist=uploaded_sources,
                            )
                        except Exception as e:
                            st.error(f"Matching error: {e}")
            else:
                st.warning("JD chunks exist, but no grouped JD list was returned. Please re-index JDs.")


    if st.session_state["cv_matches"]:
        st.subheader("Candidate Ranking")
        for i, match in enumerate(st.session_state["cv_matches"]):
            ev = match["evaluation"]
            score = ev.get("score", 0)
            badge = "GREEN" if score >= 70 else "YELLOW" if score >= 50 else "RED"
            cv_profile = match.get("cv_profile", {})

            with st.expander(
                f"{badge} #{i + 1} {match.get('candidate_name') or match.get('cv_source')} - {score}/100",
                expanded=(i == 0),
            ):
                if match.get("cv_source") == "error":
                    st.error(ev.get("summary") or ev.get("match") or "JD matching failed.")
                    continue

                st.caption(f"CV: **{match.get('cv_source')}** | JD: **{match.get('jd_title')}**")

                if cv_profile:
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Experience", cv_profile.get("experience_duration") or f"{cv_profile.get('experience_years', 0)} yrs")
                    p2.metric("Level", cv_profile.get("experience_level", "-"))
                    p3.metric("Skills", f"{cv_profile.get('total_skills', 0)} found")

                st.markdown("**Score Breakdown**")
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Technical", f"{ev.get('technical_score', '-')}/40")
                b2.metric("Experience", f"{ev.get('experience_score', '-')}/30")
                b3.metric("Education", f"{ev.get('education_score', '-')}/20")
                b4.metric("Overall Fit", f"{ev.get('fit_score', '-')}/10")

                st.markdown("---")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Matched Skills**")
                    for skill in ev.get("matched_skills", []):
                        st.markdown(f"- {skill}")
                    if not ev.get("matched_skills"):
                        st.caption("None")

                with col_b:
                    st.markdown("**Missing Skills**")
                    for skill in ev.get("missing_skills", []):
                        st.markdown(f"- {skill}")
                    if not ev.get("missing_skills"):
                        st.caption("None")

                st.info(ev.get("summary", ""))
                st.caption(
                    f"Recommendation: **{ev.get('recommendation', '')}** | "
                    f"Hybrid: {match.get('similarity_score', 0)}% | "
                    f"Dense: {match.get('dense_score', 0)}% | "
                    f"Required Skill Coverage: {match.get('bm25_score', 0)}% | "
                    f"Rerank: {match.get('rerank_score', 0)}% "
                    f"({match.get('rerank_method', 'rerank')})"
                )

                evidence = match.get("match_evidence") or []
                if evidence:
                    # Streamlit does not allow nesting expanders inside other expanders.
                    st.markdown("**Retrieved CV Evidence**")
                    for item in evidence[:5]:
                        st.markdown(
                            f"**JD {str(item.get('jd_section', 'unknown')).upper()}** "
                            f"matched **CV {str(item.get('cv_section', 'unknown')).upper()}** "
                            f"({float(item.get('score', 0)):.2f})"
                        )
                        st.caption(item.get("cv_text", ""))

                requirement_details = ev.get("requirement_match_details") or []
                if requirement_details:
                    st.markdown("**Requirement Evidence Matching**")
                    detail_rows = []
                    for item in requirement_details:
                        detail_rows.append(
                            {
                                "Requirement": item.get("requirement", ""),
                                "Status": item.get("status", ""),
                                "Confidence": item.get("confidence", 0),
                                "Source": item.get("source", ""),
                                "CV Section": item.get("cv_section", ""),
                                "Evidence": item.get("evidence", ""),
                            }
                        )
                    _display_dataframe(detail_rows, use_container_width=True, hide_index=True)



with col_chat:
    section_label("RAG Chat", "Ask focused questions about the selected resume.")
    st.header("Freeform RAG Chat")

    if active_cv:
        st.caption(f"Chatting about: **{active_cv}**")

    for role, msg in st.session_state["chat_history"]:
        st.chat_message(role).write(msg)

    user_input = st.chat_input("Ask anything about the CV...")
    if user_input:
        st.chat_message("user").write(user_input)
        with st.spinner("Searching and generating..."):
            reply = answer_question(
                f"Please answer concisely: {user_input}",
                source_filter=active_cv,
            )
            if "Aucun contenu pertinent" in reply or "not specified" in reply.lower():
                reply = "Not specified in the CV."
        st.chat_message("assistant").write(reply)
        st.session_state["chat_history"].append(("user", user_input))
        st.session_state["chat_history"].append(("assistant", reply))
        st.rerun()

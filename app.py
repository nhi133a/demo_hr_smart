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

from jd_matcher import match_cv_to_jds
from jd_store import SAMPLE_JDS, count_indexed_jds, ingest_jds
from mongo_utils import (
    count_documents,
    delete_documents_by_source,
    get_candidate_name,
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
    page_title="IAcine HR Power Tool - CV RAG Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("IAcine HR Power Tool - CV RAG Analyzer")


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
if "last_active_cv" not in st.session_state:
    st.session_state["last_active_cv"] = None
if "processed_upload_token" not in st.session_state:
    st.session_state["processed_upload_token"] = None


def _reset_cv_state():
    for key in fields:
        st.session_state[key] = ""
    st.session_state["jd_matches"] = []
    st.session_state["chat_history"] = []


st.sidebar.header("CV Library Management")


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

st.sidebar.markdown(f"**Total indexed chunks:** `{count_documents()}`")

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
            n = process_multiple_pdfs(uploaded_files)
        st.session_state["processed_upload_token"] = upload_token
        st.sidebar.success(f"Indexed {n} chunks from {len(uploaded_files)} CV(s)")
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

    candidate_name = get_candidate_name(active_cv)
    if candidate_name:
        st.session_state["Name"] = candidate_name
    else:
        name_prompt = "What is the candidate's full name mentioned at the top of the CV?"
        name_result = answer_question(name_prompt, k=3, source_filter=active_cv)
        st.session_state["Name"] = name_result if "not" not in name_result.lower() else "Name not found"

    progress = st.progress(0)
    completed = 0
    total_fields = len([f for f in fields.keys() if f != "Name"])

    field_k_values = {
        "Experience": 8,
        "Skills & Tools": 5,
        "Education": 4,
        "Contact": 3,
        "Languages": 3,
        "Location": 2,
        "Title": 3,
        "Certifications": 4,
        "Passion": 4,
    }

    for label, question in fields.items():
        if label == "Name":
            continue

        k_value = field_k_values.get(label, 3)
        ans = answer_question(question, k=k_value, source_filter=active_cv)

        if "Aucun contenu pertinent" in ans or "not specified" in ans.lower():
            ans = "Not specified in the CV."

        st.session_state[label] = ans
        completed += 1
        progress.progress(completed / total_fields)


col_profile, col_chat = st.columns([2, 1])

with col_profile:
    st.header("Candidate Profile Snapshot")

    if active_cv:
        st.caption(f"CV being analyzed: **{active_cv}**")
    else:
        st.caption("No CV selected")

    st.button("Generate Full Profile", on_click=generate_full_profile)

    data = {label: st.session_state[label] or "-" for label in fields}
    df = pd.DataFrame.from_dict(data, orient="index", columns=["Value"])
    st.table(df)

    st.markdown("---")
    st.header("JD Matching")

    if jd_count == 0:
        st.warning("No JDs indexed yet. Click **Index JDs** in the sidebar first.")
    else:
        if st.button("Find Best Matching JD"):
            if not active_cv:
                st.warning("Please select a CV first.")
            else:
                with st.spinner("Matching CV against JDs..."):
                    try:
                        cv_chunks = get_chunks_by_source_for_matching(active_cv)
                        if not cv_chunks:
                            st.warning("No indexed chunks found for this CV. Please upload/index it again.")
                        else:
                            matches = match_cv_to_jds(cv_chunks, top_k=1)
                            st.session_state["jd_matches"] = matches
                    except Exception as e:
                        st.error(f"Matching error: {e}")

    if st.session_state["jd_matches"]:
        first_match = st.session_state["jd_matches"][0]
        cv_profile = first_match.get("cv_profile", {})

        if cv_profile:
            st.markdown("**CV Profile (Auto-detected)**")
            p1, p2, p3 = st.columns(3)
            p1.metric("Experience", cv_profile.get("experience_duration") or f"{cv_profile.get('experience_years', 0)} yrs")
            p2.metric("Level", cv_profile.get("experience_level", "-"))
            p3.metric("Skills", f"{cv_profile.get('total_skills', 0)} found")
            st.markdown("---")

        for i, match in enumerate(st.session_state["jd_matches"]):
            ev = match["evaluation"]
            score = ev.get("score", 0)
            badge = "GREEN" if score >= 70 else "YELLOW" if score >= 50 else "RED"

            with st.expander(
                f"{badge} #{i + 1} {match['jd_title']} - {score}/100",
                expanded=(i == 0),
            ):
                if match.get("jd_id") == "error":
                    st.error(ev.get("summary") or ev.get("match") or "JD matching failed.")
                    continue

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
                    f"Vector Similarity: {match.get('similarity_score', 0)}%"
                )


with col_chat:
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

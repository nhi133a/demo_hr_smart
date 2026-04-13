# Phải ở trên cùng tuyệt đối trước mọi import
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
from rag_core import process_multiple_pdfs, answer_question
from mongo_utils import delete_documents_by_source, get_distinct_sources, count_documents
from jd_store import ingest_jds, get_jd_store, SAMPLE_JDS
from jd_matcher import match_cv_to_jds
import certifi
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"), tlsCAFile=certifi.where())

st.set_page_config(
    page_title="🚀 IAcine HR Power Tool – CV RAG Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 IAcine HR Power Tool – CV RAG Analyzer")


# =====================================================
# SIDEBAR — CV Library Management (GIỮ NGUYÊN GỐC)
# =====================================================

st.sidebar.header("📂 CV Library Management")

sources = get_distinct_sources()
if sources:
    to_delete = st.sidebar.selectbox("🗑️ Delete a CV", sources)
    if st.sidebar.button("Delete CV"):
        deleted = delete_documents_by_source(to_delete)
        st.sidebar.success(f"✅ Removed {deleted} chunks from {to_delete}")
else:
    st.sidebar.info("ℹ️ No CVs indexed yet.")

st.sidebar.markdown(f"**💾 Total indexed chunks:** `{count_documents()}`")

uploaded = st.sidebar.file_uploader(
    "📄 Upload one CV (PDF)",
    type="pdf",
    accept_multiple_files=False,
    key="cv_uploader"
)
if uploaded:
    with st.spinner("🔍 Indexing CV…"):
        n = process_multiple_pdfs([uploaded])
    st.sidebar.success(f"✅ Indexed {n} chunks from the CV")

active_cv = None
if sources:
    active_cv = st.sidebar.selectbox("🎯 Select CV to analyze", sources)
    st.sidebar.info(f"Analyzing: **{active_cv}**")
else:
    st.sidebar.info("Upload and index a CV to begin")


# =====================================================
# SIDEBAR — JD Management
# =====================================================

st.sidebar.markdown("---")
st.sidebar.header("📋 JD Management")

try:
    jd_db = get_jd_store()
    jd_count = jd_db._collection.count()
    st.sidebar.markdown(f"**📦 Total JD chunks:** `{jd_count}`")
except Exception:
    jd_count = 0
    st.sidebar.markdown("**📦 Total JD chunks:** `0`")

with st.sidebar.expander(f"📄 View JDs ({len(SAMPLE_JDS)})"):
    for jd in SAMPLE_JDS:
        st.markdown(f"- **{jd['title']}** `{jd['id']}`")

if st.sidebar.button("🔄 Index JDs"):
    with st.spinner("🔍 Indexing JDs into ChromaDB…"):
        try:
            ingest_jds()
            st.sidebar.success("✅ JDs indexed successfully!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")


# =====================================================
# MAIN — Session state
# =====================================================

fields = {
    "Name":           "What is the candidate's name?",
    "Title":          "What is the candidate's current title?",
    "Certifications": "Which certifications does the candidate hold?",
    "Passion":        "Provide the candidate's personal summary/passion statement.",
    "Education":      "What is the candidate's academic background?",
    "Experience":     "What professional experiences does the candidate have?",
    "Skills & Tools": "List key skills and tools mentioned.",
    "Languages":      "Which languages does the candidate speak?",
    "Contact":        "How can we contact the candidate?",
    "Location":       "Where is the candidate based?"
}

for key in fields:
    if key not in st.session_state:
        st.session_state[key] = ""

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "jd_matches" not in st.session_state:
    st.session_state["jd_matches"] = []


def generate_full_profile():
    if not active_cv:
        st.warning("Please select a CV first.")
        return

    for label, question in fields.items():
        if label == "Experience":
            prompt = "Please answer concisely and list *all* professional experiences mentioned in the CV."
            ans = answer_question(prompt, k=10)
        else:
            prompt = f"Please answer concisely: {question}"
            ans = answer_question(prompt)

        if "Aucun contenu pertinent" in ans or "not specified" in ans:
            ans = "❌ Not specified in the CV."

        st.session_state[label] = ans


# =====================================================
# MAIN — Layout
# =====================================================

col_profile, col_chat = st.columns([2, 1])

with col_profile:

    # --- Profile Snapshot (GIỮ NGUYÊN GỐC) ---
    st.header("🎯 Candidate Profile Snapshot")
    st.button("🤖 Generate Full Profile", on_click=generate_full_profile)
    data = {label: st.session_state[label] or "–" for label in fields}
    df = pd.DataFrame.from_dict(data, orient="index", columns=["Value"])
    st.table(df)

    # --- JD Matching ---
    st.markdown("---")
    st.header("🎯 JD Matching")

    if jd_count == 0:
        st.warning("⚠️ No JDs indexed yet. Click **Index JDs** in the sidebar first.")
    else:
        if st.button("🔍 Find Matching JDs"):
            if not active_cv:
                st.warning("Please select a CV first.")
            else:
                filled = [
                    v for k, v in st.session_state.items()
                    if k in fields and v and v not in ("–", "❌ Not specified in the CV.")
                ]
                if not filled:
                    st.warning("Please click **Generate Full Profile** first.")
                else:
                    with st.spinner("🔍 Matching CV against JDs…"):
                        try:
                            cv_chunks = [
                                {"section": k.lower().replace(" & ", "_"), "text": v}
                                for k, v in st.session_state.items()
                                if k in fields
                                and v
                                and v not in ("–", "❌ Not specified in the CV.")
                            ]
                            matches = match_cv_to_jds(cv_chunks, top_k=3)
                            st.session_state["jd_matches"] = matches
                        except Exception as e:
                            st.error(f"❌ Matching error: {e}")

    # Hiển thị kết quả
    if st.session_state["jd_matches"]:

        # --- CV Profile summary (từ resume_processor) ---
        first_match = st.session_state["jd_matches"][0]
        cv_profile = first_match.get("cv_profile", {})

        if cv_profile:
            st.markdown("**🧠 CV Profile (Auto-detected)**")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Experience", f"{cv_profile.get('experience_years', 0)} yrs")
            p2.metric("Level",      cv_profile.get("experience_level", "–"))
            p3.metric("Skills",     f"{cv_profile.get('total_skills', 0)} found")
            p4.metric("Fit",        cv_profile.get("fit_assessment", "–"))

            # Skill breakdown theo domain
            breakdown = cv_profile.get("skill_breakdown", {})
            if breakdown:
                st.caption("Skill domains: " + " · ".join(
                    f"**{cat}** ({cnt})" for cat, cnt in breakdown.items()
                ))

            st.markdown("---")

        # --- Từng JD result ---
        for i, match in enumerate(st.session_state["jd_matches"]):
            ev = match["evaluation"]
            score = ev.get("score", 0)
            badge = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"

            with st.expander(
                f"{badge} #{i+1}  {match['jd_title']}  —  {score}/100",
                expanded=(i == 0)
            ):
                # Score breakdown
                st.markdown("**📊 Score Breakdown**")
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Technical",   f"{ev.get('technical_score',   '–')}/40")
                b2.metric("Experience",  f"{ev.get('experience_score',  '–')}/30")
                b3.metric("Education",   f"{ev.get('education_score',   '–')}/20")
                b4.metric("Soft Skills", f"{ev.get('softskill_score',   '–')}/10")

                st.markdown("---")

                # Skills
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**✅ Matched Skills**")
                    for skill in ev.get("matched_skills", []):
                        st.markdown(f"- {skill}")
                    if not ev.get("matched_skills"):
                        st.caption("None")

                with col_b:
                    st.markdown("**❌ Missing Skills**")
                    for skill in ev.get("missing_skills", []):
                        st.markdown(f"- {skill}")
                    if not ev.get("missing_skills"):
                        st.caption("None")

                st.info(ev.get("summary", ""))
                st.caption(
                    f"Recommendation: **{ev.get('recommendation', '')}**  |  "
                    f"Vector Similarity: {match.get('similarity_score', 0)}%"
                )


with col_chat:
    st.header("💬 Freeform RAG Chat")
    for role, msg in st.session_state["chat_history"]:
        st.chat_message(role).write(msg)

    user_input = st.chat_input("Ask anything about the CV…")
    if user_input:
        st.chat_message("user").write(user_input)
        with st.spinner("🔍 Searching & Generating…"):
            reply = answer_question(f"Please answer concisely: {user_input}")
            if "Aucun contenu pertinent" in reply or "not specified" in reply:
                reply = "❌ Not specified in the CV."
        st.chat_message("assistant").write(reply)
        st.session_state["chat_history"].append(("user", user_input))
        st.session_state["chat_history"].append(("assistant", reply))
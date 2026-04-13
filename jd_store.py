import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

CHROMA_JD_DIR = "./chroma_jd"
EMBED_MODEL = "nomic-embed-text"

SAMPLE_JDS = [
    {
        "id": "jd_001",
        "title": "Tester Intern / QA Intern",
        "content": """
        Vi tri: Tester Intern / QA Intern
        Yeu cau ky nang: Manual testing, Test case design,
        Bug reporting, Jira, Postman, API testing
        Kinh nghiem: Chua can kinh nghiem, uu tien co du an thuc te
        Ky nang mem: Ti mi, can than, tieng Anh co ban
        """,
    },
    {
        "id": "jd_002",
        "title": "Backend Developer Intern",
        "content": """
        Vi tri: Backend Developer Intern
        Yeu cau: Python hoac Node.js, REST API, SQL
        Kinh nghiem: Co project ca nhan la loi the
        Ky nang mem: Teamwork, giao tiep tot
        """,
    },
    {
        "id": "jd_003",
        "title": "Frontend Developer Intern",
        "content": """
        Vi tri: Frontend Developer Intern
        Yeu cau: HTML, CSS, JavaScript, React
        Kinh nghiem: Chua can kinh nghiem, uu tien co du an thuc te
        Ky nang mem: Teamwork, giao tiep tot
        """,
    },
    {
        "id": "jd_004",
        "title": "Data Analyst Intern",
        "content": """
        Vi tri: Data Analyst Intern
        Yeu cau: Excel, SQL, Python (Pandas), Data visualization
        Kinh nghiem: Chua can kinh nghiem, uu tien co du an thuc te
        Ky nang mem: Tinh toan, chinh xac, giao tiep tot
        """,          
    }
]


@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)


def ingest_jds():
    embeddings = get_embeddings()
    docs = [
        Document(
            page_content=jd["content"],
            metadata={"jd_id": jd["id"], "title": jd["title"]}
        )
        for jd in SAMPLE_JDS
    ]
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_JD_DIR
    )
    print(f"Da index {len(docs)} JD voi model {EMBED_MODEL}")
    return vectordb


def get_jd_store():
    return Chroma(
        persist_directory=CHROMA_JD_DIR,
        embedding_function=get_embeddings()
    )
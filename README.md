# 📄 SmartHire CV with RAG | AI-Powered Resume Analysis for Recruiters

![image](https://github.com/user-attachments/assets/329d7314-e21d-4c18-a301-2515016fe893)


## 📖 Introduction

**SmartHire CV with RAG** is a next-gen AI tool that simplifies and accelerates resume screening. Upload a PDF CV and instantly extract structured data, generate summaries, and ask custom questions — all powered by a cutting-edge **Retrieval-Augmented Generation** pipeline with **AWS Bedrock embeddings**, **MongoDB vector search**, and **GPT-3.5**.

Whether you're a recruiter, HR manager, or talent specialist, **SmartHire CV** lets you assess candidates in seconds — without losing the context of the original CV. 🤖📄

---

## 🚀 Features

✔️ **One-Click Summary Table** – Auto-extracts Name, Role, Education, Experience, Skills, Certifications
✔️ **RAG-Powered Q\&A** – Ask questions like “What tech stacks?” or “Would they fit a Product Owner role?”
✔️ **AWS Bedrock Embeddings** – Uses **Titan-embed-text v2** for accurate semantic search
✔️ **MongoDB Atlas `$vectorSearch`** – High-speed vector retrieval at scale
✔️ **Concise GPT Responses** – Prompts begin with *"Please answer concisely..."* to ensure brief, focused output
✔️ **Multi-CV Management** – Upload, index, choose, and delete multiple resumes
✔️ **Streamlit Web UI** – Clean, no-code interface for non-technical users

---

## 🏗️ Technologies

* 🐍 **Python 3.12** – Backend and orchestration
* 🌐 **Streamlit** – Lightweight frontend
* 🔍 **LangChain** – RAG pipeline management
* 🧠 **OpenAI GPT-3.5** – LLM for Q\&A and summarization
* 🧆 **AWS Bedrock** – Embedding via Titan model
* 📂 **MongoDB Atlas** – Vector DB for resume chunks
* 📄 **PyMuPDF (fitz)** – PDF parsing and text extraction
* 🔐 **python-dotenv** – Environment variable handling

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Yacine-Mekideche/cv-smart-hire.git
cd cv-smart-hire
```

### 2️⃣ Create a `.env` File

```env
OPENAI_API_KEY=your_openai_api_key
MONGO_URI=your_mongodb_connection_string
AWS_PROFILE=your_aws_profile
AWS_REGION=your_aws_region
```

### 3️⃣ Set Up Your Environment

```bash
python -m venv venv
# Activate:
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Once launched in your browser, you can:

* 📄 Upload one or more PDF resumes
* ⚙️ Click *“Index CV”* to generate embeddings and store in MongoDB
* 📋 Select a CV and click *“Generate Full Profile”*
* 🗨️ Ask free-form questions in the **Chat with CV** panel

---

## 🎯 Demo

<a href="https://www.youtube.com/watch?v=-OoxQoQX86s" target="_blank">
  <img src="https://img.youtube.com/vi/-OoxQoQX86s/maxresdefault.jpg" alt="SmartHire CV Demo" style="max-width:100%; height:auto;">
</a>

---

## 🧠 AI Architecture Overview

```
PDF Resume Upload
       ↓
Parsing & Chunking (PyMuPDF)
       ↓
Embeddings
 • AWS Bedrock (Titan-embed-text v2)
 • OpenAI (fallback)
       ↓
Vector Store (MongoDB Atlas)
       ↓
RAG Pipeline (LangChain)
       ↓
GPT-3.5 Inference
       ↓
Streamlit UI (Summary + Chat)
```

---

## 📬 Contact Me

💡 **Transform your hiring pipeline with AI-powered CV insights.**

[![Website](https://img.shields.io/badge/My%20Website-%23000000.svg?style=for-the-badge\&logo=About.me\&logoColor=white)](https://iacine.tech)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?style=for-the-badge\&logo=linkedin\&logoColor=white)](https://www.linkedin.com/in/yacine-mekideche/)
[![GitHub](https://img.shields.io/badge/GitHub-%2312100E.svg?style=for-the-badge\&logo=github\&logoColor=white)](https://github.com/Yacine-Mekideche)
[![Malt](https://img.shields.io/badge/Malt-%23FF6F61.svg?style=for-the-badge\&logo=malt\&logoColor=white)](https://malt.fr/profile/yacinemekideche)
[![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge\&logo=youtube\&logoColor=white)](https://www.youtube.com/@iacine_tech)

📩 **Business inquiries:** [contact@iacine.tech](mailto:contact@iacine.tech)

---

**#SmartHire #ResumeAI #RAG #GPT #AWSBedrock #MongoDBAtlas #LangChain #Streamlit #RecruitmentTech #AIforHR #CVAnalysis #PythonProject #YacineTech #FreelanceAI**

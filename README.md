# SmartHire CV RAG Analyzer

SmartHire is a Streamlit prototype for CV analysis and JD-to-CV matching. It
indexes uploaded PDF CVs, parses CV and JD schemas, retrieves candidate CV
chunks with MongoDB Atlas Vector Search, reranks candidates with rule-based
schema scoring, and shows matched skills, gaps, and supporting evidence.

## Current Features

- Upload and index multiple CV PDFs.
- Parse CV schema fields for candidate data, skills, projects, education, and
  experience.
- Audit stored CV extraction data with raw chunks, extracted skills, and schema
  JSON in the app.
- Index sample JDs or paste a new JD.
- Match one JD against all indexed CVs or only CVs uploaded in the current
  session.
- Show score breakdown, matched skills, missing skills, and CV evidence.
- Ask RAG questions about the selected CV.

## Current Runtime

The current implementation uses:

- Streamlit for the web UI.
- MongoDB Atlas for CV chunks, candidate profiles, JD chunks, and vector search.
- Ollama with the fixed model name `qwen2.5:1.5b` in `local_llm.py` for CV
  schema parsing, JD schema parsing, and RAG answers.
- A configurable embedding provider in `bedrock_utils.py`.
  - Default: local `sentence-transformers/all-MiniLM-L6-v2`.
  - Optional: AWS Bedrock embeddings.
  - Optional: Google embeddings.
- Optional cross-encoder reranking in `jd_matcher.py`.
  - Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
  - It scores raw JD text against raw CV chunks after vector retrieval.
- Docling for PDF conversion and chunking, with a PyPDF2 text fallback.

The project currently targets a local or controlled demo deployment. A public
production deployment still needs authentication, monitoring, privacy controls,
 and measured quality benchmarks.

## Project Pipeline

```text
CV PDF
  -> PDF conversion
  -> CV schema extraction
  -> CV chunking
  -> chunk embeddings
  -> MongoDB cv_chunks + candidates_profile

JD text
  -> JD schema extraction
  -> JD section chunks
  -> JD embeddings
  -> MongoDB job_descriptions

Matching request
  -> JD vector retrieval over CV chunks
  -> schema scoring for each candidate CV
  -> hard filters
  -> rerank
  -> top K CVs with evidence
```

## Prerequisites

- Python 3.12 or a compatible tested Python environment.
- MongoDB Atlas cluster with Vector Search available.
- Ollama installed and running for the current local LLM path.
- The Ollama model used by the code:

```bash
ollama pull qwen2.5:1.5b
```

## Setup

1. Create a virtual environment.

```bash
python -m venv venv
```

2. Activate it.

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source venv/bin/activate
```

3. Install Python dependencies.

```bash
pip install -r requirements.txt
```

4. Create a local environment file from the safe template.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

5. Fill `.env` locally. Never commit `.env`.

Minimum local setting:

```env
MONGO_URI=your_mongodb_atlas_connection_string
EMBEDDING_PROVIDER=local
```

## Embedding Provider Notes

The default local embedding configuration in `.env.example` uses 384
dimensions. If `LOCAL_EMBED_LOCAL_FILES_ONLY=true`, the sentence-transformer
model must already be present in the local model cache. For a first-time local
download, set `LOCAL_EMBED_LOCAL_FILES_ONLY=false` temporarily or use Bedrock
or Google embeddings instead.

If you change embedding provider or model, create MongoDB vector indexes with a
matching vector dimension before indexing CVs and JDs.

## Cross-Encoder Reranking

The matcher can optionally rerank candidates with a cross-encoder. This step
does not replace schema scoring. It reads the JD text and the most relevant raw
CV chunks, then contributes to the final rerank score.

The same cross-encoder path can also refine matched and missing requirements.
For each JD requirement, the matcher finds candidate evidence in raw CV chunks,
scores the requirement/evidence pair, and stores the result in
`requirement_match_details` for the dashboard.

Default settings:

```env
CROSS_ENCODER_ENABLED=true
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
CROSS_ENCODER_LOCAL_FILES_ONLY=true
CROSS_ENCODER_MAX_CHUNKS=6
CROSS_ENCODER_TOP_AVG=3
CROSS_ENCODER_WEIGHT=0.25
REQUIREMENT_EVIDENCE_ENABLED=true
REQUIREMENT_EVIDENCE_THRESHOLD=0.55
REQUIREMENT_EVIDENCE_MAX_REQUIREMENTS=18
REQUIREMENT_EVIDENCE_MAX_CHUNKS=4
```

If the model is not available in the local cache and
`CROSS_ENCODER_LOCAL_FILES_ONLY=true`, the app logs a warning and continues with
the existing schema plus RAG rerank path. For first-time model download, set
`CROSS_ENCODER_LOCAL_FILES_ONLY=false` temporarily in a network-enabled
environment.

## MongoDB Collections

The app uses database `aws_rag_db` and these collections:

| Collection | Purpose |
| --- | --- |
| `cv_chunks` | Raw CV chunks and CV chunk embeddings |
| `candidates_profile` | One profile document and `cv_schema` per CV |
| `job_descriptions` | JD chunks, JD schema, and JD embeddings |

## MongoDB Atlas Vector Indexes

The default local embedding model uses 384 dimensions. Create these Atlas Vector
Search indexes before matching. Use the Atlas UI JSON editor or an equivalent
Atlas-supported index creation path.

### CV chunk vector index

Collection:

```text
aws_rag_db.cv_chunks
```

Index name:

```text
vector_index
```

Definition:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "vector_embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}
```

### JD vector index

Collection:

```text
aws_rag_db.job_descriptions
```

Index name:

```text
jd_vector_index
```

Definition:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}
```

## Run

Start Ollama first, then run the app:

```bash
streamlit run app.py
```

Inside the app:

1. Upload one or more PDF CVs in the sidebar.
2. Select a CV to inspect or chat with.
3. Open `CV Extraction Audit` to review stored raw chunks and extracted schema.
4. Index sample JDs or paste a JD.
5. Run JD-to-CV matching and review candidate ranking.

## Verification

Focused matcher tests:

```bash
python -m unittest tests.test_jd_matcher_pipeline
```

Syntax check for the Streamlit app:

```bash
python -m py_compile app.py
```

## Deployment Notes

For a demo deployment, prepare:

- Environment variables or platform secrets instead of a committed `.env`.
- Ollama runtime and model availability, or a deliberate code change to use a
  hosted LLM provider.
- MongoDB Atlas network access, collections, and vector indexes.
- Enough memory for Docling and the selected local embedding/LLM path.

For production work, add authentication, authorization, logging, monitoring,
CV data retention rules, secret rotation practices, evaluation benchmarks, and
error reporting for MongoDB, LLM, and embedding failures.

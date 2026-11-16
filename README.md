# 📄 Local RAG Document Question Answering System

A completely local, privacy-first Retrieval-Augmented Generation (RAG) application built with Python, FAISS, Hugging Face Transformers, and Streamlit. It allows users to upload multiple PDF and TXT documents, index them locally without external API dependencies, and ask context-grounded questions with source-level traceability.

---

## ✨ Key Features

* **100% Local Execution:** Runs offline using open-source models (`all-MiniLM-L6-v2` and `google/flan-t5-small`)—no OpenAI API keys required.
* **Multi-Document Ingestion:** Supports batch uploading and parsing of `.pdf` and `.txt` files.
* **Context-Preserving Chunking:** Splits extracted text into overlapping segments (`180` words per chunk with a `40`-word overlap) to retain semantic continuity across boundaries.
* **Hybrid Retrieval Scoring:** Combines dense vector similarity (`FAISS` cosine similarity) with lexical keyword overlap (`80% semantic + 20% lexical`) to improve retrieval precision.
* **Source-Level Traceability:** Tags every chunk with its originating filename and chunk index (`filename | chunk #`), ensuring every answer can be traced back to its exact source.
* **Hallucination Guardrails:** Enforces strict prompt grounding so the model declines to answer when information is missing from the uploaded documents.
* **Smart Local Caching:** Persists vector embeddings (`embeddings.npy`) and the FAISS index (`index.faiss`) on disk, automatically invalidating caches when the uploaded document set changes.

---

## 🛠️ Tech Stack

* **Language:** Python 3.8+
* **Frontend UI:** Streamlit
* **Document Parsing:** PyPDF2
* **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Vector Database:** FAISS (`faiss-cpu` / `faiss-gpu`)
* **LLM Inference:** Hugging Face `transformers` (`google/flan-t5-small`), PyTorch

---

## 📂 Project Structure

```text
├── rag.py              # Core document processing, chunking, FAISS indexing, hybrid retrieval & LLM generation
├── ui.py               # Streamlit web interface, file upload management & cache invalidation
├── docs/               # Auto-created directory storing active uploaded documents
└── README.md           # Project documentation
```

---

## ⚙️ How It Works

1. **Document Extraction & Chunking (`load_docs`):** Reads text page-by-page from PDFs and plain text files, splitting content into overlapping 180-word chunks while recording source metadata.
2. **Embedding & Vector Indexing (`create_embeddings`, `build_index`):** Encodes chunks into L2-normalized dense vectors and indexes them using `faiss.IndexFlatIP` for fast cosine similarity search.
3. **Hybrid Re-Ranking (`retrieve_docs`):** Fetches the top candidate chunks via FAISS and re-ranks them using a weighted hybrid formula combining semantic similarity and exact query-term overlap.
4. **Grounded Generation (`generate_answer_local`):** Passes the top retrieved chunks and user query into `flan-t5-small` using deterministic decoding (`do_sample=False`) to generate factual, context-bound answers.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install Dependencies
```bash
pip install streamlit faiss-cpu sentence-transformers PyPDF2 transformers torch numpy
```

### 3. Run the Application
```bash
streamlit run ui.py
```

### 4. Usage
1. Upload one or more `.pdf` or `.txt` files via the Streamlit sidebar/uploader.
2. Click **Index Documents** to extract text, generate embeddings, and build the local FAISS index.
3. Enter your question to retrieve relevant document excerpts and generate a grounded answer.

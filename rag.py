import os
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------
# Local Embeddings Model
# -------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)  # Small & fast

# -------------------------
# Load Local LLM
# -------------------------
# Example: MPT-7B-Instruct (CPU will be slow, GPU preferred)
model_name = "google/flan-t5-small" #mosaicml/mpt-7b-instruct
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
llm_model.to(device)
llm_model.eval()

def generate_answer_local(question, context, max_new_tokens=200):
    # Keep context compact so the question is not truncated for smaller models.
    compact_context = context[:3000]
    prompt = (
        "Use the context to answer the question. "
        "If the answer is not in the context, say 'I could not find it in the documents.'\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{compact_context}\n\n"
        "Answer:"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768,
    ).to(device)

    with torch.no_grad():
        output = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return answer if answer else "I could not find it in the documents."

# -------------------------
# Chunking
# -------------------------
def chunk_text(text, max_words=180, overlap=40):
    words = text.split()
    chunks = []
    if not words:
        return chunks
    step = max(1, max_words - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + max_words]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

# -------------------------
# Load documents & chunk
# -------------------------
def load_docs(folder="docs", chunk_size=180):
    chunks = []
    sources = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder, file))
            page_text = []
            for page in reader.pages:
                extracted = page.extract_text() or ""
                if extracted.strip():
                    page_text.append(extracted)
            text = "\n".join(page_text)
        elif file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                text = f.read()
        else:
            continue

        text_chunks = chunk_text(text, max_words=chunk_size, overlap=40)
        chunks.extend(text_chunks)
        sources.extend([f"{file} | chunk {i+1}" for i in range(len(text_chunks))])
    return chunks, sources

# -------------------------
# Create embeddings (local)
# -------------------------
def create_embeddings(texts, cache_path="embeddings.npy", force_rebuild=False):
    if (not force_rebuild) and os.path.exists(cache_path):
        print("Loading cached embeddings...")
        return np.load(cache_path).astype("float32")
    embeddings = embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = embeddings.astype("float32")
    np.save(cache_path, embeddings)
    print("Embeddings cached locally.")
    return embeddings

# -------------------------
# Build FAISS index
# -------------------------
def build_index(embeddings, cache_path="index.faiss", force_rebuild=False):
    if (not force_rebuild) and os.path.exists(cache_path):
        print("Loading cached FAISS index...")
        index = faiss.read_index(cache_path)
        return index
    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine via normalized vectors
    index.add(embeddings)
    faiss.write_index(index, cache_path)
    print("FAISS index saved locally.")
    return index

# -------------------------
# Retrieve top-k relevant chunks
# -------------------------
def retrieve_docs(query, texts, sources, index, top_k=4, candidate_k=10):
    query_emb = embedding_model.encode([query], normalize_embeddings=True).astype("float32")
    candidate_k = min(candidate_k, len(texts))
    scores, ids = index.search(query_emb, candidate_k)

    query_terms = set(re.findall(r"\w+", query.lower()))
    ranked = []
    for rank, idx in enumerate(ids[0]):
        if idx < 0 or idx >= len(texts):
            continue
        text = texts[idx]
        semantic = float(scores[0][rank])
        chunk_terms = set(re.findall(r"\w+", text.lower()))
        lexical = len(query_terms & chunk_terms) / max(1, len(query_terms))
        final_score = 0.8 * semantic + 0.2 * lexical
        ranked.append((final_score, text, sources[idx]))

    ranked.sort(key=lambda x: x[0], reverse=True)
    top = ranked[:top_k]
    return [
        {
            "score": score,
            "text": text,
            "source": source,
        }
        for score, text, source in top
    ]

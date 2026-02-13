import os
import streamlit as st
from rag import (
    load_docs,
    create_embeddings,
    build_index,
    retrieve_docs,
    generate_answer_local,
)

st.set_page_config(page_title="Local RAG Document Q&A", page_icon="📄")
st.title("📄 Local RAG Document Q&A (No OpenAI)")

if "texts" not in st.session_state:
    st.session_state.texts = None
if "sources" not in st.session_state:
    st.session_state.sources = None
if "index" not in st.session_state:
    st.session_state.index = None
if "uploaded_signature" not in st.session_state:
    st.session_state.uploaded_signature = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# -------------------------
# File Upload
# -------------------------
uploaded_files = st.file_uploader(
    "Upload PDFs or TXT files (multiple allowed)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    current_signature = sorted((f.name, f.size) for f in uploaded_files)
    if current_signature != st.session_state.uploaded_signature:
        os.makedirs("docs", exist_ok=True)
        # Replace previous docs so sources always match current upload set.
        for existing in os.listdir("docs"):
            existing_path = os.path.join("docs", existing)
            if os.path.isfile(existing_path):
                os.remove(existing_path)
        for file in uploaded_files:
            with open(os.path.join("docs", file.name), "wb") as f:
                f.write(file.getbuffer())

        # Invalidate on-disk caches when document set changes.
        for cache_file in ["embeddings.npy", "index.faiss"]:
            if os.path.exists(cache_file):
                os.remove(cache_file)

        st.session_state.texts = None
        st.session_state.sources = None
        st.session_state.index = None
        st.session_state.indexed_files = []
        st.session_state.uploaded_signature = current_signature
        st.success(f"{len(uploaded_files)} files uploaded successfully!")

# -------------------------
# Index Documents
# -------------------------
if st.button("Index Documents"):
    if not uploaded_files and not os.path.isdir("docs"):
        st.warning("Please upload files first.")
    else:
        with st.spinner("Processing documents..."):
            texts, sources = load_docs("docs", chunk_size=180)
            embeddings = create_embeddings(texts, force_rebuild=True)
            index = build_index(embeddings, force_rebuild=True)
            st.session_state.texts = texts
            st.session_state.sources = sources
            st.session_state.index = index
            st.session_state.indexed_files = sorted({s.split(" | chunk ")[0] for s in sources})
        st.success("Documents indexed successfully!")

if st.session_state.indexed_files:
    st.caption("Indexed files: " + ", ".join(st.session_state.indexed_files))

# -------------------------
# Ask Questions
# -------------------------
question = st.text_input("Ask a question about your documents:")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif st.session_state.index is None:
        st.warning("Please upload and index documents first.")
    else:
        with st.spinner("Generating answer..."):
            retrieved = retrieve_docs(
                question,
                st.session_state.texts,
                st.session_state.sources,
                st.session_state.index,
            )
            best_score = retrieved[0]["score"] if retrieved else -1.0
            if best_score < 0.32:
                answer = "I could not find it in the documents."
            else:
                context = "\n".join([item["text"] for item in retrieved])
                answer = generate_answer_local(question, context)

        st.subheader("Answer:")
        st.write(answer if answer else "I could not find a confident answer from the indexed documents.")

        st.subheader("Sources:")
        for item in retrieved:
            chunk = item["text"]
            source = item["source"]
            st.write(f"- [{item['score']:.3f}] {source}: {chunk[:150].replace(chr(10), ' ')}...")

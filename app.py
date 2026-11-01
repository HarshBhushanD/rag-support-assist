from pathlib import Path

import streamlit as st

from src.ingest.chunker import chunk_document_recursive
from src.ingest.loaders import load_documents
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.retriever import RetrievalEngine

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


st.set_page_config(
    page_title="RAG Support Assistant",
    page_icon="🤖",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def build_pipeline(retrieval_mode: str, top_k: int):
    """Load support documents and initialize the RAG pipeline."""
    documents = load_documents(RAW_DIR)

    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document_recursive(
            document,
            chunk_size=500,
            overlap=50,
        )
    ]

    if not chunks:
        raise ValueError("No document chunks found in data/raw.")

    retriever = RetrievalEngine(
        chunks=chunks,
        collection_name="streamlit_support_chunks",
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        max_new_tokens=160,
    )

    return pipeline, len(documents), len(chunks)


def render_sources(sources: list, retrieved_chunks: list):
    """Display source references and retrieved context."""
    if sources:
        st.markdown("**Sources**")

        for source in sources:
            if isinstance(source, dict):
                source_name = (
                    source.get("source")
                    or source.get("name")
                    or source.get("file_name")
                    or str(source)
                )
            else:
                source_name = str(source)

            st.markdown(f"- `{source_name}`")

    if retrieved_chunks:
        with st.expander(f"View retrieved document chunks ({len(retrieved_chunks)})"):
            for index, chunk in enumerate(retrieved_chunks, start=1):
                source_name = chunk.get("source", "Unknown source")
                chunk_index = chunk.get("chunk_index", "N/A")
                score = chunk.get("score")

                st.markdown(
                    f"**Chunk {index}** — `{source_name}` (chunk index: {chunk_index})"
                )

                if isinstance(score, (int, float)):
                    st.caption(f"Retrieval score: {score:.4f}")

                st.write(chunk.get("text", "No text available."))
                st.divider()


st.title("🤖 RAG Support Assistant")
st.caption(
    "Ask questions about the available support documentation. "
    "Answers are grounded in your documents."
)

with st.sidebar:
    st.header("Settings")

    retrieval_mode = st.selectbox(
        "Retrieval mode",
        options=["hybrid", "dense", "bm25"],
        index=0,
        help="Choose how relevant document chunks are retrieved.",
    )

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=10,
        value=3,
    )

    st.divider()

    st.subheader("Knowledge base")
    st.write("Source directory: `data/raw/`")
    st.write("Supported files: PDF, HTML, Markdown")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("RAG Support Assistant · Day 24")

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display existing conversation.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            st.caption(f"Retrieval mode: {message.get('retrieval_mode', 'N/A')}")
            render_sources(
                message.get("sources", []),
                message.get("retrieved_chunks", []),
            )


# Accept a new question.
if question := st.chat_input("Ask a question about your support documents..."):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    try:
        with st.spinner("Searching documents and generating an answer..."):
            pipeline, document_count, chunk_count = build_pipeline(
                retrieval_mode,
                top_k,
            )

            result = pipeline.ask(question)

        assistant_message = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "retrieval_mode": result.get("retrieval_mode", retrieval_mode),
        }

        st.session_state.messages.append(assistant_message)

        st.toast(
            f"Knowledge base: {document_count} documents, {chunk_count} chunks",
            icon="📚",
        )

    except Exception as error:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "I encountered an error while processing your question. "
                    "Please check the application logs and try again."
                ),
                "error": str(error),
            }
        )

    st.rerun()

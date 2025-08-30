import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import qrcode
from io import BytesIO
import base64
import json
import pandas as pd
import os

# Watsonx imports
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import Model
except ImportError:
    Credentials = None
    Model = None

# -------------------------
# Gamification & History state
# -------------------------
if "xp" not in st.session_state:
    st.session_state.xp = 0
if "badge" not in st.session_state:
    st.session_state.badge = "🥉 Bronze Learner"
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# Embedding model
# -------------------------
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------
# PDF extraction with page info
# -------------------------
def extract_pages(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"page_num": i+1, "text": text})
    return pages

def chunk_pages(pages, chunk_words=300):
    chunks, meta = [], []
    for p in pages:
        words = p["text"].split()
        for i in range(0, len(words), chunk_words):
            chunk = " ".join(words[i:i+chunk_words])
            if chunk.strip():
                chunks.append(chunk)
                meta.append({"page_num": p["page_num"]})
    return chunks, meta

# -------------------------
# Build FAISS index
# -------------------------
def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def retrieve_chunks_with_ids(query, chunks, index, top_k=3):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, top_k)
    ids = I[0].tolist()
    return [chunks[i] for i in ids], ids

# -------------------------
# Local QA Pipeline (fallback)
# -------------------------
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def local_generate_answer(query, context):
    if not context.strip():
        return "I couldn't find relevant info."
    result = qa_pipeline(question=query, context=context)
    return result["answer"]

# -------------------------
# IBM Watsonx Integration
# -------------------------
def init_watsonx_model():
    if Credentials is None:
        return None
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    region = os.getenv("WATSONX_REGION", "us-south")
    model_id = os.getenv("WATSONX_MODEL_ID", "mistralai/mixtral-8x7b-instruct-v0.1")

    if not api_key or not project_id:
        return None

    creds = Credentials(api_key=api_key, url=f"https://{region}.ml.cloud.ibm.com")
    gen_params = {
        "decoding_method": "greedy",
        "temperature": 0.2,
        "max_new_tokens": 300,
        "repetition_penalty": 1.05,
        "stop_sequences": ["</answer>"]
    }
    try:
        return Model(model_id=model_id, params=gen_params, credentials=creds, project_id=project_id)
    except Exception:
        return None

watsonx_model = init_watsonx_model()

SYSTEM_INSTRUCTION = (
    "You are StudyMate, a concise academic assistant.\n"
    "Answer ONLY using the provided context. "
    "If the answer is not present, say 'I couldn't find relevant info in the uploaded document.'\n"
)

def build_prompt(user_question: str, context_text: str) -> str:
    ctx = context_text[:4500]
    return (
        f"<system>\n{SYSTEM_INSTRUCTION}</system>\n"
        f"<context>\n{ctx}\n</context>\n"
        f"<question>\n{user_question}\n</question>\n"
        f"<answer>"
    )

def watsonx_generate_answer(query, context):
    if watsonx_model is None or not context.strip():
        return "Watsonx not configured or no context."
    prompt = build_prompt(query, context)
    try:
        resp = watsonx_model.generate(prompt=prompt)
        if isinstance(resp, dict) and "results" in resp and resp["results"]:
            return resp["results"][0].get("generated_text", "").strip()
        if isinstance(resp, str):
            return resp.strip()
        return "I couldn't find relevant info."
    except Exception as e:
        return f"Watsonx error: {e}"

# -------------------------
# QR code helper
# -------------------------
def make_qr(data: str):
    img = qrcode.make(data)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# -------------------------
# Streamlit UI
# -------------------------
st.title("📚 StudyMate - AI Powered PDF Q&A")
st.caption("Upload a PDF, ask questions, and get smart answers with gamification, QR sharing, and history.")

# Model selection toggle
mode = st.radio(
    "🤖 Choose Answer Mode",
    ["Local (DistilBERT)", "IBM Watsonx"],
    index=1  # 0 = Local, 1 = Watsonx (default)
)

uploaded_file = st.file_uploader("📂 Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Extracting text..."):
        pages = extract_pages(uploaded_file)
        if not pages:
            st.warning("No readable text found in PDF.")
        else:
            chunks, meta = chunk_pages(pages)
            index, embeddings = build_faiss_index(chunks)
    st.success("✅ PDF processed successfully!")

    query = st.text_input("💬 Ask a question from the document:")
    if query:
        with st.spinner("Searching & generating answer..."):
            retrieved_chunks, ids = retrieve_chunks_with_ids(query, chunks, index, top_k=3)
            context = " ".join(retrieved_chunks)

            if mode == "IBM Watsonx":
                answer = watsonx_generate_answer(query, context)
            else:
                answer = local_generate_answer(query, context)

        # Show answer
        st.markdown("### ✅ Answer:")
        st.write(answer)

        # Show references
        st.markdown("### 🔎 References (with page numbers):")
        for i in ids:
            st.write(f"- Page {meta[i]['page_num']}: {chunks[i][:150]}{'...' if len(chunks[i])>150 else ''}")

        # 🎮 Gamification update
        st.session_state.xp += 20 if context else 10
        if st.session_state.xp > 100:
            st.session_state.badge = "🥇 Gold Scholar"
        elif st.session_state.xp > 50:
            st.session_state.badge = "🥈 Silver Explorer"
        else:
            st.session_state.badge = "🥉 Bronze Learner"

        # Add to history
        st.session_state.history.append({
            "question": query,
            "answer": answer,
            "page_refs": [meta[i]['page_num'] for i in ids]
        })

        # 📲 QR sharing
        share_payload = {"question": query, "answer": answer}
        qr_buf = make_qr(json.dumps(share_payload))
        st.markdown("### 📲 Share via QR")
        st.image(qr_buf, caption="Scan to view Q&A")
        png_bytes = qr_buf.getvalue()
        st.download_button(
            label="Download QR (PNG)",
            data=png_bytes,
            file_name="studymate_answer_qr.png",
            mime="image/png"
        )

# -------------------------
# Sidebar: Gamification + History
# -------------------------
st.sidebar.markdown("## 🎮 Gamification")
st.sidebar.write(f"**XP:** {st.session_state.xp}")
st.sidebar.write(f"**Badge:** {st.session_state.badge}")

st.sidebar.markdown("## 🗂️ Q&A History")
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.sidebar.dataframe(df, use_container_width=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "⬇️ Download History (CSV)",
        data=csv_bytes,
        file_name="studymate_history.csv",
        mime="text/csv"
    )
else:
    st.sidebar.write("No Q&A yet.")


    

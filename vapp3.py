import streamlit as st
import os, tempfile, re, numpy as np, faiss, pickle
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from docx import Document as DocxDocument
from transformers import pipeline
import speech_recognition as sr

st.title("📄 Azure RAG Q&A (Fast Voice + Text, CPU)")

# ==========================
# Load environment
# ==========================
load_dotenv()
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")

if not AZURE_CONNECTION_STRING or not CONTAINER_NAME:
    st.error("Azure environment variables not loaded. Check your .env file.")
    st.stop()

# ==========================
# Load or build FAISS index
# ==========================
@st.cache_resource(show_spinner=True)
def load_index():
    if os.path.exists("faiss_index.idx") and os.path.exists("chunks.pkl"):
        index = faiss.read_index("faiss_index.idx")
        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}  # force CPU
        )
        return index, chunks, embedder

    # Build index from Azure blobs
    container = BlobServiceClient.from_connection_string(
        AZURE_CONNECTION_STRING
    ).get_container_client(CONTAINER_NAME)
    texts = []
    for blob in container.list_blobs():
        if not blob.name.lower().endswith((".pdf", ".docx", ".txt")):
            continue
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(container.get_blob_client(blob.name).download_blob().readall())
        tmp.close()
        if blob.name.endswith(".pdf"):
            text = "\n".join(p.page_content for p in PyPDFLoader(tmp.name).load())
        elif blob.name.endswith(".docx"):
            doc = DocxDocument(tmp.name)
            text = "\n".join(p.text for p in doc.paragraphs)
        else:
            text = open(tmp.name, "r", encoding="utf-8", errors="ignore").read()
        os.remove(tmp.name)
        texts.append(re.sub(r"\s+", " ", text))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = [c for t in texts for c in splitter.split_text(t)]
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectors = np.array(embedder.embed_documents(chunks), dtype="float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    # Save for future runs
    faiss.write_index(index, "faiss_index.idx")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks, embedder

index, chunks, embedder = load_index()

# ==========================
# Hugging Face generator
# ==========================
@st.cache_resource
def get_generator():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1  # CPU
    )

generator = get_generator()

# ==========================
# Voice recognition
# ==========================
recognizer = sr.Recognizer()

def add_punctuation(question):
    question = question.strip()
    if not question.endswith(("?", ".")):
        question += "?"
    return question

def process_question(question_text, placeholder):
    question_text = add_punctuation(question_text)

    # Embed query and retrieve top 5 chunks
    q_emb = np.array(embedder.embed_query(question_text), dtype="float32")
    _, I = index.search(q_emb.reshape(1, -1), 5)

    # Deduplicate chunks
    top_chunks = []
    seen = set()
    for i in I[0]:
        chunk = chunks[i].strip()
        if chunk not in seen and chunk:
            top_chunks.append(chunk)
            seen.add(chunk)

    # Limit to 3 unique chunks
    context = "\n\n".join(top_chunks[:3])
    context = re.sub(r"\d+\.", "", context)

    # Create prompt
    if "program" in question_text.lower() or "write" in question_text.lower():
        prompt = f"Use ONLY the context.\nReturn Python code only.\n\nContext:\n{context}\n\nQuestion:\n{question_text}\n\nAnswer:"
    else:
        prompt = f"Use ONLY the context.\n\nContext:\n{context}\n\nQuestion:\n{question_text}\n\nAnswer:"

    # Generate answer
    answer = generator(prompt, max_new_tokens=150, do_sample=False)[0]["generated_text"]
    placeholder.subheader("Answer")
    if "program" in question_text.lower():
        placeholder.code(answer, language="python")
    else:
        placeholder.write(answer)

# ==========================
# Voice input
# ==========================
if st.button("🎤 Ask with Voice"):
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            voice_text = recognizer.recognize_google(audio)
            if voice_text:
                st.write("You said:", voice_text)
                placeholder = st.empty()
                process_question(voice_text, placeholder)
    except Exception as e:
        st.error(f"Voice recognition failed: {e}")

# ==========================
# Text input
# ==========================
question_text = st.text_input("Or type your question")
if question_text:
    placeholder = st.empty()
    process_question(question_text, placeholder)

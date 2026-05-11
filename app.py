import os
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# ---------------- IMPORTANT ----------------
# Uncomment and verify this path exists
pytesseract.pytesseract.tesseract_cmd = r"C:\HP\Desktop\Tesseract-OCR\tesseract.exe"

# ---------------- LOAD ENV ----------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env file")
    st.stop()

genai.configure(api_key=API_KEY)

# ---------------- GEMINI MODEL ----------------
try:
    available_models = []

    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            available_models.append(m.name)

    if not available_models:
        st.error("❌ No Gemini models available.")
        st.stop()

    selected_model = available_models[0]
    model = genai.GenerativeModel(selected_model)

except Exception as e:
    st.error(f"❌ Gemini setup failed: {e}")
    st.stop()

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="Multi-Document Chat Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Document Chat Assistant")
st.write("Upload PDFs or images and ask AI-powered questions.")

# ---------------- SESSION ----------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "vectors" not in st.session_state:
    st.session_state.vectors = None

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "📄 Upload PDFs or Images",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# ---------------- PROCESS ----------------
if uploaded_files and st.button("Process Documents"):
    all_chunks = []

    with st.spinner("Reading files..."):

        for file in uploaded_files:

            # ---------- PDF ----------
            if file.type == "application/pdf":
                pdf_bytes = file.read()
                found_text = False

                # Normal text extraction
                try:
                    reader = PdfReader(file)

                    for page in reader.pages:
                        text = page.extract_text()

                        if text and text.strip():
                            found_text = True

                            text = text.replace("\n", " ").strip()

                            if len(text) > 30:
                                all_chunks.append(text)

                            parts = text.split(".")

                            cleaned = [
                                p.strip()
                                for p in parts
                                if len(p.strip()) > 5
                            ]

                            all_chunks.extend(cleaned)

                except:
                    pass

                # OCR fallback for scanned PDFs
                if not found_text:
                    st.info(f"Using OCR for: {file.name}")

                    try:
                        images = convert_from_bytes(
                            pdf_bytes,
                            poppler_path=r"C:\poppler\Library\bin"
                        )

                        for img in images:
                            text = pytesseract.image_to_string(
                                img,
                                config="--psm 6"
                            )

                            if text and text.strip():
                                text = text.replace(
                                    "\n",
                                    " "
                                ).strip()

                                if len(text) > 10:
                                    all_chunks.append(text)

                    except Exception as e:
                        st.warning(
                            f"OCR failed: {e}"
                        )

            # ---------- IMAGE ----------
            else:
                try:
                    image = Image.open(file)

                    st.image(
                        image,
                        caption=file.name,
                        width=300
                    )

                    vision_prompt = """
                    Analyze this document image.

                    Tell:
                    1. What type of document it is
                    2. Extract important visible details
                    3. Give a short summary
                    """

                    response = model.generate_content(
                        [vision_prompt, image]
                    )

                    text = response.text

                    if text and text.strip():
                        all_chunks.append(text)
                        st.success(
                            f"✅ Image analyzed: {file.name}"
                        )

                except Exception as e:
                    st.warning(
                        f"Image read failed: {e}"
                    )

    # Remove empty chunks
    all_chunks = [
        c for c in all_chunks
        if c and c.strip()
    ]

    if not all_chunks:
        st.error("❌ No readable text found.")
        st.stop()

    # TF-IDF
    st.session_state.chunks = all_chunks

    vectorizer = TfidfVectorizer(
        stop_words=None
    )

    vectors = vectorizer.fit_transform(
        all_chunks
    )

    st.session_state.vectorizer = vectorizer
    st.session_state.vectors = vectors

    st.success(
        "✅ Documents processed successfully!"
    )

# ---------------- QUESTION ----------------
query = st.text_input(
    "❓ Ask a question"
)

if query and st.session_state.vectorizer is not None:
    query_vector = (
        st.session_state.vectorizer
        .transform([query])
    )

    similarities = cosine_similarity(
        query_vector,
        st.session_state.vectors
    )

    top_indices = (
        similarities[0]
        .argsort()[-5:][::-1]
    )

    context = "\n".join(
        st.session_state.chunks[i]
        for i in top_indices
    )

    prompt = f"""
You are a helpful AI assistant.

Use the document context below to answer the question.

Document Context:
{context}

Question:
{query}
"""

    with st.spinner(
        "🤖 Gemini is thinking..."
    ):
        try:
            response = model.generate_content(
                prompt
            )

            st.subheader(
                "🤖 AI Answer"
            )
            st.write(
                response.text
            )

        except Exception as e:
            st.error(
                f"Gemini error: {e}"
            )

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(
    "Built with Python, Streamlit, Gemini API, OCR, and scikit-learn"
)
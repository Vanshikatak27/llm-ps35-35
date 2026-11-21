import os
import re
import time
import pandas as pd
import streamlit as st
from fpdf import FPDF
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai
from google.genai import types

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY", "").strip()

st.set_page_config(
    page_title="DocCompare AI | Multi-PDF RAG Engine",
    page_icon="📊",
    layout="wide"
)

st.title("📊 DocCompare AI — Multi-Document RAG Comparison Engine")
st.caption("Isolated per-PDF FAISS vector retrievers with page-level source grounding and one-click PDF report export.")

with st.sidebar:
    st.header("1. Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload 2 to 3 PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )
    top_k = st.slider("Chunks retrieved per PDF (Top-K)", min_value=2, max_value=5, value=3)
    if uploaded_files:
        st.success(f"{len(uploaded_files)} PDF(s) loaded.")
    st.divider()
    st.markdown("**Architecture:**\n- Isolated FAISS index per PDF\n- `gemini-embedding-001` vectors\n- `gemini-3.5-flash-lite` synthesis")

def create_faiss_with_retry(chunks, embeddings, max_retries=3):
    last_err = None
    for attempt in range(max_retries):
        try:
            return FAISS.from_documents(chunks, embeddings)
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", err_str)
                wait_sec = min(int(float(match.group(1))) + 2, 65) if match else 20 * (attempt + 1)
                st.toast(f"Embedding rate limit reached. Waiting {wait_sec}s and retrying...")
                time.sleep(wait_sec)
            else:
                raise
    raise last_err

def build_separate_retrievers(files, key, k_val):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=key
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    retrievers = {}
    stats = {"total_pages": 0, "total_chunks": 0, "per_doc": {}}

    for idx, file in enumerate(files):
        if idx > 0:
            time.sleep(1.5)
        file.seek(0)
        reader = PdfReader(file)
        page_docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                page_docs.append(
                    splitter.create_documents(
                        [f"[Page {i+1}] {text}"],
                        metadatas=[{"source": file.name, "page": i + 1}]
                    )
                )
        flat_chunks = [chunk for sublist in page_docs for chunk in sublist]
        vectorstore = create_faiss_with_retry(flat_chunks, embeddings)
        retrievers[file.name] = vectorstore.as_retriever(search_kwargs={"k": k_val})

        stats["total_pages"] += len(reader.pages)
        stats["total_chunks"] += len(flat_chunks)
        stats["per_doc"][file.name] = {"pages": len(reader.pages), "chunks": len(flat_chunks)}

    return retrievers, stats

def generate_fast(client, prompt):
    models_to_try = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-4-hydrogen"]
    last_error = None
    for model_name in models_to_try:
        for _ in range(2):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return resp, model_name
            except Exception as e:
                last_error = e
                time.sleep(1)
    raise last_error

def clean_text_for_pdf(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("—", "-").replace("–", "-").replace("’", "'").replace("“", '"').replace("”", '"').replace("•", "-")
    return text.encode("latin-1", "replace").decode("latin-1")

def create_comparison_pdf(question, doc_names, md_output, model_used, latency):
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(25, 60, 120)
    pdf.cell(pdf.epw, 10, "DocCompare AI - Multi-PDF Comparison Report", new_x="LMARGIN", new_y="NEXT")

    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    meta = f"Documents Compared: {', '.join(doc_names)}  |  Model: {model_used}  |  Latency: {latency}s"
    pdf.multi_cell(pdf.epw, 6, clean_text_for_pdf(meta), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Question Box
    pdf.set_x(pdf.l_margin)
    pdf.set_fill_color(240, 245, 255)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(pdf.epw, 7, clean_text_for_pdf(f"Comparison Question: {question}"), border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Parse Markdown Table & Takeaways
    table_lines = [line.strip() for line in md_output.splitlines() if line.strip().startswith("|")]
    non_table_lines = [line.strip() for line in md_output.splitlines() if line.strip() and not line.strip().startswith("|")]

    if len(table_lines) >= 3:
        headers = [clean_text_for_pdf(h.strip()) for h in table_lines[0].strip("|").split("|")]
        rows = []
        for line in table_lines[2:]:
            if set(line.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
                continue
            cells = [clean_text_for_pdf(c.strip()) for c in line.strip("|").split("|")]
            if len(cells) == len(headers):
                rows.append(cells)

        pdf.set_x(pdf.l_margin)
        with pdf.table(
            borders_layout="ALL",
            cell_fill_color=(248, 250, 252),
            cell_fill_mode="ROWS",
            line_height=6,
            text_align="LEFT",
            width=pdf.epw
        ) as table:
            pdf.set_font("Helvetica", "B", 9)
            hdr_row = table.row()
            for h in headers:
                hdr_row.cell(h)

            pdf.set_font("Helvetica", "", 9)
            for r in rows:
                row = table.row()
                for val in r:
                    row.cell(val)
        pdf.ln(5)

    # Key Takeaways / Remaining Text
    for line in non_table_lines:
        clean_line = clean_text_for_pdf(line)
        if not clean_line.strip():
            continue
        pdf.set_x(pdf.l_margin)
        if clean_line.startswith("#"):
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(25, 60, 120)
            pdf.ln(2)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.epw, 7, clean_line.lstrip("# ").strip(), new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(pdf.epw, 6, clean_line, new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())

if not api_key:
    st.error("Missing GEMINI_API_KEY in your .env file. Add GEMINI_API_KEY=your_key and save.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

if not uploaded_files:
    st.info("👈 Upload 2 to 3 PDFs from the sidebar to initialize the RAG pipeline.")
elif len(uploaded_files) < 2 or len(uploaded_files) > 3:
    st.warning("Please upload between 2 and 3 PDF files to compare.")
else:
    st.subheader("2. Cross-Document Query")
    col_q1, col_q2 = st.columns([1, 2])
    with col_q1:
        sample_q = st.selectbox(
            "Preset comparison templates:",
            [
                "Custom question...",
                "Compare participant name, course/program title, issuing authority, date, and ID.",
                "Compare skills, technical stack, experience, and qualifications across documents.",
                "Compare objectives, methodology, key results, and limitations across documents.",
                "Compare eligibility criteria, benefits, terms, and deadlines across documents."
            ]
        )
    with col_q2:
        default_text = "" if sample_q == "Custom question..." else sample_q
        question = st.text_input("Comparison question:", value=default_text, placeholder="Ask a question to compare across all uploaded PDFs...")

    if st.button("🚀 Generate Comparison Output", type="primary"):
        if not question.strip():
            st.warning("Please enter a comparison question before generating output.")
        else:
            start_time = time.time()
            file_names = [f.name for f in uploaded_files]
            file_signature = (tuple(file_names), top_k)

            if "retrievers" not in st.session_state or st.session_state.get("indexed_sig") != file_signature:
                try:
                    with st.spinner("Step 1/2: Building isolated FAISS vector indices per PDF..."):
                        retrievers, doc_stats = build_separate_retrievers(uploaded_files, api_key, top_k)
                        st.session_state.retrievers = retrievers
                        st.session_state.doc_stats = doc_stats
                        st.session_state.indexed_sig = file_signature
                except Exception as e:
                    st.error("Free-tier embedding quota temporarily reached. Please wait ~60 seconds and click Generate Comparison Output again.")
                    st.stop()

            with st.spinner("Step 2/2: Retrieving chunks per PDF & synthesizing comparison matrix..."):
                retrieved_sections = []
                for doc_name, retriever in st.session_state.retrievers.items():
                    docs = retriever.invoke(question)
                    context = "\n".join([d.page_content for d in docs])
                    retrieved_sections.append(f"=== DOCUMENT: {doc_name} ===\n{context}")

                combined_context = "\n\n".join(retrieved_sections)

                prompt = f"""You are a precise multi-document analysis engine.
Compare the uploaded documents based ONLY on the retrieved context below to answer the user's question.

User Question: {question}

Retrieved Context from Separate Retrievers:
{combined_context}

Instructions:
1. Output a clean Markdown comparison table with 'Criteria / Aspect' as the first column and each document's filename as the remaining columns.
2. Cite the page number in brackets (e.g., [Page 1]) inside table cells whenever available in the context.
3. If a document does not mention a specific criterion, write 'Not mentioned'.
4. Below the table, provide a '### Key Takeaways' section with 3 concise bullet points highlighting major differences and similarities.
"""
                client = genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(timeout=15000)
                )
                response, used_model = generate_fast(client, prompt)
                elapsed = round(time.time() - start_time, 2)

                st.session_state.last_result = {
                    "question": question,
                    "doc_names": file_names,
                    "output": response.text,
                    "sections": retrieved_sections,
                    "latency": elapsed,
                    "model": used_model
                }
                st.session_state.history.append({
                    "Question": question,
                    "Documents": ", ".join(file_names),
                    "Latency (s)": elapsed,
                    "Model": used_model
                })

    if "last_result" in st.session_state:
        res = st.session_state.last_result
        stats = st.session_state.doc_stats

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Documents Indexed", len(stats["per_doc"]))
        m2.metric("Total Pages Parsed", stats["total_pages"])
        m3.metric("Vector Chunks", stats["total_chunks"])
        m4.metric("Response Latency", f"{res['latency']} s", delta=res["model"])

        tab1, tab2, tab3 = st.tabs([
            "📊 Comparison Matrix & PDF Export",
            "🔍 RAG Inspector (Per-PDF Chunks)",
            "🕒 Session Query History"
        ])

        with tab1:
            st.markdown(res["output"])
            st.divider()
            pdf_bytes = create_comparison_pdf(
                res["question"],
                res["doc_names"],
                res["output"],
                res["model"],
                res["latency"]
            )
            st.download_button(
                label="📄 Save Comparison Report (PDF)",
                data=pdf_bytes,
                file_name="Multi_PDF_Comparison_Report.pdf",
                mime="application/pdf",
                type="primary"
            )

        with tab2:
            st.caption("Each PDF is queried via its own isolated FAISS retriever before synthesis:")
            for section in res["sections"]:
                st.code(section, language="text")

        with tab3:
            if st.session_state.history:
                st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
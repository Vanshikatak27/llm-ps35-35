# Multi-PDF Comparison Tool (PS #35)

A Retrieval-Augmented Generation (RAG) web application built with **Python, Streamlit, LangChain, FAISS, and Google Gemini API**. Upload 2–3 PDFs, build isolated vector retrievers for each document, and generate side-by-side Markdown comparison tables with page-level source citations and one-click PDF report export.

## Features
- **Isolated Per-PDF FAISS Retrievers:** Prevents cross-document chunk mixing during retrieval.
- **Structured Comparison Tables:** Generates clean side-by-side comparisons with `[Page X]` citations.
- **One-Click PDF Export:** Downloads the comparison question, matrix, and key takeaways as a formatted A4 PDF report.
- **RAG Inspector & Telemetry:** Displays response latency, chunk counts, and retrieved source chunks per PDF.

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/Vanshikatak27/llm-ps35-35.git
cd llm-ps35-35
# AutoDoc AI — HLD Intelligence

A source-grounded RAG prototype for question answering over automotive High-Level Design documents.

## Features
- Streamlit web interface
- PDF, TXT and Markdown ingestion
- Page/source metadata
- Overlapping text chunking
- Gemini embeddings + ChromaDB semantic retrieval
- Gemini answer generation with grounding instructions
- Expandable source evidence panel
- Synthetic sample HLD and evaluation question set
- Evaluation CSV generator for human review

## 1. Prerequisites
- Python 3.10 or newer (Python 3.11 is a good choice)
- Internet connection
- Gemini API key from Google AI Studio
- Windows PowerShell, macOS Terminal or Linux shell

## 2. Create a virtual environment

### Windows PowerShell
```powershell
cd AUTOSAR_HLD_Document_Analysis_Assistant
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux
```bash
cd AUTOSAR_HLD_Document_Analysis_Assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Configure Gemini API
1. Create an API key in Google AI Studio.
2. Copy `.env.example` to `.env`.
3. Replace `PASTE_YOUR_GEMINI_API_KEY_HERE` with your key.
4. Never share `.env`, commit it to GitHub, or put the key in screenshots.

## 4. Run the application
```bash
streamlit run app.py
```
Open the local URL Streamlit prints (usually `http://localhost:8501`).

## 5. First run
1. In the sidebar, upload `data/sample_autosar_hld_synthetic.txt`.
2. Click **Build knowledge index**.
3. Ask: `Who consumes the VehicleStatus interface?`
4. Expand the evidence panel and verify the source passage.
5. Ask: `What is the maximum vehicle speed?` The synthetic document does not specify this.

## 6. Run the evaluation script
Make sure the `.env` file is configured:
```bash
python scripts/run_evaluation.py
```
It creates `evaluation/evaluation_run_results.csv`. Review the answers against the expected evidence and fill the manual review columns before calculating/reporting metrics. API calls may incur usage or quota limits.

## Project structure
```text
AUTOSAR_HLD_Document_Analysis_Assistant/
├── app.py
├── rag_engine.py
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   └── sample_autosar_hld_synthetic.txt
├── evaluation/
│   ├── evaluation_questions.csv
│   └── answer_review_template.csv
├── scripts/
│   └── run_evaluation.py
└── Documentation/
    ├── PROJECT_REPORT_TEMPLATE.md
    ├── DEMO_SCRIPT.md
    └── DECLARATION_TEMPLATE.md
```

## Important notes
- The included HLD is synthetic and is not an official AUTOSAR specification.
- Use only data that you have permission to upload to the Gemini API.
- The app recreates its single active ChromaDB collection whenever a new index is built.
- Similarity values are retrieval-ranking signals, not calibrated confidence values.
- Scanned/image-only PDFs are not OCR'd by this prototype.
- Cite only metrics that you actually measured.

"""Creates an evaluation run sheet. Answers are reviewed by a human; no metric is fabricated."""
import os
import sys
import csv
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from rag_engine import extract_documents, build_index, answer_question

DATA_FILE = ROOT / "data" / "sample_autosar_hld_synthetic.txt"
QUESTIONS_FILE = ROOT / "evaluation" / "evaluation_questions.csv"
OUT_FILE = ROOT / "evaluation" / "evaluation_run_results.csv"

class SimpleUpload:
    def __init__(self, path):
        self.name = path.name
        self._bytes = path.read_bytes()
    def getvalue(self):
        return self._bytes

def main():
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY in .env first.")
    docs = extract_documents([SimpleUpload(DATA_FILE)])
    info = build_index(docs)
    print(f"Indexed {info['chunks']} chunks.")
    with QUESTIONS_FILE.open(encoding="utf-8", newline="") as f:
        questions = list(csv.DictReader(f))
    rows = []
    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {item['question']}")
        result = answer_question(item["question"], top_k=4)
        srcs = "; ".join(f"{s['source']} p.{s['page']}" for s in result["sources"])
        rows.append({
            "question": item["question"],
            "expected_answer_or_evidence": item["expected_answer_or_evidence"],
            "expected_behavior": item["expected_behavior"],
            "model_answer": result["answer"],
            "retrieved_sources": srcs,
            "citation_correct_yes_no": "",
            "grounded_yes_no": "",
            "complete_yes_no": "",
            "reviewer_notes": ""
        })
    with OUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUT_FILE}")
    print("Open the CSV and manually fill the review columns before reporting metrics.")

if __name__ == "__main__":
    main()

# course-rag-update2 (tables-only pipeline)

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Put your files here (non-recursive):
#   PDFs  -> data/pdfs/
#   PPTX  -> data/pptx/
#   HTML  -> data/html/
# Copy template and edit your eval questions:
cp data/eval/questions.TEMPLATE.jsonl data/eval/questions.jsonl

# Run everything (no graphs; outputs as CSV + Markdown)
make all
```

## Outputs
- data/ingested/raw.jsonl
- data/chunks/chunks.jsonl
- artifacts/tfidf/{vectorizer.joblib, tfidf_matrix.joblib, chunks_index.csv}
- data/eval/retrievals.jsonl
- data/eval/predictions.jsonl
- data/reports/
  - corpus_stats.csv + .md
  - retrieval_metrics.csv + .md
  - qa_metrics.csv + .md
  - final_report.md

## Windows (PowerShell) without Makefile
Run the commands from `README_WINDOWS.md`.

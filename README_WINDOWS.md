# Windows (PowerShell) run (no Makefile required)

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Ingest
python scripts\ingest_docs.py --pptx_dir data\pptx --pdf_dir data\pdfs --html_dir data\html --out_jsonl data\ingested\raw.jsonl
# Chunk
python scripts\chunk_docs.py --in_jsonl data\ingested\raw.jsonl --out_jsonl data\chunks\chunks.jsonl --chunk_words 180 --overlap_words 30
# Index
python scripts\build_indices.py --chunks_jsonl data\chunks\chunks.jsonl --index_dir artifacts\tfidf
# Retrieve
python scripts\retrieve_tfidf.py --index_dir artifacts\tfidf --chunks_jsonl data\chunks\chunks.jsonl --questions data\eval\questions.jsonl --k 5 --out data\eval\retrievals.jsonl
# Answer
python scripts\answer_extractive.py --chunks_jsonl data\chunks\chunks.jsonl --retrievals data\eval\retrievals.jsonl --out data\eval\predictions.jsonl
# Eval retrieval
python scripts\eval_retrieval.py --retrievals data\eval\retrievals.jsonl --questions data\eval\questions.jsonl --out_dir data\reports
# Eval QA
python scripts\eval_qa.py --predictions data\eval\predictions.jsonl --questions data\eval\questions.jsonl --out_dir data\reports
# Tables
python scripts\report_tables.py --raw data\ingested\raw.jsonl --chunks data\chunks\chunks.jsonl --retrievals data\eval\retrievals.jsonl --retrieval_report data\reports\retrieval_metrics.csv --qa_report data\reports\qa_metrics.csv --out_dir data\reports
```

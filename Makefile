PY := python

RAW_JSONL := data/ingested/raw.jsonl
CHUNKS_JSONL := data/chunks/chunks.jsonl
INDEX_DIR := artifacts/tfidf
QUESTIONS := data/eval/questions.jsonl
RETRIEVALS := data/eval/retrievals.jsonl
PREDICTIONS := data/eval/predictions.jsonl
REPORT_DIR := data/reports

.PHONY: dirs ingest chunk index retrieve answer eval-retrieval eval-qa report all

dirs:
	mkdir -p data/pptx data/pdfs data/html data/docx data/ingested data/chunks $(INDEX_DIR) data/eval $(REPORT_DIR)

ingest: dirs
	$(PY) scripts/ingest_docs.py --pptx_dir data/pptx --pdf_dir data/pdfs --html_dir data/html --docx_dir data/docx --out_jsonl $(RAW_JSONL)

chunk:
	$(PY) scripts/chunk_docs.py --in_jsonl $(RAW_JSONL) --out_jsonl $(CHUNKS_JSONL) --chunk_words 180 --overlap_words 30

index:
	$(PY) scripts/build_indices.py --chunks_jsonl $(CHUNKS_JSONL) --index_dir $(INDEX_DIR)

retrieve:
	$(PY) scripts/retrieve_tfidf.py --index_dir $(INDEX_DIR) --chunks_jsonl $(CHUNKS_JSONL) --questions $(QUESTIONS) --k 5 --out $(RETRIEVALS)

answer:
	$(PY) scripts/answer_extractive.py --chunks_jsonl $(CHUNKS_JSONL) --retrievals $(RETRIEVALS) --out $(PREDICTIONS)

eval-retrieval:
	$(PY) scripts/eval_retrieval.py --retrievals $(RETRIEVALS) --questions $(QUESTIONS) --out_dir $(REPORT_DIR)

eval-qa:
	$(PY) scripts/eval_qa.py --predictions $(PREDICTIONS) --questions $(QUESTIONS) --out_dir $(REPORT_DIR)

report:
	$(PY) scripts/report_tables.py --raw $(RAW_JSONL) --chunks $(CHUNKS_JSONL) --retrievals $(RETRIEVALS) --retrieval_report $(REPORT_DIR)/retrieval_metrics.csv --qa_report $(REPORT_DIR)/qa_metrics.csv --out_dir $(REPORT_DIR)

all: ingest chunk index retrieve answer eval-retrieval eval-qa report

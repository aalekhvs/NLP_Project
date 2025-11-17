#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run full RAG eval pipeline end-to-end.

Steps:
  1) ingest -> data/ingested/raw.jsonl
  2) chunk  -> data/chunks/chunks.jsonl
  3) index  -> artifacts/tfidf/*
  4) gold   -> data/eval/questions.jsonl (auto-generated unless --gold_mode=skip)
  5) retrieve -> data/eval/retrievals.jsonl
  6) answer   -> data/eval/predictions.jsonl
  7) evals    -> data/reports/*.csv, *.md
  8) console summary of key metrics
"""
import os, sys, argparse, subprocess, csv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY = sys.executable

# Default paths
PPTX_DIR   = os.path.join(ROOT, "data", "pptx")
PDF_DIR    = os.path.join(ROOT, "data", "pdfs")
HTML_DIR   = os.path.join(ROOT, "data", "html")
DOCX_DIR   = os.path.join(ROOT, "data", "docx")

RAW_JSONL  = os.path.join(ROOT, "data", "ingested", "raw.jsonl")
CHUNKS     = os.path.join(ROOT, "data", "chunks", "chunks.jsonl")
INDEX_DIR  = os.path.join(ROOT, "artifacts", "tfidf")

QUESTIONS  = os.path.join(ROOT, "data", "eval", "questions.jsonl")
RETRIEVALS = os.path.join(ROOT, "data", "eval", "retrievals.jsonl")
PREDICTS   = os.path.join(ROOT, "data", "eval", "predictions.jsonl")

REPORT_DIR = os.path.join(ROOT, "data", "reports")
RETR_CSV   = os.path.join(REPORT_DIR, "retrieval_metrics.csv")
QA_CSV     = os.path.join(REPORT_DIR, "qa_metrics.csv")

def run(cmd, cwd=None):
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd or ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)

def ensure_dirs():
    for d in [
        PPTX_DIR, PDF_DIR, HTML_DIR, DOCX_DIR,
        os.path.dirname(RAW_JSONL),
        os.path.dirname(CHUNKS),
        INDEX_DIR,
        os.path.dirname(QUESTIONS),
        REPORT_DIR
    ]:
        os.makedirs(d, exist_ok=True)

def parse_csv_single_value_table(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    for i in range(1, len(rows)):
        if len(rows[i]) >= 2:
            out[rows[i][0]] = rows[i][1]
    return out

def main():
    ap = argparse.ArgumentParser(description="Run full pipeline in one shot.")
    ap.add_argument("--pptx_dir", default=PPTX_DIR)
    ap.add_argument("--pdf_dir",  default=PDF_DIR)
    ap.add_argument("--html_dir", default=HTML_DIR)
    ap.add_argument("--docx_dir", default=DOCX_DIR)

    ap.add_argument("--chunk_words", type=int, default=180)
    ap.add_argument("--overlap_words", type=int, default=30)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--topN", type=int, default=100, help="TF-IDF pool size before rerank")

    # Gold options
    ap.add_argument("--gold_mode", choices=["full","append","skip"], default="full",
                    help="full=rebuild questions.jsonl; append=add up to target; skip=use existing file")
    ap.add_argument("--gold_target", type=int, default=150)
    ap.add_argument("--gold_filter_docs", default="", help="Comma-separated doc_id substrings to restrict gold.")

    args = ap.parse_args()
    ensure_dirs()

    # 1) Ingest
    run([PY, "scripts/ingest_docs.py",
         "--pptx_dir", args.pptx_dir,
         "--pdf_dir",  args.pdf_dir,
         "--html_dir", args.html_dir,
         "--docx_dir", args.docx_dir,
         "--out_jsonl", RAW_JSONL])

    # 2) Chunk
    run([PY, "scripts/chunk_docs.py",
         "--in_jsonl", RAW_JSONL,
         "--out_jsonl", CHUNKS,
         "--chunk_words", str(args.chunk_words),
         "--overlap_words", str(args.overlap_words)])

    # 3) Index (TF-IDF)
    run([PY, "scripts/build_indices.py",
         "--chunks_jsonl", CHUNKS,
         "--index_dir", INDEX_DIR])

    # 4) Gold
    if args.gold_mode != "skip":
        filter_docs = args.gold_filter_docs.strip()
        cmd = [PY, "scripts/make_gold_plus.py",
               "--chunks", CHUNKS,
               "--out", QUESTIONS,
               "--target", str(args.gold_target)]
        if filter_docs:
            cmd += ["--filter_docs", filter_docs]
        if args.gold_mode == "append" and os.path.exists(QUESTIONS):
            cmd += ["--append", "--existing", QUESTIONS]
        run(cmd)
    else:
        if not os.path.exists(QUESTIONS):
            print("questions.jsonl not found and --gold_mode=skip. Exiting.")
            sys.exit(2)

    # 5) Retrieve (passes topN to reranker)
    run([PY, "scripts/retrieve_tfidf.py",
         "--index_dir", INDEX_DIR,
         "--chunks_jsonl", CHUNKS,
         "--questions", QUESTIONS,
         "--k", str(args.k),
         "--topN", str(args.topN),
         "--out", RETRIEVALS])

    # 6) Answer (extractive)
    run([PY, "scripts/answer_extractive.py",
         "--chunks_jsonl", CHUNKS,
         "--retrievals", RETRIEVALS,
         "--out", PREDICTS])

    # 7) Evals
    run([PY, "scripts/eval_retrieval.py",
         "--retrievals", RETRIEVALS,
         "--questions", QUESTIONS,
         "--out_dir", REPORT_DIR])

    run([PY, "scripts/eval_qa.py",
         "--predictions", PREDICTS,
         "--questions", QUESTIONS,
         "--out_dir", REPORT_DIR])

    run([PY, "scripts/report_tables.py",
         "--raw", RAW_JSONL,
         "--chunks", CHUNKS,
         "--retrievals", RETRIEVALS,
         "--retrieval_report", RETR_CSV,
         "--qa_report", QA_CSV,
         "--out_dir", REPORT_DIR])

    # 8) Print short summary
    retr = parse_csv_single_value_table(RETR_CSV)
    qa   = parse_csv_single_value_table(QA_CSV)

    print("\n=== SUMMARY ===")
    if retr:
        print(f"Queries (N): {retr.get('Queries (N)','?')}")
        print(f"MRR:         {retr.get('MRR','?')}")
        print(f"Hit@1/3/5:   {retr.get('Hit-Rate@1','?')}  / {retr.get('Hit-Rate@3','?')}  / {retr.get('Hit-Rate@5','?')}")
        print(f"nDCG@1/3/5:  {retr.get('nDCG@1','?')}  / {retr.get('nDCG@3','?')}  / {retr.get('nDCG@5','?')}")
    else:
        print("Retrieval metrics not found.")

    if qa:
        print(f"Questions (N): {qa.get('Questions (N)','?')}")
        print(f"ExactMatch:    {qa.get('ExactMatch','?')}")
        print(f"F1:            {qa.get('F1','?')}")
    else:
        print("QA metrics not found.")

    print(f"\nReports saved to: {REPORT_DIR}")
    print("Done.")

if __name__ == "__main__":
    main()
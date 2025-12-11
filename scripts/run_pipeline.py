#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click pipeline with optional BM25+RRF fusion, cross-encoder rerank, and answer normalization.
"""
import argparse, os, subprocess, sys, shlex

def sh(cmd):
    # use the same Python interpreter that launched run_pipeline.py
    if cmd.strip().startswith("python "):
        cmd = cmd.replace("python", sys.executable, 1)
    print("$", cmd)
    subprocess.run(shlex.split(cmd), check=True)

def main():
    ap = argparse.ArgumentParser()
    # core
    ap.add_argument("--pptx_dir", default="data/pptx")
    ap.add_argument("--pdf_dir",  default="data/pdfs")
    ap.add_argument("--html_dir", default="data/html")
    ap.add_argument("--docx_dir", default="data/docx")
    ap.add_argument("--outdir",   default="data")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--topN", type=int, default=150)
    ap.add_argument("--chunk_words", type=int, default=180)
    ap.add_argument("--overlap_words", type=int, default=30)
    ap.add_argument("--gold_mode", choices=["skip","append","regen"], default="skip")
    ap.add_argument("--gold_target", type=int, default=0)

    # retrieval enhancements
    ap.add_argument("--use_bm25", action="store_true")
    ap.add_argument("--use_rrf", action="store_true")
    ap.add_argument("--rrf_K", type=int, default=60)
    ap.add_argument("--category_filter", choices=["none","soft","hard"], default="none")
    ap.add_argument("--category_boost", type=float, default=0.2)
    ap.add_argument("--head_weight", type=float, default=0.0, help="boost score for chunks with heading-like sections")

    # reranker + normalization
    ap.add_argument("--rerank_ce", action="store_true", help="apply cross-encoder rerank")
    ap.add_argument("--ce_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--normalize_answers", action="store_true", help="normalize EM/F1 evaluation")

    args = ap.parse_args()

    data = args.outdir
    raw = os.path.join(data, "ingested/raw.jsonl")
    chunks = os.path.join(data, "chunks/chunks.jsonl")
    index_dir = "artifacts/tfidf"
    questions = os.path.join(data, "eval/questions.jsonl")
    retrievals = os.path.join(data, "eval/retrievals.jsonl")
    retrievals_rer = os.path.join(data, "eval/retrievals_reranked.jsonl")
    predictions = os.path.join(data, "eval/predictions.jsonl")
    reports = os.path.join(data, "reports")

    os.makedirs(os.path.join(data, "ingested"), exist_ok=True)
    os.makedirs(os.path.join(data, "chunks"), exist_ok=True)
    os.makedirs(os.path.join(data, "eval"), exist_ok=True)
    os.makedirs(reports, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    # 1) ingest
    sh(f"python scripts/ingest_docs.py --pptx_dir {args.pptx_dir} --pdf_dir {args.pdf_dir} "
       f"--html_dir {args.html_dir} --docx_dir {args.docx_dir} --out_jsonl {raw}")

    # 2) chunk
    sh(f"python scripts/chunk_docs.py --in_jsonl {raw} --out_jsonl {chunks} "
       f"--chunk_words {args.chunk_words} --overlap_words {args.overlap_words}")

    # 3) index
    sh(f"python scripts/build_indices.py --chunks_jsonl {chunks} --index_dir {index_dir}")

    # 4) gold (optional)
    if args.gold_mode != "skip":
        if args.gold_mode == "regen":
            sh(f"python scripts/make_gold_plus.py --chunks {chunks} --out {questions} --target {args.gold_target}")
        elif args.gold_mode == "append":
            sh(f"python scripts/make_gold_plus.py --chunks {chunks} --out {questions} --target {args.gold_target} "
               f"--append --existing {questions}")

    # 5) retrieve (TF–IDF + optional BM25/RRF + category filtering + heading boost)
    retrieve_cmd = (
        f"python scripts/retrieve_tfidf.py --index_dir {index_dir} --chunks_jsonl {chunks} "
        f"--questions {questions} --k {args.k} --topN {args.topN} "
        f"--category_filter {args.category_filter} --category_boost {args.category_boost} "
        f"--head_weight {args.head_weight} "
        f"--out {retrievals}"
    )
    if args.use_bm25: retrieve_cmd += " --use_bm25"
    if args.use_rrf:  retrieve_cmd += " --use_rrf --rrf_K " + str(args.rrf_K)
    sh(retrieve_cmd)

    # 6) optional rerank
    retr_for_answer = retrievals
    if args.rerank_ce:
        sh(f"python scripts/rerank_crossencoder.py --chunks_jsonl {chunks} "
           f"--retrievals_in {retrievals} --retrievals_out {retrievals_rer} --model {args.ce_model}")
        retr_for_answer = retrievals_rer

    # 7) answer extraction
    sh(f"python scripts/answer_extractive.py --chunks_jsonl {chunks} \
  --retrievals {retr_for_answer} --out {predictions} --use_hits 1")

    # 8) metrics
    sh(f"python scripts/eval_retrieval.py --retrievals {retr_for_answer} --questions {questions} --out_dir {reports}")
    qa_cmd = f"python scripts/eval_qa.py --predictions {predictions} --questions {questions} --out_dir {reports}"
    if args.normalize_answers: qa_cmd += " --normalize"
    sh(qa_cmd)

    # 9) tables
    sh(f"python scripts/report_tables.py --raw {raw} --chunks {chunks} "
       f"--retrievals {retr_for_answer} "
       f"--retrieval_report {os.path.join(reports,'retrieval_metrics.csv')} "
       f"--qa_report {os.path.join(reports,'qa_metrics.csv')} --out_dir {reports}")

    print("\n=== Done ===")
    print(f"Reports -> {reports}")

if __name__ == "__main__":
    main()
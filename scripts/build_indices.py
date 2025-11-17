#!/usr/bin/env python3
import argparse, os, ujson as json
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump
import pandas as pd

def load_chunks(path):
    ids, texts, doc_ids = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            ids.append(rec["chunk_id"])
            texts.append(rec["text"])
            doc_ids.append(rec["doc_id"])
    return ids, texts, doc_ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_jsonl", required=True)
    ap.add_argument("--index_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.index_dir, exist_ok=True)

    chunk_ids, texts, doc_ids = load_chunks(args.chunks_jsonl)
    #vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)
    vec = TfidfVectorizer(ngram_range=(1,3), min_df=1, max_df=0.95)
    X = vec.fit_transform(texts)

    dump(vec, os.path.join(args.index_dir, "vectorizer.joblib"))
    dump(X, os.path.join(args.index_dir, "tfidf_matrix.joblib"))
    pd.DataFrame({"chunk_id":chunk_ids, "doc_id":doc_ids}).to_csv(
        os.path.join(args.index_dir, "chunks_index.csv"), index=False
    )
    print(f"Indexed {len(chunk_ids)} chunks → {args.index_dir}")

if __name__ == "__main__":
    main()

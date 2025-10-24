import argparse, json
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    chunks = read_jsonl(args.chunks)
    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    bench = read_jsonl(args.bench)
    total=hits=0; mrr=0.0
    for ex in bench:
        gold = ex.get("gold_chunk_id")
        if not gold: continue  # skip unanswerable for recall
        q = ex["query"]
        scores = bm25.get_scores(q.lower().split())
        ranked_idx = np.argsort(scores)[::-1].tolist()
        ranked_ids = [ids[i] for i in ranked_idx]
        total += 1
        if gold in ranked_ids[:args.k]: hits += 1
        rr = 0.0
        for r,cid in enumerate(ranked_ids,1):
            if cid==gold: rr=1.0/r; break
        mrr += rr

    print(f"Queries (answerable): {total}")
    print(f"Recall@{args.k}: {hits/max(total,1):.3f}")
    print(f"MRR: {mrr/max(total,1):.3f}")

if __name__ == "__main__":
    main()
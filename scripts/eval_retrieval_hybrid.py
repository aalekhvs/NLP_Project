import argparse, json, pickle, numpy as np
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def rrf(rank_lists, k=60):
    from collections import defaultdict
    scores = defaultdict(float)
    for rl in rank_lists:
        for r, id_ in enumerate(rl, start=1):
            scores[id_] += 1.0 / (k + r)
    return [x for x,_ in sorted(scores.items(), key=lambda x: -x[1])]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)     # data/processed/index
    ap.add_argument("--chunks", required=True)        # data/processed/chunks.jsonl
    ap.add_argument("--bench", required=True)         # data/benchmarks/bench_auto.jsonl
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    # load FAISS
    index = faiss.read_index(str(Path(args.index_dir) / "faiss.index"))
    with open(Path(args.index_dir) / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    ids_faiss = meta["ids"]; texts = meta["texts"]

    # load BM25 (on the same texts order)
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    # dense model for queries
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def dense_search(q, topn=50):
        qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        D, I = index.search(qv.astype("float32"), topn)
        return [ids_faiss[i] for i in I[0]]

    def bm25_search(q, topn=50):
        scores = bm25.get_scores(q.lower().split())
        ranked = np.argsort(scores)[::-1][:topn]
        return [ids_faiss[i] for i in ranked.tolist()]

    bench = read_jsonl(args.bench)
    total=hits=0; mrr=0.0
    for ex in bench:
        gold = ex.get("gold_chunk_id")
        if not gold: continue
        q = ex["query"]
        fused = rrf([bm25_search(q,50), dense_search(q,50)])
        topk = fused[:args.k]
        total += 1
        if gold in topk: hits += 1
        rr = 0.0
        for r, cid in enumerate(fused,1):
            if cid == gold:
                rr = 1.0/r; break
        mrr += rr

    print(f"Queries (answerable): {total}")
    print(f"Hybrid Recall@{args.k}: {hits/max(total,1):.3f}")
    print(f"Hybrid MRR: {mrr/max(total,1):.3f}")

if __name__ == "__main__":
    main()
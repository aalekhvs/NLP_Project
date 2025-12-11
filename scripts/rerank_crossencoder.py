#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Light cross-encoder reranker over top-N retrieved candidates.

If the model isn't available (e.g., offline), it quietly falls back to a
token-overlap reranker so the pipeline remains robust.

Usage:
  python scripts/rerank_crossencoder.py \
    --chunks_jsonl data/chunks/chunks.jsonl \
    --retrievals_in data/eval/retrievals.jsonl \
    --retrievals_out data/eval/retrievals_reranked.jsonl \
    --model cross-encoder/ms-marco-MiniLM-L-6-v2
"""
import argparse, json, os, re
from tqdm import tqdm

_WS = re.compile(r"\w+")
def tokset(s): return set(_WS.findall(s.lower()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_jsonl", required=True)
    ap.add_argument("--retrievals_in", required=True)
    ap.add_argument("--retrievals_out", required=True)
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    args = ap.parse_args()

    # load chunks
    chunk_text = {}
    with open(args.chunks_jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                chunk_text[r["chunk_id"]] = r.get("text","")

    # try cross-encoder
    use_ce = True
    try:
        from sentence_transformers import CrossEncoder  # requires torch
        ce = CrossEncoder(args.model)
    except Exception as e:
        print(f"[WARN] Cross-encoder unavailable ({e}); falling back to token-overlap rerank.")
        use_ce = False
        ce = None

    os.makedirs(os.path.dirname(args.retrievals_out) or ".", exist_ok=True)

    with open(args.retrievals_in, "r", encoding="utf-8") as inp, \
         open(args.retrievals_out, "w", encoding="utf-8") as outp:
        for line in tqdm(inp, desc="rerank"):
            if not line.strip():
                continue
            rec = json.loads(line)
            q = rec["question"]
            hits = rec.get("hits", [])
            if not hits:
                outp.write(line)  # unchanged
                continue

            if use_ce:
                pairs = [(q, chunk_text[h["chunk_id"]]) for h in hits]
                scores = ce.predict(pairs, convert_to_numpy=True).tolist()
            else:
                qset = tokset(q)
                scores = []
                for h in hits:
                    sset = tokset(chunk_text[h["chunk_id"]])
                    # overlap ratio as a crude signal
                    denom = max(1, len(qset))
                    scores.append(len(qset & sset) / denom)

            # attach and sort
            for h, s in zip(hits, scores):
                h["why"]["rerank_ce"] = float(s)
                h["score"] = float(s)  # CE score becomes the rank key

            hits.sort(key=lambda x: x["score"], reverse=True)
            rec["hits"] = hits
            outp.write(json.dumps(rec) + "\n")

    print(f"Wrote reranked retrievals -> {args.retrievals_out}")

if __name__ == "__main__":
    main()
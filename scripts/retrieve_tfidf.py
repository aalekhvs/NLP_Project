#!/usr/bin/env python3
import argparse, os, re
import ujson as json
import numpy as np
import pandas as pd
from joblib import load
from tqdm import tqdm

# stopword-aware overlap to reduce noise
STOP = set("""
a an the and or of for in on to with by from at as is are was were be been being
this that these those it its they them he she we you your our i me my mine
""".split())

def toks(s: str):
    return [w for w in re.findall(r"\w+", s.lower()) if w not in STOP]

def overlap_score(q: str, txt: str) -> float:
    qset = set(toks(q)); tset = set(toks(txt))
    if not qset or not tset:
        return 0.0
    return len(qset & tset) / max(1, len(qset))

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--chunks_jsonl", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--topN", type=int, default=100, help="TF-IDF pool size before rerank")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # load TF-IDF artifacts
    vec = load(os.path.join(args.index_dir, "vectorizer.joblib"))
    X = load(os.path.join(args.index_dir, "tfidf_matrix.joblib"))
    meta = pd.read_csv(os.path.join(args.index_dir, "chunks_index.csv"))
    id2row = {i: cid for i, cid in enumerate(meta["chunk_id"].tolist())}

    # build chunk text + heading lookup for reranking
    chunk_text, chunk_head = {}, {}
    with open(args.chunks_jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                cid = r["chunk_id"]
                chunk_text[cid] = r.get("text", "")
                chunk_head[cid] = r.get("where", "")  # headings: h1/h2/h3, "slide 3", etc.

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for q in tqdm(iter_jsonl(args.questions), desc="retrieve"):
            qtext = q["question"]
            qv = vec.transform([qtext])
            sims = (qv @ X.T).toarray()[0]

            k = max(1, min(args.k, len(sims)))
            pool = max(k, min(args.topN, len(sims)))  # initial TF-IDF pool

            if pool == 0:
                hits = []
            else:
                idx = np.argpartition(-sims, range(pool))[:pool]
                cand = []
                for i in idx:
                    i = int(i)
                    cid = id2row[i]
                    tf = float(sims[i])
                    # lexical overlap on body + heading (equal weight)
                    lex_body = overlap_score(qtext, chunk_text.get(cid, ""))
                    lex_head = overlap_score(qtext, chunk_head.get(cid, ""))
                    lex = 0.5 * lex_body + 0.5 * lex_head
                    score_blend = 0.6 * tf + 0.4 * lex
                    cand.append({
                        "chunk_id": cid,
                        "score": score_blend,      # use blended score as 'score'
                        "score_tfidf": tf,
                        "score_lex": lex
                    })

                cand.sort(key=lambda x: x["score"], reverse=True)
                hits = cand[:k]

            qid = q.get("qid") or str(n + 1)
            out.write(json.dumps({"qid": qid, "question": qtext, "hits": hits}) + "\n")
            n += 1

    print(f"Wrote retrievals for {n} queries -> {args.out}")

if __name__ == "__main__":
    main()
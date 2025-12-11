#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrieve top-k chunks per question with:
- TF–IDF (prebuilt index)
- Optional BM25 (on-the-fly) + Reciprocal Rank Fusion (RRF)
- Optional category-aware hard/soft filtering based on question intent and file metadata
- Optional heading boost to favor section headers (e.g., syllabus policy headings)

Outputs JSONL lines:
  {"qid": "...", "question": "...",
   "hits": [{"chunk_id": "...", "score": float,
             "why": {"tfidf": s, "bm25": s?, "rrf": s?, "heading_boost": w, "cat_boost": w}} ...]}
"""
import argparse, os, re, json
import numpy as np, pandas as pd
from joblib import load
from tqdm import tqdm

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

_WS = re.compile(r"\w+")
def tok(s: str):
    return _WS.findall(s.lower())

def guess_category(q: str):
    ql = q.lower()
    if any(w in ql for w in ["late","deadline","due","penalty","policy","attendance","office hour","grade","grading"]):
        return "policy"
    if any(w in ql for w in ["hw","homework","assignment","submit","submission","report","rubric"]):
        return "hw"
    if any(w in ql for w in ["lecture","topic","slide","content","definition","explain"]):
        return "topics"
    return None

def meta_from_chunk_id(chunk_id: str, row: dict):
    doc_id = (row.get("doc_id","") or "")
    src = ((row.get("where","") or "") + " | " + (row.get("type","") or ""))
    return doc_id.lower(), src.lower()

def allow_by_category(cat, doc_id, src):
    if not cat: return True, 1.0
    fname = doc_id
    if cat == "policy":
        ok = ("syllabus" in fname) or ("policy" in fname) or ("docx::" in fname) or ("docx::" in src)
        return ok, (1.25 if ok else 1.0)
    if cat == "hw":
        ok = any(k in fname for k in ["hw","homework","assignment"])
        return ok, (1.25 if ok else 1.0)
    if cat == "topics":
        ok = any(k in fname for k in ["pptx::","pdf::","lecture","topic"])
        return ok, (1.10 if ok else 1.0)
    return True, 1.0

def reciprocal_rank_fusion(rank_lists, K=60):
    fused = {}
    for rl in rank_lists:
        for r, idx in enumerate(rl):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (K + r + 1)
    return fused

def safe_top_indices(scores, topN):
    if len(scores) == 0:
        return np.array([], dtype=int)
    topN = min(topN, len(scores))
    idx = np.argpartition(-scores, range(topN))[:topN]
    return idx[np.argsort(-scores[idx])]

def looks_like_heading(where_field: str) -> bool:
    if not where_field: return False
    w = where_field.strip().lower()
    if w == "body": return False
    if re.match(r"^(page|slide)\s+\d+$", w): return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--chunks_jsonl", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--topN", type=int, default=100, help="candidate pool per method")
    ap.add_argument("--use_bm25", action="store_true")
    ap.add_argument("--use_rrf", action="store_true")
    ap.add_argument("--rrf_K", type=int, default=60)
    ap.add_argument("--category_filter", choices=["none","soft","hard"], default="none")
    ap.add_argument("--category_boost", type=float, default=0.2, help="extra multiplier for allowed docs in soft mode")
    ap.add_argument("--head_weight", type=float, default=0.0,
                    help="Multiplicative boost (score *= 1+head_weight) when a chunk's `where` looks like a section heading")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # TF–IDF index
    vec = load(os.path.join(args.index_dir, "vectorizer.joblib"))
    X = load(os.path.join(args.index_dir, "tfidf_matrix.joblib"))
    meta = pd.read_csv(os.path.join(args.index_dir, "chunks_index.csv"))
    id2row = {i: cid for i, cid in enumerate(meta["chunk_id"].tolist())}

    # chunks map
    chunks = {}
    with open(args.chunks_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            chunks[r["chunk_id"]] = r

    # optional BM25
    bm25 = None
    if args.use_bm25:
        try:
            from rank_bm25 import BM25Okapi
            corpus_order = [id2row[i] for i in range(len(id2row))]
            corpus_tokens = [tok(chunks[cid].get("text","")) for cid in corpus_order]
            bm25 = BM25Okapi(corpus_tokens)
        except Exception as e:
            print(f"[WARN] BM25 unavailable ({e}); continuing TF–IDF only.")
            args.use_bm25 = False

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for q in tqdm(iter_jsonl(args.questions), desc="retrieve"):
            qid = q.get("qid") or f"q{n+1}"
            qtext = q["question"]
            cat = guess_category(qtext)

            # TF–IDF
            qv = vec.transform([qtext])
            tfidf_scores = (qv @ X.T).toarray()[0]
            tfidf_top = safe_top_indices(tfidf_scores, args.topN)

            # BM25
            bm25_scores = None
            bm25_top = None
            if bm25 is not None:
                qtok = tok(qtext)
                bm25_scores = np.array(bm25.get_scores(qtok), dtype=float)
                bm25_top = safe_top_indices(bm25_scores, args.topN)

            # candidate pool
            cand_ids = set(tfidf_top.tolist())
            if bm25_top is not None:
                cand_ids |= set(bm25_top.tolist())
            cand_ids = list(cand_ids)

            # category filtering
            soft_weights = {}
            if args.category_filter in ("soft","hard") and cat:
                filtered = []
                for i in cand_ids:
                    cid = id2row[i]; row = chunks[cid]
                    doc_id, src = meta_from_chunk_id(cid, row)
                    allow, w = allow_by_category(cat, doc_id, src)
                    if allow:
                        filtered.append(i)
                        soft_weights[i] = w
                if args.category_filter == "hard" and filtered:
                    cand_ids = filtered

            # rankings
            tfidf_rank = sorted(cand_ids, key=lambda i: tfidf_scores[i], reverse=True)
            rankings = [tfidf_rank]
            why_mode = "tfidf"
            if bm25_top is not None:
                bm25_rank = sorted(cand_ids, key=lambda i: bm25_scores[i], reverse=True)
                rankings.append(bm25_rank)

            # fuse
            if args.use_bm25 and args.use_rrf and len(rankings) > 1:
                fused_scores = reciprocal_rank_fusion(rankings, K=args.rrf_K)
                final_sorted = sorted(cand_ids, key=lambda i: fused_scores.get(i,0.0), reverse=True)
                why_mode = "rrf"
            else:
                fused_scores = {i: float(tfidf_scores[i]) for i in cand_ids}
                final_sorted = sorted(cand_ids, key=lambda i: fused_scores[i], reverse=True)

            # soft category boost
            cat_boost_applied = set()
            if args.category_filter == "soft" and soft_weights:
                for i in final_sorted:
                    if i in soft_weights:
                        fused_scores[i] *= (1.0 + args.category_boost)
                        cat_boost_applied.add(i)
                final_sorted = sorted(final_sorted, key=lambda i: fused_scores[i], reverse=True)

            # heading boost (binary heuristic on `where`)
            head_boost_applied = set()
            if args.head_weight and args.head_weight > 0:
                for i in final_sorted:
                    cid = id2row[i]
                    where_field = (chunks[cid].get("where") or "")
                    if looks_like_heading(where_field):
                        fused_scores[i] *= (1.0 + args.head_weight)
                        head_boost_applied.add(i)
                final_sorted = sorted(final_sorted, key=lambda i: fused_scores[i], reverse=True)

            # top-k hits
            k = min(args.k, len(final_sorted))
            hits = []
            for i in final_sorted[:k]:
                cid = id2row[i]
                why = {"tfidf": float(tfidf_scores[i])}
                if bm25_top is not None:  why["bm25"] = float(bm25_scores[i])
                if why_mode == "rrf":     why["rrf"]  = float(fused_scores.get(i,0.0))
                why["heading_boost"] = float(args.head_weight) if i in head_boost_applied else 0.0
                why["cat_boost"]     = float(args.category_boost) if i in cat_boost_applied else 0.0
                hits.append({"chunk_id": cid, "score": float(fused_scores[i]), "why": why})

            json.dump({"qid": qid, "question": qtext, "hits": hits}, out)
            out.write("\n")
            n += 1

    print(f"Wrote retrievals for {n} queries -> {args.out}")

if __name__ == "__main__":
    main()
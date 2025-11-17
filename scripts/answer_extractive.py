#!/usr/bin/env python3
import argparse, ujson as json, re

def norm(s): return re.sub(r"\s+"," ",(s or "")).strip()

def load_chunks(path):
    by_id = {}
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            by_id[rec["chunk_id"]] = rec["text"]
    return by_id

def best_sentence(question, context):
    sents = re.split(r"(?<=[.!?])\s+", context)
    q = question.lower()
    def score(sent):
        import re
        a=set(re.findall(r"\w+", sent.lower()))
        b=set(re.findall(r"\w+", q))
        return len(a & b)
    if not sents: return context
    sents = [(score(s), s) for s in sents]
    sents.sort(key=lambda x:x[0], reverse=True)
    return sents[0][1] if sents else context

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_jsonl", required=True)
    ap.add_argument("--retrievals", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    chunks = load_chunks(args.chunks_jsonl)
    n=0
    with open(args.out,"w",encoding="utf-8") as out, open(args.retrievals,"r",encoding="utf-8") as r:
        for line in r:
            if not line.strip(): continue
            rec = json.loads(line)
            qid, q = rec["qid"], rec["question"]
            hits = rec.get("hits", [])
            ctx = " ".join(chunks[h["chunk_id"]] for h in hits if h["chunk_id"] in chunks) or ""
            pred = best_sentence(q, ctx) if ctx else ""
            json.dump({"qid": qid, "prediction": pred}, out); out.write("\n"); n+=1
    print(f"Wrote {n} predictions -> {args.out}")

if __name__ == "__main__":
    main()

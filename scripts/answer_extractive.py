#!/usr/bin/env python3
import argparse, ujson as json, re

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def strip_private_use(s: str) -> str:
    # drop private-use glyphs like \uf06e bullets
    return re.sub(r"[\uf000-\ufaff]", " ", s or "")

def load_chunks(path):
    by_id = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            by_id[rec["chunk_id"]] = rec.get("text","")
    return by_id

# very small span cutter to match gold phrasing for EM
SPAN_PATTERNS = [
    r"\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?",                      # times
    r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?",                          # 10/15 or 10/15/2025
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2}(?:,\s*\d{4})?",
    r"\b\d+(?:\.\d+)?\s*%",                                     # percentages
    r"\b\d+(?:\.\d+)?\s*(?:points?|pts?)\b",                    # points
    r"\b\d+\s*-\s*\d+\b",                                       # ranges
    r"\bno\s+late\s+\w+\b",                                     # “no late …”
    r"\blate\s+[^.]{0,40}penalt\w*\b",                          # late … penalty
]

def shortest_span_like(s: str) -> str:
    cands = []
    for pat in SPAN_PATTERNS:
        for m in re.finditer(pat, s):
            cands.append(m.group(0))
    if not cands:
        return ""
    cands.sort(key=lambda x: (len(x.split()), len(x)))
    return cands[0]

def best_sentence(question: str, context: str) -> str:
    # crude sentence split
    sents = re.split(r"(?<=[.!?])\s+", context)
    q = question.lower()
    def score(sent):
        a = set(re.findall(r"\w+", (sent or "").lower()))
        b = set(re.findall(r"\w+", q))
        return len(a & b)
    if not sents: 
        return context
    scored = sorted(((score(s), s) for s in sents), key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else context

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_jsonl", required=True)
    ap.add_argument("--retrievals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--use_hits", type=int, default=1, help="number of top hits to concatenate (1 is best for EM on this gold)")
    args = ap.parse_args()

    chunks = load_chunks(args.chunks_jsonl)

    n=0
    with open(args.out, "w", encoding="utf-8") as fout, open(args.retrievals, "r", encoding="utf-8") as rin:
        for line in rin:
            if not line.strip(): continue
            rec = json.loads(line)
            qid, q = rec.get("qid"), rec.get("question","")
            hits = rec.get("hits", [])
            # concatenate top-N contexts (default 1 for precision)
            ctx = " ".join(chunks[h["chunk_id"]] for h in hits[:max(1,args.use_hits)] if h["chunk_id"] in chunks)
            ctx = strip_private_use(ctx)
            pred = best_sentence(q, ctx) if ctx else ""
            pred = strip_private_use(pred)
            pred = norm_ws(pred)

            tight = shortest_span_like(pred)
            if tight and len(tight) >= 3:
                pred = norm_ws(tight)

            json.dump({"qid": qid, "prediction": pred}, fout); fout.write("\n")
            n += 1
    print(f"Wrote {n} predictions -> {args.out}")

if __name__ == "__main__":
    main()
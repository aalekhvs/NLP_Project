import argparse, json, re, random
from pathlib import Path

KEY_PAT = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /&\-\(\)]+)\s*:\s*(.+)$")
TEMPLATES = [
    "What is {key}?",
    "Give me the {key}.",
    "Please provide details about {key}.",
    "Summarize the {key}.",
    "Can you tell me about the {key}?"
]
UNANSWERABLE_POOL = [
    "What is the parking permit policy?",
    "How do I request extra credit after the final?",
    "What’s the policy for Assignment 99?",
    "Where is the lab access key pickup?",
    "When is the optional retake exam?",
    "How do I waive the course fee?",
    "What is the bonus project deadline?"
]

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def extract_keyvals(text):
    out = []
    for line in text.splitlines():
        m = KEY_PAT.match(line.strip())
        if m:
            key, val = m.group(1).strip(), m.group(2).strip()
            if 2 <= len(key) <= 80 and len(val) >= 2:
                out.append((key, f"{key}: {val}"))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_per_chunk", type=int, default=5)
    ap.add_argument("--unans", type=int, default=20)
    args = ap.parse_args()

    qid = 1
    bench = []
    keys_seen = set()

    for r in iter_jsonl(args.chunks):
        kvs = extract_keyvals(r["text"])
        random.shuffle(kvs)
        kvs = kvs[:args.max_per_chunk]
        for key, span in kvs:
            keys_seen.add(key.lower())
            bench.append({
                "qid": f"q{qid}_extract",
                "query": f"What is {key}?",
                "gold_chunk_id": r["chunk_id"],
                "gold_span": span,
                "category": "extractive",
                "doc_version": r.get("version","v1")
            }); qid += 1
            for t in random.sample(TEMPLATES, k=min(2, len(TEMPLATES))):
                bench.append({
                    "qid": f"q{qid}_paraphrase",
                    "query": t.format(key=key),
                    "gold_chunk_id": r["chunk_id"],
                    "gold_span": span,
                    "category": "paraphrase",
                    "doc_version": r.get("version","v1")
                }); qid += 1

    # add unanswerables
    unans_added = 0
    for q in UNANSWERABLE_POOL:
        if unans_added >= args.unans: break
        bench.append({
            "qid": f"q{qid}_unanswerable",
            "query": q,
            "gold_chunk_id": None,
            "gold_span": "",
            "category": "unanswerable",
            "doc_version": "v1"
        }); qid += 1; unans_added += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in bench:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(bench)} questions → {args.out}")

if __name__ == "__main__":
    main()
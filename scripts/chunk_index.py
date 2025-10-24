import argparse, json, re
from pathlib import Path

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def chunk_text(text, max_words=240, overlap=40):
    words = re.findall(r"\S+", text)
    chunks, i = [], 0
    while i < len(words):
        w = words[i:i+max_words]; chunks.append(" ".join(w))
        if i + max_words >= len(words): break
        i += max_words - overlap
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)   # data/interim/pptx_extracted.jsonl
    ap.add_argument("--out_chunks", required=True) # data/processed/chunks.jsonl
    args = ap.parse_args()

    Path(args.out_chunks).parent.mkdir(parents=True, exist_ok=True)
    recs = []
    counter = 1
    for r in iter_jsonl(args.in_jsonl):
        for ch in chunk_text(r["text"]):
            recs.append({
                "chunk_id": f"{r['doc_id']}_s{r['slide_number']}_c{counter}",
                "doc_id": r["doc_id"],
                "section_title": r.get("slide_title") or "",
                "text": ch,
                "version": r.get("version","v1"),
                "meta": {"slide_number": r["slide_number"]}
            })
            counter += 1

    with open(args.out_chunks, "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(recs)} chunks → {args.out_chunks}")

if __name__ == "__main__":
    main()
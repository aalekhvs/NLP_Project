#!/usr/bin/env python3
import argparse, ujson as json, os, re
from tqdm import tqdm

def norm(s): return re.sub(r"\s+", " ", (s or "")).strip()

def chunk_words(text, size=180, overlap=30):
    toks = text.split()
    if not toks: return []
    out = []
    step = max(1, size - overlap)
    for i in range(0, len(toks), step):
        win = toks[i:i+size]
        if not win: break
        out.append(" ".join(win))
        if i + size >= len(toks): break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--chunk_words", type=int, default=180)
    ap.add_argument("--overlap_words", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    n_in = n_out = 0
    with open(args.in_jsonl, "r", encoding="utf-8") as f,          open(args.out_jsonl, "w", encoding="utf-8") as out:
        for line in tqdm(f, desc="chunk"):
            if not line.strip(): continue
            rec = json.loads(line)
            n_in += 1
            base_id = f'{rec["type"]}::{rec["source"]}'
            where = rec.get("where","")
            section_id = f"{base_id}#{where}".strip("#")
            parts = chunk_words(norm(rec.get("text","")), args.chunk_words, args.overlap_words)
            for j, txt in enumerate(parts, start=1):
                chunk_id = f"{section_id}::chunk{j}"
                out.write(json.dumps({
                    "chunk_id": chunk_id,
                    "doc_id": base_id,
                    "where": where,
                    "type": rec.get("type",""),
                    "text": txt
                }) + "\n")
                n_out += 1
    print(f"Read {n_in} sections, wrote {n_out} chunks -> {args.out_jsonl}")

if __name__ == "__main__":
    main()

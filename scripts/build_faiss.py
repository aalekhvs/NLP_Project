import argparse, json, pickle, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    ids, texts = [], []
    for r in iter_jsonl(args.chunks):
        ids.append(r["chunk_id"]); texts.append(r["text"])

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine if normalized
    index.add(embs.astype("float32"))
    faiss.write_index(index, str(outdir / "faiss.index"))

    np.save(outdir / "embeddings.npy", embs)
    with open(outdir / "meta.pkl", "wb") as f:
        pickle.dump({"ids": ids, "texts": texts}, f)

    print(f"Saved FAISS index → {outdir}")

if __name__ == "__main__":
    main()
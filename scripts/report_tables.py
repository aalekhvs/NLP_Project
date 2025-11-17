#!/usr/bin/env python3
import argparse, os, ujson as json, pandas as pd
import numpy as np

def count_types(raw_jsonl):
    c = {}
    with open(raw_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            t = r.get("type","unknown")
            c[t] = c.get(t,0)+1
    return c

def chunk_stats(chunks_jsonl):
    lens = []
    with open(chunks_jsonl,"r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            lens.append(len(r.get("text","").split()))
    if not lens: return {"chunks":0,"avg_words":0,"p50":0,"p90":0}
    a = np.array(lens)
    return {"chunks":len(lens),"avg_words":float(a.mean()),"p50":float(np.median(a)),"p90":float(np.percentile(a,90))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--retrievals", required=True)
    ap.add_argument("--retrieval_report", required=True)
    ap.add_argument("--qa_report", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    types = count_types(args.raw)
    cstats = chunk_stats(args.chunks)
    df_corpus = pd.DataFrame([
        ["pdf", types.get("pdf",0)],
        ["pptx", types.get("pptx",0)],
        ["html", types.get("html",0)],
        ["sections_total", sum(types.values())],
        ["chunks_total", cstats["chunks"]],
        ["chunk_avg_words", round(cstats["avg_words"],1)],
        ["chunk_p50_words", round(cstats["p50"],1)],
        ["chunk_p90_words", round(cstats["p90"],1)],
    ], columns=["Item","Value"])

    df_ret = pd.read_csv(args.retrieval_report)
    df_qa  = pd.read_csv(args.qa_report)

    df_corpus.to_csv(os.path.join(args.out_dir, "corpus_stats.csv"), index=False)
    with open(os.path.join(args.out_dir, "corpus_stats.md"),"w",encoding="utf-8") as w:
        w.write("| Item | Value |\n|---|---|\n")
        for _,row in df_corpus.iterrows():
            w.write(f"| {row['Item']} | {row['Value']} |\n")

    with open(os.path.join(args.out_dir, "final_report.md"),"w",encoding="utf-8") as w:
        w.write("# RAG Evaluation Tables (no graphs)\n\n")
        w.write("## Corpus Stats\n\n")
        w.write(open(os.path.join(args.out_dir,"corpus_stats.md"),"r",encoding="utf-8").read())
        w.write("\n\n## Retrieval Metrics\n\n")
        w.write("| Metric | Value |\n|---|---|\n")
        for _,r in df_ret.iterrows():
            w.write(f"| {r['Metric']} | {r['Value']} |\n")
        w.write("\n\n## QA Metrics\n\n")
        w.write("| Metric | Value |\n|---|---|\n")
        for _,r in df_qa.iterrows():
            w.write(f"| {r['Metric']} | {r['Value']} |\n")
    print(f"Wrote tables -> {args.out_dir}")

if __name__ == "__main__":
    main()

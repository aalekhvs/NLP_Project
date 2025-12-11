#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute EM/F1 over predictions vs gold. Optional normalization.
"""
import argparse, json, os, re, math, pandas as pd
from collections import Counter

def iter_jsonl(p):
    with open(p,"r",encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                yield json.loads(ln)

def normalize(s):  # default basic
    s = s or ""
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_eval_text(s, use_norm):
    s = normalize(s)
    if use_norm:
        try:
            from answer_normalize import normalize_for_eval
            s = normalize_for_eval(s)
        except Exception:
            pass
    return s

def f1_score(pred, gold):
    pred_toks = pred.split()
    gold_toks = gold.split()
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if len(pred_toks)==0 or len(gold_toks)==0:
        return float(pred_toks==gold_toks)
    if num_same==0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--normalize", action="store_true", help="normalize pred/gold for EM/F1")
    args = ap.parse_args()

    gold = {q["qid"]: q for q in iter_jsonl(args.questions)}
    em = f1 = n = 0.0

    for p in iter_jsonl(args.predictions):
        qid = p["qid"]
        if qid not in gold: continue
        pred = norm_eval_text(p.get("answer",""), args.normalize)
        g = norm_eval_text(gold[qid].get("answer",""), args.normalize)
        n += 1
        em += 1.0 if pred == g else 0.0
        f1 += f1_score(pred, g)

    em = (em / n) if n else 0.0
    f1 = (f1 / n) if n else 0.0

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.DataFrame([["Questions (N)", int(n)],
                       ["ExactMatch", round(em, 4)],
                       ["F1", round(f1, 4)]],
                      columns=["Metric","Value"])
    df.to_csv(os.path.join(args.out_dir, "qa_metrics.csv"), index=False)
    with open(os.path.join(args.out_dir, "qa_metrics.md"), "w", encoding="utf-8") as f:
        f.write("# QA Metrics\n\n")
        for m,v in df.values:
            f.write(f"- **{m}**: {v}\n")
    print(f"QA metrics -> {os.path.join(args.out_dir, 'qa_metrics.csv')}")

if __name__ == "__main__":
    main()
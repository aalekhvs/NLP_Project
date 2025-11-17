#!/usr/bin/env python3
import argparse, ujson as json, re, string, csv, os

def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+"," ", s).strip()
    return s

def f1_score(pred, truth):
    p = normalize(pred).split()
    t = normalize(truth).split()
    if not p and not t: return 1.0
    if not p or not t: return 0.0
    common = 0
    d = {}
    for tok in t: d[tok] = d.get(tok,0)+1
    for tok in p:
        if d.get(tok,0)>0:
            common += 1
            d[tok] -= 1
    if common == 0: return 0.0
    prec = common / len(p)
    rec  = common / len(t)
    return 2 * prec * rec / (prec + rec)

def exact_match(pred, truth):
    return 1.0 if normalize(pred) == normalize(truth) else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    gold = {}
    with open(args.questions,"r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            q = json.loads(line)
            if q.get("answer"):
                gold[q["qid"]] = q["answer"]
            elif q.get("answers"):
                gold[q["qid"]] = q["answers"][0]

    N=0; em=0.0; f1=0.0
    with open(args.predictions,"r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            p = json.loads(line)
            qid = p["qid"]; pred = p.get("prediction","")
            truth = gold.get(qid, "")
            if truth == "": continue
            N+=1
            em += exact_match(pred, truth)
            f1 += f1_score(pred, truth)

    em = (em / N) if N else 0.0
    f1 = (f1 / N) if N else 0.0

    out_csv = os.path.join(args.out_dir,"qa_metrics.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as w:
        writer = csv.writer(w); writer.writerow(["Metric","Value"])
        writer.writerow(["Questions (N)", N])
        writer.writerow(["ExactMatch", round(em,4)])
        writer.writerow(["F1", round(f1,4)])

    with open(os.path.join(args.out_dir,"qa_metrics.md"),"w",encoding="utf-8") as w:
        w.write("| Metric | Value |\n|---|---|\n")
        w.write(f"| Questions (N) | {N} |\n")
        w.write(f"| ExactMatch | {round(em,4)} |\n")
        w.write(f"| F1 | {round(f1,4)} |\n")
    print(f"QA metrics -> {out_csv}")

if __name__ == "__main__":
    main()

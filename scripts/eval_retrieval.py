#!/usr/bin/env python3
import argparse, ujson as json, math, csv, os

def load_qrels(questions_path):
    qrels = {}
    with open(questions_path,"r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            q = json.loads(line)
            qid = q.get("qid")
            rel = set()
            if q.get("gold_chunk_ids"): rel |= set(q["gold_chunk_ids"])
            if q.get("gold_doc_ids"):   rel |= set(q["gold_doc_ids"])
            qrels[qid] = rel
    return qrels

def is_relevant(hit_id, relset):
    if hit_id in relset: return True
    prefixes = [rid for rid in relset if "::" in rid and "#" not in rid]
    return any(hit_id.startswith(rid) for rid in prefixes)

def dcg(rel_flags):
    return sum((1.0/math.log2(i+2)) if r else 0.0 for i,r in enumerate(rel_flags))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrievals", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    qrels = load_qrels(args.questions)
    K = [1,3,5,10]

    totals = {"N":0, "hits@k":{k:0 for k in K}, "rel_found@k":{k:0 for k in K}, "MRR":0.0, "nDCG@k":{k:0.0 for k in K}}
    with open(args.retrievals,"r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            qid = rec["qid"]; hits = rec.get("hits", [])
            relset = qrels.get(qid, set())

            flags = []
            for h in hits[:max(K)]:
                flags.append(1 if is_relevant(h["chunk_id"], relset) else 0)

            rr = 0.0
            for i, r in enumerate(flags, start=1):
                if r==1: rr = 1.0/i; break

            totals["N"] += 1
            totals["MRR"] += rr
            for k in K:
                topk = flags[:k]
                totals["hits@k"][k] += sum(topk)
                totals["rel_found@k"][k] += 1 if any(topk) else 0
                idcg = dcg(sorted(flags[:k], reverse=True)) or 1.0
                totals["nDCG@k"][k] += dcg(flags[:k]) / idcg

    out_csv = os.path.join(args.out_dir, "retrieval_metrics.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as w:
        writer = csv.writer(w)
        writer.writerow(["Metric","Value"])
        writer.writerow(["Queries (N)", totals["N"]])
        writer.writerow(["MRR", round(totals["MRR"]/max(1,totals["N"]), 4)])
        for k in K:
            hit_rate = totals["rel_found@k"][k] / max(1, totals["N"])
            avg_rel = totals["hits@k"][k] / max(1, totals["N"])
            ndcg = totals["nDCG@k"][k] / max(1, totals["N"])
            writer.writerow([f"Hit-Rate@{k}", round(hit_rate,4)])
            writer.writerow([f"AvgRelevant@{k}", round(avg_rel,4)])
            writer.writerow([f"nDCG@{k}", round(ndcg,4)])

    with open(os.path.join(args.out_dir,"retrieval_metrics.md"),"w",encoding="utf-8") as w:
        w.write(f"| Metric | Value |\n|---|---|\n")
        w.write(f"| Queries (N) | {totals['N']} |\n")
        w.write(f"| MRR | {round(totals['MRR']/max(1,totals['N']),4)} |\n")
        for k in K:
            hit_rate = totals["rel_found@k"][k] / max(1, totals["N"])
            avg_rel = totals["hits@k"][k] / max(1, totals["N"])
            ndcg = totals["nDCG@k"][k] / max(1, totals["N"])
            w.write(f"| Hit-Rate@{k} | {round(hit_rate,4)} |\n")
            w.write(f"| AvgRelevant@{k} | {round(avg_rel,4)} |\n")
            w.write(f"| nDCG@{k} | {round(ndcg,4)} |\n")
    print(f"Retrieval metrics -> {out_csv}")

if __name__ == "__main__":
    main()

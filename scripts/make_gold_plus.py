#!/usr/bin/env python3
import argparse, ujson as json, re, os, random
from collections import defaultdict

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

ADMIN_BUCKETS = [
    (r"\b(late|deadline|extension|penalt(y|ies))\b",            "What is the late submission policy?"),
    (r"\b(grade|grading|weight|percentage|points|rubric)\b",    "What is the grading breakdown?"),
    (r"\b(quiz|quizzes|exam|midterm|final|test|proctor)\b",     "What is the quiz/exam policy?"),
    (r"\b(attendance|participation)\b",                         "What is the attendance policy?"),
    (r"\b(office hours|office-hour|office\s*hours|location)\b", "What are the office hours?"),
    (r"\b(collaboration|plagiarism|integrity|cheat|honor)\b",   "What is the collaboration/integrity policy?"),
    (r"\b(submission|format|naming|filename|pdf|docx|canvas)\b","What is the submission format?"),
    (r"\b(make-?up|regrade|re-submit|resubmission)\b",          "What is the makeup/resubmission policy?"),
]

TERM_TEMPLATES = [
    ("word2vec", "What is word2vec?"),
    ("cbow", "What is CBOW in word2vec?"),
    ("skip-gram", "What is skip-gram in word2vec?"),
    ("embedding", "What is a word embedding?"),
    ("vector space", "What is the vector space model?"),
    ("tf-idf", "What is TF-IDF?"),
    ("tokenization", "What is tokenization?"),
    ("stemming", "What is stemming?"),
    ("lemmatization", "What is lemmatization?"),
    ("bag of words", "What is the bag-of-words representation?"),
    ("language model", "What is a language model?"),
    ("n-gram", "What is an n-gram language model?"),
    ("bigram", "What is a bigram model?"),
    ("trigram", "What is a trigram model?"),
    ("perplexity", "What is perplexity in language models?"),
    ("smoothing", "What is smoothing in language models?"),
    ("laplace", "What is Laplace (add-one) smoothing?"),
    ("naive bayes", "What is Naive Bayes for text classification?"),
    ("multinomial", "What is the multinomial model in Naive Bayes?"),
    ("logistic regression", "What is logistic regression?"),
    ("cosine similarity", "What is cosine similarity?"),
    ("euclidean", "What is Euclidean distance?"),
    ("softmax", "What is softmax?"),
    ("precision", "What is precision?"),
    ("recall", "What is recall?"),
    ("f1", "What is F1 score?"),
    ("roc", "What is an ROC curve?"),
    ("gradient descent", "What is gradient descent?"),
    ("sgd", "What is stochastic gradient descent (SGD)?"),
    ("rnn", "What is a Recurrent Neural Network (RNN)?"),
    ("lstm", "What is an LSTM?"),
    ("gru", "What is a GRU?"),
    ("attention", "What is attention?"),
    ("bert", "What is BERT?"),
]

STOP_TERMS = set([
    "Final Project","Contents","Table Of Contents","Overview","Introduction",
    "Natural Language Processing","NLP","Deep Learning","Machine Learning",
    "Data Set","Data Sets","Homework","Assignment","Project"
])

CAPITAL_PHRASE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b")
ACRONYM = re.compile(r"\b([A-Z]{2,6})\b")

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text or "") if s.strip()]

def best_sentence(question, context):
    q = re.findall(r"\w+", (question or "").lower())
    sents = sentences(context)
    if not sents:
        return (context[:280] + "…") if len(context) > 280 else context
    def score(sent):
        a = re.findall(r"\w+", sent.lower())
        return len(set(a) & set(q))
    sents.sort(key=lambda s: score(s), reverse=True)
    return sents[0]

def good_term(term):
    if term in STOP_TERMS: return False
    if len(term) < 3 or len(term) > 40: return False
    toks = term.split()
    if len(toks) > 5: return False
    if term.lower() == term: return False
    return True

def is_selected_doc(doc_id, filters):
    if not filters: return True
    did = doc_id.lower()
    return any(f.lower() in did for f in filters)

def build_gold(chunks, target=120, filters=None, seed=42, existing_questions=None):
    rng = random.Random(seed)
    qas, seen_q = [], set(existing_questions or [])

    # 1) Admin buckets
    for pat, qtext in ADMIN_BUCKETS:
        rx = re.compile(pat, re.I)
        for c in chunks:
            if not is_selected_doc(c["doc_id"], filters): continue
            if rx.search(c.get("text","")):
                ans = best_sentence(qtext, c.get("text",""))
                qkey = qtext.strip().lower()
                if qkey not in seen_q:
                    qas.append({"question": qtext, "answer": ans, "gold_chunk_ids":[c["chunk_id"]], "topic":"Admin"})
                    seen_q.add(qkey)
                break

    # 2) Subject by known terms
    # frequency-based caps
    def freq(term):
        tl = term.lower()
        return sum(1 for c in chunks if is_selected_doc(c["doc_id"], filters) and tl in c.get("text","").lower())
    term_caps = {}
    for term, _ in TERM_TEMPLATES:
        f = freq(term)
        if f >= 50: cap = 8
        elif f >= 20: cap = 6
        elif f >= 10: cap = 5
        elif f >= 5:  cap = 3
        elif f >= 1:  cap = 2
        else: cap = 0
        term_caps[term] = cap

    for term, qtext in TERM_TEMPLATES:
        cap = term_caps.get(term, 0)
        if cap == 0: continue
        matches = [c for c in chunks if is_selected_doc(c["doc_id"], filters) and term in c.get("text","").lower()]
        rng.shuffle(matches)
        used = 0
        for c in matches:
            if used >= cap: break
            ans = best_sentence(qtext, c.get("text",""))
            qkey = qtext.strip().lower()
            if qkey in seen_q: 
                continue
            qas.append({"question": qtext, "answer": ans, "gold_chunk_ids":[c["chunk_id"]], "topic":"Subject"})
            seen_q.add(qkey); used += 1

    # 3) Mine capitalized terms/acronyms
    candidates = []
    for c in chunks:
        if not is_selected_doc(c["doc_id"], filters): continue
        txt = c.get("text","")
        for m in CAPITAL_PHRASE.finditer(txt):
            term = m.group(1).strip()
            if good_term(term): candidates.append((term, c))
        for m in ACRONYM.finditer(txt):
            term = m.group(1).strip()
            if good_term(term): candidates.append((term, c))

    seen_terms = set()
    rng.shuffle(candidates)
    for term, c in candidates:
        if len(qas) >= target: break
        key = term.lower()
        if key in seen_terms: continue
        qtext = f"What is {term}?"
        qkey = qtext.strip().lower()
        if qkey in seen_q: 
            seen_terms.add(key); continue
        ans = best_sentence(qtext, c.get("text",""))
        if not ans: 
            seen_terms.add(key); continue
        qas.append({"question": qtext, "answer": ans, "gold_chunk_ids":[c["chunk_id"]], "topic":"Subject"})
        seen_q.add(qkey); seen_terms.add(key)

    return qas

def main():
    ap = argparse.ArgumentParser(description="Generate gold Q/A (with gold_chunk_ids) from chunks.jsonl.")
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=120)
    ap.add_argument("--filter_docs", default="", help="Comma-separated substrings; only docs whose doc_id contains any will be used.")
    ap.add_argument("--append", action="store_true", help="Append to existing out file and continue qid numbering.")
    ap.add_argument("--existing", default="", help="Optional existing questions.jsonl to avoid duplicate questions.")
    args = ap.parse_args()

    chunks = [r for r in iter_jsonl(args.chunks)]
    filters = [s.strip() for s in args.filter_docs.split(",") if s.strip()] or None

    existing_questions = set()
    if args.existing and os.path.exists(args.existing):
        for r in iter_jsonl(args.existing):
            q = r.get("question","").strip().lower()
            if q: existing_questions.add(q)

    qas = build_gold(chunks, target=args.target, filters=filters, existing_questions=existing_questions)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    start_idx = 1
    mode = "w"
    if args.append and os.path.exists(args.out):
        # count lines to continue qid
        with open(args.out, "r", encoding="utf-8") as f:
            start_idx = sum(1 for _ in f) + 1
        mode = "a"

    with open(args.out, mode, encoding="utf-8") as w:
        for i, qa in enumerate(qas, start=start_idx):
            w.write(json.dumps({"qid": f"q{i}", **qa}) + "\n")

    print(f"Wrote {len(qas)} Q/A -> {args.out} (start qid: q{start_idx})")

if __name__ == "__main__":
    main()
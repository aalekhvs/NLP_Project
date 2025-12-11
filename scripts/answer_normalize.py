# -*- coding: utf-8 -*-
import re

_num_words = {
    "zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5",
    "six":"6","seven":"7","eight":"8","nine":"9","ten":"10"
}
_num_re = re.compile(r"\b(" + "|".join(_num_words.keys()) + r")\b", re.I)

def normalize_for_eval(s: str) -> str:
    if not s: return ""
    t = s.strip()

    # collapse whitespace & lowercase
    t = re.sub(r"\s+", " ", t).strip().lower()

    # word-numbers → digits
    t = _num_re.sub(lambda m: _num_words[m.group(1).lower()], t)

    # percentages
    t = re.sub(r"\bpercent\b", "%", t)

    # hours/minutes abbreviations
    t = re.sub(r"\bhrs?\b", "hours", t)
    t = re.sub(r"\bmins?\b", "minutes", t)

    # normalize common punctuation spacing
    t = re.sub(r"\s*%\b", "%", t)
    t = re.sub(r"\s*([,.:;])\s*", r"\1 ", t)
    t = re.sub(r"\s+", " ", t).strip()

    return t
"""Build opd_explainer.html from one real dumped OPD sample.

Usage: .venv/bin/python exps/opd_explainer/build.py [sample.json]
"""

import json
import os
import sys

from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "opd_dump", "sample_1556028_0.json")
TOKENIZER = "/ai_data/models/Qwen/Qwen3.5-4B"

tok = AutoTokenizer.from_pretrained(TOKENIZER)
d = json.load(open(SAMPLE))
prompt_ids, response_ids = d["prompt_ids"], d["response_ids"]
P, R = len(prompt_ids), len(response_ids)
seq = prompt_ids + response_ids
assert all(d["teacher_ids"][i][0] == seq[i + 1] for i in range(len(seq) - 1)), "teacher alignment mismatch"


def piece(t):
    return tok.decode([t])


def topk(ids, lps, i):
    return [[piece(a), round(b, 3)] for a, b in zip(ids[i], lps[i])]


resp = []
for j, t in enumerate(response_ids):
    i = P - 1 + j  # position whose next-token prediction is response token j
    s = d["student_rollout_logprobs"][j]
    te = d["teacher_logprobs"][i][0]
    resp.append(
        {
            "id": t,
            "text": piece(t),
            "s": round(s, 4),
            "t": round(te, 4),
            "tT": topk(d["teacher_top5_ids"], d["teacher_top5_logprobs"], i),
            "sT": topk(d["student_top5_ids"], d["student_top5_logprobs"], i),
        }
    )

data = {
    # teacher logprob of prompt token k sits at index k-1; token 0 has none (vLLM returns None, verl drops it)
    "prompt": [
        {"id": t, "text": piece(t), "t": round(d["teacher_logprobs"][k - 1][0], 4) if k > 0 else None}
        for k, t in enumerate(prompt_ids)
    ],
    "response": resp,
    "P": P,
    "R": R,
    "score": d["reward_score"],
    "teacher_pos0_dropped": True,
    "source": os.path.relpath(SAMPLE, os.path.join(HERE, "..", "..")),
}

tpl = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
out = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
dst = os.path.join(HERE, "opd_explainer.html")
open(dst, "w", encoding="utf-8").write(out)
k1 = [r["s"] - r["t"] for r in resp]
print(f"wrote {dst}  P={P} R={R} mean_k1={sum(k1) / R:.3f} min_k1={min(k1):.2f}@{k1.index(min(k1))}")

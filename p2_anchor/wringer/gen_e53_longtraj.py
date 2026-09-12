"""E53 整軌長窗產料:四科 8k 整軌,截斷保留不丟棄(prereg_e53)。

用戶設計(2026-08-26):max_tokens 8192、整軌一列(pad+loss mask,
不堆砌);截斷軌跡保留——(1) 每域 8k 截斷率=P(bf16 思考>8k @temp1.0)
模型特性儀器,終結 E52 高溫長尾懸案;(2) 截斷前綴 KD 讓 student 在
長思考區段貼齊老師表徵。閉合率/截斷率/軌長分佈皆入卷。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.gen_e53_longtraj \\
    --base-url http://127.0.0.1:8000/v1 --model a1
"""
import argparse
import hashlib
import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoTokenizer

from p1_grouping.modelio import MODEL_ID, MODEL_REVISION
from p2_anchor.bleachers.bank import decontam_grams
from p2_anchor.wringer.build_calib512_v3 import scan
from p2_anchor.wringer.gen_cal4 import gen_one
from p2_anchor.wringer.gen_cal6 import BUILDERS

EV = Path("evidence/p1_grouping")
SEED = 20260827
THINK_END = 248069
MAX_TOK = 8192
ROW_LEN = 8192
PER_DOMAIN = 900       # D2:四科均分


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="a1")
    ap.add_argument("--workers", type=int, default=96)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    rows, lens, dom_stats = [], [], {}
    for di, name in enumerate(("agentic", "ifshape", "code", "math")):
        prompts, meta = BUILDERS[name](tok)
        rng = random.Random(SEED + di)
        rng.shuffle(prompts)
        prompts = prompts[:PER_DOMAIN]
        outs = []
        for w0 in range(0, len(prompts), 512):
            batch = prompts[w0:w0 + 512]
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                outs += list(ex.map(
                    lambda p: gen_one(args.base_url, args.model, p[1],
                                      MAX_TOK, SEED + di * 10000 + w0 + p[0]),
                    enumerate(batch)))
            print(f"  {name} {min(w0 + 512, len(prompts))}/{len(prompts)}",
                  flush=True)
        closed = trunc = in_win = 0
        glens = []
        for p, o in zip(prompts, outs):
            ids = tok(p + o, add_special_tokens=False)["input_ids"]
            glens.append(len(ids))
            c = "</think>" in o
            closed += c
            trunc += not c
            row = ids[:ROW_LEN]
            in_win += THINK_END in row
            L = len(row)
            rows.append(row + [0] * (ROW_LEN - L))
            lens.append(L)
        q = ([int(x) for x in statistics.quantiles(glens, n=4)]
             if len(glens) > 1 else glens)
        dom_stats[name] = {
            "bank": meta, "n_gen": len(prompts), "closed": closed,
            "closure_rate_8k": round(closed / max(len(prompts), 1), 4),
            "truncated": trunc,
            "p_exceed_8k": round(trunc / max(len(prompts), 1), 4),
            "closed_in_window": in_win, "traj_len_quartiles": q}
        print(json.dumps({name: dom_stats[name]["closure_rate_8k"],
                          "p>8k": dom_stats[name]["p_exceed_8k"]}),
              flush=True)

    t = torch.tensor(rows, dtype=torch.long)
    ln = torch.tensor(lens, dtype=torch.long)
    hits = set(scan(t, tok, decontam_grams()))
    keep = [i for i in range(t.shape[0]) if i not in hits]
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(keep), generator=g)
    idx = torch.tensor(keep)[perm]
    t, ln = t[idx], ln[idx]
    torch.save(t, EV / "calib_e53_traj.pt")
    torch.save(ln, EV / "calib_e53_len.pt")

    rec = {"version": "e53_longtraj(四科 8k 整軌,截斷保留;pad+len)",
           "sampling": {"temperature": 1.0, "top_p": 0.95,
                        "max_tokens": MAX_TOK},
           "per_domain": dom_stats, "hits_excised": len(hits),
           "rows": int(t.shape[0]), "row_len": ROW_LEN,
           "len_quartiles": [int(x) for x in
                             statistics.quantiles(ln.tolist(), n=4)],
           "closure_rows": int((t == THINK_END).any(dim=1).sum()),
           "seed": SEED, "model_revision": MODEL_REVISION,
           "sha256": hashlib.sha256(
               (EV / "calib_e53_traj.pt").read_bytes()).hexdigest(),
           "len_sha256": hashlib.sha256(
               (EV / "calib_e53_len.pt").read_bytes()).hexdigest()}
    (EV / "E53_TRAJ_MANIFEST.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({"rows": rec["rows"],
                      "closure_rows": rec["closure_rows"],
                      "sha256": rec["sha256"][:16]}, indent=1))
    print("E53_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()

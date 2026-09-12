"""E54 S0b 低溫壓回產料:temp 0.3/top_p 0.9,只留閉合軌(prereg_e54 v2)。

用戶三裁(2026-08-26 18:2x):temp 0.3/top_p 0.9、900 列/600 步、
B1 單跑+B2 雙跑。本料供 S4b 低溫 e2e 壓回段;截斷列一律丟棄
(壓回的教學對象=收得住的高機率軌跡)。per-domain 閉合率順帶
成為溫度軸儀器點(對照 temp 1.0 的 6.3%-46.7% 四點)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.gen_e54lt \\
    --base-url http://127.0.0.1:8000/v1 --model a1
"""
import argparse
import hashlib
import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer

from p1_grouping.modelio import MODEL_ID, MODEL_REVISION
from p2_anchor.bleachers.bank import decontam_grams
from p2_anchor.wringer.build_calib512_v3 import scan
from p2_anchor.wringer.gen_cal6 import BUILDERS

EV = Path("evidence/p1_grouping")
SEED = 20260828
THINK_END = 248069
MAX_TOK = 8192
ROW_LEN = 8192
PER_DOMAIN = 900
TEMP, TOP_P = 0.3, 0.9


def gen_one_lt(base_url, model, prompt, max_tokens, seed):
    r = requests.post(f"{base_url}/completions", json={
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": TEMP, "top_p": TOP_P, "seed": seed}, timeout=900)
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


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
                    lambda p: gen_one_lt(args.base_url, args.model, p[1],
                                         MAX_TOK,
                                         SEED + di * 10000 + w0 + p[0]),
                    enumerate(batch)))
            print(f"  lt {name} {min(w0 + 512, len(prompts))}/{len(prompts)}",
                  flush=True)
        closed = trunc = 0
        glens = []
        for p, o in zip(prompts, outs):
            if "</think>" not in o:          # 只留閉合軌
                trunc += 1
                continue
            closed += 1
            ids = tok(p + o, add_special_tokens=False)["input_ids"]
            glens.append(len(ids))
            row = ids[:ROW_LEN]
            L = len(row)
            rows.append(row + [0] * (ROW_LEN - L))
            lens.append(L)
        q = ([int(x) for x in statistics.quantiles(glens, n=4)]
             if len(glens) > 1 else glens)
        dom_stats[name] = {
            "bank": meta, "n_gen": len(prompts), "closed_kept": closed,
            "closure_rate_lt": round(closed / max(len(prompts), 1), 4),
            "truncated_dropped": trunc, "traj_len_quartiles": q}
        print(json.dumps({name: dom_stats[name]["closure_rate_lt"]}),
              flush=True)

    t = torch.tensor(rows, dtype=torch.long)
    ln = torch.tensor(lens, dtype=torch.long)
    hits = set(scan(t, tok, decontam_grams()))
    keep = [i for i in range(t.shape[0]) if i not in hits]
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(keep), generator=g)
    idx = torch.tensor(keep)[perm]
    t, ln = t[idx], ln[idx]
    torch.save(t, EV / "calib_e54lt_traj.pt")
    torch.save(ln, EV / "calib_e54lt_len.pt")

    rec = {"version": "e54lt(低溫壓回料:temp0.3/top_p0.9,閉合列 only)",
           "sampling": {"temperature": TEMP, "top_p": TOP_P,
                        "max_tokens": MAX_TOK},
           "per_domain": dom_stats, "hits_excised": len(hits),
           "rows": int(t.shape[0]), "row_len": ROW_LEN,
           "len_quartiles": [int(x) for x in
                             statistics.quantiles(ln.tolist(), n=4)],
           "closure_rows": int((t == THINK_END).any(dim=1).sum()),
           "seed": SEED, "model_revision": MODEL_REVISION,
           "sha256": hashlib.sha256(
               (EV / "calib_e54lt_traj.pt").read_bytes()).hexdigest(),
           "len_sha256": hashlib.sha256(
               (EV / "calib_e54lt_len.pt").read_bytes()).hexdigest()}
    (EV / "E54LT_TRAJ_MANIFEST.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({"rows": rec["rows"],
                      "sha256": rec["sha256"][:16]}, indent=1))
    print("E54LT_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()

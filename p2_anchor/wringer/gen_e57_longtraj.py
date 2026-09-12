"""E57 產料 cal10:cal8 同配方同 seed,上限 8k→16k(gen 與列窗同抬)。

用戶裁定(2026-08-28):cal8 開箱定罪——78.7% 列頂格 8192(EOS/答案尾
被列窗切除)、26.7% 列無 </think>(E53 毒性帶復現)、分科閉合率與
A₂−A₁ 增益單調對應(math 94%↔gs +4.78、code 53.5%↔he ±0)。Qwen 長
思考體質下 8k 窗教「不落地」,正中失律病灶。同 seed 同題庫只抬上限
=與 cal8 配對的閉合率劑量對照;E56 cycle-2 中止,E57 16k 全鏈接手。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.gen_e57_longtraj \\
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
SEED = 20260826
THINK_END = 248069
MAX_TOK = 16384
ROW_LEN = 16384
PER_DOMAIN = 1800      # E53 的 2×(D2 四科均分不變)


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
        if len(prompts) < PER_DOMAIN:
            print(f"  WARN {name} bank 只有 {len(prompts)} < {PER_DOMAIN}",
                  flush=True)
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
            "closure_rate_16k": round(closed / max(len(prompts), 1), 4),
            "truncated": trunc,
            "p_exceed_16k": round(trunc / max(len(prompts), 1), 4),
            "closed_in_window": in_win, "traj_len_quartiles": q}
        print(json.dumps({name: dom_stats[name]["closure_rate_16k"],
                          "p>16k": dom_stats[name]["p_exceed_16k"]}),
              flush=True)

    t = torch.tensor(rows, dtype=torch.long)
    ln = torch.tensor(lens, dtype=torch.long)
    hits = set(scan(t, tok, decontam_grams()))
    keep = [i for i in range(t.shape[0]) if i not in hits]
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(keep), generator=g)
    idx = torch.tensor(keep)[perm]
    t, ln = t[idx], ln[idx]
    torch.save(t, EV / "calib_e57_traj.pt")
    torch.save(ln, EV / "calib_e57_len.pt")

    rec = {"version": "e57_longtraj/cal10(cal8 同配方同 seed;四科 16k 整軌,截斷保留照實計數)",
           "sampling": {"temperature": 1.0, "top_p": 0.95,
                        "max_tokens": MAX_TOK},
           "per_domain": dom_stats, "hits_excised": len(hits),
           "rows": int(t.shape[0]), "row_len": ROW_LEN,
           "len_quartiles": [int(x) for x in
                             statistics.quantiles(ln.tolist(), n=4)],
           "closure_rows": int((t == THINK_END).any(dim=1).sum()),
           "seed": SEED, "model_revision": MODEL_REVISION,
           "sha256": hashlib.sha256(
               (EV / "calib_e57_traj.pt").read_bytes()).hexdigest(),
           "len_sha256": hashlib.sha256(
               (EV / "calib_e57_len.pt").read_bytes()).hexdigest()}
    (EV / "E57_TRAJ_MANIFEST.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({"rows": rec["rows"],
                      "closure_rows": rec["closure_rows"],
                      "sha256": rec["sha256"][:16]}, indent=1))
    print("E57_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()

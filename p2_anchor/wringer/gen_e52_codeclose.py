"""E52 code 閉合完整重生成:GEN_TOK 截斷根因修復(prereg_e52)。

普查(2026-08-25):calib_v6_part_code 閉合密度僅 1.5%(GEN_TOK=800
截斷)vs agentic 52.5%(1900)——冠軍語料 65% code 幾乎全為「不終止
思考」示範。本件以 max_tokens=3072 重生成 code 行為資料,僅保留自然
閉合軌跡,以閉合錨定(窗內位置抖動 [1200,1900])切 2048 窗,產出
code 域閉合窗 part。閉合率(yield)本身即截斷假說的驗收數字。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.gen_e52_codeclose \\
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
from p2_anchor.wringer.gen_cal4 import SEQ_LEN, gen_one
from p2_anchor.wringer.gen_cal6 import bank_code

EV = Path("evidence/p1_grouping")
SEED = 20260826
THINK_END = 248069
MAX_TOK = 3072
TARGET = 3200          # 除汙後目標窗數(fill 轉折佔比 ≤30% 由此上限)
JIT = (1200, 1900)     # 閉合錨定抖動帶(避免位置偽影)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="a1")
    ap.add_argument("--workers", type=int, default=96)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    prompts, meta = bank_code()
    rng = random.Random(SEED)
    rng.shuffle(prompts)

    stream, closes = [], []          # 閉合軌跡串接流 + 流內閉合位置
    n_gen = n_closed = done = 0
    while len(closes) < int(TARGET * 1.15) and done < len(prompts):
        batch = prompts[done:done + 512]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            outs = list(ex.map(
                lambda p: gen_one(args.base_url, args.model, p[1], MAX_TOK,
                                  SEED + done + p[0]),
                enumerate(batch)))
        for p, o in zip(batch, outs):
            n_gen += 1
            if "</think>" not in o:
                continue
            n_closed += 1
            ids = tok(p + o, add_special_tokens=False)["input_ids"]
            base = len(stream)
            stream.extend(ids)
            closes.extend(base + j for j, t in enumerate(ids)
                          if t == THINK_END)
        done += len(batch)
        print(f"  gen {done}/{len(prompts)}:closed {n_closed}/{n_gen}"
              f"({n_closed / max(n_gen, 1) * 100:.1f}%)"
              f",closure {len(closes)}", flush=True)

    g = random.Random(SEED + 1)
    wins, pos_in = [], []
    for c in closes:
        a = max(0, min(c - g.randint(*JIT), len(stream) - SEQ_LEN))
        rel = c - a
        if not (0 <= rel < SEQ_LEN - 16):    # 閉合須在窗內且留 post 尾
            continue
        wins.append(stream[a:a + SEQ_LEN])
        pos_in.append(rel)
    t = torch.tensor(wins, dtype=torch.long)
    hits = set(scan(t, tok, decontam_grams()))
    clean = [i for i in range(t.shape[0]) if i not in hits]
    part = t[clean[:TARGET]]
    out = EV / "calib_e52_codeclose.pt"
    torch.save(part, out)

    q = [int(x) for x in statistics.quantiles(pos_in, n=4)]
    rec = {"version": "e52_codeclose(code 重生成 max3072+閉合過濾+錨定切窗)",
           "bank": meta, "sampling": {"temperature": 1.0, "top_p": 0.95,
                                      "max_tokens": MAX_TOK},
           "n_gen": n_gen, "n_closed": n_closed,
           "closure_yield": round(n_closed / max(n_gen, 1), 4),
           "old_part_density_ref": 0.015,
           "windows_raw": int(t.shape[0]), "hits_excised": len(hits),
           "rows": int(part.shape[0]), "closure_pos_quartiles": q,
           "jitter": list(JIT), "seed": SEED, "shape": list(part.shape),
           "model_revision": MODEL_REVISION,
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    (EV / "E52_CODECLOSE_MANIFEST.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({"rows": rec["rows"], "closure_yield":
                      rec["closure_yield"], "sha256": rec["sha256"][:16]},
                     indent=1))
    print("E52_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()

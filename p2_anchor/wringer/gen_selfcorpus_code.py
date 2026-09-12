"""E38b 條件彈藥:bf16 老師自生 code SFT 語料(docstring→函式,行為級對位)。

觸發條件(prereg_e38_dualaxis.json;用戶已裁「維持條件觸發」):
  A2/A3 humaneval-A 卡 ~0.30(raw code 域的水裝進去但 humaneval 不領情)
  → 行為格式成為下一個最可疑缺口,本彈藥升級主攻(E38b 追加臂)。

種子:starcoder 抽「def 函式頭+docstring」區塊,套 sft_code smoke 檔的
SFT 格式(<|user|>\\n{block}\\n<|assistant|>\\n),bf16 老師續寫函式體。
除汙:生成後全列 8-gram 複掃(gsm8k+humaneval,同 build_calib512_v3),
命中列切除;margin 由多生成餘量提供。

兩段式:
  CPU 預驗:--extract-only(只抽種子,驗數量與格式)
  全跑(需先 serve bf16):
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.gen_selfcorpus_code \\
      --base-url http://127.0.0.1:8000/v1 --model a1
"""
import argparse
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer

from p1_grouping.calib import _sample_lines
from p1_grouping.modelio import MODEL_ID, MODEL_REVISION

EV = Path("evidence/p1_grouping")
STARCODER = ("~/projects/pretrain/data/circus_v0_2/"
             "code_starcoder.jsonl")
DEF_RE = re.compile(
    r'((?:^|\n)def \w+\([^)]*\)[^:\n]*:\n'      # 函式頭(單行簽名)
    r'\s+(?:"""|\'\'\')(?:.|\n){20,900}?(?:"""|\'\'\'))')  # docstring 區塊


def extract_seeds(n_seeds, seed, lines_per_round=8000, max_rounds=24):
    """從 starcoder 隨機行抽 def+docstring 區塊(去重、長度過濾)。"""
    rng = random.Random(seed)
    seen, seeds = set(), []
    for _ in range(max_rounds):
        for ln in _sample_lines(STARCODER, lines_per_round, rng):
            try:
                text = json.loads(ln).get("text") or ""
            except json.JSONDecodeError:
                continue
            for m in DEF_RE.finditer(text):
                block = m.group(1).strip()
                if not (80 <= len(block) <= 2000):
                    continue
                key = hashlib.sha1(block.encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                seeds.append(block)
        print(f"seeds so far: {len(seeds)}", flush=True)
        if len(seeds) >= n_seeds:
            break
    assert len(seeds) >= n_seeds, f"種子不足:{len(seeds)} < {n_seeds}"
    rng.shuffle(seeds)
    return seeds[:n_seeds]


def gen_one(base_url, model, prompt, max_tokens, seed):
    r = requests.post(f"{base_url}/completions", json={
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 1.0, "top_p": 0.95, "seed": seed}, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="a1")
    ap.add_argument("--out", default=str(EV / "calib_selfgen_code_512.pt"))
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--len", dest="seq_len", type=int, default=2048)
    ap.add_argument("--n-seeds", type=int, default=2400)
    ap.add_argument("--gen-tokens", type=int, default=700)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--extract-only", action="store_true")
    args = ap.parse_args()

    seeds = extract_seeds(args.n_seeds, args.seed)
    if args.extract_only:
        print(json.dumps({"n_seeds": len(seeds),
                          "sample": seeds[0][:300]}, ensure_ascii=False))
        return

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    from p2_anchor.wringer.build_calib512_v3 import (contamination_grams,
                                                       scan)
    bad = contamination_grams()
    prompts = [f"<|user|>\n{s}\n<|assistant|>\n" for s in seeds]
    need = args.n * args.seq_len
    ids: list[int] = []
    done = 0
    while len(ids) < need and done < len(prompts):
        batch = prompts[done:done + 512]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            outs = list(ex.map(
                lambda p: gen_one(args.base_url, args.model, p[1],
                                  args.gen_tokens, args.seed + done + p[0]),
                enumerate(batch)))
        for p, o in zip(batch, outs):
            ids.extend(tok(p + o, add_special_tokens=False)["input_ids"])
        done += len(batch)
        print(f"gen {done}/{len(prompts)}: {len(ids)}/{need} tokens",
              flush=True)
    assert len(ids) >= need, f"自生語料不足:{len(ids)} < {need}"
    n_margin = min(len(ids) // args.seq_len, args.n + 30)
    t = torch.tensor(ids[:n_margin * args.seq_len],
                     dtype=torch.long).reshape(n_margin, args.seq_len)
    hits = set(scan(t, tok, bad))
    clean = [i for i in range(t.shape[0]) if i not in hits]
    assert len(clean) >= args.n, f"除汙後不足:{len(clean)} < {args.n}"
    t = t[clean[:args.n]]
    out = Path(args.out)
    torch.save(t, out)
    rec = {"teacher": "bf16(vLLM serve)", "seed_source": "starcoder def+docstring 抽取",
           "n_seeds": len(seeds), "gen_tokens": args.gen_tokens,
           "format": "sft_code smoke 對位(<|user|>/<|assistant|>)",
           "sampling": {"temperature": 1.0, "top_p": 0.95},
           "seed": args.seed, "shape": list(t.shape),
           "hits_excised": len(hits),
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
           "note": "E38b 條件彈藥;種子非 humaneval 測題,生成後 8-gram 複掃切除",
           "model_revision": MODEL_REVISION}
    (EV / "CALIB_SELFGEN_CODE_MANIFEST.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({k: rec[k] for k in ("shape", "sha256", "hits_excised")},
                     indent=1))


if __name__ == "__main__":
    main()

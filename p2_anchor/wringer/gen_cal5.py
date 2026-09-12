"""cal5:cal4 同管線規模化(E40 規模化語料;prereg_e40_scale)。

與 cal4 配方不變,量與配比變(H_richcap FAIL → 槓桿在語料不在 rank):
  規模    — 10240 列(cal4 1023 的 10×,~21M token)
  配比    — agentic 傾斜 50/25/25(IFEval 首要判準)
  agentic — cal3 同源種子 n=2800、k=3(同 prompt 多採樣即多樣性,E39 紀律)
  code    — MBPP 974 全集 k=8 + starcoder 自足過濾 k=5
  math    — gsm8k train 7473 k=2
  margin  — code 15%(cal4 除汙率 10.5%),其餘 10%
產物 calib_selfgen_v5.pt + CAL5_MANIFEST.json;E40 訓練料由鏈內
concat(cal5, cal4) 打包(O5 accumulate:歷代合成不 replace)。

  CPU 預驗:--extract-only(只組 prompt bank,驗數量)
  全跑(先 serve bf16 於 --base-url):
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.gen_cal5 \\
      --base-url http://127.0.0.1:8000/v1 --model a1
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from p1_grouping.calib import build
from p1_grouping.modelio import MODEL_ID, MODEL_REVISION
from p2_anchor.wringer.build_calib512_v3 import (DOMAINS, contamination_grams,
                                                   scan)
from p2_anchor.wringer.gen_cal4 import SEQ_LEN, gen_domain, sft
from p2_anchor.wringer.gen_selfcorpus_code import extract_seeds

EV = Path("evidence/p1_grouping")
SEED = 20260814
ROWS = {"agentic": 5120, "code": 2560, "math": 2560}      # 50/25/25 傾斜
MARGIN = {"agentic": 0.10, "code": 0.15, "math": 0.10}
GEN_TOK = {"agentic": 1900, "code": 800, "math": 600}


def bank_code(k_mbpp=8, k_star=5):
    from datasets import load_dataset
    prompts = []
    for split in ("train", "validation", "test", "prompt"):
        for x in load_dataset("google-research-datasets/mbpp", split=split):
            p = (f"Write a Python function for the following task.\n"
                 f"{x['text']}\nYour code should pass this test:\n"
                 f"{x['test_list'][0]}")
            prompts += [sft(p)] * k_mbpp
    star = [s for s in extract_seeds(2400, 20260817)
            if ">>>" in s or "Example" in s or "example" in s]
    prompts += [sft(s) for s in star] * k_star
    return prompts, {"mbpp": 974, "mbpp_k": k_mbpp,
                     "starcoder_selfcontained": len(star), "star_k": k_star}


def bank_math(k=2):
    from datasets import load_dataset
    qs = [sft(x["question"])
          for x in load_dataset("openai/gsm8k", "main", split="train")] * k
    return qs, {"gsm8k_train": len(qs) // k, "k": k}


def bank_agentic(tok, n=2800, k=3):
    d = dict(DOMAINS["agentic"])                          # cal3 同源五切片
    t = build(d["sources"], tok, n, SEQ_LEN, d["seed"] + 2,
              lines_per_file=24000)
    seeds = [tok.decode(t[i, :256].tolist()) for i in range(n)]
    return seeds * k, {"seed_rows": n, "seed_k": k, "seed_tokens": 256,
                       "sources": d["sources"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="a1")
    ap.add_argument("--out", default=str(EV / "calib_selfgen_v5.pt"))
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--extract-only", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    banks = {}
    banks["agentic"] = bank_agentic(tok)
    banks["code"] = bank_code()
    banks["math"] = bank_math()
    if args.extract_only:
        print(json.dumps({k: {"n_prompts": len(v[0]), **v[1]}
                          for k, v in banks.items()}, ensure_ascii=False,
                         indent=1, default=str))
        return

    bad = contamination_grams()
    parts, report = [], {}
    for name, (prompts, meta) in banks.items():
        rows = ROWS[name]
        need = int(rows * (1 + MARGIN[name])) * SEQ_LEN
        print(f"== {name}:{len(prompts)} prompts → {rows} rows ==",
              flush=True)
        t = gen_domain(args, tok, prompts, GEN_TOK[name], need,
                       SEED + hash(name) % 10000)
        hits = set(scan(t, tok, bad))
        clean = [i for i in range(t.shape[0]) if i not in hits]
        assert len(clean) >= rows, f"{name} 除汙後不足:{len(clean)}"
        parts.append(t[clean[:rows]])
        report[name] = {**meta, "rows": rows, "hits_excised": len(hits),
                        "gen_tokens": GEN_TOK[name]}
    full = torch.cat(parts, dim=0)
    perm = torch.randperm(full.shape[0],
                          generator=torch.Generator().manual_seed(SEED))
    full = full[perm]
    out = Path(args.out)
    torch.save(full, out)
    rec = {"teacher": "bf16(vLLM serve)", "domains": report,
           "tilt": "50/25/25(agentic 傾斜;IFEval 首要)",
           "sampling": {"temperature": 1.0, "top_p": 0.95},
           "format": "code/math=SFT prompt bank(<|user|>/<|assistant|>);"
                     "agentic=種子續寫",
           "seed": SEED, "shape": list(full.shape),
           "decontam": "全列 8-gram(gsm8k-test+humaneval)列級切除",
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
           "model_revision": MODEL_REVISION}
    (EV / "CAL5_MANIFEST.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({"shape": rec["shape"], "sha256": rec["sha256"]},
                     indent=1))


if __name__ == "__main__":
    main()

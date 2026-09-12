"""cal4:三域 prompt bank + bf16 老師離線軌跡(E39 on-manifold 蒸餾語料)。

設計(用戶已裁:離線生成+同 prompt 多採樣;code 主力改 MBPP):
  agentic — cal3 同源 agentic 切片前 256 token 種子,老師續寫(保 IFEval 軸;
            k=1,gen 1900)
  code    — MBPP 974 題全集(k=3)+ starcoder 自足過濾種子(含 doctest/
            Examples,~134 個,k=2);SFT 格式 <|user|>/<|assistant|>
  math    — gsm8k train split 題目(k=1;與 test 隔離,生成列仍過 8-gram 掃)

打包:各域獨立打包至 n_rows/3,合併列級 shuffle;全列 8-gram 除汙
(gsm8k-test + humaneval)列級切除,margin 由超量生成提供。

  CPU 預驗:--extract-only(只組 prompt bank,驗數量)
  全跑(先 serve bf16 於 --base-url):
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.gen_cal4 \\
      --base-url http://127.0.0.1:8000/v1 --model a1
"""
import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer

from p1_grouping.calib import build
from p1_grouping.modelio import MODEL_ID, MODEL_REVISION
from p2_anchor.wringer.build_calib512_v3 import (DOMAINS, contamination_grams,
                                                   scan)
from p2_anchor.wringer.gen_selfcorpus_code import extract_seeds

EV = Path("evidence/p1_grouping")
SEED = 20260818
SEQ_LEN = 2048


def sft(p):
    return f"<|user|>\n{p}\n<|assistant|>\n"


def bank_code():
    from datasets import load_dataset
    prompts = []
    for split in ("train", "validation", "test", "prompt"):
        for x in load_dataset("google-research-datasets/mbpp", split=split):
            p = (f"Write a Python function for the following task.\n"
                 f"{x['text']}\nYour code should pass this test:\n"
                 f"{x['test_list'][0]}")
            prompts += [sft(p)] * 3                      # MBPP k=3
    star = [s for s in extract_seeds(2400, 20260817)
            if ">>>" in s or "Example" in s or "example" in s]
    prompts += [sft(s) for s in star] * 2                # 自足 starcoder k=2
    return prompts, {"mbpp": 974, "mbpp_k": 3, "starcoder_selfcontained":
                     len(star), "star_k": 2}


def bank_math():
    from datasets import load_dataset
    qs = [sft(x["question"])
          for x in load_dataset("openai/gsm8k", "main", split="train")]
    return qs, {"gsm8k_train": len(qs), "k": 1}


def bank_agentic(tok, n=450):
    d = dict(DOMAINS["agentic"])                          # cal3 同源五切片
    t = build(d["sources"], tok, n, SEQ_LEN, d["seed"] + 1,
              lines_per_file=3200)
    return ([tok.decode(t[i, :256].tolist()) for i in range(n)],
            {"seed_rows": n, "seed_tokens": 256, "sources": d["sources"]})


def gen_one(base_url, model, prompt, max_tokens, seed):
    r = requests.post(f"{base_url}/completions", json={
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 1.0, "top_p": 0.95, "seed": seed}, timeout=900)
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


def gen_domain(args, tok, prompts, gen_tokens, need_tokens, dseed):
    """依序批次生成直到 token 需求滿足(prompts 已含 k 重複,順序即多樣)。"""
    rng = random.Random(dseed)
    rng.shuffle(prompts)
    ids, done = [], 0
    while len(ids) < need_tokens and done < len(prompts):
        batch = prompts[done:done + 512]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            outs = list(ex.map(
                lambda p: gen_one(args.base_url, args.model, p[1], gen_tokens,
                                  dseed + done + p[0]),
                enumerate(batch)))
        for p, o in zip(batch, outs):
            ids.extend(tok(p + o, add_special_tokens=False)["input_ids"])
        done += len(batch)
        print(f"  gen {done}/{len(prompts)}: {len(ids)}/{need_tokens}",
              flush=True)
    assert len(ids) >= need_tokens, f"生成不足:{len(ids)} < {need_tokens}"
    n = len(ids) // SEQ_LEN
    return torch.tensor(ids[:n * SEQ_LEN], dtype=torch.long).reshape(n, SEQ_LEN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="a1")
    ap.add_argument("--out", default=str(EV / "calib_selfgen_v4.pt"))
    ap.add_argument("--n-rows", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=64)
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
    third = args.n_rows // 3
    margin = 12
    need = (third + margin) * SEQ_LEN
    gen_tok = {"agentic": 1900, "code": 800, "math": 600}
    parts, report = [], {}
    for name, (prompts, meta) in banks.items():
        print(f"== {name}:{len(prompts)} prompts ==", flush=True)
        t = gen_domain(args, tok, prompts, gen_tok[name], need,
                       SEED + hash(name) % 10000)
        hits = set(scan(t, tok, bad))
        clean = [i for i in range(t.shape[0]) if i not in hits]
        assert len(clean) >= third, f"{name} 除汙後不足:{len(clean)}"
        parts.append(t[clean[:third]])
        report[name] = {**meta, "rows": third, "hits_excised": len(hits),
                        "gen_tokens": gen_tok[name]}
    full = torch.cat(parts, dim=0)
    perm = torch.randperm(full.shape[0],
                          generator=torch.Generator().manual_seed(SEED))
    full = full[perm]
    out = Path(args.out)
    torch.save(full, out)
    rec = {"teacher": "bf16(vLLM serve)", "domains": report,
           "sampling": {"temperature": 1.0, "top_p": 0.95},
           "format": "code/math=SFT prompt bank(<|user|>/<|assistant|>);"
                     "agentic=種子續寫",
           "seed": SEED, "shape": list(full.shape),
           "decontam": "全列 8-gram(gsm8k-test+humaneval)列級切除",
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
           "model_revision": MODEL_REVISION}
    (EV / "CAL4_MANIFEST.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({"shape": rec["shape"], "sha256": rec["sha256"]},
                     indent=1))


if __name__ == "__main__":
    main()

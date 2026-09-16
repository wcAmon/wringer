"""2c 煙測:vLLM 離線載入一個 export(bf16 或 GPTQ 版面),對固定提示做 greedy 生成,存 token 序列與載入記憶體。
    python -m p2_anchor.wringer.gptq_smoke --model exports/e69_p3b_w2_gptq4 --out <dir>/smoke_gptq4.json [--compare <json>]
--compare:與另一份輸出比 token 逐位一致率(第一分歧位置的中位數)。印 SMOKE_DONE <json>。
"""
import argparse
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain in two sentences why the sky is blue.",
    "List three prime numbers greater than 100 and show they are prime.",
    "Translate to French: The quick brown fox jumps over the lazy dog.",
    "What is 17 * 23? Show the steps.",
    "Write a haiku about quantization.",
    "Give a SQL query that counts orders per customer from an orders table.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Implement binary search in Python and state its complexity.",
    "Name the capital of Australia and one fact about it.",
    "Convert 100 degrees Fahrenheit to Celsius with the formula.",
    "Write a regular expression that matches an email address and explain each part.",
    "What are the three laws of thermodynamics, briefly?",
    "Write a bash one-liner to count lines in all .py files recursively.",
    "Explain the difference between a process and a thread.",
    "Sort the list [5, 3, 9, 1, 7] using merge sort, showing the merges.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=str(ROOT / args.model), max_model_len=4096, gpu_memory_utilization=0.6, seed=0, dtype=args.dtype)
    load_s = time.time() - t0
    torch.cuda.synchronize()
    msgs = [[{"role": "user", "content": p}] for p in PROMPTS]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    toks = [list(o.outputs[0].token_ids) for o in outs]
    texts = [o.outputs[0].text for o in outs]
    rep = {"model": args.model, "dtype": args.dtype, "load_s": round(load_s, 1), "tokens": toks, "texts": texts,
           "max_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    if args.compare:
        ref = json.load(open(args.compare))["tokens"]
        firsts = []
        for a, b in zip(toks, ref):
            n = 0
            while n < min(len(a), len(b)) and a[n] == b[n]:
                n += 1
            firsts.append(n if n < min(len(a), len(b)) else -1)   # -1 = 全程一致
        rep["compare"] = {"ref": args.compare, "first_divergence": firsts,
                          "identical": sum(x == -1 for x in firsts), "n": len(firsts),
                          "mean_prefix_agree": round(sum(len(a) if f == -1 else f for a, f in zip(toks, firsts)) / len(firsts), 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("SMOKE_DONE", json.dumps({k: rep[k] for k in ("model", "load_s", "max_alloc_gb")} | ({"compare": rep["compare"]} if "compare" in rep else {})), flush=True)


if __name__ == "__main__":
    main()

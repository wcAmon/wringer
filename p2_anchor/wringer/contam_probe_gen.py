"""模型端記憶探針(guided completion):把測試集參考文字切成前綴/後綴,以 raw completions(無 chat 模板、temperature 0)
讓模型接續前綴,存下生成與參考後綴,交給 contam_probe_score 計算逐字重現。

集合:ifeval:prompt(前綴 = 前 40% 詞)、humaneval:canonical(前綴 = 題目 prompt + 解答前 40%)、gsm8k:answer(前綴 = 問題 + 解答前 40%)
輸出:<out>/<set>.jsonl,每列 {id, prefix, ref, gen}
"""
import argparse
import glob
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
HUB = Path.home() / ".cache/huggingface/hub"
TOK = re.compile(r"\S+\s*")   # 保留空白的詞切分,便於原樣拼回


def split_at(text, frac=0.4, min_prefix=8):
    parts = TOK.findall(text)
    k = max(min_prefix, int(len(parts) * frac))
    if k >= len(parts) - 3:
        return None
    return "".join(parts[:k]), "".join(parts[k:])


def load_sets():
    S = {}
    ifp = ROOT / "vendor/instruction_following_eval/data/input_data.jsonl"
    S["ifeval"] = []
    for d in map(json.loads, open(ifp)):
        sp = split_at(d["prompt"])
        if sp:
            S["ifeval"].append((str(d["key"]), sp[0], sp[1]))
    he = pd.read_parquet(glob.glob(str(HUB / "datasets--openai--openai_humaneval/snapshots/*/openai_humaneval/test-*.parquet"))[0])
    S["humaneval"] = []
    for tid, pr, sol in zip(he["task_id"], he["prompt"], he["canonical_solution"]):
        sp = split_at(sol, min_prefix=3)
        if sp:
            S["humaneval"].append((tid, pr + sp[0], sp[1]))
    gs = pd.read_parquet(glob.glob(str(HUB / "datasets--openai--gsm8k/snapshots/*/main/test-*.parquet"))[0])
    S["gsm8k"] = []
    for i, (q, a) in enumerate(zip(gs["question"], gs["answer"])):
        sp = split_at(a)
        if sp:
            S["gsm8k"].append((str(i), q + "\n" + sp[0], sp[1]))
    return S


def gen_one(base_url, model, prefix, max_tokens):
    body = {"model": model, "prompt": prefix, "max_tokens": max_tokens, "temperature": 0.0, "seed": 0}
    for attempt in range(3):
        try:
            r = requests.post(f"{base_url}/completions", json=body, timeout=600)
            r.raise_for_status()
            return r.json()["choices"][0]["text"]
        except Exception as e:
            if attempt == 2:
                raise
            print(f"retry {attempt+1}: {e}", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="a1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=192)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    S = load_sets()
    for name, items in S.items():
        f = out / f"{name}.jsonl"
        if f.exists():
            continue
        with ThreadPoolExecutor(args.workers) as ex:
            gens = list(ex.map(lambda it: gen_one(args.base_url, args.model, it[1], args.max_tokens), items))
        with f.open("w") as w:
            for (tid, pre, ref), g in zip(items, gens):
                w.write(json.dumps({"id": tid, "prefix": pre, "ref": ref, "gen": g}, ensure_ascii=False) + "\n")
        print(f"PROBE_GEN {name} n={len(items)}", flush=True)
    print("PROBE_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()

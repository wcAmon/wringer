"""A1 協定 IFEval runner——打 OpenAI 相容端點,官方腳本判分。

依據 Agents-A1 evaluation/IF/README.md:官方 google-research IFEval 腳本
(已 vendor 至 vendor/instruction_following_eval,釘 0155391 @2026-08-08)
+ Qwen3.5 對齊採樣參數(temperature=1.0, top_p=0.95, top_k=20, min_p=0,
presence_penalty=1.5, repetition_penalty=1.0)。

服務端預期 vLLM 帶 --reasoning-parser qwen3(content 已剝 <think>);
本端仍防禦性再剝一次(伺服器缺 parser 時的回退)。每題 seed = base+idx,
同 seed 同端點可重現(vLLM 逐請求 seed)。

用法:
  .venv/bin/python -m p2_anchor.a1eval.ifeval_runner \
      --base-url http://127.0.0.1:8000/v1 --model a1 \
      --out evidence/p1_grouping/a1eval/ifeval_bf16
"""
import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor/instruction_following_eval"
DATA = VENDOR / "data/input_data.jsonl"
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5,
            "extra_body": {"top_k": 20, "min_p": 0.0,
                           "repetition_penalty": 1.0}}
THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.S)


def gen_one(base_url, model, prompt, seed, max_tokens, retries=3):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "seed": seed,
            **{k: v for k, v in SAMPLING.items() if k != "extra_body"},
            **SAMPLING["extra_body"]}
    for attempt in range(retries):
        try:
            r = requests.post(f"{base_url}/chat/completions", json=body,
                              timeout=1800)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            content = msg.get("content") or ""
            return THINK_RE.sub("", content).strip()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"retry {attempt + 1}: {e}", file=sys.stderr, flush=True)
            time.sleep(5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="a1")
    ap.add_argument("--out", required=True, help="輸出目錄")
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--workers", type=int, default=128,
                    help="併發請求數;2026-08-20 起預設 128(舊座標 32,"
                         "重錨見 a1eval/reanchor_w128)")
    ap.add_argument("--skip-gen", action="store_true",
                    help="重用既有 responses.jsonl,只跑判分")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()   # subprocess cwd 在 vendor/,必須絕對
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = [json.loads(l)["prompt"] for l in DATA.open()]
    resp_path = out_dir / "responses.jsonl"

    if args.skip_gen:
        gen_s = None
        assert resp_path.exists(), f"--skip-gen 但 {resp_path} 不存在"
    else:
        print(f"{len(prompts)} prompts → {args.base_url}", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            responses = list(ex.map(
                lambda iv: gen_one(args.base_url, args.model, iv[1],
                                   args.seed + iv[0], args.max_tokens),
                enumerate(prompts)))
        gen_s = round(time.time() - t0, 1)
        print(f"generation done in {gen_s}s", flush=True)
        with resp_path.open("w") as f:
            for p, r in zip(prompts, responses):
                f.write(json.dumps({"prompt": p, "response": r},
                                   ensure_ascii=False) + "\n")

    proc = subprocess.run(
        [sys.executable, "-m",
         "instruction_following_eval.evaluation_main",
         f"--input_data={DATA}", f"--input_response_data={resp_path}",
         f"--output_dir={out_dir}"],
        cwd=ROOT / "vendor", capture_output=True, text=True)
    (out_dir / "eval_stdout.txt").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        sys.exit(1)

    # 官方輸出:兩段報告(strict / loose),各以 prompt-level / instruction-level 開頭
    accs = re.findall(r"prompt-level: ([0-9.]+)\ninstruction-level: ([0-9.]+)",
                      proc.stdout)
    summary = {"n_prompts": len(prompts), "seed": args.seed,
               "sampling": {**{k: v for k, v in SAMPLING.items()
                               if k != "extra_body"},
                            **SAMPLING["extra_body"]},
               "max_tokens": args.max_tokens, "generation_s": gen_s,
               "workers": args.workers,
               "vendor": (VENDOR / "VENDOR_NOTE.txt").read_text().strip()}
    if len(accs) == 2:
        summary["strict"] = {"prompt_level": float(accs[0][0]),
                             "instruction_level": float(accs[0][1])}
        summary["loose"] = {"prompt_level": float(accs[1][0]),
                            "instruction_level": float(accs[1][1])}
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary.get("strict"), indent=1), flush=True)
    print("wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()

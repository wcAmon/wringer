"""A1 協定三科 runner:gsm8k / mmlu / humaneval(補齊 E27 lm_eval 四科)。

與 ifeval_runner 同協定:vLLM OpenAI 端點 + Qwen3.5 對齊採樣
(temp 1.0/top_p .95/top_k 20/presence 1.5)+ chat template +
reasoning parser(防禦性再剝 <think>)+ 每題 seed=base+idx 可重現。

判分(自建,vendored 邏輯,無官方腳本可循):
  gsm8k     — 官方要求「#### 後填最終數字」;取回應最後一個 #### 數字,
              退回全文最後一個數字;數值比對(容忍逗號/貨幣/小數尾零)。
  mmlu      — 選項字母;取回應第一個獨立 A-D(含「Answer: X」樣式),
              與 gold 對照;57 科逐科分數 + 總分(micro)。
  humaneval — 抽 ```python 塊(退回全文),與官方 test + check(entry_point)
              同程序執行,10s 逾時;pass@1。

用法:
  .venv/bin/python -m p2_anchor.a1eval.subject_runner \
      --subject gsm8k --base-url http://127.0.0.1:8000/v1 --model a1 \
      --out evidence/p1_grouping/a1eval/gsm8k_bf16
"""
import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from p2_anchor.a1eval.ifeval_runner import SAMPLING, gen_one

MMLU_SUBJECTS = None  # lazy;57 configs


def load_items(subject):
    from datasets import load_dataset
    items = []
    if subject == "gsm8k":
        d = load_dataset("openai/gsm8k", "main", split="test")
        for x in d:
            items.append({
                "prompt": x["question"].strip()
                + "\n\nPlease reason step by step, and put your final "
                  "numeric answer after '####'.",
                "gold": x["answer"].split("####")[-1].strip()})
    elif subject == "mmlu":
        subjects = [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics",
            "college_medicine", "college_physics", "computer_security",
            "conceptual_physics", "econometrics", "electrical_engineering",
            "elementary_mathematics", "formal_logic", "global_facts",
            "high_school_biology", "high_school_chemistry",
            "high_school_computer_science", "high_school_european_history",
            "high_school_geography", "high_school_government_and_politics",
            "high_school_macroeconomics", "high_school_mathematics",
            "high_school_microeconomics", "high_school_physics",
            "high_school_psychology", "high_school_statistics",
            "high_school_us_history", "high_school_world_history",
            "human_aging", "human_sexuality", "international_law",
            "jurisprudence", "logical_fallacies", "machine_learning",
            "management", "marketing", "medical_genetics", "miscellaneous",
            "moral_disputes", "moral_scenarios", "nutrition", "philosophy",
            "prehistory", "professional_accounting", "professional_law",
            "professional_medicine", "professional_psychology",
            "public_relations", "security_studies", "sociology",
            "us_foreign_policy", "virology", "world_religions"]
        for sub in subjects:
            d = load_dataset("cais/mmlu", sub, split="test")
            for x in d:
                opts = "\n".join(f"{c}. {t}" for c, t in
                                 zip("ABCD", x["choices"]))
                items.append({
                    "prompt": f"{x['question'].strip()}\n{opts}\n\n"
                              "Answer with only the letter (A, B, C or D) "
                              "of the correct option.",
                    "gold": "ABCD"[x["answer"]], "subject": sub})
    elif subject == "humaneval":
        d = load_dataset("openai/openai_humaneval", split="test")
        for x in d:
            items.append({
                "prompt": "Complete the following Python function. "
                          "Reply with the FULL function implementation "
                          "(imports + def) in a single ```python code "
                          "block.\n\n```python\n" + x["prompt"] + "```",
                "gold": "", "task_id": x["task_id"],
                "he_prompt": x["prompt"], "test": x["test"],
                "entry_point": x["entry_point"]})
    else:
        raise SystemExit(f"unknown subject {subject}")
    return items


NUM_RE = re.compile(r"-?[\d,]*\.?\d+")
LETTER_RE = re.compile(r"\b([ABCD])\b")
CODE_RE = re.compile(r"```(?:python)?\n(.*?)```", flags=re.S)


def norm_num(s):
    s = s.replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
        if not math.isfinite(f):
            return s
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, OverflowError):
        return s


def score_gsm8k(resp, item):
    tail = resp.rsplit("####", 1)[-1] if "####" in resp else resp
    nums = NUM_RE.findall(tail) or NUM_RE.findall(resp)
    if not nums:
        return 0
    return int(norm_num(nums[-1]) == norm_num(item["gold"]))


def score_mmlu(resp, item):
    m = re.search(r"[Aa]nswer[^A-D]{0,10}([ABCD])\b", resp)
    if not m:
        m = LETTER_RE.search(resp)
    return int(bool(m) and m.group(1) == item["gold"])


def score_humaneval(resp, item):
    blocks = CODE_RE.findall(resp)
    code = max(blocks, key=len) if blocks else resp
    if f"def {item['entry_point']}" not in code:
        code = item["he_prompt"] + "\n" + code
    prog = (code + "\n\n" + item["test"]
            + f"\n\ncheck({item['entry_point']})\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=10)
        return int(p.returncode == 0)
    except subprocess.TimeoutExpired:
        return 0
    finally:
        Path(path).unlink(missing_ok=True)


SCORERS = {"gsm8k": score_gsm8k, "mmlu": score_mmlu,
           "humaneval": score_humaneval}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True,
                    choices=("gsm8k", "mmlu", "humaneval"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="a1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--workers", type=int, default=128,
                    help="併發請求數;2026-08-20 起預設 128(舊座標 32)")
    ap.add_argument("--limit", type=int, default=0,
                    help="triage 層:seed 抽 N 題(seed=base+原始索引,"
                         "為全量語義嚴格子集);只授權提前止損,不授權裁決")
    ap.add_argument("--skip-gen", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_items(args.subject)
    n_full = len(items)
    idxs = list(range(n_full))
    if args.limit and args.limit < n_full:
        import random
        idxs = sorted(random.Random(args.seed).sample(idxs, args.limit))
        items = [items[i] for i in idxs]
    resp_path = out_dir / "responses.jsonl"

    if args.skip_gen:
        gen_s = None
        responses = [json.loads(l)["response"] for l in resp_path.open()]
        assert len(responses) == len(items)
    else:
        print(f"{args.subject}: {len(items)} prompts → {args.base_url}",
              flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            responses = list(ex.map(
                lambda iv: gen_one(args.base_url, args.model,
                                   iv[1]["prompt"], args.seed + iv[0],
                                   args.max_tokens),
                zip(idxs, items)))
        gen_s = round(time.time() - t0, 1)
        print(f"generation done in {gen_s}s", flush=True)
        with resp_path.open("w") as f:
            for it, r in zip(items, responses):
                f.write(json.dumps({"prompt": it["prompt"], "response": r},
                                   ensure_ascii=False) + "\n")

    scorer = SCORERS[args.subject]
    scores = [scorer(r, it) for r, it in zip(responses, items)]
    summary = {"subject": args.subject, "n": len(items),
               "score": round(sum(scores) / len(scores), 6),
               "seed": args.seed,
               "sampling": {**{k: v for k, v in SAMPLING.items()
                               if k != "extra_body"},
                            **SAMPLING["extra_body"]},
               "max_tokens": args.max_tokens, "generation_s": gen_s,
               "workers": args.workers,
               "scoring": "self-built A1-protocol scorer(見模組 docstring)"}
    if args.limit and args.limit < n_full:
        summary["triage"] = {"limit": args.limit, "n_full": n_full,
                             "discipline": "只授權提前止損,終裁一律全量"}
    if args.subject == "mmlu":
        per = {}
        for it, s in zip(items, scores):
            a, b = per.get(it["subject"], (0, 0))
            per[it["subject"]] = (a + s, b + 1)
        summary["per_subject"] = {k: round(a / b, 4)
                                  for k, (a, b) in sorted(per.items())}
    (out_dir / "scores.jsonl").write_text(
        "\n".join(str(s) for s in scores))
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("subject", "n", "score")},
                     indent=1), flush=True)
    print("wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()

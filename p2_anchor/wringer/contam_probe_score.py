"""模型端記憶探針計分:對 contam_probe_gen 的輸出算 (a) 生成 vs 參考後綴的逐字前綴長(詞級)、(b) 最長共同詞串;
另對官方 HumanEval 跑的 responses.jsonl 算回應 vs canonical_solution 的最長共同詞串(解答重現)。
輸出 evidence/p1_grouping/corkscrew/contam_probe_<tag>.{json,md}

    python -m p2_anchor.wringer.contam_probe_score --tag qwen3_4b --probe evidence/p1_grouping/a1eval/probe_qwen3_4b \
        --he-run evidence/p1_grouping/a1eval/humaneval_qwen3_4b_bf16 [--control-tag a1 ...]
"""
import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HUB = Path.home() / ".cache/huggingface/hub"
OUT = ROOT / "evidence/p1_grouping/corkscrew"
TOK = re.compile(r"\w+|[^\w\s]")


def toks(s):
    return [t.lower() for t in TOK.findall(s)]


def exact_prefix(gen, ref):
    g, r = toks(gen), toks(ref)
    n = 0
    while n < min(len(g), len(r)) and g[n] == r[n]:
        n += 1
    return n


def longest_common_run(a, b):
    A, B = toks(a), toks(b)
    if not A or not B:
        return 0
    idx = {}
    for j, t in enumerate(B):
        idx.setdefault(t, []).append(j)
    best = 0
    prev = {}
    for i, t in enumerate(A):
        cur = {}
        for j in idx.get(t, ()):
            L = prev.get(j - 1, 0) + 1
            cur[j] = L
            best = max(best, L)
        prev = cur
    return best


def score_probe(pdir):
    res = {}
    for name in ("ifeval", "humaneval", "gsm8k"):
        f = Path(pdir) / f"{name}.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in f.open()]
        ep = [exact_prefix(r["gen"], r["ref"]) for r in rows]
        lr = [longest_common_run(r["gen"], r["ref"]) for r in rows]
        res[name] = {"n": len(rows),
                     "exact_prefix_mean": round(sum(ep) / len(ep), 2),
                     "exact_prefix>=8": sum(x >= 8 for x in ep), "exact_prefix>=20": sum(x >= 20 for x in ep),
                     "lcr_mean": round(sum(lr) / len(lr), 2), "lcr>=13": sum(x >= 13 for x in lr), "lcr>=25": sum(x >= 25 for x in lr),
                     "top": sorted([(x, r["id"]) for x, r in zip(ep, rows)], reverse=True)[:10]}
    return res


def score_he_run(run_dir):
    he = pd.read_parquet(glob.glob(str(HUB / "datasets--openai--openai_humaneval/snapshots/*/openai_humaneval/test-*.parquet"))[0])
    sols = dict(zip(he["task_id"], he["canonical_solution"]))
    rows = [json.loads(l) for l in (Path(run_dir) / "responses.jsonl").open()]
    tids = list(he["task_id"])
    lr = [longest_common_run(r.get("response", ""), sols[t]) for r, t in zip(rows, tids)]
    return {"n": len(rows), "lcr_mean": round(sum(lr) / len(lr), 2), "lcr>=13": sum(x >= 13 for x in lr),
            "lcr>=25": sum(x >= 25 for x in lr), "lcr>=50": sum(x >= 50 for x in lr),
            "top": sorted([(x, t) for x, t in zip(lr, tids)], reverse=True)[:10]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--he-run", required=True)
    ap.add_argument("--control-tag", default=None)
    ap.add_argument("--control-probe", default=None)
    ap.add_argument("--control-he-run", default=None)
    args = ap.parse_args()
    rep = {"model": {"tag": args.tag, "probe": score_probe(ROOT / args.probe), "he_run": score_he_run(ROOT / args.he_run)}}
    if args.control_probe:
        rep["control"] = {"tag": args.control_tag, "probe": score_probe(ROOT / args.control_probe), "he_run": score_he_run(ROOT / args.control_he_run)}
    (OUT / f"contam_probe_{args.tag}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    L = [f"# 模型端記憶探針 {args.tag}" + (f"(對照 {args.control_tag})" if args.control_probe else ""), "",
         "guided completion:參考文字前 40% 詞當前綴(raw completions,temperature 0),量生成對參考後綴的逐字前綴長(exact_prefix,詞)與最長共同詞串(lcr)。",
         "he_run:官方 HumanEval 回應 vs canonical_solution 的最長共同詞串。", "",
         "| 模型 | 集合 | n | exact_prefix 均值 | ≥8 | ≥20 | lcr 均值 | ≥13 | ≥25 |", "|---|---|---|---|---|---|---|---|---|"]
    for who in ("model", "control"):
        if who not in rep:
            continue
        for name, r in rep[who]["probe"].items():
            L.append(f"| {rep[who]['tag']} | {name} | {r['n']} | {r['exact_prefix_mean']} | {r['exact_prefix>=8']} | {r['exact_prefix>=20']} | {r['lcr_mean']} | {r['lcr>=13']} | {r['lcr>=25']} |")
    L += ["", "| 模型 | HumanEval 回應 vs canonical | lcr 均值 | ≥13 | ≥25 | ≥50 |", "|---|---|---|---|---|---|"]
    for who in ("model", "control"):
        if who in rep:
            r = rep[who]["he_run"]
            L.append(f"| {rep[who]['tag']} | n={r['n']} | {r['lcr_mean']} | {r['lcr>=13']} | {r['lcr>=25']} | {r['lcr>=50']} |")
    L += ["", "top exact_prefix(model):"] + [f"- {name}: {r['top'][:5]}" for name, r in rep["model"]["probe"].items()]
    (OUT / f"contam_probe_{args.tag}.md").write_text("\n".join(L) + "\n")
    print("PROBE_SCORE_DONE " + json.dumps({k: {n: (v["exact_prefix>=20"], v["lcr>=25"]) for n, v in rep[k]["probe"].items()} for k in rep}), flush=True)


if __name__ == "__main__":
    main()

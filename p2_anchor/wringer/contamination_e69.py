"""E69 污染檢查:蓄水/校準語料 vs 三科官方測試集的 n-gram 重疊。

語料來源(兩路各自報告):
  calib   evidence/p1_grouping/calib_e58r1b_traj.pt(384 列 ×16k token,fill 全用;quantize 用前 128 列)
          → 以模型 tokenizer 解碼回文字
  bank    evidence/p1_grouping/bleachers-e58r2/{code,ifshape,math}.jsonl 的 prompt + teacher_solution
測試集(HF cache):IFEval prompt;HumanEval prompt / canonical_solution / test;GSM8K question / answer。

指標(詞級 token = \\w+ 或單一標點,小寫):
  hit13   測試項含任一 13-gram 出現於語料(GPT-3 式判準)
  run≥N   測試項與語料的最長連續共同 token 串 ≥ N(以 8-gram 鏈接估計)
  he_def  HumanEval entry_point 以 `def name(` 形式出現在語料的題數(語意撞題參考)
輸出:evidence/p1_grouping/corkscrew/contamination_e69.{json,md}
"""
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "evidence/p1_grouping"
OUT = EV / "corkscrew"
HUB = Path.home() / ".cache/huggingface/hub"
TOK = re.compile(r"\w+|[^\w\s]")
NG = 8            # 鏈接用 n-gram
RUNS = (13, 25, 50, 100)


def toks(s):
    return [t.lower() for t in TOK.findall(s)]


def grams(ts, n=NG):
    return (hash(tuple(ts[i:i + n])) for i in range(len(ts) - n + 1))


def load_corpus():
    src = {}
    # bank
    for dom in ("code", "ifshape", "math"):
        for i, line in enumerate((EV / "bleachers-e58r2" / f"{dom}.jsonl").open()):
            d = json.loads(line)
            src[f"bank:{dom}:{d.get('id', i)}"] = d.get("prompt", "") + "\n" + d.get("teacher_solution", "")
    # calib(解碼)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "exports/e69_p3b_w"))
    data = torch.load(EV / "calib_e58r1b_traj.pt", weights_only=False)
    lens = torch.load(EV / "calib_e58r1b_len.pt", weights_only=False)
    for r in range(data.shape[0]):
        ids = data[r, : int(lens[r])].tolist()
        src[f"calib:{r}"] = tok.decode(ids, skip_special_tokens=False)
    return src


def build_index(src, prefix):
    idx = {}
    ntok = 0
    for k, text in src.items():
        if not k.startswith(prefix):
            continue
        ts = toks(text)
        ntok += len(ts)
        for h in grams(ts):
            idx.setdefault(h, k)
    return idx, ntok


def longest_run(ts, idx):
    """最長連續被語料 8-gram 覆蓋的 token 串長(及來源)。"""
    best, best_src = 0, None
    cur, cur_src = 0, None
    for i in range(len(ts) - NG + 1):
        s = idx.get(hash(tuple(ts[i:i + NG])))
        if s is not None:
            cur = cur + 1 if cur else NG
            cur_src = cur_src or s
            if cur > best:
                best, best_src = cur, cur_src
        else:
            cur, cur_src = 0, None
    return best, best_src


def has13(ts, idx):
    for i in range(len(ts) - 13 + 1):
        w = ts[i:i + 13]
        # 13-gram 命中 ⇔ 其內 6 個相鄰 8-gram 全命中
        if all(hash(tuple(w[j:j + NG])) in idx for j in range(13 - NG + 1)):
            return True
    return False


def load_tests():
    T = {}
    ifp = glob.glob(str(HUB / "datasets--google--IFEval/snapshots/*/ifeval_input_data.jsonl"))[0]
    T["ifeval:prompt"] = [(str(d["key"]), d["prompt"]) for d in map(json.loads, open(ifp))]
    he = pd.read_parquet(glob.glob(str(HUB / "datasets--openai--openai_humaneval/snapshots/*/openai_humaneval/test-*.parquet"))[0])
    for col in ("prompt", "canonical_solution", "test"):
        T[f"humaneval:{col}"] = list(zip(he["task_id"], he[col]))
    gs = pd.read_parquet(glob.glob(str(HUB / "datasets--openai--gsm8k/snapshots/*/main/test-*.parquet"))[0])
    for col in ("question", "answer"):
        T[f"gsm8k:{col}"] = [(str(i), s) for i, s in enumerate(gs[col])]
    he_entry = list(zip(he["task_id"], he["entry_point"]))
    return T, he_entry


def main():
    src = load_corpus()
    T, he_entry = load_tests()
    report = {"corpus": {}, "tables": {}, "examples": {}, "he_def": {}}
    for prefix in ("bank", "calib"):
        idx, ntok = build_index(src, prefix)
        report["corpus"][prefix] = {"docs": sum(k.startswith(prefix) for k in src), "word_tokens": ntok, "ngrams8": len(idx)}
        print(f"[{prefix}] docs={report['corpus'][prefix]['docs']} tokens={ntok} 8grams={len(idx)}", flush=True)
        # he_def:entry_point 撞名
        alltext = "\n".join(v for k, v in src.items() if k.startswith(prefix))
        hits = [tid for tid, ep in he_entry if re.search(rf"\bdef\s+{re.escape(ep)}\s*\(", alltext)]
        report["he_def"][prefix] = {"n": len(hits), "ids": hits}
        for tname, items in T.items():
            n = len(items)
            c13 = 0
            runs = []
            ex = []
            for tid, text in items:
                ts = toks(text)
                r, s = longest_run(ts, idx)
                h = has13(ts, idx) if r >= 13 else False
                c13 += h
                runs.append(r)
                if r >= 13:
                    ex.append({"id": tid, "run": r, "src": s, "snippet": " ".join(ts[:40])})
            row = {"n": n, "hit13": c13, "hit13_frac": round(c13 / n, 4), "max_run": max(runs), "mean_run": round(sum(runs) / n, 2)}
            for N in RUNS:
                row[f"run>={N}"] = sum(r >= N for r in runs)
            report["tables"][f"{prefix}|{tname}"] = row
            report["examples"][f"{prefix}|{tname}"] = sorted(ex, key=lambda e: -e["run"])[:10]
            print(f"  {tname:28s} {row}", flush=True)
        del idx
    (OUT / "contamination_e69.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    # markdown
    L = ["# E69 污染檢查(calib_e58r1b / bleachers-e58r2 vs IFEval / HumanEval / GSM8K)", "",
         "詞級 token(\\w+ 或單標點,小寫);hit13 = 含任一 13-gram 出現於語料;run≥N = 最長連續共同串 ≥ N token。", ""]
    for p, c in report["corpus"].items():
        L.append(f"- 語料 `{p}`:{c['docs']} 文件,{c['word_tokens']:,} 詞,{c['ngrams8']:,} 個 8-gram;HumanEval entry_point `def name(` 撞名 {report['he_def'][p]['n']}/164")
    L += ["", "| 語料 | 測試欄位 | n | hit13 | hit13% | run≥13 | run≥25 | run≥50 | run≥100 | max_run | mean_run |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in report["tables"].items():
        p, t = k.split("|")
        L.append(f"| {p} | {t} | {r['n']} | {r['hit13']} | {r['hit13_frac']*100:.1f} | {r['run>=13']} | {r['run>=25']} | {r['run>=50']} | {r['run>=100']} | {r['max_run']} | {r['mean_run']} |")
    L += ["", "## 命中樣例(run ≥ 13,每欄最多 10)"]
    for k, ex in report["examples"].items():
        if not ex:
            continue
        L.append(f"### {k}")
        for e in ex:
            L.append(f"- id={e['id']} run={e['run']} src={e['src']} :: `{e['snippet'][:160]}`")
    (OUT / "contamination_e69.md").write_text("\n".join(L) + "\n")
    print("CONTAM_DONE", flush=True)


if __name__ == "__main__":
    main()

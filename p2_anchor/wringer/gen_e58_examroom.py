"""E58 產線 v4:還原考場行為(用戶 08-29 裁定必修)。

cal4→cal10 全系譜定罪:sft() 偽模板 `<|user|>` 非特殊 token(tokenize 成
5 個普通字元),教材無 system/im_start/<think> 前綴、教師在非對話上下文
不發 <|im_end|>(=eos)→ EOS 饑荒 76.7% 的頭號嫌疑;student 學到 teacher
不熟的生成格式 = 失律候選遠因。

四落實點(ROADMAP_post_e57 §2b):
  1. 真模板:tok.apply_chat_template(add_generation_prompt=True)
     (含 system + <|im_start|>user…<|im_end|> + assistant <think>\\n 前綴)
  2. 採樣對齊考場:temp 1.0/top_p .95/top_k 20/min_p 0/presence 1.5
     (照 a1eval ifeval_runner.SAMPLING;採樣改軌跡抽樣不改 KD 條件分佈)
  3. 終止訊號入列:eos=<|im_end|>(248046),finish_reason=="stop" 時
     手動補 im_end token 於列尾(vLLM 回文不含 eos);length 列照實
     不補假 EOS,入手術/淘汰池
  4. 儀器:manifest 記分科 finish_reason 分佈(natural_stop_rate)、
     </think> 閉合、長度分位;--probe N 產前探針

域:ifshape/code/math 沿 gen_cal6 題池(去 sft 取原題);agentic 改接
27B 出題槽(bleachers v3,任務 #73)——本腳本 --domains 不含 agentic,
題池就緒後以 --agentic-bank 傳入 jsonl(欄位 prompt)。

  探針(50 列/域,~10min):
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.gen_e58_examroom \\
      --probe 50 --domains ifshape,code,math
  正式(E58 R1 FROZEN):
    --per-domain-counts code:6700,ifshape:4400,math:11000 --pack --out-tag e58r1

--pack(E58 R1 凍結裁定):整集(episode)打包進 16384 窗——v4 自然列中位
1.3-5.8k,遮罩填充下 8k+ 位置零梯度=KL 裁定的尾段增益(12-16k 比 2-2.8×)
整段餓死,且 fill 步數 3×;打包=域內整集順序集裝(不跨集切割),
finish!=stop 集淘汰照實計數(v4 探針 98-100% stop,損耗 ~2%)。
"""
import argparse
import hashlib
import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer

from p1_grouping.modelio import MODEL_ID, MODEL_REVISION

EV = Path("evidence/p1_grouping")
SEED = 20260829
THINK_END = 248069          # </think>
IM_END = 248046             # <|im_end|> = eos
MAX_TOK = 16384
ROW_LEN = 16384

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "min_p": 0.0, "presence_penalty": 1.5}   # = a1eval 考場參數


def wrap_exam(tok, q):
    return tok.apply_chat_template([{"role": "user", "content": q}],
                                   add_generation_prompt=True,
                                   tokenize=False)


def gen_one_exam(base_url, model, prompt, max_tokens, seed):
    r = requests.post(f"{base_url}/completions", json={
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "seed": seed, **SAMPLING}, timeout=1800)
    r.raise_for_status()
    c = r.json()["choices"][0]
    return c["text"], c.get("finish_reason", "?")


def bank_ifshape():
    xs = []
    for d in (EV / "bleachers-20260814", EV / "bleachers-e42"):
        p = d / "ifshape.jsonl"
        if p.exists():
            xs += [json.loads(ln)["prompt"]
                   for ln in p.read_text().splitlines()]
    return xs


def bank_code():
    from datasets import load_dataset
    from p2_anchor.wringer.gen_selfcorpus_code import extract_seeds
    qs = []
    for split in ("train", "validation", "test", "prompt"):
        for x in load_dataset("google-research-datasets/mbpp", split=split):
            qs.append(f"Write a Python function for the following task.\n"
                      f"{x['text']}\nYour code should pass this test:\n"
                      f"{x['test_list'][0]}")
    for d in (EV / "bleachers-20260814", EV / "bleachers-e42"):
        for name in ("code.jsonl", "code_mbpp.jsonl"):
            p = d / name
            if p.exists():
                qs += ["Complete this Python function:\n\n"
                       + json.loads(ln)["prompt"]
                       for ln in p.read_text().splitlines()]
    qs += [s for s in extract_seeds(2400, 20260817)
           if ">>>" in s or "Example" in s or "example" in s]
    return qs


def bank_math():
    from datasets import load_dataset
    qs = [x["question"]
          for x in load_dataset("openai/gsm8k", "main", split="train")]
    qs = random.Random(SEED).sample(qs, 6600)
    for d in (EV / "bleachers-20260814", EV / "bleachers-e42"):
        p = d / "math.jsonl"
        if p.exists():
            qs += [json.loads(ln)["prompt"]
                   for ln in p.read_text().splitlines()]
    return qs


BANKS = {"ifshape": bank_ifshape, "code": bank_code, "math": bank_math}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="a1")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--per-domain", type=int, default=1800)
    ap.add_argument("--per-domain-counts", default=None,
                    help="分域題數,如 code:6700,ifshape:4400,math:11000")
    ap.add_argument("--pack", action="store_true",
                    help="整集打包 16k 窗(stop 集 only;見模組 docstring)")
    ap.add_argument("--probe", type=int, default=0,
                    help="探針模式:每域 N 列,只出統計不落盤語料")
    ap.add_argument("--domains", default="ifshape,code,math")
    ap.add_argument("--agentic-bank", default=None,
                    help="27B 出題槽 jsonl(欄位 prompt);就緒後併入")
    ap.add_argument("--out-tag", default="e58")
    args = ap.parse_args()
    n_dom = args.probe or args.per_domain
    dom_counts = {}
    if args.per_domain_counts:
        dom_counts = {k: int(v) for k, v in
                      (t.split(":") for t in
                       args.per_domain_counts.split(","))}

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    global THINK_END, IM_END                      # E70:特殊 token id 隨模型 tokenizer(A1 248069/248046;Qwen3 151668/151645)
    THINK_END = tok.convert_tokens_to_ids("</think>")
    IM_END = tok.convert_tokens_to_ids("<|im_end|>")
    print(f"special ids: </think>={THINK_END} <|im_end|>={IM_END} model={MODEL_ID}", flush=True)
    rows, lens, dom_stats = [], [], {}
    names = [d for d in args.domains.split(",") if d]
    for di, name in enumerate(names):
        if name == "agentic":
            assert args.agentic_bank, "agentic 需 --agentic-bank(27B 出題槽)"
            qs = [json.loads(ln)["prompt"] for ln
                  in Path(args.agentic_bank).read_text().splitlines()]
        else:
            qs = BANKS[name]()
        pool_n = len(qs)
        rng = random.Random(SEED + di)
        rng.shuffle(qs)
        qs = qs[:dom_counts.get(name, n_dom)]
        prompts = [wrap_exam(tok, q) for q in qs]
        outs = []
        for w0 in range(0, len(prompts), 512):
            batch = prompts[w0:w0 + 512]
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                outs += list(ex.map(
                    lambda p: gen_one_exam(args.base_url, args.model, p[1],
                                           MAX_TOK,
                                           SEED + di * 10000 + w0 + p[0]),
                    enumerate(batch)))
            print(f"  {name} {min(w0 + 512, len(prompts))}/{len(prompts)}",
                  flush=True)
        closed = stopped = in_win = culled = 0
        glens, episodes = [], []
        rows0 = len(rows)
        for p, (o, fr) in zip(prompts, outs):
            ids = tok(p + o, add_special_tokens=False)["input_ids"]
            closed += "</think>" in o
            glens.append(len(ids))
            if fr == "stop":
                ids = ids + [IM_END]      # 終止訊號入列(vLLM 回文不含 eos)
                stopped += 1
            if args.pack:
                if fr != "stop" or len(ids) > ROW_LEN:
                    culled += 1           # 無 EOS/超窗集淘汰,照實計數
                    continue
                in_win += THINK_END in ids
                episodes.append(ids)
            else:
                row = ids[:ROW_LEN]
                in_win += THINK_END in row
                rows.append(row + [0] * (ROW_LEN - len(row)))
                lens.append(len(row))
        if args.pack:                     # 域內整集順序集裝,不跨集切割
            cur = []
            for ep in episodes:
                if len(cur) + len(ep) > ROW_LEN:
                    rows.append(cur + [0] * (ROW_LEN - len(cur)))
                    lens.append(len(cur))
                    cur = []
                cur += ep
            if cur:
                rows.append(cur + [0] * (ROW_LEN - len(cur)))
                lens.append(len(cur))
        q = ([int(x) for x in statistics.quantiles(glens, n=4)]
             if len(glens) > 1 else glens)
        dom_stats[name] = {
            "n_gen": len(qs), "bank_pool": pool_n,
            "natural_stop": stopped,
            "natural_stop_rate": round(stopped / max(len(qs), 1), 4),
            "closed_think": closed,
            "closure_rate": round(closed / max(len(qs), 1), 4),
            "closed_in_window": in_win, "traj_len_quartiles": q,
            "culled": culled, "packed_rows": len(rows) - rows0}
        print(f"{name}: stop {stopped}/{len(qs)}"
              f"({dom_stats[name]['natural_stop_rate']:.1%}) "
              f"closure {dom_stats[name]['closure_rate']:.1%} len_q {q}",
              flush=True)

    if args.probe:
        print("E58_PROBE_DONE:" + json.dumps(
            {d: s["natural_stop_rate"] for d, s in dom_stats.items()}))
        Path(EV / f"E58_PROBE_{args.probe}.json").write_text(
            json.dumps(dom_stats, ensure_ascii=False, indent=1))
        return

    t = torch.tensor(rows, dtype=torch.long)
    L = torch.tensor(lens, dtype=torch.long)
    perm = torch.randperm(len(rows),
                          generator=torch.Generator().manual_seed(SEED))
    t, L = t[perm], L[perm]
    torch.save(t, EV / f"calib_{args.out_tag}_traj.pt")
    torch.save(L, EV / f"calib_{args.out_tag}_len.pt")
    sha = hashlib.sha256(t.numpy().tobytes()).hexdigest()
    man = {"recipe": "examroom v4(真模板+考場採樣+im_end 入列"
                     + ("+整集打包 16k 窗" if args.pack else "") + ")",
           "seed": SEED, "sampling": SAMPLING, "max_tok": MAX_TOK,
           "row_len": ROW_LEN, "rows": len(rows),
           "tokens_valid": int(L.sum()), "sha256": sha,
           "per_domain": dom_stats}
    Path(EV / f"E58_TRAJ_MANIFEST_{args.out_tag}.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=1))
    print(f"rows {len(rows)}  tokens {int(L.sum())/1e6:.1f}M  sha {sha[:8]}")
    print("E58_GEN_DONE")


if __name__ == "__main__":
    main()

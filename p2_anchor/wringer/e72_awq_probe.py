"""E72 AWQ falsifier 查證探針:Qwen3-4B awq4 GS 1319 題中 484 題 content 為空(usage 250–16k token、finish stop)。
假說:整段輸出留在思考段、以 EOS 結束而未吐 </think>,reasoning parser 把 content 留空。
本探針直接打 chat/completions,同 runner 採樣與 seed(seed=base+idx),抓 reasoning_content + content,
  --mode think   :同協定;統計 content 空 / 思考段非空且 finish stop(=EOS in think)/ 抽取正確率
  --mode nothink :chat_template_kwargs enable_thinking=False;同題正確率(數學能力是否完好的儀器)
輸出 JSON:n, empty_content, eos_in_think, acc, tok_mean, samples(前 5 筆空答的思考段尾 200 字)。儀器,不入裁決。
"""
import argparse, json, statistics, time
from concurrent.futures import ThreadPoolExecutor
import requests
from p2_anchor.a1eval.ifeval_runner import SAMPLING, THINK_RE
from p2_anchor.a1eval.subject_runner import load_items, score_gsm8k


def one(base_url, model, prompt, seed, max_tokens, nothink):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "seed": seed,
            **{k: v for k, v in SAMPLING.items() if k != "extra_body"}, **SAMPLING["extra_body"]}
    if nothink:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    for attempt in range(3):
        try:
            r = requests.post(f"{base_url}/chat/completions", json=body, timeout=1800)
            r.raise_for_status()
            j = r.json(); msg = j["choices"][0]["message"]
            return {"content": THINK_RE.sub("", msg.get("content") or "").strip(),
                    "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                    "finish": j["choices"][0].get("finish_reason"),
                    "tok": (j.get("usage") or {}).get("completion_tokens")}
        except Exception as e:
            if attempt == 2:
                return {"content": "", "reasoning": "", "finish": f"error:{e}", "tok": None}
            time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1"); ap.add_argument("--model", default="a1")
    ap.add_argument("--n", type=int, default=150); ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--max-tokens", type=int, default=16384); ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--mode", choices=("think", "nothink"), default="think"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = load_items("gsm8k")[: a.n]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rs = list(ex.map(lambda iv: one(a.base_url, a.model, iv[1]["prompt"], a.seed + iv[0], a.max_tokens, a.mode == "nothink"), enumerate(items)))
    empty = [r for r in rs if not r["content"]]
    eos_in_think = [r for r in empty if r["reasoning"].strip() and r["finish"] == "stop"]
    acc = sum(score_gsm8k(r["content"], it) for r, it in zip(rs, items)) / len(items)
    toks = [r["tok"] for r in rs if isinstance(r["tok"], int)]
    res = {"mode": a.mode, "n": len(items), "empty_content": len(empty), "eos_in_think": len(eos_in_think),
           "empty_len_trunc": sum(1 for r in empty if r["finish"] == "length"),
           "acc": round(acc, 4), "tok_mean": round(statistics.mean(toks), 1) if toks else None,
           "finish": {k: sum(1 for r in rs if r["finish"] == k) for k in set(r["finish"] for r in rs)},
           "samples": [{"finish": r["finish"], "tok": r["tok"], "reasoning_tail": r["reasoning"][-200:]} for r in empty[:5]]}
    json.dump({"result": res, "rows": rs}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("PROBE_RESULT " + json.dumps({k: v for k, v in res.items() if k != "samples"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

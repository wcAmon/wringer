"""自生成退火語料(E33-pre-2;LLM-QAT 路線):
bf16 teacher 對 cal2 各序列前 seed-tokens 個 token 續寫,seed+續寫重新
tokenize 打包成 n×len。行為分佈錨定:KD 語料=模型自己的軌跡,非原始語料。
IFEval prompts 不進種子(防污染;種子只來自 agentic 校準集)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.gen_selfcorpus \
    --base-url http://127.0.0.1:8000/v1 --model a1
"""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer

from p1_grouping.modelio import MODEL_ID, MODEL_REVISION

EV = Path("evidence/p1_grouping")


def gen_one(base_url, model, prompt, max_tokens, seed):
    r = requests.post(f"{base_url}/completions", json={
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 1.0, "top_p": 0.95, "seed": seed}, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="a1")
    ap.add_argument("--src", default=str(EV / "calib_train_512_v2.pt"))
    ap.add_argument("--out", default=str(EV / "calib_selfgen_512.pt"))
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--len", dest="seq_len", type=int, default=2048)
    ap.add_argument("--seed-tokens", type=int, default=256)
    ap.add_argument("--gen-tokens", type=int, default=1900)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260811)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    src = torch.load(args.src, weights_only=True)
    need = args.n * args.seq_len
    ids: list[int] = []
    rounds = 0
    while len(ids) < need and rounds < 4:
        off = rounds * args.seed_tokens          # 每輪換種子切片,避免重複
        prompts = [tok.decode(src[i, off:off + args.seed_tokens].tolist())
                   for i in range(src.shape[0])]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            outs = list(ex.map(
                lambda p: gen_one(args.base_url, args.model, p[1],
                                  args.gen_tokens, args.seed + p[0]),
                enumerate(prompts)))
        for p, o in zip(prompts, outs):
            ids.extend(tok(p + o, add_special_tokens=False)["input_ids"])
        rounds += 1
        print(f"round {rounds}: {len(ids)}/{need} tokens", flush=True)
    assert len(ids) >= need, f"自生語料不足:{len(ids)} < {need}"
    t = torch.tensor(ids[:need], dtype=torch.long).reshape(args.n, args.seq_len)
    out = Path(args.out)
    torch.save(t, out)
    rec = {"teacher": "bf16(vLLM serve)", "seed_src": args.src,
           "seed_tokens": args.seed_tokens, "gen_tokens": args.gen_tokens,
           "sampling": {"temperature": 1.0, "top_p": 0.95},
           "rounds": rounds, "seed": args.seed, "shape": list(t.shape),
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
           "note": "E33-pre-2 自生語料;種子=cal2 前 256 token 切片,"
                   "IFEval prompts 不進種子(防污染)",
           "model_revision": MODEL_REVISION}
    (EV / "CALIB_SELFGEN_MANIFEST.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({k: rec[k] for k in ("shape", "sha256", "rounds")},
                     indent=1))


if __name__ == "__main__":
    main()

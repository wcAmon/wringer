"""cal5 v2:bleachers-20260814 題庫 + 人寫庫 → 4B bf16 作答 → 打包。

四域 10240 列(prereg_e40_scale v2):
  agentic 3584 — cal3 同源種子續寫(E39 已證配方,不動)
  ifshape 1536 — bleachers ifshape 題面(官方檢查器已驗)×k3
  code    2560 — MBPP 指令形 k3 + bleachers code/code_mbpp(humaneval 形)
                 + starcoder 自足 k3
  math    2560 — gsm8k-train 子採樣 + bleachers math ×k2(約對半)
行為資料 = 4B bf16 對題面的續寫;Max 解答只在題庫,不進本檔。
除汙 = gsm8k-test + humaneval + IFEval 全集(bank.decontam_grams)。

  CPU 預驗:--extract-only
  分域流水(題庫先到先產,GPU 不空等;--domains 逗號清單,存 part 檔):
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.gen_cal5v2 \\
      --base-url http://127.0.0.1:8000/v1 --model a1 --domains agentic,ifshape
  四域 part 齊後合裝(列級 shuffle → calib_selfgen_v5.pt + manifest):
    ... gen_cal5v2 --assemble
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from p1_grouping.calib import build
from p1_grouping.modelio import MODEL_ID, MODEL_REVISION
from p2_anchor.bleachers.bank import OUT_DIR as BLE_DIR
from p2_anchor.bleachers.bank import decontam_grams
from p2_anchor.wringer.build_calib512_v3 import DOMAINS
from p2_anchor.wringer.gen_cal4 import SEQ_LEN, gen_domain, sft
from p2_anchor.wringer.gen_selfcorpus_code import extract_seeds

EV = Path("evidence/p1_grouping")
SEED = 20260814
ROWS = {"agentic": 3584, "ifshape": 1536, "code": 2560, "math": 2560}
MARGIN = {"agentic": 0.10, "ifshape": 0.12, "code": 0.15, "math": 0.10}
GEN_TOK = {"agentic": 1900, "ifshape": 900, "code": 800, "math": 600}


def _grams8(text):
    import re
    words = re.sub(r"\s+", " ", text.lower()).strip().split(" ")
    return {" ".join(words[i:i + 8]) for i in range(len(words) - 7)}


def load_bank(name):
    p = BLE_DIR / f"{name}.jsonl"
    return [json.loads(ln) for ln in p.read_text().splitlines()]


def bank_agentic(tok, n=2000, k=3):
    d = dict(DOMAINS["agentic"])
    t = build(d["sources"], tok, n, SEQ_LEN, d["seed"] + 2,
              lines_per_file=24000)
    seeds = [tok.decode(t[i, :256].tolist()) for i in range(n)]
    return seeds * k, {"seed_rows": n, "seed_k": k, "source": "cal3 同源"}


def bank_ifshape(k=3):
    xs = load_bank("ifshape")
    return [sft(x["prompt"]) for x in xs] * k, \
        {"bleachers_ifshape": len(xs), "k": k}


def bank_code(k_mbpp=3, k_star=3, k_ble=2):
    from datasets import load_dataset
    prompts = []
    n_mb = 0
    for split in ("train", "validation", "test", "prompt"):
        for x in load_dataset("google-research-datasets/mbpp", split=split):
            p = (f"Write a Python function for the following task.\n"
                 f"{x['text']}\nYour code should pass this test:\n"
                 f"{x['test_list'][0]}")
            prompts += [sft(p)] * k_mbpp
            n_mb += 1
    ble = load_bank("code") + load_bank("code_mbpp")
    prompts += [sft("Complete this Python function:\n\n" + x["prompt"])
                for x in ble] * k_ble
    star = [s for s in extract_seeds(2400, 20260817)
            if ">>>" in s or "Example" in s or "example" in s]
    prompts += [sft(s) for s in star] * k_star
    return prompts, {"mbpp": n_mb, "mbpp_k": k_mbpp,
                     "bleachers_code": len(ble), "ble_k": k_ble,
                     "starcoder": len(star), "star_k": k_star}


def bank_math(k_ble=4, n_gsm=6600):
    from datasets import load_dataset
    qs = [x["question"]
          for x in load_dataset("openai/gsm8k", "main", split="train")]
    rng = random.Random(SEED)
    qs = rng.sample(qs, n_gsm)                 # 子採樣至與題庫約對半
    ble = load_bank("math")
    prompts = [sft(q) for q in qs] + [sft(x["prompt"]) for x in ble] * k_ble
    return prompts, {"gsm8k_train_sub": n_gsm, "bleachers_math": len(ble),
                     "ble_k": k_ble}


BUILDERS = {"agentic": lambda tok: bank_agentic(tok),
            "ifshape": lambda tok: bank_ifshape(),
            "code": lambda tok: bank_code(),
            "math": lambda tok: bank_math()}


def _part(name):
    return EV / f"calib_v5_part_{name}.pt"


def assemble(out_path):
    parts, report = [], {}
    for name in ROWS:                          # 固定域序,shuffle 前可重現
        p = _part(name)
        assert p.exists(), f"缺 part:{p}"
        parts.append(torch.load(p))
        report[name] = json.loads(
            p.with_suffix(".json").read_text())
        assert parts[-1].shape[0] == ROWS[name]
    full = torch.cat(parts, dim=0)
    perm = torch.randperm(full.shape[0],
                          generator=torch.Generator().manual_seed(SEED))
    full = full[perm]
    out = Path(out_path)
    torch.save(full, out)
    ble_stats = {f.stem: json.loads(f.read_text()).get("sha256")
                 for f in BLE_DIR.glob("*_stats.json")}
    rec = {"teacher": "bf16(vLLM serve);題面來源=bleachers-20260814+人寫庫",
           "version": "v2(bleachers 重設計,分域流水)", "domains": report,
           "tilt": "35/15/25/25(agentic/ifshape/code/math)",
           "bleachers_sha": ble_stats,
           "sampling": {"temperature": 1.0, "top_p": 0.95,
                        "per_request_seed": "dseed+done+idx(同題多 run 獨立採樣)"},
           "seed": SEED, "shape": list(full.shape),
           "decontam": "8-gram(gsm8k-test+humaneval+IFEval 全集)列級切除",
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
           "model_revision": MODEL_REVISION}
    (EV / "CAL5_MANIFEST.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({"shape": rec["shape"], "sha256": rec["sha256"]},
                     indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="a1")
    ap.add_argument("--out", default=str(EV / "calib_selfgen_v5.pt"))
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--domains", default="",
                    help="逗號清單;空=全四域")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()
    if args.assemble:
        assemble(args.out)
        return

    names = [d for d in (args.domains.split(",") if args.domains else ROWS)
             if d]
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    banks = {n: BUILDERS[n](tok) for n in names}
    if args.extract_only:
        print(json.dumps({k: {"n_prompts": len(v[0]), **v[1]}
                          for k, v in banks.items()}, ensure_ascii=False,
                         indent=1, default=str))
        return

    from p2_anchor.wringer.build_calib512_v3 import scan
    bad = decontam_grams()                     # 含 IFEval 全集(補洞)
    for name, (prompts, meta) in banks.items():
        rows = ROWS[name]
        need = int(rows * (1 + MARGIN[name])) * SEQ_LEN
        print(f"== {name}:{len(prompts)} prompts → {rows} rows ==",
              flush=True)
        dseed = SEED + int(hashlib.sha256(name.encode())
                           .hexdigest()[:6], 16) % 10000   # 跨進程穩定
        t = gen_domain(args, tok, prompts, GEN_TOK[name], need, dseed)
        hits = set(scan(t, tok, bad))
        clean = [i for i in range(t.shape[0]) if i not in hits]
        assert len(clean) >= rows, f"{name} 除汙後不足:{len(clean)}"
        part = t[clean[:rows]]
        torch.save(part, _part(name))
        pj = {**meta, "rows": rows, "hits_excised": len(hits),
              "gen_tokens": GEN_TOK[name],
              "sha256": hashlib.sha256(_part(name).read_bytes()).hexdigest()}
        _part(name).with_suffix(".json").write_text(
            json.dumps(pj, indent=1, ensure_ascii=False, default=str))
        print(f"PART_DONE:{name}", flush=True)


if __name__ == "__main__":
    main()

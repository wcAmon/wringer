"""cal6:語料 v3(prereg_e42 FROZEN)——新舊題庫合池 + 人寫庫 → 4B bf16 作答 → 打包。

四域 22,977 列 + cal4 1,023 = 24,000 列(E40 train 11,263 的 ~2.1×):
  code    12,600 — tilt 52.5%(全語料占比;MBPP 人寫 k8 + 新舊 bleachers k10
                   + starcoder 自足 k3;池 7,185;同題多 run 獨立採樣)
  ifshape  3,700 — 新舊 bleachers 合池 5,796 ×k4
  math     3,600 — gsm8k-train 子採樣 6,600 + 新舊 bleachers 4,531 ×k6
  agentic  3,077 — cal3 同源種子續寫(E39 已證配方,不動)
題庫池 = bleachers-20260814(Qwen)+ bleachers-e42(DeepSeek 三線 @831e002);
兩庫產線已 cross-dedup。行為資料 = 4B bf16 續寫;決律同 cal5v2:
8-gram 除汙(gsm8k-test+humaneval+IFEval 全集)列級切除。

  CPU 預驗:--extract-only
  分域流水(--domains 逗號清單,存 part 檔 calib_v6_part_*.pt):
    PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.gen_cal6 \\
      --base-url http://127.0.0.1:8000/v1 --model a1 --domains code
  四域 part 齊後合裝(+cal4 併入、列級 shuffle → calib_e42_train.pt):
    ... gen_cal6 --assemble
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
from p2_anchor.bleachers.bank import decontam_grams
from p2_anchor.wringer.build_calib512_v3 import DOMAINS
from p2_anchor.wringer.gen_cal4 import SEQ_LEN, gen_domain, sft
from p2_anchor.wringer.gen_selfcorpus_code import extract_seeds

EV = Path("evidence/p1_grouping")
BLE_DIRS = [EV / "bleachers-20260814", EV / "bleachers-e42"]
CAL4 = EV / "calib_selfgen_v4.pt"              # 1,023 列,歷代成分不 replace
SEED = 20260816
ROWS = {"agentic": 3077, "ifshape": 3700, "code": 12600, "math": 3600}
MARGIN = {"agentic": 0.10, "ifshape": 0.12, "code": 0.15, "math": 0.10}
GEN_TOK = {"agentic": 1900, "ifshape": 900, "code": 800, "math": 600}


def load_banks(name):
    xs = []
    for d in BLE_DIRS:
        p = d / f"{name}.jsonl"
        if p.exists():
            xs += [json.loads(ln) for ln in p.read_text().splitlines()]
    return xs


def bank_agentic(tok, n=2000, k=3):
    d = dict(DOMAINS["agentic"])
    t = build(d["sources"], tok, n, SEQ_LEN, d["seed"] + 2,
              lines_per_file=24000)
    seeds = [tok.decode(t[i, :256].tolist()) for i in range(n)]
    return seeds * k, {"seed_rows": n, "seed_k": k, "source": "cal3 同源"}


def bank_ifshape(k=4):
    xs = load_banks("ifshape")
    return [sft(x["prompt"]) for x in xs] * k, \
        {"bleachers_ifshape_pool": len(xs), "k": k}


def bank_code(k_mbpp=8, k_star=4, k_ble=10):
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
    ble = load_banks("code") + load_banks("code_mbpp")
    prompts += [sft("Complete this Python function:\n\n" + x["prompt"])
                for x in ble] * k_ble
    star = [s for s in extract_seeds(2400, 20260817)
            if ">>>" in s or "Example" in s or "example" in s]
    prompts += [sft(s) for s in star] * k_star
    return prompts, {"mbpp": n_mb, "mbpp_k": k_mbpp,
                     "bleachers_code_pool": len(ble), "ble_k": k_ble,
                     "starcoder": len(star), "star_k": k_star}


def bank_math(k_ble=6, n_gsm=6600):
    from datasets import load_dataset
    qs = [x["question"]
          for x in load_dataset("openai/gsm8k", "main", split="train")]
    rng = random.Random(SEED)
    qs = rng.sample(qs, n_gsm)
    ble = load_banks("math")
    prompts = [sft(q) for q in qs] + [sft(x["prompt"]) for x in ble] * k_ble
    return prompts, {"gsm8k_train_sub": n_gsm,
                     "bleachers_math_pool": len(ble), "ble_k": k_ble}


BUILDERS = {"agentic": lambda tok: bank_agentic(tok),
            "ifshape": lambda tok: bank_ifshape(),
            "code": lambda tok: bank_code(),
            "math": lambda tok: bank_math()}


def _part(name):
    return EV / f"calib_v6_part_{name}.pt"


def assemble(out_path):
    parts, report = [], {}
    for name in ROWS:
        p = _part(name)
        assert p.exists(), f"缺 part:{p}"
        parts.append(torch.load(p))
        report[name] = json.loads(p.with_suffix(".json").read_text())
        assert parts[-1].shape[0] == ROWS[name]
    cal4 = torch.load(CAL4)
    assert cal4.shape == (1023, SEQ_LEN), f"cal4 形狀異常:{cal4.shape}"
    parts.append(cal4)
    full = torch.cat(parts, dim=0)
    perm = torch.randperm(full.shape[0],
                          generator=torch.Generator().manual_seed(SEED))
    full = full[perm]
    out = Path(out_path)
    torch.save(full, out)
    ble_stats = {f"{d.name}/{f.stem}": json.loads(f.read_text()).get("sha256")
                 for d in BLE_DIRS for f in d.glob("*_stats.json")}
    rec = {"teacher": "bf16(vLLM serve);題面=bleachers 新舊合池+人寫庫",
           "version": "cal6(prereg_e42:tilt code 52.5%,池 17,512)",
           "domains": report, "cal4_rows": 1023,
           "tilt": "12.8/15.4/52.5/15.0/4.3(agentic/ifshape/code/math/cal4)",
           "bleachers_sha": ble_stats,
           "sampling": {"temperature": 1.0, "top_p": 0.95,
                        "per_request_seed": "dseed+done+idx"},
           "seed": SEED, "shape": list(full.shape),
           "decontam": "8-gram(gsm8k-test+humaneval+IFEval 全集)列級切除",
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
           "model_revision": MODEL_REVISION}
    (EV / "E42_TRAIN_MANIFEST.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    drain = full[:128]
    dp = EV / "calib_drain_e42.pt"
    torch.save(drain, dp)
    print(json.dumps({"shape": rec["shape"], "sha256": rec["sha256"],
                      "drain_sha256": hashlib.sha256(
                          dp.read_bytes()).hexdigest()}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="a1")
    ap.add_argument("--out", default=str(EV / "calib_e42_train.pt"))
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--domains", default="", help="逗號清單;空=全四域")
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
    bad = decontam_grams()
    for name, (prompts, meta) in banks.items():
        rows = ROWS[name]
        need = int(rows * (1 + MARGIN[name])) * SEQ_LEN
        print(f"== {name}:{len(prompts)} prompts → {rows} rows ==",
              flush=True)
        dseed = SEED + int(hashlib.sha256(name.encode())
                           .hexdigest()[:6], 16) % 10000
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

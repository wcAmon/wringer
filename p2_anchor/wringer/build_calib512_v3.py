"""建 cal3:512×2048 三域均分語料(E38 語料軸;prereg_e38_dualaxis)。

三域各自 build() 再列級合併:
  agentic 171 seqs — 五源全新切片(與 cal1/cal2 檔案級零重疊,深度取中層)
  code    171 seqs — code_starcoder(repo 續寫格式,BigCode 已除汙)
  math    170 seqs — math_nemotron

除汙後置閘(入料前必過):對建成 token 列全量 detokenize,掃
gsm8k test(question+answer)與 humaneval(prompt+canonical_solution+test)
的 word 8-gram;採**列級切除**——每域多建 MARGIN 列,命中列剔除,
取前 n_seqs 條乾淨列;乾淨列不足即 FAIL。
(實測:math_nemotron 有 ~2% 列含 gsm8k 測題文本,seed 重抽不可行,
 必須切除;命中數入 manifest 留痕。)

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.build_calib512_v3
"""
import hashlib
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from p1_grouping.calib import build
from p1_grouping.modelio import MODEL_ID, MODEL_REVISION

EV = Path("evidence/p1_grouping")
SEQ_LEN = 2048
CLEAN = "~/projects/pretrain/data/circus_stage2/agentic/clean"
V02 = "~/projects/pretrain/data/circus_v0_2"
DOMAINS = {
    "agentic": {
        "n_seqs": 171, "seed": 20260813, "lines_per_file": 1600,
        "sources": [
            f"{CLEAN}/nemotron_v1_interactive_depth2.jsonl",
            f"{CLEAN}/nemotron_v2_toolcalling_depth2.jsonl",
            f"{CLEAN}/nemotron_v2_search_depth1.jsonl",
            f"{CLEAN}/toucan_depth2.jsonl",
            f"{CLEAN}/swe_v3_depth2.jsonl",
        ]},
    "code": {
        "n_seqs": 171, "seed": 20260814, "lines_per_file": 8000,
        "sources": [f"{V02}/code_starcoder.jsonl"]},
    "math": {
        "n_seqs": 170, "seed": 20260815, "lines_per_file": 8000,
        "sources": [f"{V02}/math_nemotron.jsonl"]},
}
SHUFFLE_SEED = 20260816
MARGIN = 30
NGRAM = 8


def _grams(text):
    words = re.sub(r"\s+", " ", text.lower()).strip().split(" ")
    return {" ".join(words[i:i + NGRAM])
            for i in range(len(words) - NGRAM + 1)}


def contamination_grams():
    from datasets import load_dataset
    g = set()
    for x in load_dataset("openai/gsm8k", "main", split="test"):
        g |= _grams(x["question"]) | _grams(x["answer"])
    for x in load_dataset("openai/openai_humaneval", split="test"):
        g |= _grams(x["prompt"] + "\n" + x["canonical_solution"]
                    + "\n" + x["test"])
    return g


def scan(t, tok, bad):
    hits = []
    for i in range(t.shape[0]):
        if _grams(tok.decode(t[i].tolist())) & bad:
            hits.append(i)
    return hits


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    bad = contamination_grams()
    print(f"contamination 8-gram set: {len(bad)}", flush=True)
    parts, report = {}, {}
    for name, d in DOMAINS.items():
        t = build(d["sources"], tok, d["n_seqs"] + MARGIN, SEQ_LEN,
                  d["seed"], lines_per_file=d["lines_per_file"])
        hits = set(scan(t, tok, bad))
        clean = [i for i in range(t.shape[0]) if i not in hits]
        print(f"{name} seed={d['seed']} built={t.shape[0]} "
              f"hits_excised={len(hits)}", flush=True)
        if len(clean) < d["n_seqs"]:
            raise SystemExit(f"DECONTAM_FAIL:{name} 乾淨列不足 "
                             f"{len(clean)} < {d['n_seqs']}")
        parts[name] = t[clean[:d["n_seqs"]]]
        report[name] = {"seed": d["seed"], "built": int(t.shape[0]),
                        "hits_excised": len(hits),
                        "n_seqs": d["n_seqs"],
                        "lines_per_file": d["lines_per_file"],
                        "sources": d["sources"]}
    full = torch.cat([parts["agentic"], parts["code"], parts["math"]], dim=0)
    perm = torch.randperm(full.shape[0],
                          generator=torch.Generator().manual_seed(SHUFFLE_SEED))
    full = full[perm]
    out = EV / "calib_train_512_v3.pt"
    torch.save(full, out)
    rec = {"domains": report, "shuffle_seed": SHUFFLE_SEED,
           "shape": list(full.shape),
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
           "decontam": f"word {NGRAM}-gram vs gsm8k-test + humaneval "
                       f"(prompt+solution+test);全列掃描零命中",
           "note": "E38 cal3:token 級三域均分(agentic/code/math),"
                   "agentic 與 cal1/cal2 檔案級零重疊;列級 shuffle 混域",
           "model_revision": MODEL_REVISION}
    (EV / "CALIB512V3_MANIFEST.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({"shape": rec["shape"], "sha256": rec["sha256"]},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()

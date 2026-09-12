"""建第二份 512×2048 退火語料(E33 sweep2 用;與 CALIB_MANIFEST 五源檔案級不相交)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.build_calib512_v2
"""
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from p1_grouping.calib import build
from p1_grouping.modelio import MODEL_ID, MODEL_REVISION

EV = Path("evidence/p1_grouping")
SEED = 20260811
N_SEQS, SEQ_LEN = 512, 2048
LINES_PER_FILE = 1600
CLEAN = "~/projects/pretrain/data/circus_stage2/agentic/clean"
# 與 calib_train_512(CALIB_MANIFEST 五源)檔案級不相交;同域對位:
# toolcalling×2 / interactive / toucan / swe,深度偏深(揭露)
SOURCES_V2 = [
    f"{CLEAN}/nemotron_v1_toolcalling_depth1.jsonl",
    f"{CLEAN}/nemotron_v1_toolcalling_depth3plus.jsonl",
    f"{CLEAN}/nemotron_v2_interactive_depth1.jsonl",
    f"{CLEAN}/toucan_depth3plus.jsonl",
    f"{CLEAN}/swe_v3_depth3plus.jsonl",
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    t = build(SOURCES_V2, tok, N_SEQS, SEQ_LEN, SEED,
              lines_per_file=LINES_PER_FILE)
    out = EV / "calib_train_512_v2.pt"
    torch.save(t, out)
    rec = {"sources": SOURCES_V2, "seed": SEED,
           "lines_per_file": LINES_PER_FILE,
           "shape": list(t.shape),
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
           "note": "E33 sweep2 語料;與 calib_train_512 檔案級不相交(零行重疊),"
                   "同 agentic 域五源對位,深度偏深(depth3plus 佔 3/5)",
           "model_revision": MODEL_REVISION}
    (EV / "CALIB512V2_MANIFEST.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({k: rec[k] for k in ("shape", "sha256", "seed")}, indent=1))


if __name__ == "__main__":
    main()

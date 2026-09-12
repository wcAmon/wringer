"""建 512×2048 校準集(Corkscrew 劑量;CALIB_MANIFEST 同五源、新種子)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.build_calib512
"""
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from p1_grouping.calib import build
from p1_grouping.modelio import MODEL_ID, MODEL_REVISION

EV = Path("evidence/p1_grouping")
SEED = 20260809
N_SEQS, SEQ_LEN = 512, 2048
LINES_PER_FILE = 1600


def main():
    man = json.loads((EV / "CALIB_MANIFEST.json").read_text())
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    t = build(man["sources"], tok, N_SEQS, SEQ_LEN, SEED,
              lines_per_file=LINES_PER_FILE)
    out = EV / "calib_train_512.pt"
    torch.save(t, out)
    rec = {"sources": man["sources"], "seed": SEED,
           "lines_per_file": LINES_PER_FILE,
           "shape": list(t.shape),
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
           "note": "Corkscrew ST 劑量(512×2048);抽樣偏差同 CALIB_MANIFEST",
           "model_revision": man["model_revision"]}
    (EV / "CALIB512_MANIFEST.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({k: rec[k] for k in ("shape", "sha256", "seed")}, indent=1))


if __name__ == "__main__":
    main()

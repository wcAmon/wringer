"""校準窗 → 純文字(llama-imatrix -f 用):逐列 decode(去 pad),模型 tokenizer 隨 WRINGER_MODEL。
    python -m p2_anchor.wringer.calib_to_text --calib evidence/p1_grouping/calib_e70_traj.pt --len evidence/p1_grouping/calib_e70_len.pt --out <txt>
"""
import argparse

import torch
from transformers import AutoTokenizer

from p1_grouping.modelio import MODEL_ID, MODEL_REVISION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--len", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    t = torch.load(args.calib, weights_only=False)
    L = torch.load(args.len, weights_only=False)
    n_tok = 0
    with open(args.out, "w") as f:
        for row, n in zip(t, L):
            ids = row[: int(n)].tolist()
            n_tok += len(ids)
            f.write(tok.decode(ids) + "\n\n")
    print(f"CALIB_TEXT_DONE rows={len(L)} tokens={n_tok} out={args.out}", flush=True)


if __name__ == "__main__":
    main()

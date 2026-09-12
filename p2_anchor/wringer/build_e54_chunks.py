"""E54 chunk 視圖:cal8(8k 整軌 pad+len)→ 有效區 2048 切片(prereg_e54)。

蓄水/逐窗(res_fill/train_res)與 GPTQ cov 走 2048 régime,pad 位若
入 H/Σ 會汙染統計;本工具只取每列有效長度內的「滿 2048」切片,
pad 尾與不足一片的殘段一律丟棄(零 pad、零 mask 需求、零引擎改動)。
e2e 末段才用 8k 整軌 + --lengths(已上崗補丁)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.build_e54_chunks
"""
import hashlib
import json
from pathlib import Path

import torch

EV = Path("evidence/p1_grouping")
CHUNK = 2048
SEED = 20260826


def main():
    t = torch.load(EV / "calib_e54_traj.pt", weights_only=False)
    ln = torch.load(EV / "calib_e54_len.pt", weights_only=False)
    assert t.shape[0] == ln.shape[0] and t.shape[1] % CHUNK == 0
    chunks, src = [], []
    for i in range(t.shape[0]):
        L = int(ln[i])
        for j in range(L // CHUNK):
            chunks.append(t[i, j * CHUNK:(j + 1) * CHUNK])
            src.append(i)
    out = torch.stack(chunks)
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(out.shape[0], generator=g)
    out = out[perm]
    src = [src[int(k)] for k in perm]
    torch.save(out, EV / "calib_e54_chunks.pt")
    rec = {"version": "e54_chunks(有效區滿 2048 切片,殘段/pad 丟棄)",
           "chunk_len": CHUNK, "seed": SEED,
           "src_rows": int(t.shape[0]), "n_chunks": int(out.shape[0]),
           "rows_with_zero_chunks": int((ln < CHUNK).sum()),
           "tokens_kept": int(out.numel()),
           "tokens_valid_total": int(ln.sum()),
           "sha256": hashlib.sha256(
               (EV / "calib_e54_chunks.pt").read_bytes()).hexdigest()}
    (EV / "E54_CHUNKS_MANIFEST.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps(rec, indent=1, ensure_ascii=False))
    print("E54_CHUNKS_DONE", flush=True)


if __name__ == "__main__":
    main()

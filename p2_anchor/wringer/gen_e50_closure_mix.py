"""E50 閉合鏈尾料:hec65 轉折窗偏置(prereg_e50)。

含 </think>(token 248069)窗 2,512(13.0%)×3 上採樣 + 等量非轉折窗
= 15,072 列(轉折佔比 50%);dose 1200×8=9,600 抽 → 有效 epoch 0.64,
轉折窗期望曝露 ~1.9 次(遠離 E43 小池毒性線)。
"""
import hashlib
import json
from pathlib import Path

import torch

EV = Path("evidence/p1_grouping")
SEED = 20260825
THINK_END = 248069


def main():
    t = torch.load(EV / "calib_e42_train_hec65.pt")
    has = (t == THINK_END).any(dim=1)
    T, N = t[has], t[~has]
    g = torch.Generator().manual_seed(SEED)
    n_pair = T.shape[0] * 3
    idx = torch.randperm(N.shape[0], generator=g)[:n_pair]
    full = torch.cat([T.repeat(3, 1), N[idx]], dim=0)
    perm = torch.randperm(full.shape[0],
                          generator=torch.Generator().manual_seed(SEED + 1))
    full = full[perm]
    out = EV / "calib_e50_closure.pt"
    torch.save(full, out)
    rec = {"version": "e50_closure(hec65 轉折窗 ×3 + 等量非轉折)",
           "transition_rows": int(T.shape[0]), "upsample": 3,
           "non_transition_rows": n_pair, "total": int(full.shape[0]),
           "transition_share": 0.5, "seed": SEED,
           "source": "calib_e42_train_hec65.pt(sha 0d792995…)",
           "think_end_token": THINK_END, "shape": list(full.shape),
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    (EV / "E50_CLOSURE_MANIFEST.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({"total": rec["total"], "sha256": rec["sha256"][:16]},
                     indent=1))


if __name__ == "__main__":
    main()

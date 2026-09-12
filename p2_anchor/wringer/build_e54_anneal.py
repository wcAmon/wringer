"""E54 S4b 壓回料篩選(prereg_e54 amendment_2):cal8 自然終止列。

cal8lt 判死(EOS 饑荒 79.6%);壓回教材改為高溫料中 len<8192 的列
(=EOS 在窗內收尾的完整軌跡)。溫度對齊 eval(temp 1.0)、
mode-concentrated 由選擇實現。定額 900 列(用戶三裁),SEED 20260828
抽樣;不足 900 全取照實。域標籤未存,偏 math/ifshape 混雜已預記。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.build_e54_anneal
"""
import hashlib
import json
import statistics
from pathlib import Path

import torch

EV = Path("evidence/p1_grouping")
SEED = 20260828
N_TARGET = 900
ROW_LEN = 8192
THINK_END = 248069


def main():
    t = torch.load(EV / "calib_e54_traj.pt", weights_only=False)
    ln = torch.load(EV / "calib_e54_len.pt", weights_only=False)
    assert t.shape[0] == ln.shape[0]

    nat = (ln < ROW_LEN).nonzero(as_tuple=True)[0]
    closed = (t[nat] == THINK_END).any(dim=1)
    nat = nat[closed]                     # 自然終止且思考閉合(雙保險)
    print(f"cal8 {t.shape[0]} 列:自然終止+閉合 {nat.shape[0]} 列", flush=True)

    if nat.shape[0] > N_TARGET:
        g = torch.Generator().manual_seed(SEED)
        nat = nat[torch.randperm(nat.shape[0], generator=g)[:N_TARGET]]
    sel_t, sel_ln = t[nat], ln[nat]
    torch.save(sel_t, EV / "calib_e54an_traj.pt")
    torch.save(sel_ln, EV / "calib_e54an_len.pt")

    rec = {"version": "e54an(amendment_2:cal8 自然終止列壓回料)",
           "source": "calib_e54_traj.pt(temp1.0)", "filter":
           f"len<{ROW_LEN} 且含 </think>(id {THINK_END})",
           "pool": int(t.shape[0]), "natural_closed": None,
           "rows": int(sel_t.shape[0]), "seed": SEED,
           "len_quartiles": [int(x) for x in
                             statistics.quantiles(sel_ln.tolist(), n=4)],
           "sha256": hashlib.sha256(
               (EV / "calib_e54an_traj.pt").read_bytes()).hexdigest(),
           "len_sha256": hashlib.sha256(
               (EV / "calib_e54an_len.pt").read_bytes()).hexdigest()}
    rec["natural_closed"] = int(closed.sum())
    (EV / "E54AN_MANIFEST.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({"rows": rec["rows"],
                      "len_q": rec["len_quartiles"],
                      "sha256": rec["sha256"][:16]}, indent=1))
    print("E54AN_DONE", flush=True)


if __name__ == "__main__":
    main()

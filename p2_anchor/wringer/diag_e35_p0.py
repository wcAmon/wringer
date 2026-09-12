"""E35 P0 前置診斷(CPU,零 GPU):st7m 檔位佔用與 herding 工作量預估。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.diag_e35_p0

7元(c=0.5)值集 {0, ±0.5, ±1, ±1.5};7→3 走廊的兩類搬運:
  ridge(±0.5 = 碼 T1=0,T2≠0):終態刀鋒族群,訓練分流 0 或 ±1
  outer(±1.5 = 碼 T1=T2≠0):確定性壓回 ±1,α 承受振幅折衷
輸出 evidence/p1_grouping/corkscrew/diag_e35_p0.json,數字進 prereg。
"""
import json
from pathlib import Path

import torch

from p2_anchor.wringer.state import qs_dir

EVC = Path("evidence/p1_grouping/corkscrew")


def main():
    st = {}
    for p in sorted(qs_dir("st7m").glob("layer*.pt")):
        st.update(torch.load(p, map_location="cpu", weights_only=False))
    assert st, "qs_st7m 空"
    tot = {"n": 0, "v0": 0, "v05": 0, "v1": 0, "v15": 0}
    per_mod = {}
    blk_outer_dom = blk_n = 0
    for n, s in st.items():
        assert int(s.get("grid", 0)) == 7 and abs(s["c"] - 0.5) < 1e-9, \
            f"{n} 非 7元 c=0.5 態:grid={s.get('grid')} c={s.get('c')}"
        va = (s["T1"].float() + 0.5 * s["T2"].float()).abs()   # (M,nB,B)
        cnt = {"v0": int((va == 0).sum()), "v05": int((va == 0.5).sum()),
               "v1": int((va == 1.0).sum()), "v15": int((va == 1.5).sum())}
        nn_ = va.numel()
        assert sum(cnt.values()) == nn_, f"{n} 佔用不齊:{cnt} vs {nn_}"
        for k, v in cnt.items():
            tot[k] += v
        tot["n"] += nn_
        per_mod[n] = {k: round(v / nn_, 4) for k, v in cnt.items()}
        blk15 = (va == 1.5).float().mean(-1)                   # (M,nB)
        blk_outer_dom += int((blk15 > 0.5).sum())
        blk_n += blk15.numel()
    frac = {k: round(v / tot["n"], 6) for k, v in tot.items() if k != "n"}
    ridge_sorted = sorted(per_mod.items(), key=lambda kv: -kv[1]["v05"])
    out = {
        "state": "st7m(E34 morph 產物,7元 c=0.5)",
        "n_weights": tot["n"],
        "occupancy": frac,
        "ridge_mass": frac["v05"],
        "outer_mass": frac["v15"],
        "moving_mass": round(frac["v05"] + frac["v15"], 6),
        "static_mass": round(frac["v0"] + frac["v1"], 6),
        "blocks_outer_dominant_frac": round(blk_outer_dom / blk_n, 6),
        "ridge_top5_modules": {k: v["v05"] for k, v in ridge_sorted[:5]},
        "note": "moving_mass = 走廊必須搬運的質量比例(herding 工作量);"
                "blocks_outer_dominant = ±1.5 過半的 block 佔比(α 補償壓力)",
    }
    EVC.mkdir(parents=True, exist_ok=True)
    (EVC / "diag_e35_p0.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items()
                      if k != "ridge_top5_modules"},
                     indent=1, ensure_ascii=False))
    print("→", EVC / "diag_e35_p0.json")


if __name__ == "__main__":
    main()

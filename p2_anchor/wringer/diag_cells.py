"""零 GPU 診斷:k=2 網格的格點佔用、跨 3元 邊界率、逐通道誤差集中度。

三問(#36 通道級混配 go/no-go 與 7元 損傷落點預測):
  A 反號率:T₁·T₂ < 0 的權重佔比 —— 這些格點正是 7元 禁止的
    (7元 約束 T₁·T₂ ≥ 0),故此比率 = 降到 7元 時「必須搬家」的權重量。
    逐層報,看損傷會集中在哪。
  B 跨 3元 邊界率:比對兩個 state 的 T₁,分「粗翻(T₁ 變)」與
    「細調(T₁ 不變、T₂ 變)」。粗翻率高 = 訓練在重構表示而非微調。
  C 逐通道誤差集中度:每個輸出通道的重建誤差佔比,取 top-1% 通道
    的誤差質量份額 + 峰度。份額高 = 少數通道主導誤差 = 通道級混配
    (給這些通道更高檔位)有本錢。

用法:
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.diag_cells \
    --state stmix --init gmix32 --out evidence/p1_grouping/corkscrew/diag_cells.json
"""
import argparse
import json
from pathlib import Path

import torch

from p2_anchor.wringer.modelio import N_LAYERS
from p2_anchor.wringer.state import qs_dir

EV = Path("evidence/p1_grouping/corkscrew")


def cell_stats(T1, T2):
    """9 格佔用 + 反號率 + T₂ 動用率。T 為 int8,轉 int16 防溢位。"""
    a = T1.to(torch.int16).reshape(-1)
    b = T2.to(torch.int16).reshape(-1)
    n = a.numel()
    cells = {}
    for t1 in (-1, 0, 1):
        for t2 in (-1, 0, 1):
            cells[f"{t1},{t2}"] = int(((a == t1) & (b == t2)).sum())
    opp = int((a * b < 0).sum())
    return {"n": n, "cells": cells,
            "opp_sign_frac": opp / n,
            "t2_used_frac": float((b != 0).sum()) / n,
            "t1_zero_frac": float((a == 0).sum()) / n}


def flip_stats(T1a, T2a, T1b, T2b):
    """init→trained 的翻轉分解。"""
    a1, a2 = T1a.reshape(-1), T2a.reshape(-1)
    b1, b2 = T1b.reshape(-1), T2b.reshape(-1)
    n = a1.numel()
    coarse = (a1 != b1)
    fine = (~coarse) & (a2 != b2)
    return {"n": n,
            "coarse_flip_frac": float(coarse.sum()) / n,   # T₁ 變 = 跨 3元 邊界
            "fine_flip_frac": float(fine.sum()) / n,       # 只有 T₂ 變
            "any_flip_frac": float((coarse | (a2 != b2)).sum()) / n}


def chan_stats(c, topk_frac=0.01):
    """逐輸出通道的重建量能集中度。

    無原始 W 可比時,用 α·(T₁+cT₂) 的逐通道 L2 當代理量能,再取
    通道能量分佈的 top-1% 份額與峰度。峰度高/份額高 = 少數通道主導。
    """
    a = c["a0"].float()                       # (M, nB, 1)
    v = c["T1"].float() + c["c"] * c["T2"].float()
    e = ((a * v) ** 2).sum(dim=(1, 2))        # (M,) 逐輸出通道能量
    M = e.numel()
    k = max(1, int(M * topk_frac))
    top = torch.topk(e, k).values.sum()
    mu, sd = e.mean(), e.std().clamp_min(1e-12)
    kurt = float((((e - mu) / sd) ** 4).mean())
    return {"M": M, "top1pct_energy_share": float(top / e.sum().clamp_min(1e-12)),
            "kurtosis": kurt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="已訓 state tag")
    ap.add_argument("--init", default=None, help="初始 state tag(算翻轉率;可略)")
    ap.add_argument("--out", default=str(EV / "diag_cells.json"))
    args = ap.parse_args()

    layers = []
    for li in range(N_LAYERS):
        p = qs_dir(args.state) / f"layer{li:02d}.pt"
        if not p.exists():
            continue
        st = torch.load(p, map_location="cpu", weights_only=False)
        init = None
        if args.init:
            pi = qs_dir(args.init) / f"layer{li:02d}.pt"
            if pi.exists():
                init = torch.load(pi, map_location="cpu", weights_only=False)
        for name, c in st.items():
            rec = {"layer": li, "name": name, "grid": c.get("grid")}
            rec.update(cell_stats(c["T1"], c["T2"]))
            rec.update(chan_stats(c))
            if init and name in init:
                ic = init[name]
                rec["flip"] = flip_stats(ic["T1"], ic["T2"], c["T1"], c["T2"])
            layers.append(rec)
        print(f"layer{li:02d} done", flush=True)
        del st, init

    tot = sum(r["n"] for r in layers)
    agg = {"state": args.state, "init": args.init, "n_weights": tot,
           "opp_sign_frac_global": sum(r["opp_sign_frac"] * r["n"]
                                       for r in layers) / tot,
           "t2_used_frac_global": sum(r["t2_used_frac"] * r["n"]
                                      for r in layers) / tot}
    if args.init:
        agg["coarse_flip_frac_global"] = sum(
            r["flip"]["coarse_flip_frac"] * r["n"] for r in layers
            if "flip" in r) / tot
        agg["fine_flip_frac_global"] = sum(
            r["flip"]["fine_flip_frac"] * r["n"] for r in layers
            if "flip" in r) / tot
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"agg": agg, "per_module": layers},
                              indent=1, ensure_ascii=False))
    print(json.dumps(agg, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

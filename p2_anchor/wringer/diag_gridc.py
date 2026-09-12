"""零 GPU 診斷:k=2 網格的 c 重優化(E34 候選前置)。

Corkscrew 的值集由 c 決定:
  9元 {0, ±c, ±(1−c), ±1, ±(1+c)}   7元 {0, ±c, ±1, ±(1+c)}
c=0.6 是從 3⊂5⊂7⊂9 嵌套需求繼承的(c=0.5 會使 1−c=c 碰撞退化成 7 值)。
但**從頭量化的純 7元 不使用嵌套**,c 在那裡是自由參數。

本工具量 u = W/α₀ 的經驗分佈(α₀ = TWN 逐 block 初始),再在 (c, s)
平面上掃描量化 MSE,s 為網格整體縮放(等價於重解 α 的一階近似)。

限制(必須連同結果一起讀):此處是「直接四捨五入到網格」的 MSE,
不含 GPTQ 的誤差補償、不含 scale_joint 的 Σ-加權 α 重解、不含後續
訓練。故絕對值不等於實際量化誤差;**只用於 c 之間的相對比較**。

用法:
  OMP_NUM_THREADS=2 PYTHONPATH=. nice -n 19 .venv/bin/python \
    -m p2_anchor.wringer.diag_gridc --out evidence/p1_grouping/corkscrew/diag_gridc.json
"""
import argparse
import json
from pathlib import Path

import torch

from p2_anchor.wringer.ktier import BLOCK, init_alpha
from p2_anchor.wringer.modelio import enumerate_targets

EV = Path("evidence/p1_grouping/corkscrew")
LO, HI, NBINS = -4.0, 4.0, 8000


def grid_vals(level, c, s):
    base = {9: [0, c, 1 - c, 1, 1 + c], 7: [0, c, 1, 1 + c],
            5: [0, c, 1], 3: [0, 1]}[level]
    v = sorted({0.0} | {x for b in base for x in (b, -b)})
    return torch.tensor(v, dtype=torch.float64) * s


def mse_on_hist(centers, p, vals):
    """Σ p_i · min_g (x_i − g)²。centers/p: (NBINS,);vals: (G,)。"""
    d = (centers.unsqueeze(1) - vals.unsqueeze(0)).abs().min(dim=1).values
    return float((p * d ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="7,9")
    ap.add_argument("--c-lo", type=float, default=0.05)
    ap.add_argument("--c-hi", type=float, default=0.95)
    ap.add_argument("--c-step", type=float, default=0.005)
    ap.add_argument("--s-lo", type=float, default=0.6)
    ap.add_argument("--s-hi", type=float, default=1.6)
    ap.add_argument("--s-step", type=float, default=0.01)
    ap.add_argument("--out", default=str(EV / "diag_gridc.json"))
    args = ap.parse_args()

    from p1_grouping.modelio import load_model
    model, _ = load_model(device="cpu")
    model.requires_grad_(False)
    names = enumerate_targets(model)
    print(f"targets: {len(names)}", flush=True)

    hist = torch.zeros(NBINS, dtype=torch.float64)
    n_clip = 0
    n_tot = 0
    for i, n in enumerate(names):
        W = model.get_submodule(n).weight.data.float()
        M, K = W.shape
        a = init_alpha(W)                                # (M, nB, 1)
        u = (W.reshape(M, K // BLOCK, BLOCK) / a).reshape(-1)
        hist += torch.histc(u, bins=NBINS, min=LO, max=HI).double()
        n_clip += int(((u < LO) | (u > HI)).sum())
        n_tot += u.numel()
        if i % 25 == 0:
            print(f"{i}/{len(names)} {n}", flush=True)
        del W, a, u
    del model

    edges = torch.linspace(LO, HI, NBINS + 1, dtype=torch.float64)
    centers = (edges[1:] + edges[:-1]) / 2
    p = hist / hist.sum()
    var_u = float((p * centers ** 2).sum())              # E[u²](u 近似零均值)
    print(f"clip 率 {n_clip/n_tot:.2e}  E[u²]={var_u:.4f}", flush=True)

    cs = torch.arange(args.c_lo, args.c_hi + 1e-9, args.c_step, dtype=torch.float64)
    ss = torch.arange(args.s_lo, args.s_hi + 1e-9, args.s_step, dtype=torch.float64)
    res = {}
    for level in [int(x) for x in args.levels.split(",")]:
        best = None
        curve = []
        for c in cs.tolist():
            if level == 9 and abs(c - 0.5) < 1e-9:
                continue                                  # 1−c = c 碰撞
            bc = None
            for s in ss.tolist():
                m = mse_on_hist(centers, p, grid_vals(level, c, s))
                if bc is None or m < bc[0]:
                    bc = (m, s)
            curve.append({"c": round(c, 4), "mse": bc[0], "s": round(bc[1], 4)})
            if best is None or bc[0] < best["mse"]:
                best = {"c": round(c, 4), "s": round(bc[1], 4), "mse": bc[0]}
        # c=0.6 基線(s 同樣最佳化,公平比較)
        b06 = min(({"s": s, "mse": mse_on_hist(centers, p, grid_vals(level, 0.6, s))}
                   for s in ss.tolist()), key=lambda x: x["mse"])
        res[str(level)] = {
            "best": best, "baseline_c0.6": {"s": round(b06["s"], 4),
                                            "mse": b06["mse"]},
            "rel_rms_best": (best["mse"] / var_u) ** 0.5,
            "rel_rms_c0.6": (b06["mse"] / var_u) ** 0.5,
            "mse_reduction_pct": (1 - best["mse"] / b06["mse"]) * 100,
            "curve": curve}
        print(f"level {level}: best c={best['c']} s={best['s']} "
              f"rel_rms {(best['mse']/var_u)**0.5*100:.2f}%  |  "
              f"c=0.6 rel_rms {(b06['mse']/var_u)**0.5*100:.2f}%  |  "
              f"MSE −{(1-best['mse']/b06['mse'])*100:.1f}%", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"n_weights": n_tot, "clip_frac": n_clip / n_tot, "E_u2": var_u,
         "block": BLOCK, "note": "直接捨入 MSE;無 GPTQ 補償/無 joint-α/無訓練",
         "levels": res}, indent=1, ensure_ascii=False))
    print("wrote", out)


if __name__ == "__main__":
    main()

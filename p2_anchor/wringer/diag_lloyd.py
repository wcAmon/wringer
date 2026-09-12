"""零 GPU 診斷:自由碼本(Lloyd-Max)上限——「統計調刻度」的天花板。

diag_gridc 量的是 Corkscrew 網格族 {0,±c,±1,±(1+c)}·s 內的最優(只有
c、s 兩個自由度)。本工具放掉全部結構約束,對 u = W/α₀ 的經驗直方圖
直接解 L 點最優量化器(Lloyd-Max),得到「任何統計調刻度方案」的 MSE
下界。族內最優與自由碼本之間的差 = k=2 結構本身付出的幾何代價。

讀法:
  - 若自由碼本增益也只有幾個百分點 → 統計刻度這條路整條封頂,
    幾何軸徹底出清,配方軸(訓練驅趕/課程)是唯一開放的槓桿。
  - 若增益大 → k=2 結構代價高,LUT 化(棄雙三元累加器)才值得討論。

限制(與 diag_gridc 相同,必須連同結果讀):直接捨入 MSE,不含 GPTQ
補償、不含 Σ-加權 α 重解、不含訓練。只用於方案間相對比較。

直方圖會存檔重用(evidence/.../weight_hist_u.json),之後的幾何探測
不必再載模型。

用法:
  OMP_NUM_THREADS=2 PYTHONPATH=. nice -n 19 .venv/bin/python \
    -m p2_anchor.wringer.diag_lloyd
"""
import argparse
import json
from pathlib import Path

import torch

from p2_anchor.wringer.ktier import BLOCK, init_alpha
from p2_anchor.wringer.modelio import enumerate_targets

EV = Path("evidence/p1_grouping/corkscrew")
LO, HI, NBINS = -4.0, 4.0, 8000


def build_or_load_hist(hist_path):
    if hist_path.exists():
        d = json.load(open(hist_path))
        assert d["lo"] == LO and d["hi"] == HI and d["nbins"] == NBINS
        print(f"直方圖已存在,重用 {hist_path}", flush=True)
        return (torch.tensor(d["hist"], dtype=torch.float64),
                d["n_weights"], d["clip_frac"])
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
        a = init_alpha(W)
        u = (W.reshape(M, K // BLOCK, BLOCK) / a).reshape(-1)
        hist += torch.histc(u, bins=NBINS, min=LO, max=HI).double()
        n_clip += int(((u < LO) | (u > HI)).sum())
        n_tot += u.numel()
        if i % 25 == 0:
            print(f"{i}/{len(names)} {n}", flush=True)
        del W, a, u
    del model
    hist_path.write_text(json.dumps(
        {"lo": LO, "hi": HI, "nbins": NBINS, "n_weights": n_tot,
         "clip_frac": n_clip / n_tot, "block": BLOCK,
         "note": "u = W/α₀ 全 200 target linear 的經驗直方圖(α₀=TWN 逐 block)",
         "hist": hist.tolist()}))
    print(f"直方圖已存 {hist_path}", flush=True)
    return hist, n_tot, n_clip / n_tot


def hist_mse(centers, p, codes):
    d = (centers.unsqueeze(1) - codes.unsqueeze(0)).abs().min(dim=1).values
    return float((p * d ** 2).sum())


def lloyd(centers, p, L, sym=True, iters=1000, tol=1e-14):
    """對直方圖解 L 點 Lloyd-Max。sym=True 每輪投影回對稱碼本
    (Corkscrew 族是對稱的,公平比較要求自由碼本同樣對稱)。"""
    cdf = torch.cumsum(p, 0)
    q = torch.tensor([(i + 0.5) / L for i in range(L)], dtype=torch.float64)
    idx = torch.searchsorted(cdf.contiguous(), q).clamp(max=NBINS - 1)
    codes = centers[idx].clone()
    prev = None
    for _ in range(iters):
        codes, _ = torch.sort(codes)
        b = (codes[1:] + codes[:-1]) / 2
        cell = torch.bucketize(centers, b)
        num = torch.zeros(L, dtype=torch.float64).scatter_add_(0, cell, p * centers)
        den = torch.zeros(L, dtype=torch.float64).scatter_add_(0, cell, p)
        codes = torch.where(den > 0, num / den.clamp(min=1e-300), codes)
        if sym:
            codes = (codes - codes.flip(0)) / 2      # 對稱化(奇 L 中點歸零)
        m = hist_mse(centers, p, codes)
        if prev is not None and abs(prev - m) < tol * max(m, 1e-300):
            break
        prev = m
    return torch.sort(codes)[0], m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="3,5,7,9")
    ap.add_argument("--out", default=str(EV / "diag_lloyd.json"))
    args = ap.parse_args()

    hist, n_tot, clip_frac = build_or_load_hist(EV / "weight_hist_u.json")
    edges = torch.linspace(LO, HI, NBINS + 1, dtype=torch.float64)
    centers = (edges[1:] + edges[:-1]) / 2
    p = hist / hist.sum()
    var_u = float((p * centers ** 2).sum())
    print(f"n={n_tot:.3e} clip={clip_frac:.2e} E[u²]={var_u:.4f}", flush=True)

    gridc = {}
    gp = EV / "diag_gridc.json"
    if gp.exists():
        gridc = json.load(open(gp))["levels"]

    res = {}
    for L in [int(x) for x in args.levels.split(",")]:
        codes_s, mse_s = lloyd(centers, p, L, sym=True)
        codes_f, mse_f = lloyd(centers, p, L, sym=False)
        row = {
            "lloyd_sym": {"codes": [round(v, 5) for v in codes_s.tolist()],
                          "mse": mse_s, "rel_rms": (mse_s / var_u) ** 0.5},
            "lloyd_free": {"codes": [round(v, 5) for v in codes_f.tolist()],
                           "mse": mse_f, "rel_rms": (mse_f / var_u) ** 0.5},
        }
        g = gridc.get(str(L))
        if g:
            row["family_best"] = {"c": g["best"]["c"], "s": g["best"]["s"],
                                  "mse": g["best"]["mse"],
                                  "rel_rms": g["rel_rms_best"]}
            row["family_c0.6"] = {"mse": g["baseline_c0.6"]["mse"],
                                  "rel_rms": g["rel_rms_c0.6"]}
            row["structure_gap_mse_pct"] = (1 - mse_s / g["best"]["mse"]) * 100
        res[str(L)] = row
        fam = (f"  族內最優 rel_rms {g['rel_rms_best']*100:.2f}%  "
               f"結構代價 MSE −{row['structure_gap_mse_pct']:.1f}%") if g else ""
        print(f"L={L}: Lloyd(sym) rel_rms {(mse_s/var_u)**0.5*100:.2f}%  "
              f"codes {[round(v,3) for v in codes_s.tolist()]}{fam}", flush=True)

    out = Path(args.out)
    out.write_text(json.dumps(
        {"n_weights": n_tot, "clip_frac": clip_frac, "E_u2": var_u,
         "note": "直接捨入 MSE;無 GPTQ 補償/無 joint-α/無訓練;"
                 "structure_gap = 族內最優(c,s) vs 自由對稱碼本",
         "levels": res}, indent=1, ensure_ascii=False))
    print("wrote", out)


if __name__ == "__main__":
    main()

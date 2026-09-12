"""零 GPU 診斷:9→7 硬投影誤差的逐通道集中度(#36 通道級混配 go/no-go)。

7元 約束 T₁·T₂ ≥ 0,砍掉的正是值 ±(1−c) 的兩個反號格。對一個既有 9元
state,每個權重被迫搬到最近的 7元 值,位移量 = α·|v − v'|。整段誤差
完全由 state 內部決定:不需要原始 bf16 權重,不需要 GPU。

問的是「這些誤差集中在少數通道嗎」——若集中,把高誤差通道留在 9元、
其餘降 7元 的通道級混配就有本錢;若彌散,通道級配置無利可圖(與 E32
已否證的「層級配置溢價」同型的檢驗,只是換到通道粒度)。

輸出兩軸:
  row  = 輸出通道(α 的所屬列,共 M)
  col  = 輸入通道(α block 的所屬行,共 K)

用法:
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.diag_proj7 \
    --state g9 --out evidence/p1_grouping/corkscrew/diag_proj7_g9.json
"""
import argparse
import json
from pathlib import Path

import torch

from p2_anchor.wringer.ktier import grid_tensors
from p2_anchor.wringer.modelio import N_LAYERS
from p2_anchor.wringer.state import qs_dir

EV = Path("evidence/p1_grouping/corkscrew")


def concentration(e, fracs=(0.001, 0.01, 0.05)):
    """誤差質量的頭部份額 + 基尼係數。e: 1-D 非負。"""
    tot = e.sum().clamp_min(1e-30)
    n = e.numel()
    srt = torch.sort(e, descending=True).values
    out = {f"top{f*100:g}pct_share": float(srt[:max(1, int(n * f))].sum() / tot)
           for f in fracs}
    asc = torch.sort(e).values
    idx = torch.arange(1, n + 1, dtype=torch.float64)
    out["gini"] = float((2 * (idx * asc.double()).sum() / (n * asc.double().sum())
                         - (n + 1) / n))
    out["n"] = n
    return out


def module_proj_err(c, level_from=9, level_to=7):
    """回傳 (err2_row (M,), err2_col (K,), tot_err2, tot_w2)。"""
    cc = c["c"]
    v = c["T1"].float() + cc * c["T2"].float()          # (M, nB, B)
    a = c["a0"].float()                                  # (M, nB, 1)
    vals_to, _, _ = grid_tensors(level_to, cc, device="cpu")
    # 最近 7元 值(值域小,直接逐值比較)
    d = (v.unsqueeze(-1) - vals_to.view(1, 1, 1, -1)).abs()
    dv = d.min(dim=-1).values                            # (M, nB, B) |v - v'|
    e2 = (a * dv) ** 2                                   # 位移平方
    w2 = (a * v) ** 2
    M, nB, B = v.shape
    return (e2.sum(dim=(1, 2)),                          # 逐輸出通道
            e2.reshape(M, nB * B).sum(dim=0),            # 逐輸入通道
            float(e2.sum()), float(w2.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--from-level", type=int, default=9)
    ap.add_argument("--to-level", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or EV / f"diag_proj7_{args.state}.json")

    mods, tot_e2, tot_w2 = [], 0.0, 0.0
    for li in range(N_LAYERS):
        p = qs_dir(args.state) / f"layer{li:02d}.pt"
        if not p.exists():
            continue
        st = torch.load(p, map_location="cpu", weights_only=False)
        for name, c in st.items():
            if c.get("grid") != args.from_level:
                continue
            er, ec, e2, w2 = module_proj_err(c, args.from_level, args.to_level)
            rec = {"layer": li, "name": name,
                   "rel_rms": (e2 / max(w2, 1e-30)) ** 0.5,
                   "row": concentration(er), "col": concentration(ec)}
            mods.append(rec)
            tot_e2 += e2
            tot_w2 += w2
        print(f"layer{li:02d} done", flush=True)
        del st

    if not mods:
        raise SystemExit(f"qs_{args.state} 無 grid={args.from_level} 模組")
    n = len(mods)

    def med(path):
        vs = sorted(m[path[0]][path[1]] for m in mods)
        return vs[n // 2]

    agg = {"state": args.state, "from": args.from_level, "to": args.to_level,
           "n_modules": n,
           "rel_rms_global": (tot_e2 / max(tot_w2, 1e-30)) ** 0.5,
           "row_top1pct_share_median": med(("row", "top1pct_share")),
           "row_gini_median": med(("row", "gini")),
           "col_top1pct_share_median": med(("col", "top1pct_share")),
           "col_gini_median": med(("col", "gini"))}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"agg": agg, "per_module": mods},
                              indent=1, ensure_ascii=False))
    print(json.dumps(agg, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

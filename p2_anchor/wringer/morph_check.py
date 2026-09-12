"""E34 形變降元的零 GPU 數學驗證:c 滑移 0.6→0.5 即 9→7 降元。

驗證四件事(全 CPU,不碰訓練):
  V1 連續性:soft staircase 對 c 連續(滑移不產生跳變)。
  V2 收斂性:9元(c→0.5⁺) 階梯逐點收斂到 7元(c=0.5) 階梯(消失階
     高度 2c−1→0,門檻極限 {±0.25,±0.75,±1.25} 與均勻 7元 重合)。
  V3 恆等極限:∀c∈(0.5,0.6],s→0 時 f(u)→u(SoftGrid assert 沿途成立)。
  V4 零訓練 null 流向:從 weight_hist_u.json 算「準靜態滑移+最近捨入」
     下 ±0.4 盆地質量的去向——此 null 恰等於直接投影到 7元(c=0.5),
     故 E34 判準 M3 的操作化 = 訓練臂流向與此基線的偏離。

用法:PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.morph_check
"""
import json
from pathlib import Path

import torch

from p2_anchor.wringer.stq import SoftGrid, soft_staircase

EV = Path("evidence/p1_grouping/corkscrew")
U = torch.linspace(-2.2, 2.2, 20001, dtype=torch.float32)


def stair(n, c, s):
    g = SoftGrid(c=c, device="cpu", n=n)
    return soft_staircase(U, s, g, g.theta), g


def main():
    # V1 連續性:相鄰 c 的階梯逐點差 → 0
    cs = [0.600, 0.575, 0.550, 0.525, 0.5125, 0.50625, 0.503125]
    prev = None
    print("V1 連續性(s=8):")
    for c in cs:
        f, g = stair(9, c, 8.0)
        gap = float((f - prev).abs().max()) if prev is not None else None
        h_vanish = float(g.h.min())
        print(f"  c={c:.6f}  消失階高 2c−1={2*c-1:.4f} (h_min={h_vanish:.4f})"
              + (f"  Δ階梯={gap:.5f}" if gap is not None else ""))
        prev = f
    # V2 收斂性:9元(c→0.5⁺) vs 7元(0.5)
    print("V2 收斂性(s=8):")
    f7, g7 = stair(7, 0.5, 8.0)
    for c in [0.55, 0.52, 0.505, 0.5005]:
        f9, _ = stair(9, c, 8.0)
        print(f"  c={c}:  max|9元(c) − 7元(0.5)| = {float((f9-f7).abs().max()):.5f}")
    # V3 恆等極限沿途成立(SoftGrid 建構時已 assert Σhθ=0;再量 s→0)
    print("V3 恆等極限:")
    for c in [0.6, 0.55, 0.505]:
        f, _ = stair(9, c, 0.05)
        print(f"  c={c}: s=0.05 時 max|f(u)−u| = {float((f-U).abs().max()):.5f}")
    # V4 null 流向(零訓練基線,直方圖)
    d = json.load(open(EV / "weight_hist_u.json"))
    hist = torch.tensor(d["hist"], dtype=torch.float64)
    edges = torch.linspace(d["lo"], d["hi"], d["nbins"] + 1, dtype=torch.float64)
    ctr = (edges[1:] + edges[:-1]) / 2
    p = hist / hist.sum()
    # 起點:9元 c=0.6,+0.4 的盆地 = [0.2, 0.5](負側鏡像)
    basin = (ctr.abs() >= 0.2) & (ctr.abs() < 0.5)
    m_basin = float(p[basin].sum())
    # 終點:7元 c=0.5,門檻 0.25 → [0.2,0.25) 流向 0,[0.25,0.5) 流向 ±0.5
    to_zero = basin & (ctr.abs() < 0.25)
    print(f"V4 null 流向(準靜態滑移 ≡ 直接投影,M3 的零訓練基線):")
    print(f"  ±0.4 盆地質量 = {m_basin*100:.2f}% 全權重")
    print(f"  → 0     : {float(p[to_zero].sum())/m_basin*100:.1f}%")
    print(f"  → ±0.5  : {(m_basin-float(p[to_zero].sum()))/m_basin*100:.1f}%")
    print("  訓練臂流向與此偏離 = 「驅趕確實發生」的機制證據(M3)。")


if __name__ == "__main__":
    main()

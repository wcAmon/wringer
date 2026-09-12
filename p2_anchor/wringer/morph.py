"""E34 可微網格機制:h(c)/θ(c) 閉式建構(c 三態生命週期的載體)。

grid_tensors 走 python sort/dedupe,c 不可微;本模組在**值序固定的
象限內**用閉式張量建構,讓 autograd 直通 c:

  9元 c∈(0.5,1):  [-(1+c), -1, -c, -(1-c), 0, (1-c), c, 1, 1+c]
  7元 c∈(0,1):    [-(1+c), -1, -c, 0, c, 1, 1+c]
  5元 c∈(0,1):    [-1, -c, 0, c, 1]

三態(E34_DESCENT_MORPH.md §2b):
  態一 自由學習:c = 可訓張量,直接進 vals。
  態二 降元走廊:c = corridor(c_free, c_tgt, logit, m)。
  態三 收歸 α:3元 無 c,不經本模組。

獨立於 stq.py(E33 運行期間不改既有訓練路徑);收卷後由 train_st
接線。CPU 自檢:python -m p2_anchor.wringer.morph
"""
import math

import torch

_LAYOUTS = {
    9: lambda c: [-(1 + c), -1 + 0 * c, -c, -(1 - c), 0 * c,
                  (1 - c), c, 1 + 0 * c, 1 + c],
    7: lambda c: [-(1 + c), -1 + 0 * c, -c, 0 * c, c, 1 + 0 * c, 1 + c],
    5: lambda c: [-1 + 0 * c, -c, 0 * c, c, 1 + 0 * c],
}
_DOMAIN = {9: (0.5, 1.0), 7: (0.0, 1.0), 5: (0.0, 1.0)}


def grid_from_c(n, c):
    """c: 標量張量(可帶梯度)→ (vals, theta, h, H),全部可微。

    象限外拋錯而非裁剪——走廊排程有責任把 c 關在域內。
    """
    lo, hi = _DOMAIN[n]
    cv = float(c.detach())
    assert lo < cv < hi, f"c={cv} 出 {n}元 值序象限 ({lo},{hi})"
    vals = torch.stack(_LAYOUTS[n](c))
    theta = (vals[1:] + vals[:-1]) / 2
    h = vals[1:] - vals[:-1]
    return vals, theta, h, vals[-1] - vals[0]


def soft_staircase_c(u, s, n, c, dtheta_scaled=None):
    """與 stq.soft_staircase 同語義,但 (h, θ) 由 c 閉式生成、對 c 可微。

    dtheta_scaled: 選配,已縮放的 θ 偏移(THETA_RANGE·h·tanh(dθ) 由
    呼叫端算好傳入,保持與 stq.theta_eff 相同語義)。
    """
    _, theta, h, H = grid_from_c(n, c)
    if dtheta_scaled is not None:
        theta = theta + dtheta_scaled
    out = torch.zeros_like(u)
    for j in range(theta.shape[0]):
        out = out + (h[j] / 2) * torch.tanh(s * (u - theta[j]))
    return out / math.tanh(s * float(H.detach()) / 2)


def corridor(c_free, c_tgt, logit, m):
    """態二走廊:c(m) = c_tgt + (c_sched(m) − c_tgt) · 2σ(logit)。

    c_sched(m) = c_free + m·(c_tgt − c_free)(線性收縮)。
    m=1 時無論 logit 為何,c ≡ c_tgt(終點被排程夾死);m<1 時
    σ 乘子∈(0,2) 讓各模組先衝或滯後。c_free 可為態一學到的 c*。
    """
    c_sched = c_free + m * (c_tgt - c_free)
    return c_tgt + (c_sched - c_tgt) * 2 * torch.sigmoid(logit)


def _selfcheck():
    from p2_anchor.wringer.ktier import grid_tensors
    # C1 數值等價:固定 c 下 vals/θ/h 與 grid_tensors 逐位一致
    for n, cv in [(9, 0.6), (9, 0.55), (7, 0.6), (7, 0.5), (5, 0.6)]:
        c = torch.tensor(cv, dtype=torch.float64)
        vals, theta, h, H = grid_from_c(n, c)
        ref, _, _ = grid_tensors(n, cv, device="cpu")
        assert torch.allclose(vals.float(), ref, atol=1e-6), (n, cv)
        assert abs(float((h * theta).sum())) < 1e-12          # 恆等極限條件
    print("C1 數值等價 grid_tensors:PASS(9/7/5元 各 c)")
    # C2 c 梯度存在且有限(態一)
    u = torch.linspace(-2, 2, 4001, dtype=torch.float64)
    c = torch.tensor(0.58, dtype=torch.float64, requires_grad=True)
    loss = (soft_staircase_c(u, 6.0, 9, c) - u).pow(2).mean()
    loss.backward()
    g = float(c.grad)
    assert math.isfinite(g) and abs(g) > 0
    print(f"C2 ∂loss/∂c = {g:+.6f}(有限、非零):PASS")
    # C3 走廊語義:m=1 夾死終點;m<1 時 logit 有梯度
    logit = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    cf = torch.tensor(0.62, dtype=torch.float64)
    c1 = corridor(cf, 0.5, logit, 1.0)
    assert abs(float(c1.detach()) - 0.5) < 1e-12
    cm = corridor(cf, 0.5, logit, 0.4)
    (soft_staircase_c(u, 6.0, 9, cm) - u).pow(2).mean().backward()
    assert math.isfinite(float(logit.grad)) and abs(float(logit.grad)) > 0
    print(f"C3 走廊:m=1 → c≡0.5;m=0.4 → c={float(cm):.4f},"
          f"∂loss/∂logit = {float(logit.grad):+.2e}:PASS")
    # C5(E35)7→3 走廊域:c<0.5 值序等價 + 梯度 + m=1 夾 0(c 收歸 α)
    for cv in (0.49, 0.3, 0.1, 0.05):
        c = torch.tensor(cv, dtype=torch.float64)
        vals, theta, h, H = grid_from_c(7, c)
        ref, _, _ = grid_tensors(7, cv, device="cpu")
        assert torch.allclose(vals.float(), ref, atol=1e-6), (7, cv)
        assert abs(float((h * theta).sum())) < 1e-12
    c = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    (soft_staircase_c(u, 6.0, 7, c) - u).pow(2).mean().backward()
    assert math.isfinite(float(c.grad)) and abs(float(c.grad)) > 0
    c0 = corridor(torch.tensor(0.5, dtype=torch.float64), 0.0,
                  torch.tensor(-0.9, dtype=torch.float64), 1.0)
    assert abs(float(c0)) < 1e-12
    print(f"C5 7→3 域:值序 4 點等價、∂loss/∂c|₇,₀.₃ = {float(c.grad):+.6f}"
          "、m=1 → c≡0:PASS")
    # C4 MSE 梯度方向實測:9元 在 c=0.6 的 ∂MSE/∂c 指向增大(背對 0.5)
    import json
    from pathlib import Path
    hp = Path("evidence/p1_grouping/corkscrew/weight_hist_u.json")
    if hp.exists():
        d = json.load(open(hp))
        hist = torch.tensor(d["hist"], dtype=torch.float64)
        edges = torch.linspace(d["lo"], d["hi"], d["nbins"] + 1,
                               dtype=torch.float64)
        ctr = (edges[1:] + edges[:-1]) / 2
        p = hist / hist.sum()
        c = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
        vals, _, _, _ = grid_from_c(9, c)
        dmin = (ctr.unsqueeze(1) - vals.unsqueeze(0)).abs()
        w = torch.softmax(-80.0 * dmin, dim=1)             # 軟指派(可微)
        mse = (p * (w * (ctr.unsqueeze(1) - vals.unsqueeze(0)) ** 2)
               .sum(1)).sum()
        mse.backward()
        print(f"C4 ∂MSE/∂c|c=0.6 = {float(c.grad):+.6f} "
              f"({'指向 0.66,背對 0.5——合併必須課程強制' if c.grad < 0 else '指向減小'})"
              ":量測完成")
    print("morph selfcheck 全過")


if __name__ == "__main__":
    _selfcheck()

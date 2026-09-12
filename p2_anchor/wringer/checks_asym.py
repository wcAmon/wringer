"""E63 非對稱三元 等價閘/數學閘(合成資料,CPU 為主;A2/A3 用少量 cuda 驗 state/export 路)。

A1 r≡1 退化:asym_V/dense_asym 逐位 == 對稱。
A2 state.dense_weight:rneg=1 逐位 == 無 rneg;rneg 隨機時等於手算。
A3 export.build_planes/decode_tensor 含 rneg 與 dense_weight bf16 逐位一致。
A4 fit_alpha_r:交替軌跡單調不增;asym 終值 ≤ sym(起點即 sym 解)。
A5 round_ternary(r):逐元最近值(對 {−rα,0,α} 暴力最近)。
A6 gptq_ternary(r=None) 與 ktier.gptq_grid(level 3) 的 T1 逐位相同。
A7 solve_r 為逐列 argmin:r±ε 誤差皆不低。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.checks_asym
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from p2_anchor.wringer import ktier
from p2_anchor.wringer.asym import (asym_V, dense_asym, err_h, fit_alpha_r,
                                      gptq_ternary, round_ternary, solve_r)
from p2_anchor.wringer.export import build_planes, decode_tensor
from p2_anchor.wringer.ktier import gptq_grid, init_alpha
from p2_anchor.wringer.state import dense_weight

torch.manual_seed(0)
fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


M, nB, B = 64, 8, 32
K = nB * B
dev = "cpu"
W = torch.randn(M, K) * 0.02
xs = torch.randn(4096, K) * (1 + torch.rand(K) * 3)
Sig = xs.T @ xs
a0 = init_alpha(W)
T3 = torch.sign(W.reshape(M, nB, B)) * (W.reshape(M, nB, B).abs() > 0.6 * a0)
ones = torch.ones(M, 1, 1)

# ---- A1 ----
check("A1 asym_V r=1 逐位", torch.equal(asym_V(T3, ones), T3))
check("A1 dense_asym r=1 逐位", torch.equal(dense_asym(T3, a0, ones), (a0 * T3).reshape(M, K)))

# ---- A2 / A3(cuda 小張量)----
if torch.cuda.is_available():
    inv = torch.randperm(K)
    r_rand = (0.7 + 0.6 * torch.rand(M, 1))
    ent = {"T1": T3.to(torch.int8), "T2": torch.zeros_like(T3, dtype=torch.int8), "inv": inv,
           "a0": a0, "c": 0.6, "grid": 3}
    w_sym = dense_weight(ent)
    w_r1 = dense_weight({**ent, "rneg": torch.ones(M, 1)})
    check("A2 dense_weight rneg=1 逐位", torch.equal(w_sym, w_r1))
    w_rr = dense_weight({**ent, "rneg": r_rand})
    man = dense_asym(T3, a0, r_rand.reshape(M, 1, 1))[:, inv].cuda()
    check("A2 dense_weight rneg 隨機 = 手算", torch.equal(w_rr, man),
          f"maxabs={(w_rr - man).abs().max():.2e}")
    tmp = Path(tempfile.mkdtemp())
    try:
        n = "model.language_model.layers.0.mlp.down_proj"
        planes = build_planes({n: {**ent, "rneg": r_rand}}, None, tmp, c=0.6, grid=3)
        z = np.load(planes)
        wd = torch.from_numpy(decode_tensor(z, n)).cuda()
        check("A3 位面包 rneg 逐位", torch.equal(wd.to(torch.bfloat16), w_rr.to(torch.bfloat16))
              and "rneg" in "".join(z.files), f"maxabs={(wd - w_rr).abs().max():.2e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 逐 block r + 2 bit 碼書
    from p2_anchor.wringer.asym import quantize_r
    r_blk = (0.7 + 0.6 * torch.rand(M, nB, 1))
    rq, cb = quantize_r(r_blk, 2)
    ent_b = {**ent, "rneg": rq.reshape(M, nB), "rneg_cb": cb}
    w_b = dense_weight(ent_b)
    man_b = dense_asym(T3, a0, rq)[:, inv].cuda()
    check("A2b dense_weight rneg 逐 block = 手算", torch.equal(w_b, man_b))
    tmp = Path(tempfile.mkdtemp())
    try:
        planes = build_planes({n: ent_b}, None, tmp, c=0.6, grid=3)
        z = np.load(planes)
        wd = torch.from_numpy(decode_tensor(z, n)).cuda()
        check("A3b 位面包 rneg_idx+cb 逐位", torch.equal(wd.to(torch.bfloat16), w_b.to(torch.bfloat16))
              and f"{n}.rneg_idx" in z.files and z[f"{n}.rneg_idx"].dtype == np.uint8 and len(cb) == 4,
              f"idx dtype={z[f'{n}.rneg_idx'].dtype} cb={len(cb)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
else:
    print("[SKIP] A2/A3 需 cuda")

# ---- A4 ----
W3 = W.reshape(M, nB, B)
al_s, _, tr_s = fit_alpha_r(W3, T3, Sig, asym=False)
al_a, r_a, tr_a = fit_alpha_r(W3, T3, Sig, alt=4)
mono = all(tr_a[i + 1] <= tr_a[i] * (1 + 1e-6) for i in range(len(tr_a) - 1))
check("A4 交替單調不增", mono, f"trace={[round(t, 6) for t in tr_a]}")
check("A4 asym ≤ sym(起點=sym 解)", tr_a[0] <= tr_s[0] * (1 + 1e-6) and tr_a[-1] <= tr_s[0],
      f"sym={tr_s[0]:.6f} asym0={tr_a[0]:.6f} asymK={tr_a[-1]:.6f} gain={1 - tr_a[-1] / tr_s[0]:.4f}")

# ---- A5 ----
r5 = 0.5 + torch.rand(M, 1, 1)
al5 = a0 * (torch.randn(M, nB, 1).sign())      # 含負 α
Tq = round_ternary(W3, al5, r5)
vals = torch.stack([-r5.expand(M, nB, B) * al5, torch.zeros(M, nB, B), al5.expand(M, nB, B)], -1)
best = (vals - W3.unsqueeze(-1)).abs().argmin(-1).float() - 1.0
ties = ((vals - W3.unsqueeze(-1)).abs().sort(-1).values[..., :2].diff(dim=-1).abs() < 1e-9).squeeze(-1)
check("A5 非對稱逐元最近值", torch.equal(Tq[~ties], best[~ties]),
      f"mismatch={(Tq != best)[~ties].sum().item()} ties={int(ties.sum())}")

# ---- A6 ----
ktier.set_block(B)
a0g, T1g, _ = gptq_grid(W, Sig, 3)
Tg = gptq_ternary(W, Sig, init_alpha(W), None)
check("A6 gptq_ternary(r=None) == gptq_grid(3)", torch.equal(Tg, T1g),
      f"diff={(Tg != T1g).float().mean():.2e}")

# ---- A7 ----
r_opt = solve_r(W3, T3, al_a, Sig)
Wf = W3.reshape(M, K)
e_opt = err_h(Wf - dense_asym(T3, al_a, r_opt), Sig)
ok7 = True
for eps in (0.02, -0.02):
    e_p = err_h(Wf - dense_asym(T3, al_a, (r_opt * (1 + eps)).clamp(0.25, 4)), Sig)
    ok7 &= e_p >= e_opt * (1 - 1e-7)
check("A7 solve_r argmin", ok7, f"e_opt={e_opt:.6f}")

print("ASYM_CHECKS", "FAIL" if fails else "PASS", fails)

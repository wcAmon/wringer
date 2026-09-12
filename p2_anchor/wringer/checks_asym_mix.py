"""E64 S1 混格非對稱 等價閘(合成資料;M2/M3 用 cuda 驗 state/export 路)。

M1 三元退化:asym_mix 各函數對 T2≡0 與 asym.py 逐位/allclose 一致(solve_r/solve_r_blk/fit/dense/round)。
M2 state.dense_weight:grid 9 + 逐 block rneg 隨機 = dense_asym_m 手算;rneg=1 = 無 rneg。
M3 export.build_planes/decode_tensor:grid 9 + rneg_idx/cb 位面包與 dense_weight bf16 逐位一致。
M4 fit_alpha_r_m(grid 9):交替軌跡單調不增;asym 終值 ≤ sym。
M5 round_grid_asym(grid 9, r 逐 block):逐元 == 暴力最近(非平局)。
M6 solve_r_blk_m 為逐 block argmin:r_g±ε 誤差皆不低(grid 9)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.checks_asym_mix
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from p2_anchor.wringer import ktier
from p2_anchor.wringer.asym import (dense_asym, err_h, fit_alpha_r, quantize_r,
                                      round_ternary, solve_r, solve_r_blk)
from p2_anchor.wringer.asym_mix import (asym_Vm, code_V0, dense_asym_m, fit_alpha_r_m,
                                          neg_mask, round_grid_asym, solve_r_blk_m, solve_r_m)
from p2_anchor.wringer.export import build_planes, decode_tensor
from p2_anchor.wringer.ktier import grid_pairs, init_alpha
from p2_anchor.wringer.state import dense_weight

torch.manual_seed(0)
fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


M, nB, B = 64, 8, 32
K = nB * B
C = 0.6
W = torch.randn(M, K) * 0.02
xs = torch.randn(4096, K) * (1 + torch.rand(K) * 3)
Sig = xs.T @ xs
W3 = W.reshape(M, nB, B)
a0 = init_alpha(W)
T3 = torch.sign(W3) * (W3.abs() > 0.6 * a0)


def rand_codes(level):
    pairs = torch.tensor(grid_pairs(level), dtype=torch.float32)
    idx = torch.randint(0, len(pairs), (M, nB, B))
    return pairs[idx, 0], pairs[idx, 1]


# ---- M1 三元退化 ----
V0, neg = code_V0(T3, torch.zeros_like(T3), C), neg_mask(T3)
r_row = solve_r(W3, T3, a0, Sig)
r_row_m = solve_r_m(W3, V0, neg, a0, Sig)
check("M1 solve_r 三元退化", torch.allclose(r_row, r_row_m, atol=1e-6), f"maxabs={(r_row-r_row_m).abs().max():.2e}")
r_blk = solve_r_blk(W3, T3, a0, Sig)
r_blk_m = solve_r_blk_m(W3, V0, neg, a0, Sig)
check("M1 solve_r_blk 三元退化", torch.allclose(r_blk, r_blk_m, atol=1e-6), f"maxabs={(r_blk-r_blk_m).abs().max():.2e}")
al_a, r_a, tr_a = fit_alpha_r(W3, T3, Sig, alt=3, blk=True)
al_m, r_m, tr_m = fit_alpha_r_m(W3, V0, neg, Sig, alt=3, blk=True)
check("M1 fit_alpha_r blk 三元退化", torch.allclose(al_a, al_m, atol=1e-6) and torch.allclose(r_a, r_m, atol=1e-6)
      and all(abs(x - y) <= 1e-6 * max(1.0, abs(x)) for x, y in zip(tr_a, tr_m)),
      f"trace_a={[round(t,6) for t in tr_a]} trace_m={[round(t,6) for t in tr_m]}")
check("M1 dense 三元退化(allclose,次序改與 dense_weight 同)", torch.allclose(dense_asym(T3, al_a, r_a), dense_asym_m(V0, neg, al_a, r_a), atol=1e-7, rtol=1e-6))
Tq = round_ternary(W3, al_a, r_a)
T1q, T2q = round_grid_asym(W3, al_a, r_a, 3, C)
vals3 = torch.stack([-r_a.expand(M, nB, B) * al_a, torch.zeros(M, nB, B), al_a.expand(M, nB, B)], -1)
ties = ((vals3 - W3.unsqueeze(-1)).abs().sort(-1).values[..., :2].diff(dim=-1).abs() < 1e-9).squeeze(-1)
check("M1 round 三元退化(非平局)", torch.equal(Tq[~ties], T1q[~ties]) and bool((T2q == 0).all()),
      f"mismatch={(Tq != T1q)[~ties].sum().item()} ties={int(ties.sum())}")

# ---- M2 / M3(cuda)----
T1_9, T2_9 = rand_codes(9)
V9, neg9 = code_V0(T1_9, T2_9, C), neg_mask(T1_9)
if torch.cuda.is_available():
    inv = torch.randperm(K)
    ent = {"T1": T1_9.to(torch.int8), "T2": T2_9.to(torch.int8), "inv": inv, "a0": a0, "c": C, "grid": 9}
    w_sym = dense_weight(ent)
    w_r1 = dense_weight({**ent, "rneg": torch.ones(M, nB)})
    check("M2 dense_weight grid9 rneg=1 逐位", torch.equal(w_sym, w_r1))
    r_b = 0.7 + 0.6 * torch.rand(M, nB, 1)
    rq, cb = quantize_r(r_b, 2)
    ent_b = {**ent, "rneg": rq.reshape(M, nB), "rneg_cb": cb}
    w_b = dense_weight(ent_b)
    man_b = dense_asym_m(V9, neg9, a0, rq)[:, inv].cuda()
    check("M2 dense_weight grid9 逐 block rneg = 手算", torch.equal(w_b, man_b), f"maxabs={(w_b-man_b).abs().max():.2e}")
    ent5 = {**ent, "grid": 5}
    T1_5, T2_5 = rand_codes(5)
    ent5.update({"T1": T1_5.to(torch.int8), "T2": T2_5.to(torch.int8), "rneg": rq.reshape(M, nB), "rneg_cb": cb})
    w5 = dense_weight(ent5)
    man5 = dense_asym_m(code_V0(T1_5, T2_5, C), neg_mask(T1_5), a0, rq)[:, inv].cuda()
    check("M2 dense_weight grid5 逐 block rneg = 手算", torch.equal(w5, man5))
    tmp = Path(tempfile.mkdtemp())
    try:
        n = "model.language_model.layers.0.mlp.down_proj"
        planes = build_planes({n: ent_b}, None, tmp, c=C, grid=9)
        z = np.load(planes)
        wd = torch.from_numpy(decode_tensor(z, n)).cuda()
        check("M3 位面包 grid9 rneg_idx+cb 逐位", torch.equal(wd.to(torch.bfloat16), w_b.to(torch.bfloat16))
              and f"{n}.rneg_idx" in z.files, f"maxabs={(wd - w_b).abs().max():.2e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
else:
    print("[SKIP] M2/M3 需 cuda")

# ---- M4 ----
al_s9, _, tr_s9 = fit_alpha_r_m(W3, V9, neg9, Sig, asym=False)
al_9, r_9, tr_9 = fit_alpha_r_m(W3, V9, neg9, Sig, alt=4, blk=True)
mono = all(tr_9[i + 1] <= tr_9[i] * (1 + 1e-6) for i in range(len(tr_9) - 1))
check("M4 grid9 交替單調不增", mono, f"trace={[round(t, 6) for t in tr_9]}")
check("M4 grid9 asym ≤ sym", tr_9[0] <= tr_s9[0] * (1 + 1e-6) and tr_9[-1] <= tr_s9[0],
      f"sym={tr_s9[0]:.6f} asymK={tr_9[-1]:.6f} gain={1 - tr_9[-1] / tr_s9[0]:.4f}")

# ---- M5 ----
r5 = 0.5 + torch.rand(M, nB, 1)
al5 = a0 * torch.randn(M, nB, 1).sign()
T1r, T2r = round_grid_asym(W3, al5, r5, 9, C)
vals, t1g, t2g = ktier.grid_tensors(9, C, "cpu")
cand = torch.where(t1g.view(1, 1, 1, -1) < 0, vals.view(1, 1, 1, -1) * r5.unsqueeze(-1),
                   vals.view(1, 1, 1, -1).expand(M, nB, 1, -1)) * al5.unsqueeze(-1)
d = (cand - W3.unsqueeze(-1)).abs()
best = d.argmin(-1)
ties5 = (d.sort(-1).values[..., :2].diff(dim=-1).abs() < 1e-9).squeeze(-1)
ok5 = torch.equal(T1r[~ties5], t1g[best][~ties5]) and torch.equal(T2r[~ties5], t2g[best][~ties5])
check("M5 grid9 非對稱逐元最近值", ok5, f"ties={int(ties5.sum())}")

# ---- M6 ----
r_opt = solve_r_blk_m(W3, V9, neg9, al_9, Sig)
Wf = W3.reshape(M, K)
e_opt = err_h(Wf - dense_asym_m(V9, neg9, al_9, r_opt), Sig)
ok6 = True
for eps in (0.02, -0.02):
    e_p = err_h(Wf - dense_asym_m(V9, neg9, al_9, (r_opt * (1 + eps)).clamp(0.25, 4)), Sig)
    ok6 &= e_p >= e_opt * (1 - 1e-7)
check("M6 solve_r_blk_m argmin(grid9)", ok6, f"e_opt={e_opt:.6f}")

print("ASYM_MIX_CHECKS", "FAIL" if fails else "PASS", fails)

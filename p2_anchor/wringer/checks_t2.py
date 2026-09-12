"""步驟 3 T2 探針閘(合成;小 cuda 走 state/export 路)。
D1 round_t2 = 對 9 值暴力最近(T1 固定下三選一)。
D2 t2fix 強制 T2=0 時 == ctrl 逐位(α 閉式同式)。
D3 grid 9 條目 → dense_weight → build_planes/decode bf16 逐位;混合 grid 字典(3+9)export meta.grid=0 且各 tensor 逐位。
D4 gptq_grid level 9 回傳 T1,T2 ∈{−1,0,1} 且 V 值域 9 值。
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.checks_t2
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from p2_anchor.wringer import ktier
from p2_anchor.wringer.export import build_planes, decode_tensor
from p2_anchor.wringer.ktier import gptq_grid, grid_tensors, init_alpha, scale_joint
from p2_anchor.wringer.probe_t2 import round_t2
from p2_anchor.wringer.state import dense_weight

torch.manual_seed(0)
fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


M, nB, B = 64, 8, 32
K = nB * B
c = 0.6
ktier.set_block(B)
W = torch.randn(M, K) * 0.02
xs = torch.randn(4096, K) * (1 + torch.rand(K) * 3)
Sig = xs.T @ xs
a0 = init_alpha(W)
W3 = W.reshape(M, nB, B)
T1 = torch.sign(W3) * (W3.abs() > 0.6 * a0)
# D1
T2 = round_t2(W3, T1, a0, c)
u = W3 / a0
cand = torch.stack([T1 + c * t for t in (-1.0, 0.0, 1.0)], -1)
best = (cand - u.unsqueeze(-1)).abs().argmin(-1).float() - 1.0
ties = ((cand - u.unsqueeze(-1)).abs().sort(-1).values[..., :2].diff(dim=-1).abs() < 1e-9).squeeze(-1)
check("D1 round_t2 = 暴力最近", torch.equal(T2[~ties], best[~ties]), f"ties={int(ties.sum())}")
# D2
al_c = scale_joint(W3, T1, Sig)
al_0 = scale_joint(W3, T1 + c * torch.zeros_like(T1), Sig)
check("D2 T2=0 ⇒ α 閉式與 ctrl 逐位", torch.equal(al_c, al_0))
# D3
if torch.cuda.is_available():
    inv = torch.randperm(K)
    e9 = {"T1": T1.to(torch.int8), "T2": T2.to(torch.int8), "inv": inv, "a0": al_c, "c": c, "grid": 9}
    e3 = {"T1": T1.to(torch.int8), "T2": torch.zeros_like(T1, dtype=torch.int8), "inv": inv, "a0": al_c, "c": c, "grid": 3}
    w9 = dense_weight(e9)
    man = (al_c * (T1 + c * T2)).reshape(M, K)[:, inv].cuda()
    check("D3 grid9 dense_weight = 手算", torch.equal(w9, man))
    tmp = Path(tempfile.mkdtemp())
    try:
        n9 = "model.language_model.layers.0.mlp.down_proj"
        n3 = "model.language_model.layers.0.mlp.up_proj"
        z = np.load(build_planes({n9: e9, n3: e3}, None, tmp, c=c, grid=0))
        d9 = torch.from_numpy(decode_tensor(z, n9)).cuda()
        d3 = torch.from_numpy(decode_tensor(z, n3)).cuda()
        check("D3 混合 grid 位面 decode bf16 逐位", torch.equal(d9.to(torch.bfloat16), w9.to(torch.bfloat16))
              and torch.equal(d3.to(torch.bfloat16), dense_weight(e3).to(torch.bfloat16))
              and int(z["meta.grid"]) == 0 and int(z[f"{n9}.grid"]) == 9 and int(z[f"{n3}.grid"]) == 3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
else:
    print("[SKIP] D3 需 cuda")
# D4
_, g1, g2 = gptq_grid(W, Sig, 9, c=c)
vals, _, _ = grid_tensors(9, c, "cpu")
V = (g1 + c * g2).reshape(-1)
check("D4 gptq_grid(9) T ∈{−1,0,1}、V 9 值", bool(torch.isin(g1, torch.tensor([-1., 0., 1.])).all())
      and bool(torch.isin(g2, torch.tensor([-1., 0., 1.])).all()) and int(V.unique().numel()) <= 9,
      f"uniq={int(V.unique().numel())}")
# D5/D6 t2beta 殘差位面
from p2_anchor.wringer.probe_t2 import fit_t2beta
water = torch.randn(M, K) * 0.003
W3t = (W + water).reshape(M, nB, B)
T2b, al_b, beta, tr = fit_t2beta(W3t, T1, al_c, Sig, 3)
e_c = float((((W3t - al_c * T1).reshape(M, K) @ Sig) * (W3t - al_c * T1).reshape(M, K)).sum())
check("D5 fit_t2beta 單調且 < ctrl", all(tr[i + 1] <= tr[i] * (1 + 1e-6) for i in range(len(tr) - 1)) and tr[-1] < e_c,
      f"ratio={tr[-1]/e_c:.4f} t2_nz={float((T2b != 0).float().mean()):.3f} beta/alpha p50={float((beta/al_b).abs().median()):.3f}")
if torch.cuda.is_available():
    inv = torch.randperm(K)
    eb = {"T1": T1.to(torch.int8), "T2": T2b.to(torch.int8), "inv": inv, "a0": al_b, "c": c, "grid": 9,
          "t2beta": beta.reshape(M, nB)}
    wb = dense_weight(eb)
    manb = (al_b * T1 + beta * T2b).reshape(M, K)[:, inv].cuda()
    check("D6 t2beta dense_weight = 手算", torch.equal(wb, manb))
    # β = c·α 時與舊 k2 路逐位一致
    ec = {**eb, "t2beta": (c * al_b).reshape(M, nB)}
    e9c = {k: v for k, v in eb.items() if k != "t2beta"}
    check("D6 β=c·α ⇒ 與 k2_weight 舊路 bf16 逐位", torch.equal(dense_weight(ec).to(torch.bfloat16), dense_weight(e9c).to(torch.bfloat16)))
    tmp = Path(tempfile.mkdtemp())
    try:
        nb_ = "model.language_model.layers.1.mlp.down_proj"
        z = np.load(build_planes({nb_: eb, "model.language_model.layers.1.mlp.up_proj": e3}, None, tmp, c=c, grid=0))
        db = torch.from_numpy(decode_tensor(z, nb_)).cuda()
        check("D6 t2beta 位面 decode bf16 逐位", torch.equal(db.to(torch.bfloat16), wb.to(torch.bfloat16)) and f"{nb_}.t2beta" in z.files)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
print("T2_CHECKS", "FAIL" if fails else "PASS", fails)

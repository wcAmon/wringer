"""E62b S1 等價閘(輸出側 L 桶;對應 E61 G1–G5)。

H1 L=I:STQParam(lmix_block=128) hard 前向 逐位 == STQParam(lmix_block=0)(mix 同開)。
H2 隨機 L:parametrize 前向(einsum 路)vs state.dense_weight(block_diag 左乘路)bf16 ≤1 ulp;隨機 x 前傳 rel<1e-3。
H3 落盤/位面包:條目含 mix+lmix → dense_weight;export.build_planes/decode + mix/lmix 與 apply_k2 路 bf16 逐位一致。
H4 梯度:soft 前向 loss.backward 後 lmix.grad 非零、且 I 塊外亦非零。
H5 solve_left_ls 一致性:對 Z=C·Bd(X) 解 L 後 ‖W_t − L Z‖_H 不增;L=I 起點、cap 大時 gain>0。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.checks_e62b
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from p2_anchor.wringer.export import build_planes, decode_tensor
from p2_anchor.wringer.mixls import solve_left_ls
from p2_anchor.wringer.state import dense_weight, mix_dense
from p2_anchor.wringer.stq import STQParam, SoftGrid

torch.manual_seed(0)
st = torch.load("data/qs_st54r/layer00.pt", weights_only=False)
n = "model.language_model.layers.0.mlp.down_proj"
B = 128
fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


c = st[n]
W0 = dense_weight(c).to(torch.bfloat16)
Mo, K = W0.shape
grid = SoftGrid(c=float(c["c"]), n=int(c["grid"]))


def make(lmix_block):
    lin = nn.Linear(K, Mo, bias=False).cuda().to(torch.bfloat16)
    lin.weight.data.copy_(W0)
    p = STQParam(c["a0"].cuda(), grid, r=64, c=float(c["c"]),
                 learn_theta=True, use_comp=True, mix_block=B, lmix_block=lmix_block)
    parametrize.register_parametrization(lin, "weight", p)
    p.mode = "hard"
    with torch.no_grad():
        _ = lin.weight
    return lin, p


# ---- H1 ----
lin0, p0 = make(0)
lin1, p1 = make(B)
with torch.no_grad():
    w0, w1 = lin0.weight.clone(), lin1.weight.clone()
check("H1 L=I 逐位", torch.equal(w0, w1), f"maxabs={(w0.float()-w1.float()).abs().max():.3e}")

# ---- H2 ----
with torch.no_grad():
    Lr = torch.eye(B, device="cuda").expand(Mo // B, B, B).clone() + 0.05 * torch.randn(Mo // B, B, B, device="cuda")
    Mr = torch.eye(B, device="cuda").expand(K // B, B, B).clone() + 0.05 * torch.randn(K // B, B, B, device="cuda")
    p1.lmix.copy_(Lr)
    p1.mix.copy_(Mr)
    p0.mix.copy_(Mr)
    w_param = lin1.weight.clone()
    T1, T2, al, w_hard = p1.extract(lin1.parametrizations.weight.original)
    entry = {"T1": T1.cpu(), "T2": T2.cpu(), "inv": torch.arange(K), "a0": al.cpu(),
             "c": p1.c, "grid": p1.grid.n, "mix": Mr.cpu(), "lmix": Lr.cpu()}
    w_dense = dense_weight(entry).to(torch.bfloat16)
    ulp = ((w_param.float() - w_dense.float()).abs() / w_dense.float().abs().clamp_min(1e-6)).max()
    x = torch.randn(8, 64, K, device="cuda", dtype=torch.bfloat16)
    y1 = F.linear(x, w_param).float(); y2 = F.linear(x, w_dense).float()
    rel = float((y1 - y2).norm() / y2.norm())
check("H2 隨機 L/M 材化一致", float(ulp) <= 2 ** -7 and rel < 1e-3, f"rel_ulp={float(ulp):.2e} fwd_rel={rel:.2e}")

# ---- H3 ----
tmp = Path(tempfile.mkdtemp())
try:
    planes = build_planes({n: entry}, None, tmp, c=entry["c"], grid=entry["grid"])
    z = np.load(planes)
    w = torch.from_numpy(decode_tensor(z, n)).cuda()
    w = w @ mix_dense(torch.from_numpy(z[f"{n}.mix"]).cuda())
    w = mix_dense(torch.from_numpy(z[f"{n}.lmix"]).cuda()) @ w
    check("H3 落盤/位面包 lmix", torch.equal(w.to(torch.bfloat16), w_dense),
          f"maxabs={(w.to(torch.bfloat16).float()-w_dense.float()).abs().max():.3e}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---- H4 ----
p1.mode = "soft"; p1.s = 30.0
x = torch.randn(2, 32, K, device="cuda", dtype=torch.bfloat16)
loss = lin1(x).float().pow(2).mean()
loss.backward()
g = p1.lmix.grad
off = g.clone()
idx = torch.arange(B, device="cuda")
off[:, idx, idx] = 0
check("H4 L 梯度", g is not None and float(g.abs().max()) > 0 and float(off.abs().max()) > 0,
      f"|g|max={float(g.abs().max()):.2e} offdiag={float(off.abs().max()):.2e}")

# ---- H5 ----
with torch.no_grad():
    C = w_hard.float()
    Z = C @ mix_dense(Mr)
    xs = torch.randn(4096, K, device="cuda"); H = xs.T @ xs
    W_t = W0.float() + 0.02 * torch.randn_like(W0.float())
    L0 = torch.eye(Mo, device="cuda")
    Ln, stt = solve_left_ls(Z, W_t, H, L0, B, ridge=0.0, cap=1.0, iters=40)
    check("H5 solve_left_ls gain>0 且不增", stt["obj_after"] <= stt["obj_before"] and stt["ls_gain"] > 0,
          f"gain={stt['ls_gain']} t={stt['t']} sv_min={stt['sv_min']} cg={stt['cg_iters']}")
    # 塊外恆零
    Lb = Ln.reshape(Mo // B, B, Mo // B, B)
    offblk = Lb.clone(); ii = torch.arange(Mo // B, device="cuda"); offblk[ii, :, ii, :] = 0
    check("H5b L 塊外恆零", float(offblk.abs().max()) == 0.0)

print("E62B_CHECKS", "FAIL" if fails else "PASS", fails)

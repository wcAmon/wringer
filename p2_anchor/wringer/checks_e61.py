"""E61 S0 等價閘(prereg_e61_bucket engineering_S0.gates_before_S1)。

G1 M=I:STQParam(mix_block=128) hard 前向 逐位 == STQParam(mix_block=0)。
G2 隨機 M:parametrize 前向權重(einsum 路)vs state.dense_weight(block_diag 路)
   → bf16 逐元 ≤1 ulp、隨機 x 前傳 rel<1e-3。
G3 comp 座標恆等式:‖W_t − w_hard Bd‖_H == ‖W_t Bd⁻¹ − w_hard‖_{Bd H Bdᵀ}。
G4 落盤/解碼:qs 條目含 mix → load_state → dense_weight 一致;
   export.build_planes/decode_tensor + mix 與 apply_k2 路 bf16 逐位一致。
G5 梯度:soft 前向 loss.backward 後 mix.grad 非零、且 I 塊外亦非零。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.checks_e61
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
from p2_anchor.wringer.state import dense_weight, load_state, mix_dense
from p2_anchor.wringer.stq import STQParam, SoftGrid

torch.manual_seed(0)
st = torch.load("data/qs_st54r/layer00.pt", weights_only=False)
names = ["model.language_model.layers.0.linear_attn.in_proj_z",
         "model.language_model.layers.0.mlp.down_proj"]
B = 128
fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


for n in names:
    c = st[n]
    W0 = dense_weight(c).to(torch.bfloat16)          # 起點(冠軍態材化)
    Mo, K = W0.shape
    grid = SoftGrid(c=float(c["c"]), n=int(c["grid"]))

    def make(mix_block):
        lin = nn.Linear(K, Mo, bias=False).cuda().to(torch.bfloat16)
        lin.weight.data.copy_(W0)
        p = STQParam(c["a0"].cuda(), grid, r=64, c=float(c["c"]),
                     learn_theta=True, use_comp=True, mix_block=mix_block)
        parametrize.register_parametrization(lin, "weight", p)
        p.mode = "hard"
        with torch.no_grad():
            _ = lin.weight                           # lazy init
        return lin, p

    # ---- G1 ----
    lin0, p0 = make(0)
    lin1, p1 = make(B)
    with torch.no_grad():
        w0, w1 = lin0.weight, lin1.weight
    check(f"G1 M=I 逐位 {n.split('.')[-1]}", torch.equal(w0, w1),
          f"maxabs {float((w0.float() - w1.float()).abs().max()):.3e}")

    # ---- G2 ----
    with torch.no_grad():
        p1.mix.add_(0.05 * torch.randn_like(p1.mix))
        w_par = lin1.weight                          # einsum 路 → bf16
        T1, T2, al, w_hard = p1.extract(lin1.parametrizations.weight.original)
        entry = {"T1": T1.cpu(), "T2": T2.cpu(), "inv": torch.arange(K),
                 "a0": al.cpu(), "c": float(c["c"]), "grid": int(c["grid"]),
                 "mix": p1.mix.detach().cpu()}
        w_den = dense_weight(entry).to(torch.bfloat16)   # block_diag 路
        d = (w_par.float() - w_den.float()).abs()
        ulp = (w_den.float().abs() * 2 ** -7).clamp_min(1e-30)
        frac_ne = float((w_par != w_den).float().mean())
        x = torch.randn(4, 64, K, device="cuda", dtype=torch.bfloat16)
        y1 = F.linear(x, w_par).float()
        y2 = F.linear(x, w_den).float()
        rel = float((y1 - y2).norm() / y2.norm())
    check(f"G2 隨機M 材化一致 {n.split('.')[-1]}",
          bool((d <= ulp * 1.01).all()) and rel < 1e-3,
          f"不等元 {frac_ne:.2e} maxulp {float((d / ulp).max()):.2f} fwd rel {rel:.2e}")

    # ---- G3 ----
    with torch.no_grad():
        xs = torch.randn(2 * K, K, device="cuda")     # 滿秩 H(K=9216 時 4096 樣本會奇異)
        H = xs.T @ xs
        W_t = W0.float() + 0.01 * torch.randn_like(W0.float())
        Bd = p1.mix_dense()
        Rd = W_t - w_hard @ Bd
        lhs = float(((Rd @ H) * Rd).sum())
        Rc = W_t @ p1.mix_dense_inv() - w_hard
        Hp = Bd @ H @ Bd.T
        rhs = float(((Rc @ Hp) * Rc).sum())
    check(f"G3 comp 恆等式 {n.split('.')[-1]}", abs(lhs - rhs) / lhs < 1e-4,
          f"lhs {lhs:.6e} rhs {rhs:.6e} rel {abs(lhs - rhs) / lhs:.2e}")

    # ---- G4 ----
    tmp = Path(tempfile.mkdtemp(dir="/tmp/claude-1000")) if Path(
        "/tmp/claude-1000").exists() else Path(tempfile.mkdtemp())
    qs = Path("data") / "qs_e61chk"
    qs.mkdir(parents=True, exist_ok=True)
    torch.save({n: entry}, qs / "layer00.pt")
    st2 = load_state("e61chk", [n])
    w_re = dense_weight(st2[n]).to(torch.bfloat16)
    planes = build_planes(st2, None, tmp, c=float(c["c"]),
                          grid=int(c["grid"]))
    z = np.load(planes)
    w_dec = torch.from_numpy(decode_tensor(z, n)).cuda()
    w_dec = (w_dec @ mix_dense(torch.from_numpy(z[f"{n}.mix"]).cuda())
             ).to(torch.bfloat16)
    check(f"G4 落盤/位面包 {n.split('.')[-1]}",
          torch.equal(w_re, w_den) and torch.equal(w_dec, w_den),
          f"reload {torch.equal(w_re, w_den)} planes {torch.equal(w_dec, w_den)}")
    shutil.rmtree(qs, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)

    # ---- G5 ----
    p1.mode, p1.s = "soft", 2.0
    x = torch.randn(2, 32, K, device="cuda", dtype=torch.bfloat16)
    y = lin1(x)
    loss = y.float().pow(2).mean()
    loss.backward()
    g = p1.mix.grad
    off = g.clone()
    idx = torch.arange(B, device=g.device)
    off[:, idx, idx] = 0
    check(f"G5 mix 梯度 {n.split('.')[-1]}",
          g is not None and float(g.norm()) > 0 and float(off.norm()) > 0,
          f"|g| {float(g.norm()):.3e} |g_offdiag| {float(off.norm()):.3e}")
    # ---- G6(amendment_1)零水殘差對 M 不變:R_code(M≠I, water=0) == W0q − w_hard ----
    with torch.no_grad():
        W0q = lin1.parametrizations.weight.original
        _, _, _, w_h = p1.extract(W0q)
        Bdi = p1.mix_dense_inv()
        water = torch.zeros_like(W0q.float())
        R_fix = W0q.float() + water @ Bdi - w_h
        R_ref = W0q.float() - w_h
        R_old = W0q.float() @ Bdi - w_h              # 舊式(S1 st61a 所用)
        spur = float((R_old - R_ref).norm() / R_ref.norm().clamp_min(1e-12))
    check(f"G6 零水殘差 M-不變 {n.split('.')[-1]}", torch.equal(R_fix, R_ref),
          f"新式 diff {float((R_fix - R_ref).norm()):.2e} | 舊式偽殘差/真殘差 {spur:.3f}")
    # ---- G7–G9(st61c)(α,M) 閉式重解 ----
    from p2_anchor.wringer.mixls import _bd_apply, _bd_proj, refit_alpha, solve_mix_ls
    from p2_anchor.wringer.ktier import scale_joint
    with torch.no_grad():
        Hh = H / H.diagonal().mean()
        Xc = p1.mix.detach().float().clone()
        Cm = w_h.float()
        Wt = W0.float() + 0.02 * torch.randn_like(W0.float())   # 錨定目標(含「水」)
        # G7 目標單調 + 大 ridge 不動
        Xn, st7 = solve_mix_ls(Cm, Wt, Hh, Xc, ridge=1.0, cap=0.05, iters=25)
        Xr, st7r = solve_mix_ls(Cm, Wt, Hh, Xc, ridge=1e8, cap=0.05, iters=5)
        mono = st7["obj_after"] <= st7["obj_before"] * (1 + 1e-6)
        still = float((Xr - Xc).norm() / Xc.norm()) < 1e-4
    check(f"G7 mixLS 目標單調/大ridge不動 {n.split('.')[-1]}", mono and still,
          f"gain {st7['ls_gain']:.4f} step {st7['step_max']:.4f} t {st7['t']} cg {st7['cg_iters']} | ridge1e8 rel {float((Xr - Xc).norm() / Xc.norm()):.1e}")
    with torch.no_grad():
        # G8 法方程殘差(生產設定 ridge 1.0 / 25 迭代;無 cap)
        Xf, _ = solve_mix_ls(Cm, Wt, Hh, Xc, ridge=1.0, cap=1e9, iters=25, tol=1e-8)
        CtC = Cm.T @ Cm
        lam = 1.0 * float(CtC.diagonal().mean() * Hh.diagonal().mean())
        lhs = _bd_proj(_bd_apply(CtC, Xf) @ Hh, *Xf.shape[:2]) + lam * Xf
        rhs = _bd_proj((Cm.T @ Wt) @ Hh, *Xf.shape[:2]) + lam * Xc
        rel8 = float((lhs - rhs).norm() / rhs.norm())
    check(f"G8 mixLS 法方程殘差 {n.split('.')[-1]}", rel8 < 1e-2, f"rel {rel8:.2e}")
    with torch.no_grad():
        # G9 α 重解往返:未夾處 α_eff == scale_joint 解;Σ' 目標不升
        Bd = p1.mix_dense()
        Sp = Bd @ Hh @ Bd.T
        Wc = Wt @ p1.mix_dense_inv()
        _, _, _, w_h2 = p1.extract(lin1.parametrizations.weight.original)
        V3 = (w_h2 / (2.0 * torch.sigmoid(p1.logit) * p1.a0).expand(-1, -1, w_h2.shape[1] // p1.a0.shape[1]).reshape(w_h2.shape)).reshape(p1.a0.shape[0], p1.a0.shape[1], -1)
        al_ref = scale_joint(Wc.reshape(V3.shape), V3, Sp)
        e0 = Wc - w_h2.float()
        o0 = float(((e0 @ Sp) * e0).sum())
        al_new, st9 = refit_alpha(p1, Wc, V3, Sp)
        e1 = Wc - (al_new * V3).reshape(Wc.shape)
        o1 = float(((e1 @ Sp) * e1).sum())
        unclipped = ((al_ref / (2 * p1.a0)) > 0.02) & ((al_ref / (2 * p1.a0)) < 0.98)
        rt = float(((al_new - al_ref)[unclipped]).abs().max() / al_ref[unclipped].abs().max())
    check(f"G9 αLS 往返/目標不升 {n.split('.')[-1]}", rt < 1e-4 and o1 <= o0 * (1 + 1e-6),
          f"roundtrip {rt:.1e} obj {o0:.4e}→{o1:.4e} clip {st9['alpha_clip_frac']:.4f} Δα {st9['alpha_rel_change']:.4f}")
    del lin0, lin1, p0, p1
    torch.cuda.empty_cache()

print("E61_CHECKS_" + ("PASS" if not fails else "FAIL:" + ",".join(fails)),
      flush=True)

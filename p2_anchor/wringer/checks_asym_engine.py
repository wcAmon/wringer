"""E63 S1 引擎閘(合成資料;CPU 為主,C3 用小 cuda 走 state/export 路)。

C1  greedy_comp_grid(rneg=None) 與 HEAD 版逐位相同(comp/R/回傳);rneg≡1 亦逐位相同。
C1b STQParam(rneg=None) hard/soft 前向與 HEAD 版逐位相同(同參數)。
C2  閉式段:(α,r) 交替 → 2 bit 量化 → α 重擬合 後 err_h ≤ greedy 後;r_q 值全在碼書;碼書 4 值。
C3  set_rneg 後 hard 前向 == α·V(T,r_q) 手算;extract→entry→dense_weight/export decode bf16 逐位。
C5  rneg 存在時 greedy 的部署位移與潛位移分離:comp 落格點(u2==vals[tgt])且 R 以部署值更新。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.checks_asym_engine <old_dir>
"""
import copy
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from p2_anchor.wringer import ktier
from p2_anchor.wringer.asym import asym_V, dense_asym, err_h, quantize_r, solve_r_blk
from p2_anchor.wringer.comp import greedy_comp_grid
from p2_anchor.wringer.ktier import init_alpha, round_with_mids, scale_joint
from p2_anchor.wringer.mixls import refit_alpha
from p2_anchor.wringer.stq import STQParam, SoftGrid

old_dir = Path(sys.argv[1])


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


comp_old = load("comp_old", old_dir / "comp_old.py")
stq_old = load("stq_old", old_dir / "stq_old.py")
torch.manual_seed(0)
fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


dev = "cpu"
M, K = 64, 256
ktier.set_block(32)
nB = K // 32
W = torch.randn(M, K) * 0.02
xs = torch.randn(4096, K) * (1 + torch.rand(K) * 3)
H = xs.T @ xs
water = torch.randn(M, K) * 0.004
a0 = init_alpha(W)
grid = SoftGrid(c=0.6, device=dev, n=3)


def make(cls, seed=1):
    torch.manual_seed(seed)
    p = cls(a0.clone(), grid, r=8, c=0.6, learn_theta=True, use_comp=True)
    p.mode = "hard"
    with torch.no_grad():
        p(W)                        # lazy init
        p.lora_B.normal_(std=1e-3)
        p.logit.normal_(std=0.1)
        p.dtheta.normal_(std=0.1)
    return p


# ---- C1 ----
p_new = make(STQParam)
p_old = make(STQParam)
with torch.no_grad():
    u, al = p_new._u(W)
    _, _, vh = round_with_mids(u, p_new.theta_eff(), grid.vals, grid.t1, grid.t2)
    R0 = (W + water) - (al * vh).reshape(M, K)
    c_new = greedy_comp_grid(p_new, W, R0.clone(), H, rounds=3, cap=0.03)
    c_old = comp_old.greedy_comp_grid(p_old, W, R0.clone(), H, rounds=3, cap=0.03)
check("C1 greedy rneg=None == HEAD 逐位",
      torch.equal(p_new.comp, p_old.comp) and c_new == c_old, f"flips={c_new['flips']} absorb={c_new['absorb']}")
p_one = make(STQParam)
with torch.no_grad():
    c_one = greedy_comp_grid(p_one, W, R0.clone(), H, rounds=3, cap=0.03,
                             rneg=torch.ones(M, nB, 1))
check("C1 greedy rneg≡1 == None 逐位", torch.equal(p_one.comp, p_new.comp) and c_one == c_new)

# ---- C1b ----
q_new = make(STQParam)
q_old = make(stq_old.STQParam)
with torch.no_grad():
    q_old.lora_A.copy_(q_new.lora_A); q_old.lora_B.copy_(q_new.lora_B)
    q_old.logit.copy_(q_new.logit); q_old.dtheta.copy_(q_new.dtheta)
    q_old.comp.copy_(q_new.comp)
    ok = True
    for mode, s in (("hard", 0.0), ("soft", 30.0)):
        q_new.mode = q_old.mode = mode; q_new.s = q_old.s = s
        ok &= torch.equal(q_new(W), q_old(W))
check("C1b STQParam(rneg=None) hard/soft == HEAD 逐位", ok)

# ---- C2 ----
q = make(STQParam)
W_t = W + water
with torch.no_grad():
    u_, al_ = q._u(W)
    _, _, v_h = round_with_mids(u_, q.theta_eff(), grid.vals, grid.t1, grid.t2)
    T3 = v_h.float(); W3 = W_t.reshape(T3.shape)
    alpha = al_.float()
    e_g = err_h(W_t - (alpha * T3).reshape(M, K), H)
    r = None
    tr = [e_g]
    for _ in range(2):
        r = solve_r_blk(W3, T3, alpha, H)
        alpha = scale_joint(W3, asym_V(T3, r), H)
        tr.append(err_h(W_t - (alpha * asym_V(T3, r)).reshape(M, K), H))
    r_q, cb = quantize_r(r, 2)
    _, ast = refit_alpha(q, W_t, asym_V(T3, r_q), H)
    q.set_rneg(r_q, cb)
    _, al_n = q._u(W)
    e_a = err_h(W_t - (al_n * q._apply_rneg(T3)).reshape(M, K), H)
in_cb = bool(torch.isin(r_q.reshape(-1), cb).all())
check("C2 閉式段 err_h 單調且量化後 ≤ greedy 後", all(tr[i + 1] <= tr[i] * (1 + 1e-6) for i in range(len(tr) - 1)) and e_a <= e_g,
      f"e_g={e_g:.5f} alt={[round(t, 5) for t in tr[1:]]} e_a(q2+α)={e_a:.5f} gain={1 - e_a / e_g:.3f} clip={ast['alpha_clip_frac']}")
check("C2 r_q ∈ 碼書、碼書 4 值", in_cb and cb.numel() == 4, f"cb={[round(float(x), 3) for x in cb]}")

# ---- C3 ----
with torch.no_grad():
    q.mode = "hard"
    out = q(W)
    u_n, _ = q._u(W)
    _, _, v_n = round_with_mids(u_n, q.theta_eff(), grid.vals, grid.t1, grid.t2)
    man = dense_asym(v_n.float(), al_n, r_q)
check("C3 hard 前向 == α·V(T',r_q) 手算 逐位(T'=α 重擬合後碼)", torch.equal(out.float(), man),
      f"refit_flip={float((v_n != v_h).float().mean()):.4f}")
with torch.no_grad():
    T1, T2, alx, w_hard = q.extract(W)
check("C3 extract w_hard == 前向", torch.equal(w_hard, out.float()))
if torch.cuda.is_available():
    from p2_anchor.wringer.export import build_planes, decode_tensor
    from p2_anchor.wringer.state import dense_weight
    entry = {"T1": T1, "T2": T2, "inv": torch.arange(K), "a0": alx, "c": 0.6, "grid": 3,
             "rneg": q.rneg.reshape(M, nB), "rneg_cb": q.rneg_cb}
    wd = dense_weight(entry)
    check("C3 entry→dense_weight bf16 逐位", torch.equal(wd.to(torch.bfloat16), out.cuda().to(torch.bfloat16)))
    tmp = Path(tempfile.mkdtemp())
    try:
        n = "model.language_model.layers.0.mlp.down_proj"
        z = np.load(build_planes({n: entry}, None, tmp, c=0.6, grid=3))
        wdd = torch.from_numpy(decode_tensor(z, n)).cuda()
        check("C3 位面包 rneg_idx+cb decode bf16 逐位", torch.equal(wdd.to(torch.bfloat16), wd.to(torch.bfloat16))
              and z[f"{n}.rneg_idx"].dtype == np.uint8)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
else:
    print("[SKIP] C3 state/export 需 cuda")

# ---- C5 ----
q5 = make(STQParam)
with torch.no_grad():
    q5.set_rneg(r_q, cb)
    u, al = q5._u(W)
    _, _, vh = round_with_mids(u, q5.theta_eff(), grid.vals, grid.t1, grid.t2)
    R5 = W_t - (al * q5._apply_rneg(vh)).reshape(M, K)
    e_before = err_h(R5, H)
    c5 = greedy_comp_grid(q5, W, R5.clone(), H, rounds=3, cap=0.03, rneg=q5.rneg)
    u2, al2 = q5._u(W)
    _, _, vh2 = round_with_mids(u2, q5.theta_eff(), grid.vals, grid.t1, grid.t2)
    R5b = W_t - (al2 * q5._apply_rneg(vh2)).reshape(M, K)
    e_after = err_h(R5b, H)
    moved = (q5.comp != 0).reshape(M, nB, 32)
    on_grid = bool(((u2[moved] - torch.round(u2[moved])).abs() < 1e-5).all()) if moved.any() else True
    n_ch = int((vh2 != vh).sum())
check("C5 rneg 下 greedy:部署 err_h 下降、落格點(1e-5)、每步皆真翻碼", e_after < e_before and on_grid and c5["flips"] == n_ch > 0,
      f"before={e_before:.5f} after={e_after:.5f} clog_after={c5['obj_after']:.5f} flips={c5['flips']}")
print("ASYM_ENGINE_CHECKS", "FAIL" if fails else "PASS", fails)

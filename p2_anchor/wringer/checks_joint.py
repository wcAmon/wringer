"""E64 S3 聯合求解器 CPU 閘(合成 toy,與 checks_e37_cpu 同構)。

C4 joint_comp_grid(d=1, allowed=None) 逐位 == greedy_comp_grid(flips/obj/comp 相同)。
C5 joint_comp_grid(d=4, allowed=全 True) == joint_comp_grid(d=4, allowed=None)。
C6 joint(d=4) 的 obj_after ≤ greedy(相關欄 H 上聯合打分不應更差;報告,不硬裁)。
C7 allowed=隨機子集:事件後所有 d 元組 ∈ allowed(conform 生效),且 obj 有限。
C8 rneg 非 None 路徑可跑且 d=1 仍 == greedy(rneg)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.checks_joint
"""
import torch

from p2_anchor.wringer.comp import greedy_comp_grid
from p2_anchor.wringer.comp_joint import build_tuple_codebook, joint_comp_grid, tuple_ids
from p2_anchor.wringer.ktier import init_alpha, round_with_mids
from p2_anchor.wringer.stq import STQParam, SoftGrid

torch.manual_seed(0)
fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


M, nB, B = 16, 4, 32
K = nB * B
W = torch.randn(M, K) * 0.02
a0 = init_alpha(W)
Xc = torch.randn(256, K)
for j in range(0, K, 4):                       # 四欄一組強相關(H 非對角)
    Xc[:, j + 1] = 0.8 * Xc[:, j] + 0.2 * Xc[:, j + 1]
    Xc[:, j + 3] = 0.6 * Xc[:, j + 2] + 0.4 * Xc[:, j + 3]
H = Xc.T @ Xc
delta = torch.randn(M, K) * 0.015             # 「水」


def make_p(n=3, rneg=None):
    grid = SoftGrid(c=0.6, device="cpu", n=n)
    p = STQParam(a0, grid, r=4, c=0.6, learn_theta=False, use_comp=True)
    p._lazy_init(W)
    if rneg is not None:
        p.set_rneg(rneg, None) if hasattr(p, "set_rneg") else None
    return p, grid


def r_eff(p, grid):
    with torch.no_grad():
        u_, al_ = p._u(W)
        _, _, vh = round_with_mids(u_, p.theta_eff(), grid.vals, grid.t1, grid.t2)
        vh = p._apply_rneg(vh) if hasattr(p, "_apply_rneg") else vh
        return W + delta - (al_ * vh).reshape(M, K)


def run(fn, **kw):
    p, grid = make_p()
    log = fn(p, W, r_eff(p, grid), H, rounds=3, cap=0.5, **kw)
    return log, p.comp.clone()


# ---- C4 ----
lg, cg = run(greedy_comp_grid)
lj1, cj1 = run(joint_comp_grid, d=1)
check("C4 joint d=1 == greedy", lg["flips"] == lj1["flips"] and abs(lg["obj_after"] - lj1["obj_after"]) <= 1e-9 * max(1, abs(lg["obj_after"]))
      and torch.equal(cg, cj1), f"greedy={lg} joint1={{flips:{lj1['flips']},obj:{lj1['obj_after']:.6g}}}")

# ---- C5 ----
n_vals = 3
lj4, cj4 = run(joint_comp_grid, d=4)
allowed_all = torch.ones(n_vals ** 4, dtype=torch.bool)
lj4a, cj4a = run(joint_comp_grid, d=4, allowed=allowed_all)
check("C5 joint d=4 allowed=all == allowed=None", lj4["flips"] == lj4a["flips"] and torch.equal(cj4, cj4a),
      f"flips={lj4['flips']}/{lj4a['flips']} conform={lj4a['conform_flips']}")

# ---- C6 ----
check("C6 joint d=4 obj_after ≤ greedy(報告)", lj4["obj_after"] <= lg["obj_after"] * (1 + 1e-6),
      f"greedy obj={lg['obj_after']:.6g} absorb={lg['absorb']} | joint4 obj={lj4['obj_after']:.6g} absorb={lj4['absorb']} flips={lj4['flips']}")

# ---- C7 ----
p7, g7 = make_p()
with torch.no_grad():
    u_, _ = p7._u(W)
    idx0 = torch.bucketize(u_.reshape(M, K).float(), g7.theta)
allowed = build_tuple_codebook(idx0, 4, 40, n_vals)
l7 = joint_comp_grid(p7, W, r_eff(p7, g7), H, rounds=3, cap=0.5, d=4, allowed=allowed)
with torch.no_grad():
    u_, _ = p7._u(W)
    idx1 = torch.bucketize(u_.reshape(M, K).float(), g7.theta)
ids1 = tuple_ids(idx1.reshape(M, K // 4, 4), n_vals)
check("C7 conform:事件後全部元組 ∈ allowed", bool(allowed[ids1].all()) and torch.isfinite(torch.tensor(l7["obj_after"])),
      f"conform_flips={l7['conform_flips']} flips={l7['flips']} absorb={l7['absorb']} allowed={int(allowed.sum())}/81")

# ---- C8 ----
rneg = 0.7 + 0.6 * torch.rand(M, nB, 1)
pg, gg = make_p()
pg.set_rneg(rneg, None)
lg8 = greedy_comp_grid(pg, W, r_eff(pg, gg), H, rounds=3, cap=0.5, rneg=pg.rneg)
pj, gj = make_p()
pj.set_rneg(rneg, None)
lj8 = joint_comp_grid(pj, W, r_eff(pj, gj), H, rounds=3, cap=0.5, rneg=pj.rneg, d=1)
check("C8 rneg 路 d=1 == greedy(rneg)", lg8["flips"] == lj8["flips"] and torch.equal(pg.comp, pj.comp), f"flips={lg8['flips']}/{lj8['flips']}")
pj4, gj4 = make_p()
pj4.set_rneg(rneg, None)
lj84 = joint_comp_grid(pj4, W, r_eff(pj4, gj4), H, rounds=3, cap=0.5, rneg=pj4.rneg, d=4, allowed=allowed)
check("C8 rneg 路 d=4+碼書 可跑", torch.isfinite(torch.tensor(lj84["obj_after"])), f"{lj84}")

print("JOINT_CHECKS", "FAIL" if fails else "PASS", fails)

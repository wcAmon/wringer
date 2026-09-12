"""E36 CPU 三檢(prereg_e36_ternary_comp.cpu_checks;全 CPU,GPU 讓渡 E35-C)。"""
import torch

from p2_anchor.wringer.comp import greedy_comp
from p2_anchor.wringer.ktier import grid_tensors, k2_weight, round_with_mids
from p2_anchor.wringer.morph import corridor
from p2_anchor.wringer.stq import STQParam, SoftGrid

torch.manual_seed(20260812)
M, K, nB = 6, 8, 2
B = K // nB

# ---------- ① δW 恆等式 + corridor c 讀值 ----------
a0 = (torch.rand(M, nB, 1) * 0.5 + 0.1)
T1 = torch.randint(-1, 2, (M, nB, B)).to(torch.int8)
T2 = torch.randint(-1, 2, (M, nB, B)).to(torch.int8)
c1v, c2v = 0.37, 0.29
W1 = k2_weight(T1, T2, a0, c1v)
W2 = k2_weight(T1, T2, a0, c2v)
dW_pred = ((c2v - c1v) * a0 * T2.float()).reshape(M, K)
assert torch.allclose(W2 - W1, dW_pred, atol=1e-6), "δW 恆等式 FAIL"

g7 = SoftGrid(c=0.5, device="cpu", n=7)
p1 = STQParam(a0, g7, r=2, c=0.5, learn_theta=False, use_comp=True)
p1.enable_morph(c_tgt=0.0, to=3)
p1.set_m(0.4)
c_exp = float(corridor(torch.tensor(0.5), 0.0, torch.tensor(0.0), 0.4))
assert abs(float(p1._c_now().detach()) - c_exp) < 1e-9, "corridor 讀值 FAIL"
print(f"① δW 恆等式逐位 PASS;corridor m=0.4 → c={c_exp:.4f} 讀值一致 PASS")

# ---------- ② solver 健全性(相關輸入 toy,總虧欠殘差) ----------
# 情境:走廊中途 c 已滑到 0.3,ridge u 停在 0.45(原 ±0.5α 位置)——
# 硬指派 v=0.3,每權重虧欠 0.15α;6 個 donor 欄與 1 個閒置欄(u≈0.02)
# 強相關 → 聚合虧欠跨過整格步的增益判準,閒置欄應被 recruit 到 +1。
cv = 0.3
gs = SoftGrid(c=cv, device="cpu", n=7)
p = STQParam(a0, gs, r=2, c=cv, learn_theta=False, use_comp=True)
u_row = torch.tensor([0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.02, -0.45])
u0 = u_row.unsqueeze(0).repeat(M, 1) + torch.randn(M, K) * 0.01
W0 = (a0 * u0.reshape(M, nB, B)).reshape(M, K)   # logit=0 → α=a0
p.mode = "soft"; p.s = 2.0
_ = p(W0)                                         # lazy init(lora+comp)
assert p.comp is not None and float(p.comp.abs().sum()) == 0.0

# 共同因子輸入:欄 0-6 同向強相關、欄 7 反向(代班地圖 + 反相虧欠同號增強)
N = 256
f = torch.randn(N, 1)
X = f.repeat(1, K) + torch.randn(N, K) * 0.3
X[:, 7] = -f[:, 0] + torch.randn(N) * 0.3
H = X.T @ X

# R = W_fp − W_hard(總虧欠,直接量測;與 train_st 事件碼同式)
with torch.no_grad():
    u_, al_ = p._u(W0)
    vals, t1g, t2g = grid_tensors(7, cv, device="cpu")
    th_ = (vals[1:] + vals[:-1]) / 2
    _, _, v_h = round_with_mids(u_, th_, vals, t1g, t2g)
    R = W0.float() - (al_ * v_h).reshape(M, K)
assert float(R.abs().sum()) > 0, "toy 虧欠全零"

b = (1.0 + cv) / 2
cell_before = torch.where(u_.reshape(M, K).abs() < b,
                          torch.zeros(M, K), u_.reshape(M, K).sign())
obj0 = float(((R @ H) * R).sum())

# 逐輪單調:rounds=1 連跑 3 次,殘差手動回饋
Rk, objs, flips = R.clone(), [obj0], 0
for _ in range(3):
    c_pre = p.comp.clone()
    log = greedy_comp(p, W0, Rk, H, cv, rounds=1, cap=0.34)
    dW = p.comp - c_pre
    Rk = Rk - dW
    objs.append(float(((Rk @ H) * Rk).sum()))
    flips += log["flips"]
for i in range(len(objs) - 1):
    assert objs[i + 1] <= objs[i] + 1e-8, f"absorb 非單調:{objs}"
assert flips > 0, "solver 零翻轉"
assert objs[-1] < obj0, "H-目標未優於零補償基線"

# 異胞防護 + 落胞心 + 硬投影穩定
u_after, _ = p._u(W0)
u2a = u_after.reshape(M, K)
cell_after = torch.where(u2a.abs() < b, torch.zeros(M, K), u2a.sign())
moved = (p.comp.abs() > 1e-9)
assert bool((cell_after[moved] != cell_before[moved]).all()), "同胞回填(退化解)"
tgt = cell_after[moved]
assert float((u2a[moved] - tgt).abs().max()) < 1e-5, "未落胞心"
_, _, v_hard = round_with_mids(u_after, th_, vals, t1g, t2g)
assert torch.allclose(v_hard.reshape(M, K)[moved], tgt, atol=1e-6), "硬投影漂移"
print(f"② solver PASS:obj {obj0:.4f}→{objs[-1]:.4f}"
      f"(absorb {1 - objs[-1] / obj0:.2%},flips={flips},逐輪單調)"
      ";異胞防護/落胞心/硬投影穩定 ✓")

# ---------- ③ comp 生命週期 + 旗標關閉逐位等價 ----------
t1e, t2e, ale, w_hard = p.extract(W0)
w_ref = k2_weight(t1e, t2e, ale, p.c)
assert torch.allclose(w_hard, w_ref, atol=1e-5), "extract ≠ k2_weight(部署形態破壞)"

pm = STQParam(a0, SoftGrid(c=0.5, device="cpu", n=7), r=2, c=0.5,
              learn_theta=True, use_comp=True)
pm.enable_morph(c_tgt=0.0, to=3)
pm.mode = "soft"; pm.s = 2.0
_ = pm(W0)
pm.comp += 0.01                       # 走廊中寫 comp
pm.set_m(1.0)                         # collapse → 3元
assert pm.grid.n == 3 and pm.c == 0.0 and not pm._morph
t13, t23, al3, wh3 = pm.extract(W0)
assert int(t23.abs().sum()) == 0, "3元 T2 非零"
assert torch.allclose(wh3, k2_weight(t13, t23, al3, 0.0), atol=1e-5)

p_off = STQParam(a0, gs, r=2, c=cv, learn_theta=False)     # 預設 use_comp=False
p_off.mode = "soft"; p_off.s = 2.0
_ = p_off(W0)
assert p_off.comp is None, "旗標關仍配置 comp"
w_off = p_off(W0)
p_on0 = STQParam(a0, gs, r=2, c=cv, learn_theta=False, use_comp=True)
p_on0.mode = "soft"; p_on0.s = 2.0
_ = p_on0(W0)                         # lora_A 隨機不同 → 拷貝後比對
p_on0.lora_A.data.copy_(p_off.lora_A.data)
p_on0.lora_B.data.copy_(p_off.lora_B.data)
w_on0 = p_on0(W0)
assert torch.allclose(w_off, w_on0, atol=0), "comp=0 前向 ≠ 基線(逐位)"
print("③ 生命週期 PASS:extract≡k2_weight、collapse 後 comp 吸收、"
      "旗標關 comp=None、comp=0 前向逐位等價")
print("E36_CPU_CHECKS_PASS")

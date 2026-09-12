"""E64 S1:非對稱負尺度 r 泛化到混格(3/5/9,E63 asym.py 只支援純三元;原函數不動,證據保存)。

部署語義與 state.dense_weight / export.decode_tensor 逐位一致:
  v = T1 + c·T2,w = α·v,r 只作用在 T1 < 0 的元素(w ← r·w),非 v < 0。
  三元(T2≡0)時 T1<0 ⇔ v<0,逐位退化為 asym.py。

碼以 (V0, neg) 表示:V0 = T1 + c·T2 (M,nB,B) float,neg = T1<0 (bool)。
  V(r)  = where(neg, r·V0, V0) = vpos − r·n,  vpos = V0·[¬neg],n = −V0·[neg] ≥ 0
  閉式(固定碼):α ← scale_joint(W_t, V(r), Σ);r_g ← s_g/α_g,s = scale_joint(P − W_t, n, Σ),P = α·vpos
  重投影:對合法 (t1,t2) 候選值 v·(r if t1<0 else 1) 取逐元最近(brute force,r 逐 block 廣播)。
"""
import torch

from p2_anchor.wringer.asym import R_MAX, R_MIN, err_h
from p2_anchor.wringer.ktier import grid_tensors, scale_joint


def code_V0(T1, T2, c):
    return T1.float() + float(c) * T2.float()


def neg_mask(T1):
    return T1 < 0


def asym_Vm(V0, neg, r):
    """r None ⇒ 對稱;r (M,1,1) 或 (M,nB,1) 廣播。"""
    if r is None:
        return V0
    return torch.where(neg, V0 * r, V0)


def dense_asym_m(V0, neg, alpha, r):
    """運算次序與 state.dense_weight 逐位一致:w = α·v 後負側再乘 r。"""
    M, nB, B = V0.shape
    w = alpha * V0
    if r is not None:
        w = torch.where(neg, w * r, w)
    return w.reshape(M, nB * B)


@torch.no_grad()
def solve_r_m(W3, V0, neg, alpha, Sig):
    """固定 (碼, α),逐列閉式 r (M,1,1);無負碼列 r=1。"""
    M, nB, B = W3.shape
    K = nB * B
    zero = torch.zeros_like(V0)
    P = (alpha * torch.where(neg, zero, V0)).reshape(M, K)
    N = (alpha * torch.where(neg, -V0, zero)).reshape(M, K)
    W = W3.reshape(M, K)
    num = (((P - W) @ Sig) * N).sum(-1)
    den = ((N @ Sig) * N).sum(-1)
    r = torch.where(den > 0, num / den.clamp_min(1e-30), torch.ones_like(num))
    r = torch.where(torch.isfinite(r), r, torch.ones_like(r))
    return r.clamp(R_MIN, R_MAX).reshape(M, 1, 1)


@torch.no_grad()
def solve_r_blk_m(W3, V0, neg, alpha, Sig):
    """固定 (碼, α),逐 (row, block) 閉式 r_g (M,nB,1)。"""
    zero = torch.zeros_like(V0)
    n = torch.where(neg, -V0, zero)
    P = alpha * torch.where(neg, zero, V0)
    s = scale_joint(P - W3, n, Sig)
    r = s / alpha
    empty = n.sum(-1, keepdim=True) == 0
    r = torch.where(empty | ~torch.isfinite(r), torch.ones_like(r), r)
    return r.clamp(R_MIN, R_MAX)


@torch.no_grad()
def fit_alpha_r_m(W3, V0, neg, Sig, alt=3, r0=None, asym=True, blk=False):
    """固定碼的 (α, r) 交替閉式;回傳 (alpha (M,nB,1), r|None, trace)。"""
    M, nB, B = W3.shape
    K = nB * B
    Wf = W3.reshape(M, K)
    if not asym:
        alpha = scale_joint(W3, V0, Sig)
        return alpha, None, [err_h(Wf - (alpha * V0).reshape(M, K), Sig)]
    r = torch.ones(M, 1, 1, device=W3.device) if r0 is None else r0
    alpha = scale_joint(W3, asym_Vm(V0, neg, r), Sig)
    trace = [err_h(Wf - dense_asym_m(V0, neg, alpha, r), Sig)]
    for _ in range(alt):
        r = (solve_r_blk_m if blk else solve_r_m)(W3, V0, neg, alpha, Sig)
        alpha = scale_joint(W3, asym_Vm(V0, neg, r), Sig)
        trace.append(err_h(Wf - dense_asym_m(V0, neg, alpha, r), Sig))
    return alpha, r, trace


@torch.no_grad()
def round_grid_asym(W3, alpha, r, level, c, row_chunk=1024):
    """逐元最近值重投影(無誤差回饋)到 level 網格,負側候選值乘 r(r None ⇒ 1)。
    回傳 (T1, T2) float (M,nB,B)。候選值退化(重複)時取 |t2| 較小者(grid_tensors 慣例)。"""
    M, nB, B = W3.shape
    vals, t1, t2 = grid_tensors(level, c, W3.device)
    G = vals.numel()
    T1 = torch.empty_like(W3)
    T2 = torch.empty_like(W3)
    for m0 in range(0, M, row_chunk):
        m1 = min(M, m0 + row_chunk)
        u = W3[m0:m1] / alpha[m0:m1]                                  # (m,nB,B)
        if r is None:
            cand = vals.view(1, 1, 1, G).expand(m1 - m0, nB, 1, G)
        else:
            rr = r[m0:m1].expand(m1 - m0, nB, 1).unsqueeze(-1)          # (m,nB,1,1)
            cand = torch.where(t1.view(1, 1, 1, G) < 0, vals.view(1, 1, 1, G) * rr,
                               vals.view(1, 1, 1, G).expand(m1 - m0, nB, 1, G))
        idx = (u.unsqueeze(-1) - cand).abs().argmin(-1)                # (m,nB,B)
        T1[m0:m1] = t1[idx]
        T2[m0:m1] = t2[idx]
    return T1, T2

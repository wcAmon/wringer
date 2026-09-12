"""E61 st61c:(α, M) 閉式重解——把連續載體從 Adam 步長節流中解開。

錨定目標(部署座標,與排水 loss 的 Y 同錨):W_t = W0q + (1−γ)·BA。
載體:C·Bd,C = α⊙V(碼座標硬載體),Bd = blockdiag(M)。

M 解:min_X ‖W_t − C X‖_H² + λ·s·‖X − X_cur‖_F²,X 限塊對角 (k,b,b)。
  法方程 P_bd(Cᵀ C X H) + λ s X = P_bd(Cᵀ W_t H) + λ s X_cur,共軛梯度解
  (H 全 K×K,跨塊相關保留;不用塊對角近似)。
  信賴域:每事件每塊 ‖ΔM_b‖_F/‖I_b‖_F ≤ cap,超出則整體縮步;
  sv_min(M_b) < sv_floor 時回退該塊步長。
α 解:固定 V 與 Bd,碼座標目標 W_t·Bd⁻¹、度量 Σ' = Bd H Bdᵀ,
  scale_joint 逐行聯合 α;寫回 logit = logit⁻¹(α/(2a0))(夾 (0.02,0.98))。
"""
import torch

from p2_anchor.wringer.ktier import scale_joint


def _bd_apply(C, X):
    """C (M,K) @ blockdiag(X) (k,b,b) → (M,K)。"""
    M, K = C.shape
    k, b, _ = X.shape
    return torch.einsum("mkb,kbc->mkc", C.reshape(M, k, b), X).reshape(M, K)


def _bd_proj(G, k, b):
    """(K,K) → 塊對角投影 (k,b,b)。"""
    K = G.shape[0]
    Gb = G.reshape(k, b, k, b)
    idx = torch.arange(k, device=G.device)
    return Gb[idx, :, idx, :]


@torch.no_grad()
def solve_mix_ls(C, W_t, H, X_cur, ridge=1.0, cap=0.05, iters=25,
                 sv_floor=0.3, tol=1e-5, G=None):
    """回傳 (X_new, stats)。C, W_t: (M,K) fp32;H: (K,K);X_cur: (k,b,b)。

    G(E62b 漂移感知目標,可選):交叉共變 Σ x_id x_drᵀ (K,K)。給定時法方程
    右端 W_t·H → W_t·G,即目標由「理想權重作用在漂移輸入」改為「理想權重
    作用在理想輸入」(與窗 loss 的 Y 同義);None 逐位還原舊行為。"""
    M, K = C.shape
    k, b, _ = X_cur.shape
    CtC = C.T @ C                                      # (K,K)
    # 尺度:算子對角量級 ≈ (CᵀC)_ii · H_jj 均值
    s = float(CtC.diagonal().mean() * H.diagonal().mean())
    lam = ridge * s

    def A(X):
        Y = _bd_apply(CtC, X) @ H                      # CᵀC·Bd(X)·H
        return _bd_proj(Y, k, b) + lam * X

    rhs = _bd_proj((C.T @ W_t) @ (H if G is None else G), k, b) + lam * X_cur
    X = X_cur.clone()
    r = rhs - A(X)
    p = r.clone()
    rs = float((r * r).sum())
    r0 = rs ** 0.5
    it = 0
    for it in range(iters):
        Ap = A(p)
        a = rs / max(float((p * Ap).sum()), 1e-30)
        X = X + a * p
        r = r - a * Ap
        rs_new = float((r * r).sum())
        if rs_new ** 0.5 <= tol * max(r0, 1e-30):
            break
        p = r + (rs_new / max(rs, 1e-30)) * p
        rs = rs_new
    # 信賴域:塊步長上限(相對 ‖I_b‖_F = √b)
    dX = X - X_cur
    step = dX.reshape(k, -1).norm(dim=1) / (b ** 0.5)     # (k,)
    smax = float(step.max())
    t = min(1.0, cap / max(smax, 1e-30))
    X_new = X_cur + t * dX
    # 奇異值地板:逐塊回退
    sv = torch.linalg.svdvals(X_new)
    bad = sv.min(dim=1).values < sv_floor
    n_bad = int(bad.sum())
    if n_bad:
        X_new[bad] = X_cur[bad] + 0.5 * t * dX[bad]
        sv2 = torch.linalg.svdvals(X_new[bad])
        still = sv2.min(dim=1).values < sv_floor
        if int(still.sum()):
            idx = bad.nonzero().squeeze(1)[still]
            X_new[idx] = X_cur[idx]

    def obj(Xq):
        E = W_t - _bd_apply(C, Xq)
        return float(((E @ H) * E).sum())

    o0, o1 = obj(X_cur), obj(X_new)
    return X_new, {"obj_before": o0, "obj_after": o1,
                   "ls_gain": round(1.0 - o1 / max(o0, 1e-30), 4),
                   "step_max": round(smax, 5), "t": round(t, 4),
                   "cg_iters": it + 1, "sv_floor_hits": n_bad,
                   "sv_min": round(float(torch.linalg.svdvals(X_new).min()), 4)}


@torch.no_grad()
def refit_alpha(q, W_t_code, V3, Sigma_p, lo=0.02, hi=0.98):
    """固定碼 V 與 Bd,碼座標目標 W_t·Bd⁻¹ 下 Σ'-加權聯合 α,寫回 q.logit。
    W_t_code: (M,K);V3: (M,nB,B);Sigma_p: (K,K)。回傳 (α_new, 統計)。"""
    M, K = W_t_code.shape
    nB = V3.shape[1]
    B = K // nB
    al = scale_joint(W_t_code.reshape(M, nB, B), V3.float(), Sigma_p)
    al_old = 2.0 * torch.sigmoid(q.logit) * q.a0
    ratio = (al / (2.0 * q.a0)).clamp(lo, hi)
    q.logit.copy_(torch.log(ratio / (1.0 - ratio)))
    al_eff = 2.0 * torch.sigmoid(q.logit) * q.a0
    rel = float((al_eff - al_old).norm() / al_old.norm().clamp_min(1e-12))
    clip = float(((al / (2.0 * q.a0)) <= lo).float().mean()
                 + ((al / (2.0 * q.a0)) >= hi).float().mean())
    return al_eff, {"alpha_rel_change": round(rel, 5),
                    "alpha_clip_frac": round(clip, 5)}


def bd_mask(K, b, device):
    """列塊對角遮罩 (K,K) bool。"""
    idx = torch.arange(K, device=device) // b
    return idx[:, None] == idx[None, :]


@torch.no_grad()
def solve_left_ls(Z, W_t, H, L_cur, block, ridge=0.0, cap=0.05, iters=40,
                  sv_floor=0.3, tol=1e-5, G=None):
    """E62b 輸出側 L:min_L ‖W_t − L Z‖_H² + λ s ‖L − L_cur‖²,L 列塊對角(塊 block)。
    Z = C·Bd(X) (M,K) 部署形態載體;L_cur (M,M) 密集(僅塊內非零)。
    法方程 (L Zᵀ... ) 以 Gz = Z H Zᵀ (M,M):A(L) = (L Gz)⊙mask + λ s L;
    rhs = (W_t (H|G) Zᵀ)⊙mask + λ s L_cur。信賴域與 sv 地板逐塊同 solve_mix_ls。"""
    M, K = Z.shape
    mk = bd_mask(M, block, Z.device)
    Gz = Z @ H @ Z.T
    s = float(Gz.diagonal().mean())
    lam = ridge * s
    A = lambda L: (L @ Gz) * mk + lam * L
    rhs = ((W_t @ (H if G is None else G)) @ Z.T) * mk + lam * L_cur
    L = L_cur * mk
    r = rhs - A(L)
    p = r.clone()
    rs = float((r * r).sum())
    r0 = rs ** 0.5
    it = 0
    for it in range(iters):
        Ap = A(p)
        a = rs / max(float((p * Ap).sum()), 1e-30)
        L = L + a * p
        r = r - a * Ap
        rs_new = float((r * r).sum())
        if rs_new ** 0.5 <= tol * max(r0, 1e-30):
            break
        p = r + (rs_new / max(rs, 1e-30)) * p
        rs = rs_new
    k = M // block
    dL = (L - L_cur).reshape(k, block, k, block)
    idx = torch.arange(k, device=Z.device)
    dLb = dL[idx, :, idx, :]                               # (k,b,b)
    step = dLb.reshape(k, -1).norm(dim=1) / (block ** 0.5)
    smax = float(step.max())
    t = min(1.0, cap / max(smax, 1e-30))
    L_new = L_cur + t * (L - L_cur)
    Lb = L_new.reshape(k, block, k, block)[idx, :, idx, :]
    sv = torch.linalg.svdvals(Lb)
    bad = sv.min(dim=1).values < sv_floor
    n_bad = int(bad.sum())
    if n_bad:
        Lc = L_cur.reshape(k, block, k, block)[idx, :, idx, :]
        Lb[bad] = Lc[bad] + 0.5 * t * dLb[bad]
        still = torch.linalg.svdvals(Lb[bad]).min(dim=1).values < sv_floor
        if int(still.sum()):
            Lb[bad.nonzero().squeeze(1)[still]] = Lc[bad.nonzero().squeeze(1)[still]]
        L_new = torch.block_diag(*Lb.unbind(0))

    def obj(Lq):
        E = W_t - Lq @ Z
        return float(((E @ H) * E).sum())

    o0, o1 = obj(L_cur), obj(L_new)
    return L_new, {"obj_before": o0, "obj_after": o1,
                   "ls_gain": round(1.0 - o1 / max(o0, 1e-30), 4),
                   "step_max": round(smax, 5), "t": round(t, 4),
                   "cg_iters": it + 1, "sv_floor_hits": n_bad,
                   "sv_min": round(float(torch.linalg.svdvals(Lb).min()), 4)}

"""k=2 雙位面表示:W = α ⊙ (T₁ + c·T₂) — 統一網格族(9/7/5/3)。

移植自 guava-qat src/gqat/{ktier,quant}.py(公開 repo,語義逐位對齊;
唯一差異:無 HIP fused 路徑,CUDA eager)。

T₂ 約束階梯(c=0.6 時網格嵌套 3 ⊂ 5 ⊂ 7 ⊂ 9):
  9:T₂ 自由;7:T₁·T₂ ≥ 0;5:T₂≠0 僅限 T₁=0;3:T₂=0。
加法路徑帳:任一檔位 = 兩條 ternary 加法累加器,每 block 兩次縮放(α, c·α)。
"""
import torch
import torch.nn as nn

BLOCK = 32      # α group 大小;set_block() 全程序生效(E31 起 g16 臂)
C_K2 = 0.6      # 雙射必要條件(c=0.5 使 1−c=c 碰撞退化 7 值)


def set_block(b):
    global BLOCK
    BLOCK = int(b)
GPTQ_DAMP = 0.01
GPTQ_LAZY = 128


INT_LEVELS = {4: (-2, 1), 8: (-4, 3), 16: (-8, 7), 256: (-128, 127)}   # E68:整數格(T2≡0,vals=T1;8 級留零丟 +4);E69:4 級 {-2..1} 2 bit


def grid_pairs(level):
    if level in INT_LEVELS:
        lo, hi = INT_LEVELS[level]
        return [(t, 0) for t in range(lo, hi + 1)]
    pairs = []
    for t1 in (-1, 0, 1):
        for t2 in (-1, 0, 1):
            if level == 9:
                ok = True
            elif level == 7:
                ok = t1 * t2 >= 0
            elif level == 5:
                ok = t2 == 0 or t1 == 0
            elif level == 3:
                ok = t2 == 0
            else:
                raise ValueError(level)
            if ok:
                pairs.append((t1, t2))
    return pairs


def grid_tensors(level, c=C_K2, device="cuda"):
    """回傳 (vals 升冪 (G,), t1 (G,), t2 (G,))。值重複時保留 |t2| 較小者。"""
    pairs = grid_pairs(level)
    seen = {}
    for t1, t2 in pairs:
        v = t1 + c * t2
        if v not in seen or abs(t2) < abs(seen[v][1]):
            seen[v] = (t1, t2)
    vals = sorted(seen)
    t1 = torch.tensor([seen[v][0] for v in vals], device=device,
                      dtype=torch.float32)
    t2 = torch.tensor([seen[v][1] for v in vals], device=device,
                      dtype=torch.float32)
    vals = torch.tensor(vals, device=device, dtype=torch.float32)
    assert len(vals) == level, f"網格值退化:{len(vals)} != {level}(c={c})"
    return vals, t1, t2


def round_to_grid(U, vals, t1, t2):
    """U 任意形狀 → 最近網格點的 (T1, T2, V)(預設中點門檻)。"""
    mids = (vals[1:] + vals[:-1]) / 2
    return round_with_mids(U, mids, vals, t1, t2)


def round_with_mids(U, mids, vals, t1, t2):
    """自訂門檻(θ 可學臂用;mids 需升冪)。"""
    idx = torch.bucketize(U.contiguous(), mids)
    return t1[idx], t2[idx], vals[idx]


def init_alpha(W):
    """TWN 式初始 α₀,逐 (row, block)。W: (M, K) fp32。回傳 (M, nB, 1)。"""
    M, K = W.shape
    Wb = W.reshape(M, K // BLOCK, BLOCK).abs()
    delta = 0.7 * Wb.mean(-1, keepdim=True)
    mask = Wb > delta
    a0 = (Wb * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)
    return a0.clamp_min(1e-8)


def alpha_search(W, vals, wdiag=None, n_cand=24, lo=0.6, hi=1.15, orient=True):
    """GGUF 式逐 block 尺度搜尋(E68 P0-rtn / 整數格 α 初值):α 候選 = sgn·s·max|w|/max|vals|,
    s ∈ [lo,hi] n_cand 點;誤差 Σ_i wdiag_i (w_i − α q_i)²(wdiag = diag(Σ) = E[x²],imatrix 同義;None=均權)。
    orient:非對稱格(如 {−4..3})允許 α<0 = 格翻向 {−3..4},逐 block 免費方向位。回傳 α (M,nB,1) 帶號。"""
    M, K = W.shape
    B = BLOCK
    nB = K // B
    Wb = W.reshape(M, nB, B)
    wd = (torch.ones(K, device=W.device) if wdiag is None else wdiag.to(W.device).float()).reshape(1, nB, B)
    vmax = float(vals.abs().max())
    amax = Wb.abs().amax(-1, keepdim=True).clamp_min(1e-8) / vmax
    asym = abs(float(vals.min()) + float(vals.max())) > 1e-6
    signs = (1.0, -1.0) if (orient and asym) else (1.0,)
    best_err = best_a = None
    mids = (vals[1:] + vals[:-1]) / 2
    for sgn in signs:
        for sc in torch.linspace(lo, hi, n_cand).tolist():
            a = sgn * sc * amax
            q = vals[torch.bucketize(Wb / a, mids)]
            err = (wd * (Wb - a * q) ** 2).sum(-1, keepdim=True)
            if best_err is None:
                best_err, best_a = err, a
            else:
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_a = torch.where(better, a, best_a)
    return best_a


def gptq_grid(W, Sigma, level, c=C_K2, damp=GPTQ_DAMP, a0=None):
    """GPTQ 列序循序,rounding 到 k=2 網格。回傳 (α₀, T1, T2)(T 皆 (M,K) fp32)。
    a0 (M,nB,1) 給定則用之(E68:alpha_search 帶號 α;None=TWN init_alpha 舊行為)。"""
    M, K = W.shape
    assert K % BLOCK == 0, f"K={K} 不整除 BLOCK={BLOCK}"
    vals, gt1, gt2 = grid_tensors(level, c, W.device)
    a0 = init_alpha(W) if a0 is None else a0
    A = a0.expand(-1, -1, BLOCK).reshape(M, K)

    H = Sigma.clone()
    diag = torch.diagonal(H)
    dead = diag <= 0
    diag[dead] = 1.0
    diag += damp * diag.mean()
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(Hinv, upper=True)
    del L, Hinv, H

    Wc = W.clone()
    T1 = torch.zeros_like(W)
    T2 = torch.zeros_like(W)
    for b0 in range(0, K, GPTQ_LAZY):
        b1 = min(b0 + GPTQ_LAZY, K)
        Err = torch.zeros(M, b1 - b0, device=W.device)
        for j in range(b0, b1):
            w = Wc[:, j]
            u1, u2, v = round_to_grid(w / A[:, j], vals, gt1, gt2)
            T1[:, j] = u1
            T2[:, j] = u2
            e = (w - v * A[:, j]) / U[j, j]
            Err[:, j - b0] = e
            if j + 1 < b1:
                Wc[:, j + 1:b1] -= e.unsqueeze(1) * U[j, j + 1:b1].unsqueeze(0)
        if b1 < K:
            Wc[:, b1:] -= Err @ U[b0:b1, b1:]
    return a0, T1, T2


def round_rd(U, vals, t1, t2, bits, lam):
    """率失真 rounding(E66-RD):argmin_g (U − vals_g)² + λ·bits_g。U 任意形狀;bits (G,) 由直方圖 −log₂p 給。
    λ=0 ⇒ 與 round_to_grid 逐位相同(argmin 平方距離)。"""
    cost = (U.unsqueeze(-1) - vals) ** 2 + lam * bits
    idx = cost.argmin(-1)
    return t1[idx], t2[idx], vals[idx]


def gptq_grid_rd(W, Sigma, level, A, bits, lam, c=C_K2, damp=GPTQ_DAMP):
    """GPTQ 列序循序 + 率罰 rounding,α 固定(A (M,K) 已展開)。回傳 (T1, T2) (M,K) fp32。
    率罰在 Σ 加權誤差單位:cost = ((w − v·A_j)/U_jj)² / s_mod + λ·bits_v,s_mod = mean_j(mean_m A_mj² / U_jj²)
    (模組內歸一化 ⇒ λ 無量綱、跨模組同等傾斜;模組內按 OBS 顯著度逐元連續選擇,避免 u 落格時的階梯)。
    目標精確可表示且 λ=0 時每列誤差 0、無回饋 ⇒ 碼逐位還原(E66-RD R0 儀器檢查)。"""
    M, K = W.shape
    vals, gt1, gt2 = grid_tensors(level, c, W.device)
    bits = torch.as_tensor(bits, device=W.device, dtype=torch.float32)
    assert bits.numel() == vals.numel(), f"bits {bits.numel()} != grid {vals.numel()}"
    H = Sigma.clone()
    diag = torch.diagonal(H)
    dead = diag <= 0
    diag[dead] = 1.0
    diag += damp * diag.mean()
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(Hinv, upper=True)
    del L, Hinv, H
    d = torch.diagonal(U)
    s_mod = float(((A ** 2).mean(0) / d ** 2).mean()) if lam > 0 else 1.0
    Wc = W.clone()
    T1 = torch.zeros_like(W)
    T2 = torch.zeros_like(W)
    for b0 in range(0, K, GPTQ_LAZY):
        b1 = min(b0 + GPTQ_LAZY, K)
        Err = torch.zeros(M, b1 - b0, device=W.device)
        for j in range(b0, b1):
            w = Wc[:, j]
            if lam > 0:
                cost = ((w.unsqueeze(-1) - vals * A[:, j].unsqueeze(-1)) / d[j]) ** 2 / s_mod + lam * bits
                idx = cost.argmin(-1)
                u1, u2, v = gt1[idx], gt2[idx], vals[idx]
            else:
                u1, u2, v = round_to_grid(w / A[:, j], vals, gt1, gt2)
            T1[:, j] = u1
            T2[:, j] = u2
            e = (w - v * A[:, j]) / U[j, j]
            Err[:, j - b0] = e
            if j + 1 < b1:
                Wc[:, j + 1:b1] -= e.unsqueeze(1) * U[j, j + 1:b1].unsqueeze(0)
        if b1 < K:
            Wc[:, b1:] -= Err @ U[b0:b1, b1:]
    return T1, T2


def scale_joint(W, T, Sigma, ridge=1e-6, prior=None, lam=0.0):
    """固定 T,逐行解 Σ-加權聯合 α:A α = b。

    W, T: (M, nB, B);Sigma: (K, K)。回傳 α (M, nB, 1) fp32。B 由 shape 推導。
    prior (M,nB,1) + lam > 0(E65-X):Tikhonov 拉向先驗——(A + λ·diag(A)) α = b + λ·diag(A)·prior;
    目標精確可表示時解 = prior(修 E65-X U0 零空間飄移:校準誤差 0 但權重變 44%)。prior=None 行為逐位不變。"""
    M, nB, B = W.shape
    K = nB * B
    Wf = W.reshape(M, K)
    Tf = T.reshape(M, K)

    G = Wf @ Sigma
    bvec = (G.reshape(M, nB, B) * T).sum(-1)
    del G

    Amat = torch.empty(M, nB, nB, dtype=torch.float32, device=Wf.device)
    Sb = Sigma.reshape(nB, B, K)
    for b in range(nB):
        P = Tf[:, b * B:(b + 1) * B] @ Sb[b]
        Amat[:, b, :] = (P.reshape(M, nB, B) * T).sum(-1)
        del P

    tr = torch.diagonal(Amat, dim1=1, dim2=2).sum(-1, keepdim=True) / nB
    eye = torch.eye(nB, device=Wf.device).unsqueeze(0)
    Amat = Amat + eye * (ridge * tr.clamp_min(1e-12)).unsqueeze(-1)
    if prior is not None and lam > 0:
        dA = torch.diagonal(Amat, dim1=1, dim2=2)                       # (M, nB)
        Amat = Amat + eye * (lam * dA).unsqueeze(-1)
        bvec = bvec + lam * dA * prior.reshape(M, nB).float()
    alpha = torch.empty(M, nB, 1, dtype=torch.float32, device=Wf.device)
    CHUNK = 1024
    for m0 in range(0, M, CHUNK):
        m1 = min(m0 + CHUNK, M)
        Ac = Amat[m0:m1]
        try:
            Lc = torch.linalg.cholesky(Ac)
            alpha[m0:m1] = torch.cholesky_solve(bvec[m0:m1].unsqueeze(-1), Lc)
        except torch.linalg.LinAlgError:
            alpha[m0:m1] = torch.linalg.lstsq(
                Ac, bvec[m0:m1].unsqueeze(-1)).solution

    a0 = init_alpha(Wf) if prior is None else prior.reshape(M, nB, 1).float()
    zero_blk = (T != 0).sum(-1, keepdim=True) == 0
    alpha = torch.where(zero_blk | ~torch.isfinite(alpha), a0, alpha)
    return alpha


def k2_weight(T1, T2, alpha, c=C_K2):
    """(M,nB,32) 三件 → (M,K) fp32。"""
    M, nB, B = T1.shape
    V = T1.float() + c * T2.float()
    return (alpha * V).reshape(M, nB * B)


class AlphaParamK2(nn.Module):
    """weight = bf16((α ⊙ (T₁ + c·T₂))[:, inv]);僅 α 可學(打磨段用)。"""

    def __init__(self, T1, T2, inv, a0, c=C_K2):
        super().__init__()
        self.register_buffer("T1", T1)   # int8 (M, nB, 32)
        self.register_buffer("T2", T2)
        self.register_buffer("inv", inv)
        self.alpha = nn.Parameter(a0.detach().float().clone())
        self.c = c

    def _inv_identity(self):
        if not hasattr(self, "_inv_is_identity"):
            self._inv_is_identity = bool(
                (self.inv == torch.arange(len(self.inv),
                                          device=self.inv.device)).all())
        return self._inv_is_identity

    def forward(self, _w):
        M, nB, B = self.T1.shape
        V = self.T1.float() + self.c * self.T2.float()
        w = (self.alpha * V).reshape(M, nB * B)
        if self._inv_identity():
            return w.to(torch.bfloat16)
        return w[:, self.inv].to(torch.bfloat16)

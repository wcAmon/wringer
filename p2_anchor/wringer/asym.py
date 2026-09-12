"""非對稱三元(E63):逐列負尺度 r 的閉式擬合與投影(裝置無關,CPU 可測)。

碼空間:w = α_{i,g} · (T⁺ − r_i · T⁻),T ∈ {−1,0,+1},α 逐 (row, block),r 逐 row。
  推論仍是 3 符號(1.58 b/w);每列多 1 個 fp16(down_proj 0.0016 b/w);
  雙累加器核:y_i = Σ_g α_{ig}(x·T⁺_{ig}) − r_i Σ_g α_{ig}(x·T⁻_{ig}),r 在列外提出。
  r ≡ 1 逐位退化為對稱三元(等價閘 checks_asym H1)。

擬合(固定 T):交替
  α ← scale_joint(W_t, V(r), Σ)             (逐列 Σ-加權聯合 α,既有閉式)
  r ← ⟨P − W_t, N⟩_Σ / ⟨N, N⟩_Σ  逐列      (P = α∘T⁺, N = α∘T⁻;1-D 加權 LS)
每步皆為該變數的 Σ-加權最小平方閉式解 ⇒ 目標 tr(ΔW Σ ΔWᵀ) 單調不增(H3)。

投影(重新落碼,無誤差回饋):u = W_t/α;T=+1 若 u>½,T=−1 若 u<−r/2(對稱:−½)。
GPTQ(列序誤差回饋)版:同 ktier.gptq_grid 結構,rounding 換成逐列非對稱門檻。
"""
import torch

from p2_anchor.wringer.ktier import GPTQ_DAMP, GPTQ_LAZY, init_alpha, scale_joint

R_MIN, R_MAX = 0.25, 4.0


def asym_V(T3, r):
    """T3 (M,nB,B) ∈{−1,0,1} float;r (M,1,1) → V = T⁺ − r·T⁻。r=None ⇒ 對稱(V=T3)。"""
    if r is None:
        return T3
    return T3.clamp(min=0) - r * (T3 < 0).float()


def err_h(dW, Sig):
    """tr(ΔW Σ ΔWᵀ)(GPTQ 目標;與 t2drain.err_h 同度量)。"""
    return float(((dW @ Sig) * dW).sum())


@torch.no_grad()
def solve_r(W3, T3, alpha, Sig):
    """固定 (T, α),逐列閉式 r;無負碼列 r=1。回傳 (M,1,1)。"""
    M, nB, B = W3.shape
    K = nB * B
    P = (alpha * T3.clamp(min=0)).reshape(M, K)
    N = (alpha * (T3 < 0).float()).reshape(M, K)
    W = W3.reshape(M, K)
    num = (((P - W) @ Sig) * N).sum(-1)
    den = ((N @ Sig) * N).sum(-1)
    r = torch.where(den > 0, num / den.clamp_min(1e-30), torch.ones_like(num))
    r = torch.where(torch.isfinite(r), r, torch.ones_like(r))
    return r.clamp(R_MIN, R_MAX).reshape(M, 1, 1)


@torch.no_grad()
def solve_r_blk(W3, T3, alpha, Sig):
    """診斷上界:逐 (row, block) 的 r_g(位元 +0.5 b/w,不可交付)。
    固定 (T, α):min_s ‖(W_t − P) + Σ_g s_g T⁻_g‖_Σ = scale_joint(P − W_t, T⁻, Σ) ⇒ r_g = s_g/α_g。"""
    M, nB, B = W3.shape
    Tneg = (T3 < 0).float()
    P = alpha * T3.clamp(min=0)
    s = scale_joint(P - W3, Tneg, Sig)
    r = s / alpha
    empty = Tneg.sum(-1, keepdim=True) == 0
    r = torch.where(empty | ~torch.isfinite(r), torch.ones_like(r), r)
    return r.clamp(R_MIN, R_MAX)


@torch.no_grad()
def fit_alpha_r(W3, T3, Sig, alt=3, r0=None, asym=True, blk=False):
    """固定 T 的 (α, r) 交替閉式擬合。asym=False ⇒ 只解 α(r=None);blk=True ⇒ r 逐 block(診斷)。

    回傳 (alpha (M,nB,1), r (M,1,1)|None, trace [err ...])。"""
    M, nB, B = W3.shape
    K = nB * B
    Wf = W3.reshape(M, K)
    if not asym:
        alpha = scale_joint(W3, T3, Sig)
        return alpha, None, [err_h(Wf - (alpha * T3).reshape(M, K), Sig)]
    r = torch.ones(M, 1, 1, device=W3.device) if r0 is None else r0
    trace = []
    alpha = scale_joint(W3, asym_V(T3, r), Sig)
    trace.append(err_h(Wf - (alpha * asym_V(T3, r)).reshape(M, K), Sig))
    for _ in range(alt):
        r = (solve_r_blk if blk else solve_r)(W3, T3, alpha, Sig)
        alpha = scale_joint(W3, asym_V(T3, r), Sig)
        trace.append(err_h(Wf - (alpha * asym_V(T3, r)).reshape(M, K), Sig))
    return alpha, r, trace


def round_ternary(W3, alpha, r=None):
    """逐元投影(無誤差回饋)。r=None 對稱門檻 ±½;否則負側門檻 −r/2。"""
    u = W3 / alpha
    neg_th = 0.5 if r is None else r / 2
    return (u > 0.5).float() - (u < -neg_th).float()


@torch.no_grad()
def gptq_ternary(W, Sigma, alpha3, r=None, damp=GPTQ_DAMP):
    """GPTQ 列序誤差回饋,三元 rounding 用給定 α (M,nB,1) 與逐列 r (M,1,1)|None。
    與 ktier.gptq_grid 同結構(lazy batch、cholesky 上三角);回傳 T (M,K) float。"""
    M, K = W.shape
    nB = alpha3.shape[1]
    B = K // nB
    A = alpha3.expand(-1, -1, B).reshape(M, K)
    rr = None if r is None else r.reshape(M)
    H = Sigma.clone()
    diag = torch.diagonal(H)
    diag[diag <= 0] = 1.0
    diag += damp * diag.mean()
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(Hinv, upper=True)
    del L, Hinv, H
    Wc = W.clone()
    T = torch.zeros_like(W)
    for b0 in range(0, K, GPTQ_LAZY):
        b1 = min(b0 + GPTQ_LAZY, K)
        Err = torch.zeros(M, b1 - b0, device=W.device)
        for j in range(b0, b1):
            w = Wc[:, j]
            a = A[:, j]
            u = w / a
            neg_th = 0.5 if rr is None else rr / 2
            t = (u > 0.5).float() - (u < -neg_th).float()
            T[:, j] = t
            v = torch.where(t < 0, -(1.0 if rr is None else rr), t) * a
            e = (w - v) / U[j, j]
            Err[:, j - b0] = e
            if j + 1 < b1:
                Wc[:, j + 1:b1] -= e.unsqueeze(1) * U[j, j + 1:b1].unsqueeze(0)
        if b1 < K:
            Wc[:, b1:] -= Err @ U[b0:b1, b1:]
    return T


@torch.no_grad()
def quantize_r(r, bits, iters=12):
    """逐 block r (M,nB,1) → 每模組 2^bits 值碼書(log 域 Lloyd)。回傳 (r_q, codebook)。
    位元帳:bits/32 b/w(g32);bits=0 ⇒ 全 1(對稱)。"""
    if bits == 0:
        return torch.ones_like(r), torch.ones(1, device=r.device)
    x = r.reshape(-1).log()
    k = 2 ** bits
    qs = torch.quantile(x[torch.randperm(x.numel(), device=x.device)[:2_000_000]],
                        (torch.arange(k, device=x.device) + 0.5) / k)
    cb = qs.clone()
    for _ in range(iters):
        idx = (x[:, None] - cb[None]).abs().argmin(-1)
        for j in range(k):
            m = idx == j
            if m.any():
                cb[j] = x[m].mean()
    idx = (x[:, None] - cb[None]).abs().argmin(-1)
    return cb[idx].exp().reshape_as(r), cb.exp()


def dense_asym(T3, alpha, r):
    """(M,nB,B) 碼 + α + r → (M,K)(儲存欄序;inv 由呼叫端套)。"""
    M, nB, B = T3.shape
    return (alpha * asym_V(T3.float(), r)).reshape(M, nB * B)


def r_stats(r):
    if r is None:
        return None
    x = r.reshape(-1).float()
    q = torch.quantile(x, torch.tensor([0.1, 0.5, 0.9], device=x.device))
    return {"mean": round(float(x.mean()), 4), "p10": round(float(q[0]), 4),
            "p50": round(float(q[1]), 4), "p90": round(float(q[2]), 4),
            "frac_far": round(float(((x < 0.8) | (x > 1.25)).float().mean()), 4),
            "frac_clamped": round(float(((x <= R_MIN) | (x >= R_MAX)).float().mean()), 4)}

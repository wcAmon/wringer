"""E65-X 兩端夾擊階梯(零訓練;prereg_e65_x.json):三種模式共用一個逐層前傳骨架。

  fold   從上臂(U):起點 g9(全九元)。--fold-layers 內的層 T2→0,對 W9 目標閉式解 (α, r_blk)+逐元
         非對稱重投影(level 3),r 量化 2 bit;其餘層碼固定、α 閉式重擬(--refit-others)。
         **循序學生輸入**:層 li 的 Σ 在「層 < li 已折/已重擬並寫回模型」的前傳下量測;
         每層解完寫回模型再前傳一次供下一層(GPTQ 式循序;Codex §8.3 跨層偏移的第一版補償)。
  t2add  從下臂(D):起點 pa64_asymrefitq2 + res62s2 水。--t2-spec 選中的三元模組加第二位面 T2β
         (W ≈ α·T1(負側×r) + β·T2,β 逐 block;r 固定;E63 fit_t2beta 的 r 感知版)。Σ 於 pa64 部署形態單趟。
  rate   率失真台階(E66-RD):起點 g9。全模組(或 --rate-layers)對 W9 本尊做 GPTQ + 率罰 rounding
         (argmin (u−v)² + λ·bits,bits 由 --rd-bits 給的 9 格碼長;α 固定;循序學生輸入,解完寫回)。
         --rate-level 3 + --rate-layers:對照臂 S1g(該層 9→3 GPTQ 直解,α TWN 先驗 + 閉式重擬兩輪)。
  alpha  α 臂(A):起點同上。全模組 α 以 --alpha-block 重分組(32/16→128)閉式重解、--alpha-bits 量化
         (逐模組 log 域碼書,保號);--drop-r 撤 r(block 改變時必撤)。

每點存完整 200 模組態 data/qs_<save-tag>(未動的模組原樣複製)+ 位元帳(固定率與熵兩欄)寫入 --out。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.ladder_e65x --mode fold --state-tag g9 --fold-layers 0-15 --save-tag ux_u1
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.ladder_e65x --mode t2add --t2-spec "down_proj:0-15" --save-tag ux_d1
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.ladder_e65x --mode alpha --alpha-block 128 --alpha-bits 8 --drop-r --save-tag ux_a2
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from p2_anchor.wringer import ktier
from p2_anchor.wringer.asym import err_h, quantize_r, r_stats
from p2_anchor.wringer.asym_mix import (asym_Vm, code_V0, dense_asym_m, fit_alpha_r_m,
                                          neg_mask, round_grid_asym)
from p2_anchor.wringer.comp import HAccum
from p2_anchor.wringer.engine import capture_layer0, fwd_chain
from p2_anchor.wringer.ktier import scale_joint
from p2_anchor.wringer.modelio import (LAYER_PREFIX, N_LAYERS, cov_source_for,
                                         enumerate_targets, get_module,
                                         layer_index, load_model)
from p2_anchor.wringer.state import dense_weight, load_state

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"
FIXED_RATE = {3: 1.6, 5: 7 / 3, 9: 3.2, 4: 2.0, 8: 3.0, 16: 4.0, 256: 8.0}      # 固定率打包 b/w:5 trit/byte、3 五元/7 bit、兩位面;E68 整數格 3/4/8 bit


def fam(n):
    return n.split(".")[-1]


def parse_layers(s):
    out = set()
    for part in (s or "").split(","):
        part = part.strip()
        if not part or part.lower() == "none":
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def parse_t2spec(s):
    """'down_proj:0-15;all:8-15' → [(families|None, layers)]"""
    rules = []
    for item in (s or "").split(";"):
        item = item.strip()
        if not item:
            continue
        f, l = item.split(":")
        fams = None if f == "all" else set(f.split("+"))
        rules.append((fams, parse_layers(l)))
    return rules


def t2_selected(n, rules):
    li, f = layer_index(n), fam(n)
    return any((fams is None or f in fams) and li in layers for fams, layers in rules)


def entropy_bits(t):
    t = t.reshape(-1).to(torch.int64)
    cnt = torch.bincount(t - min(int(t.min()), -1), minlength=3).float()     # 三元 −1..1 舊行為;整數格自動擴 bin
    p = cnt / cnt.sum()
    return float(-(p[p > 0] * p[p > 0].log2()).sum())


def sym9(c):
    return (c["T1"].reshape(-1).to(torch.int64) + 1) * 3 + (c["T2"].reshape(-1).to(torch.int64) + 1)


def entropy_joint(c):
    cnt = torch.bincount(sym9(c), minlength=9).float()
    p = cnt / cnt.sum()
    return float(-(p[p > 0] * p[p > 0].log2()).sum())


def g9_bits(st):
    """全態 9 格直方圖 → 碼長 −log₂p(格序同 grid_tensors(9) 升冪:idx = 3(T1+1)+(T2+1) 對應值 T1+0.6T2 需重排)。"""
    cnt = torch.zeros(9, dtype=torch.float64)
    for c in st.values():
        if int(c.get("grid", 9)) == 9 and c.get("t2beta") is None:
            cnt += torch.bincount(sym9(c), minlength=9).double()
    p = (cnt / cnt.sum()).clamp_min(1e-12)
    bits_by_sym = -p.log2()
    vals, t1, t2 = ktier.grid_tensors(9, 0.6, "cpu")
    order = ((t1 + 1) * 3 + (t2 + 1)).long()                       # 升冪格 g ↔ sym 索引
    return [float(bits_by_sym[i]) for i in order]


def ledger(st):
    """整態位元帳(文字主幹 200 模組):固定率碼 / 熵碼 / α / β / r。"""
    tot = 0
    code_fixed = code_ent = code_joint = alpha_b = beta_b = r_b = 0.0
    for c in st.values():
        M, nB, B = c["T1"].shape
        n = M * nB * B
        tot += n
        g = int(c.get("grid", 9))
        has_t2 = bool((c["T2"] != 0).any())
        if c.get("t2beta") is not None:
            code_fixed += n * 3.2
            beta_b += n * 16 / B
        else:
            code_fixed += n * (FIXED_RATE[g] if g in ktier.INT_LEVELS else (FIXED_RATE[3] if not has_t2 else FIXED_RATE[g]))
        code_ent += n * (entropy_bits(c["T1"]) + (entropy_bits(c["T2"]) if has_t2 else 0.0))
        code_joint += n * (entropy_joint(c) if has_t2 else entropy_bits(c["T1"]))
        ab = float(c.get("a0_bits", 16))
        alpha_b += n * (ab / B + (16.0 / (nB * B) if ab < 16 else 0.0))   # E69:α int8 逐列 fp16 尺度另計
        if c.get("rneg") is not None:
            per_block = c["rneg"].numel() > M
            k = int(c["rneg_cb"].numel()) if c.get("rneg_cb") is not None else 0
            rb = math.log2(k) if k > 1 else 16.0
            r_b += n * (rb / B if per_block else rb / (nB * B))     # 逐列 r 幾乎免費
    d = {"code_fixed": code_fixed / tot, "code_entropy": code_ent / tot, "code_entropy_joint": code_joint / tot, "alpha": alpha_b / tot,
         "beta": beta_b / tot, "r": r_b / tot, "weights": tot}
    d["total_fixed"] = d["code_fixed"] + d["alpha"] + d["beta"] + d["r"]
    d["total_entropy"] = d["code_entropy"] + d["alpha"] + d["beta"] + d["r"]
    d["total_entropy_joint"] = d["code_entropy_joint"] + d["alpha"] + d["beta"] + d["r"]
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def to_storage(W_t, inv, Sig):
    M, K = W_t.shape
    q = torch.argsort(inv)
    ident = torch.equal(inv, torch.arange(K, device=inv.device))
    return W_t[:, q], (Sig if ident else Sig[q][:, q])


class CrossAccum:
    """老師/學生同批配對:H[n] = X3ᵀX3(學生輸入二階矩)、C[n] = X9ᵀX3(老師輸入 × 學生輸入)。
    teacher 鉤存本批 x9;student 鉤累積。mask (B,S) bool。"""

    def __init__(self):
        self.H, self.C, self.H9, self.x9 = {}, {}, {}, {}
        self.mask = None
        self._hooks = []

    def _x(self, args):
        x = args[0].detach().reshape(-1, args[0].shape[-1])
        return x[self.mask.reshape(-1)] if self.mask is not None else x

    def attach_teacher(self, named):
        for n, m in named:
            def pre(mod, args, _n=n):
                self.x9[_n] = self._x(args)
            self._hooks.append(m.register_forward_pre_hook(pre))

    def attach_student(self, named):
        for n, m in named:
            def pre(mod, args, _n=n):
                x3 = self._x(args).float()
                h = x3.T @ x3
                self.H[_n] = (self.H[_n] + h) if _n in self.H else h
                if _n in self.x9:
                    x9 = self.x9[_n].float()
                    c = x9.T @ x3
                    h9 = x9.T @ x9
                    self.C[_n] = (self.C[_n] + c) if _n in self.C else c
                    self.H9[_n] = (self.H9[_n] + h9) if _n in self.H9 else h9
            self._hooks.append(m.register_forward_pre_hook(pre))

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self.x9.clear()

    def cross_map(self, n, damp=0.01, mix=1.0):
        """Cm = I + mix·(C − Σ33)(Σ33 + λI)⁻¹:W_t = W9 @ Cm 使 X3 W_tᵀ ≈ X9 W9ᵀ(最小平方,增量形式;
        無漂移時 C = Σ33 ⇒ Cm = I 精確,阻尼只作用在增量)。"""
        H = self.H[n]
        lam = damp * float(torch.diagonal(H).mean())
        A = H + lam * torch.eye(H.shape[0], device=H.device)
        D = torch.linalg.solve(A, (self.C[n] - H).T).T
        return torch.eye(H.shape[0], device=H.device) + mix * D

    def mismatch(self, n, W, W9):
        """‖X3 Wᵀ − X9 W9ᵀ‖² / ‖X9 W9ᵀ‖²(校準集上,學生用 W 的輸出對老師輸出的相對誤差能量)。"""
        H, C, H9 = self.H[n], self.C[n], self.H9[n]
        t99 = float(((W9 @ H9) * W9).sum())
        v = float(((W @ H) * W).sum()) - 2.0 * float(((W @ C.T) * W9).sum()) + t99
        return v / max(t99, 1e-30)


from p2_anchor.wringer.asym import R_MAX, R_MIN


@torch.no_grad()
def fit_alpha_r_prior(W3, V0, neg, Sig, a_prior, alt, lam, r0=None):
    """(α, r_blk) 交替閉式,兩者皆 Tikhonov 拉向先驗(α → a_prior;r → 1,即 s → α)。"""
    M, nB, B = W3.shape
    r = torch.ones(M, nB, 1, device=W3.device) if r0 is None else r0
    al = scale_joint(W3, asym_Vm(V0, neg, r), Sig, prior=a_prior, lam=lam)
    zero = torch.zeros_like(V0)
    for _ in range(alt):
        n = torch.where(neg, -V0, zero)
        P = al * torch.where(neg, zero, V0)
        s_ = scale_joint(P - W3, n, Sig, prior=al, lam=lam)
        r = s_ / al
        empty = n.sum(-1, keepdim=True) == 0
        r = torch.where(empty | ~torch.isfinite(r), torch.ones_like(r), r).clamp(R_MIN, R_MAX)
        al = scale_joint(W3, asym_Vm(V0, neg, r), Sig, prior=a_prior, lam=lam)
    return al, r


@torch.no_grad()
def solve_rate(c, W_t, Sig, bits, lam, level=9, refit_alpha=False, prior_lam=0.01, scan=None):
    """E66-RD:α 固定,GPTQ + 率罰 rounding 對 W_t(=W9 本尊)。level 3 = S1g 對照(α TWN 先驗 + 閉式重擬,兩輪)。"""
    M, K = W_t.shape
    Wt_s, Sig_s = to_storage(W_t, c["inv"].cuda(), Sig)
    nB, B = c["T1"].shape[1], c["T1"].shape[2]
    ktier.set_block(B)
    W3 = Wt_s.reshape(M, nB, B)
    e_ref = err_h(Wt_s, Sig_s)
    s_old = sym9(c).cuda()
    cc = float(c["c"])
    scan_rec = []
    if level == 9:
        al = c["a0"].cuda().float()
        A = al.expand(M, nB, B).reshape(M, K)
        for lam_s in (scan or []):                                 # λ 掃描(smoke 校準用):只記 H/flips,不存
            T1s, T2s = ktier.gptq_grid_rd(Wt_s, Sig_s, 9, A, bits, lam_s, cc)
            es = {"T1": T1s.reshape(M, nB, B).to(torch.int8).cpu(), "T2": T2s.reshape(M, nB, B).to(torch.int8).cpu()}
            scan_rec.append({"lam": lam_s, "H": entropy_joint(es), "flips": float((sym9(es).cuda() != s_old).float().mean())})
        T1, T2 = ktier.gptq_grid_rd(Wt_s, Sig_s, 9, A, bits, lam, cc)
        if refit_alpha:
            al = scale_joint(W3, (T1 + cc * T2).reshape(M, nB, B), Sig_s, prior=al, lam=prior_lam)
        ent = {"T1": T1.reshape(M, nB, B).to(torch.int8).cpu(), "T2": T2.reshape(M, nB, B).to(torch.int8).cpu(), "inv": c["inv"].clone(),
               "a0": al.cpu(), "c": cc, "grid": 9}
    else:
        al = ktier.init_alpha(Wt_s)
        bits0 = [0.0] * 3
        for _ in range(2):
            A = al.expand(M, nB, B).reshape(M, K)
            T1, T2 = ktier.gptq_grid_rd(Wt_s, Sig_s, 3, A, bits0, 0.0, cc)
            al = scale_joint(W3, T1.reshape(M, nB, B), Sig_s, prior=al, lam=prior_lam)
        ent = {"T1": T1.reshape(M, nB, B).to(torch.int8).cpu(), "T2": torch.zeros_like(T1).reshape(M, nB, B).to(torch.int8).cpu(),
               "inv": c["inv"].clone(), "a0": al.cpu(), "c": cc, "grid": 3}
    s_new = sym9(ent).cuda()
    e = err_h(Wt_s - dense_weight(ent)[:, torch.argsort(c["inv"].cuda())], Sig_s)
    rec = {"e": e, "rel_rms": math.sqrt(e / max(e_ref, 1e-30)), "flips": float((s_new != s_old).float().mean()),
           "entropy_joint": entropy_joint(ent), "level": level, "lam": lam, **({"scan": scan_rec} if scan_rec else {})}
    return ent, rec


# ---------------- 三種模組級求解 ----------------
@torch.no_grad()
def solve_fold(c, W_t, Sig, alt, r_bits, block=0, lam=0.01):
    """九元條目 → 三元 + α + r_q(對 W_t 目標;block>0 ⇒ 折層以該 block 重分組解 α/r)。回傳 (entry, rec)。"""
    M, K = W_t.shape
    Wt_s, Sig_s = to_storage(W_t, c["inv"].cuda(), Sig)
    nB0, B0 = c["T1"].shape[1], c["T1"].shape[2]
    B = int(block) if block else B0
    assert K % B == 0, f"K={K} 不整除 block {B}"
    nB = K // B
    ktier.set_block(B)
    W3 = Wt_s.reshape(M, nB, B)
    T1 = c["T1"].cuda().float().reshape(M, nB, B)
    a9 = c["a0"].cuda().float().reshape(M, nB0, 1).expand(M, nB0, B0).reshape(M, nB, B)
    e0 = err_h(Wt_s - (a9 * T1).reshape(M, K), Sig_s)             # 只丟 T2、α 不動
    e_ref = err_h(Wt_s, Sig_s)                                     # 目標輸出能量(相對量級用)
    V0, neg = T1, neg_mask(T1)
    a9p = a9.mean(-1, keepdim=True)                                # 先驗 (M,nB,1);重分組時取 block 內均值
    al, r = fit_alpha_r_prior(W3, V0, neg, Sig_s, a9p, alt, lam)
    T1n = T1
    for _ in range(alt):
        T1n, _ = round_grid_asym(W3, al, r, 3, float(c["c"]))
        al, r = fit_alpha_r_prior(W3, T1n, neg_mask(T1n), Sig_s, a9p, 1, lam, r0=r)
    rq, cb = quantize_r(r, r_bits)
    negn = neg_mask(T1n)
    al_q = scale_joint(W3, asym_Vm(T1n, negn, rq), Sig_s, prior=a9p, lam=lam)
    e = err_h(Wt_s - dense_asym_m(T1n, negn, al_q, rq), Sig_s)
    ent = {"T1": T1n.to(torch.int8).cpu(), "T2": torch.zeros_like(T1n).to(torch.int8).cpu(),
           "inv": c["inv"].clone(), "a0": al_q.cpu(), "c": c["c"], "grid": 3,
           "rneg": rq.reshape(M, -1).cpu(), "rneg_cb": cb.cpu()}
    rec = {"e0_dropT2": e0, "e": e, "absorb": 1 - e / max(e0, 1e-30), "rel_rms": math.sqrt(e / max(e_ref, 1e-30)),
           "flips_t1": float((T1n != T1).float().mean()), "r": r_stats(rq), "codebook": [round(float(v), 4) for v in cb]}
    return ent, rec


@torch.no_grad()
def solve_refit(c, W_t, Sig, lam=0.01):
    """碼固定(含 T2、rneg),α 閉式重擬(拉向原 α 先驗)。"""
    M, K = W_t.shape
    Wt_s, Sig_s = to_storage(W_t, c["inv"].cuda(), Sig)
    nB, B = c["T1"].shape[1], c["T1"].shape[2]
    W3 = Wt_s.reshape(M, nB, B)
    T1, T2 = c["T1"].cuda().float(), c["T2"].cuda().float()
    V0, neg = code_V0(T1, T2, c["c"]), neg_mask(T1)
    r = None if c.get("rneg") is None else c["rneg"].cuda().float().reshape(M, -1, 1)
    a_prior = c["a0"].cuda().float()
    e_before = err_h(Wt_s - dense_asym_m(V0, neg, a_prior, r), Sig_s)
    al = scale_joint(W3, asym_Vm(V0, neg, r), Sig_s, prior=a_prior, lam=lam)
    e = err_h(Wt_s - dense_asym_m(V0, neg, al, r), Sig_s)
    ent = dict(c)
    ent["a0"] = al.cpu()
    return ent, {"e_before": e_before, "e": e, "ratio": (e / e_before if e_before > 1e-12 * max(e, 1.0) else 1.0)}


@torch.no_grad()
def solve_t2add(c, W_t, Sig, alt):
    """三元(+r)條目加第二位面 T2β(r 感知):W ≈ α·V(r) + β·T2。"""
    assert int(c["grid"]) == 3 and c.get("t2beta") is None, "t2add 只對三元模組"
    M, K = W_t.shape
    Wt_s, Sig_s = to_storage(W_t, c["inv"].cuda(), Sig)
    nB, B = c["T1"].shape[1], c["T1"].shape[2]
    W3 = Wt_s.reshape(M, nB, B)
    T1 = c["T1"].cuda().float()
    r = None if c.get("rneg") is None else c["rneg"].cuda().float().reshape(M, -1, 1)
    Vr = asym_Vm(T1, neg_mask(T1), r)
    al = scale_joint(W3, Vr, Sig_s)
    e_c = err_h(Wt_s - (al * Vr).reshape(M, K), Sig_s)
    Rr = W3 - al * Vr
    thr = 0.7 * Rr.abs().mean(-1, keepdim=True)
    T2 = torch.sign(Rr) * (Rr.abs() > thr).float()
    beta = scale_joint(Rr, T2, Sig_s)
    for _ in range(alt):
        ok = beta.abs() > 1e-12
        T2 = torch.where(ok, (Rr / torch.where(ok, beta, torch.ones_like(beta))).clamp(-1, 1).round(), torch.zeros_like(T2))
        beta = scale_joint(Rr, T2, Sig_s)
        al = scale_joint(W3 - beta * T2, Vr, Sig_s)
        Rr = W3 - al * Vr
    e = err_h(Wt_s - (al * Vr + beta * T2).reshape(M, K), Sig_s)
    ent = dict(c)
    ent.update({"T2": T2.to(torch.int8).cpu(), "a0": al.cpu(), "t2beta": beta.reshape(M, nB).cpu(), "grid": 9})
    return ent, {"e_ctrl": e_c, "e": e, "ratio": e / max(e_c, 1e-30), "t2_nz": float((T2 != 0).float().mean()),
                 "beta_over_alpha_p50": round(float((beta / al).abs().median()), 4)}


@torch.no_grad()
def solve_alpha(c, W_t, Sig, block, bits, drop_r):
    """α 重分組 + 量化(保號 log 碼書);碼固定。"""
    M, K = W_t.shape
    Wt_s, Sig_s = to_storage(W_t, c["inv"].cuda(), Sig)
    nB0, B0 = c["T1"].shape[1], c["T1"].shape[2]
    Bn = int(block) if block else B0
    assert K % Bn == 0, f"K={K} 不整除 block {Bn}"
    ktier.set_block(Bn)                                            # scale_joint fallback 用全域 BLOCK
    T1 = c["T1"].cuda().float().reshape(M, K // Bn, Bn)
    T2 = c["T2"].cuda().float().reshape(M, K // Bn, Bn)
    W3 = Wt_s.reshape(M, K // Bn, Bn)
    V0, neg = code_V0(T1, T2, c["c"]), neg_mask(T1)
    keep_r = (not drop_r) and c.get("rneg") is not None
    assert not (keep_r and Bn != B0), "block 改變時必須 --drop-r"
    r = c["rneg"].cuda().float().reshape(M, -1, 1) if keep_r else None
    r0 = None if c.get("rneg") is None else c["rneg"].cuda().float().reshape(M, -1, 1)
    e_before = err_h(Wt_s - dense_asym_m(code_V0(c["T1"].cuda().float(), c["T2"].cuda().float(), c["c"]),
                                         neg_mask(c["T1"].cuda().float()), c["a0"].cuda().float(), r0), Sig_s)
    al = scale_joint(W3, asym_Vm(V0, neg, r), Sig_s)
    cb = None
    if bits:
        mag = al.abs().clamp_min(1e-8)
        mq, cb = quantize_r(mag, int(bits))
        al = torch.sign(al) * mq
        al = torch.where(al == 0, mq, al)
    e = err_h(Wt_s - dense_asym_m(V0, neg, al, r), Sig_s)
    ent = {"T1": T1.to(torch.int8).cpu(), "T2": T2.to(torch.int8).cpu(), "inv": c["inv"].clone(),
           "a0": al.cpu(), "c": c["c"], "grid": c["grid"], "a0_bits": int(bits) if bits else 16}
    if cb is not None:
        ent["a0_cb"] = cb.cpu()
    if keep_r:
        ent["rneg"] = c["rneg"].clone()
        if c.get("rneg_cb") is not None:
            ent["rneg_cb"] = c["rneg_cb"].clone()
    return ent, {"e_before": e_before, "e": e, "ratio": e / max(e_before, 1e-30), "block": Bn, "bits": int(bits) if bits else 16}


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fold", "t2add", "alpha", "rate"])
    ap.add_argument("--state-tag", default=None)
    ap.add_argument("--res-tag", default="res62s2")
    ap.add_argument("--target-state-tag", default=None,
                    help="t2add/alpha 的目標基底態(W_t = dense(此態) + 水);None = 同 --state-tag。pa64 已含水 ⇒ 必給 st51c_e2e(amendment_3)")
    ap.add_argument("--fold-layers", default="")
    ap.add_argument("--no-refit-others", action="store_true")
    ap.add_argument("--target", default="cross", choices=["cross", "w9"],
                    help="fold 目標:cross = W9·C·Σ33⁻¹(老師輸出對學生輸入,補跨層偏移);w9 = W9 本尊")
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--cross-mix", type=float, default=1.0, help="交叉補償增量比例(1 = 全量最小平方)")
    ap.add_argument("--r-bits", type=int, default=2)
    ap.add_argument("--fold-block", type=int, default=0, help="折層 α/r 的 block(0 = 沿用 32/16;128 = A 臂)")
    ap.add_argument("--prior-lam", type=float, default=0.01, help="fold/refit 的 α、r 先驗正則(相對 diag(A));0 = 舊行為")
    ap.add_argument("--t2-spec", default="")
    ap.add_argument("--alpha-block", type=int, default=0)
    ap.add_argument("--alpha-bits", type=int, default=0)
    ap.add_argument("--drop-r", action="store_true")
    ap.add_argument("--alt", type=int, default=3)
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--calib", default=str(EV / "calib_e58r1b_traj.pt"))
    ap.add_argument("--lengths", default=str(EV / "calib_e58r1b_len.pt"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-layer", type=int, default=N_LAYERS - 1)
    ap.add_argument("--save-tag", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lam", type=float, default=0.0, help="rate:率罰 λ(u 單位²/bit)")
    ap.add_argument("--rd-bits", default="", help="rate:9 格碼長 JSON 列表或含 'bits' 鍵的 json 檔;空 = 由 g9 直方圖 −log2p")
    ap.add_argument("--rate-layers", default="", help="rate:只動這些層(空 = 全部)")
    ap.add_argument("--rate-level", type=int, default=9, choices=[9, 3], help="rate:目標格點;3 = S1g 對照(GPTQ 9→3)")
    ap.add_argument("--lam-scan", default="", help="rate:逗號 λ 列表,每模組額外掃描記 H/flips(smoke 校準)")
    ap.add_argument("--rate-refit-alpha", action="store_true", help="rate(level 9):碼解完後 α 閉式重擬(先驗 lam);預設關(只動碼)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.state_tag is None:
        args.state_tag = "g9" if args.mode in ("fold", "rate") else "pa64_asymrefitq2"
    if args.smoke:
        args.n_calib, args.alt = 16, 2
        args.max_layer = min(args.max_layer, 2)
    args.out = args.out or str(EVC / f"ladder_{args.save_tag}.json")
    fold_layers = parse_layers(args.fold_layers)
    t2rules = parse_t2spec(args.t2_spec)
    rate_layers = parse_layers(args.rate_layers) if args.rate_layers else set(range(N_LAYERS))
    rd_bits = None
    if args.mode == "rate":
        if args.rd_bits:
            rd_bits = json.loads(Path(args.rd_bits).read_text())["bits"] if args.rd_bits.endswith(".json") else json.loads(args.rd_bits)
        print(f"rate: lam={args.lam} level={args.rate_level} layers={sorted(rate_layers)[:4]}..{max(rate_layers)} bits={rd_bits}", flush=True)
    t0 = time.time()
    torch.manual_seed(0)

    model, _ = load_model()
    model.requires_grad_(False)
    targets = enumerate_targets(model)
    st = load_state(args.state_tag, targets)
    st_t = load_state(args.target_state_tag, targets) if args.target_state_tag else None
    res = {}
    if args.mode == "rate" and rd_bits is None:
        rd_bits = g9_bits(st)
        print(f"rate: bits from g9 hist {[round(b, 3) for b in rd_bits]}", flush=True)
    if args.mode not in ("fold", "rate"):
        for li in range(N_LAYERS):
            p = Path("data") / f"res_{args.res_tag}" / f"layer{li:02d}.pt"
            if p.exists():
                res.update(torch.load(p, map_location="cpu", weights_only=False))
        assert all(n in res for n in targets), "水庫模組不齊"
    for n in targets:
        get_module(model, n).weight.data.copy_(dense_weight(st[n]).to(torch.bfloat16))
    cross = args.mode == "fold" and args.target == "cross"
    model_t = None
    if cross:                                                      # 老師副本(g9 本尊,權重不變)
        model_t, _ = load_model()
        model_t.requires_grad_(False)
        for n in targets:
            get_module(model_t, n).weight.data.copy_(dense_weight(st[n]).to(torch.bfloat16))
    calib = torch.load(args.calib, weights_only=False)[:args.n_calib]
    lens = torch.load(args.lengths, weights_only=False)[:calib.shape[0]]
    S = calib.shape[1]
    ar = torch.arange(S)
    Mk = [(ar[None] < lens[i:i + args.batch, None]).cuda() for i in range(0, calib.shape[0], args.batch)]
    X, kw = capture_layer0(model, calib, batch=args.batch, to_cpu=True)
    Xt = [x.clone() for x in X] if cross else None
    layers = [model.get_submodule(f"{LAYER_PREFIX}.{li}") for li in range(N_LAYERS)]
    layers_t = [model_t.get_submodule(f"{LAYER_PREFIX}.{li}") for li in range(N_LAYERS)] if cross else None
    print(f"load+capture {time.time()-t0:.0f}s batches {len(X)} seq {S} mode={args.mode} target={args.target if args.mode=='fold' else '-'}", flush=True)

    out = {"mode": args.mode, "state": args.state_tag, "hparams": vars(args), "modules": {}}
    save_dir = Path("data") / f"qs_{args.save_tag}"
    save_dir.mkdir(parents=True, exist_ok=True)
    new_state = {}

    def advance(layer, Xs, acc=None):
        """就地前傳(60G RAM:不留兩份 X)。acc 給定則同時累積統計。"""
        for bi in range(len(Xs)):
            if acc is not None:
                acc.mask = Mk[bi]
            Xs[bi] = fwd_chain([layer], Xs[bi], kw).to("cpu", torch.bfloat16)
        if acc is not None:
            acc.mask = None

    for li in range(N_LAYERS):
        mods = [n for n in targets if layer_index(n) == li]
        if li > args.max_layer:
            for n in mods:
                new_state[n] = st[n]
            torch.save({n: new_state[n] for n in mods}, save_dir / f"layer{li:02d}.pt")
            continue
        layer = layers[li]
        if args.mode == "fold":
            todo = {n: ("fold" if li in fold_layers else ("refit" if (cross and not args.no_refit_others) else None)) for n in mods}
        elif args.mode == "rate":
            todo = {n: ("rate" if (li in rate_layers and int(st[n]["grid"]) == 9 and st[n].get("t2beta") is None) else None) for n in mods}
        elif args.mode == "t2add":
            todo = {n: ("t2add" if (t2_selected(n, t2rules) and int(st[n]["grid"]) == 3) else None) for n in mods}
        else:
            todo = {n: "alpha" for n in mods}
        need = [n for n in mods if todo[n]]
        srcs = sorted({cov_source_for(n) for n in need})
        acc = CrossAccum() if cross else HAccum()
        if need:
            if cross:
                acc.attach_teacher([(s_, get_module(model_t, s_)) for s_ in srcs])
                acc.attach_student([(s_, get_module(model, s_)) for s_ in srcs])
                for bi in range(len(X)):                           # 同批:老師先(存 x9、就地推進)、學生後(累積、丟棄輸出)
                    acc.mask = Mk[bi]
                    Xt[bi] = fwd_chain([layers_t[li]], Xt[bi], kw).to("cpu", torch.bfloat16)
                    fwd_chain([layer], X[bi], kw)
                acc.mask = None
            elif args.mode in ("fold", "rate"):
                acc.attach([(s_, get_module(model, s_)) for s_ in srcs])
                for bi in range(len(X)):
                    acc.mask = Mk[bi]
                    fwd_chain([layer], X[bi], kw)
                acc.mask = None
            else:                                                  # 單趟:統計 + 就地推進(權重不寫回)
                acc.attach([(s_, get_module(model, s_)) for s_ in srcs])
                advance(layer, X, acc)
            acc.detach()
        elif cross:
            advance(layers_t[li], Xt)
        line = []
        for n in mods:
            c = st[n]
            if not todo[n]:
                new_state[n] = c
                continue
            src = cov_source_for(n)
            Sig = acc.H[src]
            ktier.set_block(int(c["T1"].shape[2]))
            W_cur = dense_weight(c)
            if args.mode == "fold":
                W_t = (W_cur @ acc.cross_map(src, args.damp, args.cross_mix)) if cross else W_cur
                ent, rec = (solve_fold(c, W_t, Sig, args.alt, args.r_bits, args.fold_block, args.prior_lam) if todo[n] == "fold"
                            else solve_refit(c, W_t, Sig, args.prior_lam))
                rec["drift_rel"] = math.sqrt(err_h(W_t - W_cur, Sig) / max(err_h(W_cur, Sig), 1e-30))
                if cross:                                            # 對老師輸出的相對誤差能量:原碼 / 補償目標 / 折後
                    rec["out_err_w9"] = acc.mismatch(src, W_cur, W_cur)
                    rec["out_err_target"] = acc.mismatch(src, W_t, W_cur)
                    rec["out_err_final"] = acc.mismatch(src, dense_weight(ent), W_cur)
            elif args.mode == "rate":
                W_t = W_cur
                ent, rec = solve_rate(c, W_t, Sig, rd_bits, args.lam, args.rate_level, args.rate_refit_alpha, args.prior_lam,
                                      scan=[float(x) for x in args.lam_scan.split(",")] if args.lam_scan else None)
            else:
                water = (res[n]["B"].cuda().float() @ res[n]["A"].cuda().float()) / float(res[n]["r"])
                W_base = dense_weight(st_t[n]) if st_t is not None else W_cur
                W_t = W_base + water
                ent, rec = (solve_t2add(c, W_t, Sig, args.alt) if args.mode == "t2add"
                            else solve_alpha(c, W_t, Sig, args.alpha_block, args.alpha_bits, args.drop_r))
            rec.update({"op": todo[n], "family": fam(n), "layer": li,
                        **({"base_rel_gap": float((W_base - W_cur).norm() / W_cur.norm())} if args.mode in ("t2add", "alpha") else {}),
                        "dense_rel_change": float((dense_weight(ent) - W_cur).norm() / W_cur.norm())})
            out["modules"][n] = rec
            new_state[n] = ent
            if args.mode in ("fold", "rate"):
                get_module(model, n).weight.data.copy_(dense_weight(ent).to(torch.bfloat16))
            key = "absorb" if "absorb" in rec else ("rel_rms" if "rel_rms" in rec and "ratio" not in rec else "ratio")
            line.append(f"{fam(n)[:9]:9s} {todo[n]}:{rec[key]:.3f}" + (f" flips={rec['flips_t1']:.4f}" if "flips_t1" in rec else "")
                        + (f" drift={rec['drift_rel']:.3f}" if "drift_rel" in rec else "")
                        + f" dW={rec['dense_rel_change']:.3f}"
                        + (f" out_err w9/tgt/final={rec['out_err_w9']:.3f}/{rec['out_err_target']:.3f}/{rec['out_err_final']:.3f}" if "out_err_w9" in rec else "")
                        + (f" nz={rec['t2_nz']:.3f}" if "t2_nz" in rec else "")
                        + (f" flips={rec['flips']:.4f} H={rec['entropy_joint']:.3f}" if "entropy_joint" in rec else "")
                        + ((" scan " + " ".join(f"{q['lam']}:{q['H']:.3f}/{q['flips']:.3f}" for q in rec["scan"])) if "scan" in rec else ""))
            del Sig, W_cur, W_t
        acc.H.clear()
        if hasattr(acc, "C"):
            acc.C.clear()
            acc.H9.clear()
        if args.mode in ("fold", "rate") or not need:              # fold/rate:寫回後就地推進(循序學生輸入);非 fold 有工作者已於統計趟推進
            advance(layer, X)
        torch.cuda.empty_cache()
        torch.save({n: new_state[n] for n in mods}, save_dir / f"layer{li:02d}.pt")
        print(f"LADDER_LAYER {li} ({time.time()-t0:.0f}s)" + ("\n   " + "\n   ".join(line) if line else ""), flush=True)
        Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))

    out["ledger"] = ledger(new_state)
    out["runtime_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("LADDER_LEDGER " + json.dumps(out["ledger"]), flush=True)
    print(f"LADDER_DONE {args.save_tag} {out['runtime_s']}s", flush=True)


if __name__ == "__main__":
    main()

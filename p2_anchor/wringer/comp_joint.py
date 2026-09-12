"""E64 S3:comp 事件內的 d 欄聯合求解器(joint_comp_grid)+ 配對碼書(build_tuple_codebook)。

B 路線收編為排水求解器升級的 A/B 工具(prereg_e64_probes S3):
  G   greedy_comp_grid(現行,逐元 Δobj 只用 H 對角)
  J4  joint_comp_grid(d=4, allowed=None):相鄰 d 欄為一組,對 2^d 個「動/不動」子集用 d×d H 子塊精確打分
      (含欄間交叉項),每組取最佳子集,再走與 greedy 同款的 top-k 階梯回退(全 H 目標驗證)+ 殘差回饋。
      隔離「求解器變好」。d=1 逐位退化為 greedy(checks_joint C4)。
  V4  joint_comp_grid(d=4, allowed=碼書):只允許落到碼書內的 d 元組(n_vals^d 中取 K 種),
      事件起點先做 conform(組外元組以精確 d×d 打分移到最佳碼書元組)。隔離「格子變圓」。
      allowed 含全部元組時 ≡ J4(checks_joint C5)。
非對稱 rneg 語義與 greedy_comp_grid 同(部署位移 delta 打分、潛位移 delta_lat 寫 comp)。
E36/E37 原函數不動(證據保存)。
"""
from itertools import product

import torch


def _subsets(d, device):
    return torch.tensor(list(product((0, 1), repeat=d)), dtype=torch.float32, device=device)  # (2^d, d)


def tuple_ids(idx_q, n_vals):
    """idx_q (..., d) long → 元組 id (...,) = Σ idx_i · n_vals^i。"""
    d = idx_q.shape[-1]
    w = (n_vals ** torch.arange(d, device=idx_q.device)).to(idx_q.dtype)
    return (idx_q * w).sum(-1)


@torch.no_grad()
def build_tuple_codebook(idx, d, keep, n_vals):
    """idx (M,K) long 現行格點索引;以相鄰 d 欄元組頻率取前 keep 種。回傳 allowed (n_vals^d,) bool。"""
    M, K = idx.shape
    ids = tuple_ids(idx.reshape(M, K // d, d), n_vals).reshape(-1)
    counts = torch.bincount(ids, minlength=n_vals ** d)
    keep = min(int(keep), counts.numel())
    allowed = torch.zeros(counts.numel(), dtype=torch.bool, device=idx.device)
    allowed[torch.topk(counts, keep).indices] = True
    return allowed


@torch.no_grad()
def joint_comp_grid(p, W0, R, H, rounds=3, cap=0.03, rneg=None, d=4, allowed=None,
                    row_chunk=256):
    """與 comp.greedy_comp_grid 同介面;d=1 且 allowed=None 逐位同 greedy。回傳 dict(含 conform_flips)。"""
    assert p.comp is not None, "STQParam 未開 use_comp"
    M, K = R.shape
    assert K % d == 0, f"K={K} 不整除 d={d}"
    Q = K // d
    u, alpha = p._u(W0)
    nB = p.a0.shape[1]
    u2 = u.reshape(M, K).float()
    a2 = alpha.expand(-1, -1, K // nB).reshape(M, K).float()
    theta = p.theta_eff().detach().float()
    vals = p.grid.vals.float()
    n_vals = vals.shape[0]
    idx = torch.bucketize(u2, theta)
    hd = torch.diagonal(H).clamp_min(1e-12)
    rk = (None if rneg is None else rneg.detach().float().expand(-1, -1, K // nB).reshape(M, K))
    feff = (lambda x: x) if rk is None else (lambda x: torch.where(x < 0, x * rk, x))
    dev = R.device
    S = _subsets(d, dev)                                            # (2^d, d)
    eye = torch.eye(d, device=dev)
    SS_off = (S[:, :, None] * S[:, None, :]) * (1 - eye)             # (2^d, d, d)
    ar = torch.arange(Q, device=dev)
    Hb = H.reshape(Q, d, Q, d)[ar, :, ar, :]                          # (Q, d, d) 對角子塊
    Hb_off = Hb * (1 - eye)
    hdq = hd.reshape(Q, d)
    n_flip = 0
    conform_flips = 0
    obj0 = None

    def apply(dsel_flat, dlat_flat, idx_new_flat, selm_flat):
        nonlocal R, u2, idx
        dsel = dsel_flat.reshape(M, K)
        dlat = dlat_flat.reshape(M, K)
        p.comp += dlat
        R = R - dsel
        u2 = u2 + dlat / a2
        idx = torch.where(selm_flat.reshape(M, K), idx_new_flat.reshape(M, K), idx)

    # ---- conform:組外元組移到最佳碼書元組(精確 d×d 打分) ----
    if allowed is not None and not bool(allowed.all()):
        cand = torch.tensor(list(product(range(n_vals), repeat=d)), device=dev, dtype=torch.long)  # (n^d, d)
        cand = cand[allowed[tuple_ids(cand, n_vals)]]                                               # (nc, d)
        nc = cand.shape[0]
        RH = R @ H
        obj0 = float((RH * R).sum())
        cur_ids = tuple_ids(idx.reshape(M, Q, d), n_vals)                                            # (M,Q)
        bad = ~allowed[cur_ids]                                                                        # (M,Q)
        if bool(bad.any()):
            D_all = torch.zeros(M, K, device=dev)
            Dlat_all = torch.zeros(M, K, device=dev)
            new_idx = idx.clone()
            for m0 in range(0, M, row_chunk):
                m1 = min(M, m0 + row_chunk)
                bq = bad[m0:m1]                                                                       # (m,Q)
                if not bool(bq.any()):
                    continue
                uq = u2[m0:m1].reshape(m1 - m0, Q, d)
                aq = a2[m0:m1].reshape(m1 - m0, Q, d)
                rhq = RH[m0:m1].reshape(m1 - m0, Q, d)
                vc = vals[cand]                                                                        # (nc, d)
                if rk is None:
                    fvc = vc.view(1, 1, nc, d)
                    fu = uq.unsqueeze(2)
                else:
                    rkq = rk[m0:m1].reshape(m1 - m0, Q, 1, d)
                    fvc = torch.where(vc.view(1, 1, nc, d) < 0, vc.view(1, 1, nc, d) * rkq, vc.view(1, 1, nc, d).expand(m1 - m0, Q, nc, d))
                    fu = torch.where(uq < 0, uq * rk[m0:m1].reshape(m1 - m0, Q, d), uq).unsqueeze(2)
                dl = aq.unsqueeze(2) * (vc.view(1, 1, nc, d) - uq.unsqueeze(2))                       # 潛位移 (m,Q,nc,d)
                dd = aq.unsqueeze(2) * (fvc - fu)                                                      # 部署位移
                lin = (dd * dd * hdq.view(1, Q, 1, d) - 2.0 * dd * rhq.unsqueeze(2)).sum(-1)          # (m,Q,nc)
                cross = torch.einsum("mqci,mqcj,qij->mqc", dd, dd, Hb_off)
                dobj = lin + cross
                best = dobj.argmin(-1)                                                                 # (m,Q)
                sel = bq
                bi = best[sel]
                mi, qi = sel.nonzero(as_tuple=True)
                D_all[m0:m1].reshape(m1 - m0, Q, d)[mi, qi] = dd[mi, qi, bi]
                Dlat_all[m0:m1].reshape(m1 - m0, Q, d)[mi, qi] = dl[mi, qi, bi]
                new_idx[m0:m1].reshape(m1 - m0, Q, d)[mi, qi] = cand[bi]
            moved = (new_idx != idx)
            conform_flips = int(moved.sum())
            apply(D_all, Dlat_all, new_idx, torch.ones_like(idx, dtype=torch.bool))
            del D_all, Dlat_all, new_idx

    cap_round = max(int(cap / rounds * R.numel()), 1)
    cap_q = cap_round if d == 1 else max(cap_round // 2, 1)
    for _ in range(rounds):
        RH = R @ H
        cur = float((RH * R).sum())
        if obj0 is None:
            obj0 = cur
        step = torch.sign(RH).long()
        tgt = (idx + step).clamp_(0, n_vals - 1)
        move = tgt != idx
        delta_lat = a2 * (vals[tgt] - u2)
        delta = (delta_lat if rk is None else a2 * (feff(vals[tgt]) - feff(u2)))
        delta = torch.where(move, delta, torch.zeros_like(delta))
        delta_lat = torch.where(move, delta_lat, torch.zeros_like(delta_lat))
        D = delta.reshape(M, Q, d)
        RHq = RH.reshape(M, Q, d)
        lin = D * D * hdq.view(1, Q, d) - 2.0 * D * RHq                       # (M,Q,d)
        if d == 1:
            dobj = lin                                                           # (M,Q,1) ≡ 子集 {1}
            gain = torch.where(move.reshape(M, Q, d), -dobj, torch.zeros_like(dobj)).reshape(M, Q)
            best_s = torch.ones(M, Q, dtype=torch.long, device=dev)
        else:
            P = D.unsqueeze(-1) * D.unsqueeze(-2) * Hb_off.view(1, Q, d, d)        # (M,Q,d,d)
            dobj = lin @ S.T + torch.einsum("mqij,sij->mqs", P, SS_off)            # (M,Q,2^d)
            if allowed is not None:
                idq = idx.reshape(M, Q, d)
                stq = torch.where(move, step, torch.zeros_like(step)).reshape(M, Q, d)
                ids_s = tuple_ids(idq.unsqueeze(2) + (S.long().view(1, 1, -1, d) * stq.unsqueeze(2)), n_vals)  # (M,Q,2^d)
                dobj = torch.where(allowed[ids_s], dobj, torch.full_like(dobj, float("inf")))
            best_s = dobj.argmin(-1)                                             # (M,Q)
            gain = -dobj.gather(-1, best_s.unsqueeze(-1)).squeeze(-1)
            gain = torch.where(torch.isfinite(gain), gain, torch.zeros_like(gain))
        k = min(cap_q, int((gain > 0).sum()))
        if k == 0:
            break
        order = torch.topk(gain.reshape(-1), k).indices
        Ssel = S[best_s.reshape(-1)]                                              # (M*Q, d) 每組最佳子集
        dsel_q = (D.reshape(M * Q, d) * Ssel)                                    # 只動子集內元素
        dlat_q = (delta_lat.reshape(M * Q, d) * Ssel)
        best_j, best_obj = 0, cur
        j = k
        while j >= 1:
            Dt = torch.zeros(M * Q, d, device=dev)
            Dt[order[:j]] = dsel_q[order[:j]]
            Rt = R - Dt.reshape(M, K)
            objt = float(((Rt @ H) * Rt).sum())
            if objt < best_obj:
                best_j, best_obj = j, objt
            j //= 2
        if best_j == 0:
            break
        selq = torch.zeros(M * Q, dtype=torch.bool, device=dev)
        selq[order[:best_j]] = True
        selm = (selq.unsqueeze(-1) & (Ssel > 0.5) & move.reshape(M * Q, d))         # (M*Q, d) 實際移動元素
        dsel = torch.where(selm, dsel_q, torch.zeros_like(dsel_q))
        dlat = torch.where(selm, dlat_q, torch.zeros_like(dlat_q))
        apply(dsel, dlat, tgt.reshape(M * Q, d), selm)
        n_flip += int(selm.sum())
    obj1 = float(((R @ H) * R).sum())
    if obj0 is None:
        obj0 = obj1
    return {"flips": n_flip, "conform_flips": conform_flips,
            "absorb": round(1.0 - obj1 / max(obj0, 1e-30), 4),
            "obj_before": obj0, "obj_after": obj1}

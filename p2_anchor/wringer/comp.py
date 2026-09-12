"""E36 合併即補償(error-feedback collapse;prereg_e36_ternary_comp)。

7→3 走廊的 c 遞減 = 對 c·T₂ 平面的結構化剪枝(恆等式:
W_hard(c₂) − W_hard(c₁) = (c₂−c₁)·α⊙T₂,作為理論來源與事件跳過閘)。
殘差採直接量測:R = W_fp − W_hard(當下)= 對 fp 功能的總虧欠
(CPU 檢②裁定:逐事件 Δc 切片每權重 ~0.1α,永遠低於整格步 α 的
增益判準——總虧欠才會隨走廊累積到「跨相關欄聚合 ≥ 半步」而觸發)。
本模組用 H = Σxxᵀ 驅動的貪婪整格跳躍把 R 重分配到其他權重
(0 胞→±1 胞 = 顯式 recruitment),經 STQParam.comp 潛空間位移落地
(extract 全數吸收,部署形態不變)。

退化解防護(prereg math.degenerate_guard):僅允許「異胞」跳躍——
同胞內回填只是把 ridge u 推回刀口,c=0 時停在門檻 0.5 上;遮罩排除。

單移動增益:obj(R) = Tr(R·H·Rᵀ);位移 δ 於 (m,k) 的變化
  Δobj = δ²·H_kk − 2δ·(R·H)[m,k]
δ = α·(v_tgt − u) 落胞心(硬投影穩定);僅收 Δobj<0,逐輪殘差回饋。
"""
import torch


class HAccum:
    """逐模組輸入二階矩累積器(與老師前傳共用同一趟 no-grad 前傳)。

    mask:(B,S) bool,設定後只累積有效位置(E54 8k pad 遮罩);
    None = 舊行為逐位不變。呼叫端於每批前傳前設 self.mask。"""

    def __init__(self):
        self.H = {}
        self._hooks = []
        self.mask = None

    def attach(self, named_modules):
        for n, m in named_modules:
            def pre(mod, args, _n=n):
                x = args[0].detach().reshape(-1, args[0].shape[-1])
                if self.mask is not None:
                    x = x[self.mask.reshape(-1)]
                x = x.float()
                h = x.T @ x
                self.H[_n] = (self.H[_n] + h) if _n in self.H else h
            self._hooks.append(m.register_forward_pre_hook(pre))

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def pop(self, n):
        return self.H.pop(n, None)


@torch.no_grad()
def greedy_comp(p, W0, R, H, c_now, rounds=3, cap=0.03):
    """對單模組吸收殘差 R(M,K,weight 空間):貪婪異胞整格跳躍寫入 p.comp。

    cell 依 3元 存續界 b=(1+c)/2 判(prereg terminal_boundary);
    目標胞 = cell + sign(R·H) 夾 [−1,1],同胞遮罩。每輪封頂
    cap/rounds,總量 ≤ cap。回傳 log(flips / absorb / obj 前後)。
    """
    assert p.comp is not None, "STQParam 未開 use_comp"
    M, K = R.shape
    u, alpha = p._u(W0)
    nB = p.a0.shape[1]
    u2 = u.reshape(M, K).float()
    a2 = alpha.expand(-1, -1, K // nB).reshape(M, K).float()
    b = (1.0 + float(c_now)) / 2
    cell = torch.where(u2.abs() < b, torch.zeros_like(u2), torch.sign(u2))
    hd = torch.diagonal(H).clamp_min(1e-12)                   # (K,)
    obj0 = None
    n_flip = 0
    cap_round = max(int(cap / rounds * R.numel()), 1)
    for _ in range(rounds):
        RH = R @ H                                            # (M,K)
        cur = float((RH * R).sum())
        if obj0 is None:
            obj0 = cur
        v_tgt = (cell + torch.sign(RH)).clamp_(-1.0, 1.0)
        move = v_tgt != cell                                  # 異胞防護
        delta = a2 * (v_tgt - u2)                             # 落胞心位移
        dobj = delta * delta * hd - 2.0 * delta * RH          # 單移動近似
        gain = torch.where(move, -dobj, torch.zeros_like(dobj))
        k = min(cap_round, int((gain > 0).sum()))
        if k == 0:
            break
        # 聯合超衝防護:獨立評分在相關欄上超加性——選擇集大小走
        # 階梯回退(k, k/2, …, 1),逐一算「聯合」obj,取實降最多者。
        order = torch.topk(gain.reshape(-1), k).indices
        dflat = delta.reshape(-1)
        best_j, best_obj = 0, cur
        j = k
        while j >= 1:
            D = torch.zeros_like(dflat)
            D[order[:j]] = dflat[order[:j]]
            Rt = R - D.reshape(R.shape)
            objt = float(((Rt @ H) * Rt).sum())
            if objt < best_obj:
                best_j, best_obj = j, objt
            j //= 2
        if best_j == 0:
            break
        selm = torch.zeros_like(dflat, dtype=torch.bool)
        selm[order[:best_j]] = True
        selm = selm.reshape(R.shape)
        d = torch.where(selm, delta, torch.zeros_like(delta))
        p.comp += d
        R = R - d
        u2 = u2 + d / a2
        cell = torch.where(selm, v_tgt, cell)
        n_flip += int(best_j)
    obj1 = float(((R @ H) * R).sum())
    if obj0 is None:
        obj0 = obj1
    return {"flips": n_flip,
            "absorb": round(1.0 - obj1 / max(obj0, 1e-30), 4),
            "obj_before": obj0, "obj_after": obj1}


@torch.no_grad()
def greedy_comp_grid(p, W0, R, H, rounds=3, cap=0.03, rneg=None):
    """E37 泛化版:任意 SoftGrid(3/5/9 混元)上的貪婪鄰格跳躍。

    與 greedy_comp 同數學(Δobj 打分、階梯回退、殘差回饋),差異:
    胞語義由「3元 存續界」換成模組自身 grid——現行指派 idx 由
    theta_eff 分桶,目標 = 鄰格 vals[idx±1](夾界),落點即格點值
    (硬投影穩定:格點在門檻之間,保序由 THETA_RANGE 保證)。
    E36 原函數不動(證據保存)。R 為呼叫端定義的虧欠(E37:R_eff)。
    E63 S1:rneg (M,nB,1) 非對稱負尺度——部署值 f(x)=x·r (x<0);Δobj 以部署位移
    delta 打分、潛位移 delta_lat 落格點寫 comp;rneg=None 時兩者同物件,逐位舊行為。
    """
    assert p.comp is not None, "STQParam 未開 use_comp"
    M, K = R.shape
    u, alpha = p._u(W0)
    nB = p.a0.shape[1]
    u2 = u.reshape(M, K).float()
    a2 = alpha.expand(-1, -1, K // nB).reshape(M, K).float()
    theta = p.theta_eff().detach().float()
    vals = p.grid.vals.float()
    n_vals = vals.shape[0]
    idx = torch.bucketize(u2, theta)                          # (M,K)∈[0,n−1]
    hd = torch.diagonal(H).clamp_min(1e-12)
    rk = (None if rneg is None else
          rneg.detach().float().expand(-1, -1, K // nB).reshape(M, K))
    feff = (lambda x: x) if rk is None else (
        lambda x: torch.where(x < 0, x * rk, x))
    obj0 = None
    n_flip = 0
    cap_round = max(int(cap / rounds * R.numel()), 1)
    for _ in range(rounds):
        RH = R @ H
        cur = float((RH * R).sum())
        if obj0 is None:
            obj0 = cur
        step = torch.sign(RH).long()
        tgt = (idx + step).clamp_(0, n_vals - 1)
        move = tgt != idx
        delta_lat = a2 * (vals[tgt] - u2)                     # 落格點(潛)
        delta = (delta_lat if rk is None else
                 a2 * (feff(vals[tgt]) - feff(u2)))           # 部署位移
        dobj = delta * delta * hd - 2.0 * delta * RH
        gain = torch.where(move, -dobj, torch.zeros_like(dobj))
        k = min(cap_round, int((gain > 0).sum()))
        if k == 0:
            break
        order = torch.topk(gain.reshape(-1), k).indices
        dflat = delta.reshape(-1)
        best_j, best_obj = 0, cur
        j = k
        while j >= 1:
            D = torch.zeros_like(dflat)
            D[order[:j]] = dflat[order[:j]]
            Rt = R - D.reshape(R.shape)
            objt = float(((Rt @ H) * Rt).sum())
            if objt < best_obj:
                best_j, best_obj = j, objt
            j //= 2
        if best_j == 0:
            break
        selm = torch.zeros_like(dflat, dtype=torch.bool)
        selm[order[:best_j]] = True
        selm = selm.reshape(R.shape)
        d = torch.where(selm, delta, torch.zeros_like(delta))
        d_lat = d if rk is None else torch.where(selm, delta_lat,
                                                 torch.zeros_like(delta_lat))
        p.comp += d_lat
        R = R - d
        u2 = u2 + d_lat / a2
        idx = torch.where(selm, tgt, idx)
        n_flip += int(best_j)
    obj1 = float(((R @ H) * R).sum())
    if obj0 is None:
        obj0 = obj1
    return {"flips": n_flip,
            "absorb": round(1.0 - obj1 / max(obj0, 1e-30), 4),
            "obj_before": obj0, "obj_after": obj1}

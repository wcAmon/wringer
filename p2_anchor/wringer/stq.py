"""ST 軟退火量化(Corkscrew 核心;移植 guava-qat src/gqat/stq.py + θ 擴充)。

f(u;s) = Σ_j (h_j/2)·tanh(s(u−θ_j)) / tanh(s·H/2)
s→0 恆等、s→∞ 最近網格;硬階段 = 最近網格投影 + STE。
可訓參數 = LoRA(r=64,量化前加在權重上)+ δ_α logit(sigmoid×2 乘子)
         [+ 選配 dθ:門檻逐模組可學(circus 擴充,部署自由——烘焙後無痕)]。
部署形態不變:W = (2σ(logit)·α₀) ⊙ (T₁ + c·T₂),LoRA/θ 全數被吸收。

θ 擴充註記:θ_j = mid_j + 0.45·gap_j·tanh(dθ_j),界內保序;軟/STE/extract
三路一律用同一 θ_eff(訓練態=烘焙態)。dθ=0 時逐位還原 guava 語義。
θ 可學會微破 s→0 恆等極限的對稱消去——dθ 初值 0、s 由小走大,漂移漸進,
prereg 以 G-disc 系列監測。
"""
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from p2_anchor.wringer.ktier import C_K2, grid_tensors, round_with_mids
from p2_anchor.wringer.morph import corridor, grid_from_c, soft_staircase_c

THETA_RANGE = 0.45   # dθ 界:±0.45×相鄰檔位間距(嚴格保序)


class SoftGrid:
    """軟階梯網格;n=9/7/5/3 同一機制(門檻數 = n−1)。"""

    def __init__(self, c=C_K2, device="cuda", n=9):
        vals, t1, t2 = grid_tensors(n, c, device)
        self.n = n
        self.vals, self.t1, self.t2 = vals, t1, t2
        self.theta = (vals[1:] + vals[:-1]) / 2          # (n-1,) 中點門檻
        self.h = vals[1:] - vals[:-1]                    # (n-1,) 檔位間距
        self.H = float(vals[-1] - vals[0])
        # 恆等極限成立條件:Σh_j/H = 1(自動)、對稱網格 Σh_jθ_j = 0;E68 整數格 {−4..3} 非對稱 ⇒ 常數偏移,不斷言
        from p2_anchor.wringer.ktier import INT_LEVELS
        if n not in INT_LEVELS:
            assert abs(float((self.h * self.theta).sum())) < 1e-6


def soft_staircase(u, s, grid: SoftGrid, theta):
    out = torch.zeros_like(u)
    for j in range(theta.shape[0]):                      # 逐門檻累加,免大張量
        out = out + (grid.h[j] / 2) * torch.tanh(s * (u - theta[j]))
    return out / math.tanh(s * grid.H / 2)


class STQParam(nn.Module):
    """parametrize 掛在 target Linear 的 weight 上。

    mode: 'off'(fp,算重建目標用)/ 'soft'(可微退火)/ 'hard'(STE)。
    """

    def __init__(self, a0, grid: SoftGrid, r=64, c=C_K2, learn_theta=False,
                 use_comp=False, mix_block=0, lmix_block=0):
        super().__init__()
        self.grid = grid
        self.c = c
        # E61 擴桶:輸入維逐塊混合 M(k,b,b),部署形態 W_eff = W_hard @ blockdiag(M)
        # 不被吸收、不進碼;0 = 關(逐位還原舊行為)
        self._mix_block = int(mix_block)
        self.mix = None
        # E62b 輸出側桶:列塊混合 L(M/b,b,b),W_eff = blockdiag(L) @ W_hard @ blockdiag(M)
        # 0 = 關(逐位還原);探針 @probe_e62b_win:down_proj 上 0.11 b/w(int8)拿八成吸水
        self._lmix_block = int(lmix_block)
        self.lmix = None
        self.register_buffer("a0", a0.detach().float().clone())   # (M,nB,1)
        self.mode = "off"
        self.s = 0.0
        self._shape = None
        self._r = r
        self.lora_A = None   # 延遲 init(需知 K)
        self.lora_B = None
        self._use_comp = use_comp
        self.comp = None     # E36 補償位移(潛空間,非訓練;延遲 init)
        self.res_B = None    # E37 水庫旁路(量化後、凍結;load_reservoir 載入)
        self.res_A = None
        self._r_res = 1
        self.gamma = 0.0     # 排水閥;僅 res 載入後有意義
        self.logit = nn.Parameter(torch.zeros(a0.shape, device=a0.device))
        self.dtheta = (nn.Parameter(torch.zeros(grid.n - 1,
                                                device=a0.device))
                       if learn_theta else None)
        self._morph = False          # E34 態二走廊(enable_morph 開啟)
        self._dtheta_idx = None      # 收斂後 dθ 索引遮罩(9→7 門檻映射)
        self.full_ckpt = False       # 引擎壓榨:整前向 checkpoint(預設關=舊行為)
        # E63 S1:非對稱三元負尺度 r 逐 (row, block)(M,nB,1),buffer 非參數(STE 不學);
        # None=關(逐位還原對稱);由 comp 事件閉式 set_rneg 寫入;rneg_cb=每模組碼書(2^bits)
        self.register_buffer("rneg", None)
        self.rneg_cb = None
        # E67-B 率項:rate_bits (G,) 升冪格碼長(−log₂p);設定後每次前向記 last_rate = mean bits_interp(v_soft)
        self.rate_bits = None
        self.last_rate = None
        self.rate_s = 30.0           # hard 模式下算率用的軟階梯銳度

    def _rate_impl(self, v):
        """v (M,nB,B) 軟格值 → 分段線性內插碼長的均值(對 v 可微)。"""
        vals = self.grid.vals
        b = self.rate_bits
        j = torch.bucketize(v.detach(), vals).clamp(1, vals.numel() - 1)
        v0, v1 = vals[j - 1], vals[j]
        t = ((v - v0) / (v1 - v0)).clamp(0.0, 1.0)
        return (b[j - 1] + (b[j] - b[j - 1]) * t).mean()

    def _rate_of(self, v):
        """checkpoint 包裝:內插中間張量(每模組 2-3 份全尺寸 float)不留給 backward,
        09-09 煙測 OOM 定罪(84→94 GB)後改為 backward 重算。"""
        return checkpoint(self._rate_impl, v, use_reentrant=False)

    # ---- E34/E35 形變降元(態二走廊;morph.py 為數學載體) ----

    # (from, to) → 走廊終點 c 與收斂後 dθ 存活索引(值序間隙寬度推導:
    #   9→7 c→0.5:idx 2/5 間隙 (2c−1)→0 消失;
    #   7→3 c→0:間隙寬 [c,1−c,c,c,1−c,c],僅 idx 1/4 存活)
    _MORPH_STEPS = {(9, 7): {"c_tgt": 0.5, "dtheta_idx": [0, 1, 3, 4, 6, 7]},
                    (7, 3): {"c_tgt": 0.0, "dtheta_idx": [1, 4]}}

    def enable_morph(self, c_tgt=0.5, to=7):
        """c 由 corridor 排程滑向 c_tgt;m≥1 時碰撞收斂切換檔位。

        9→7(E34):±(1−c) 撞 ±c。7→3(E35):±c 併 0、±(1+c) 併 ±1,
        c 收歸 α(態三)。7元 layout 值序在 (0,1) 全域穩定,無換位點。
        """
        key = (self.grid.n, to)
        assert key in self._MORPH_STEPS, f"morph 不支援 {key[0]}→{key[1]}"
        assert abs(float(c_tgt) - self._MORPH_STEPS[key]["c_tgt"]) < 1e-12
        self._morph = True
        self._morph_from = self.grid.n
        self._morph_to = to
        self._c_tgt = float(c_tgt)
        self._c_free = float(self.c)
        self._m = 0.0
        self.logit_c = nn.Parameter(torch.zeros((), device=self.a0.device))

    def set_m(self, m):
        """設走廊排程 m∈[0,1];已收斂參數為 no-op(窗重疊天然處理)。"""
        if not self._morph:
            return
        self._m = float(m)
        if self._m >= 1.0:
            self._collapse()

    def _collapse(self):
        """合併點:值集收斂、c 夾到終點;dθ 走索引遮罩(優化器不重建)。

        存活索引見 _MORPH_STEPS 註記。E35 量測義務②:收斂時存 logit_c
        終值(修 E34 量測缺口——各模組衝/滯證據)。
        """
        self._morph = False
        self.logit_c_final = float(self.logit_c.detach())
        self.c = self._c_tgt
        self.grid = SoftGrid(c=self._c_tgt, device=self.a0.device,
                             n=self._morph_to)
        if self.dtheta is not None:
            idx = self._MORPH_STEPS[(self._morph_from, self._morph_to)][
                "dtheta_idx"]
            self._dtheta_idx = torch.tensor(idx, device=self.dtheta.device)

    def _c_now(self):
        return corridor(self._c_free, self._c_tgt, self.logit_c, self._m)

    def theta_eff(self):
        if self.dtheta is None:
            return self.grid.theta
        d = (self.dtheta if self._dtheta_idx is None
             else self.dtheta[self._dtheta_idx])
        return self.grid.theta + THETA_RANGE * self.grid.h * torch.tanh(d)

    def _lazy_init(self, W):
        M, K = W.shape
        self._shape = (M, K)
        A = torch.empty(self._r, K, device=W.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        self.lora_A = nn.Parameter(A)
        self.lora_B = nn.Parameter(torch.zeros(M, self._r, device=W.device,
                                               dtype=torch.float32))
        if self._use_comp:
            self.comp = torch.zeros(M, K, device=W.device,
                                    dtype=torch.float32)
        if self._mix_block > 0 and self.mix is None:
            b = self._mix_block
            assert K % b == 0, f"in_dim {K} 不整除 mix_block {b}"
            self.mix = nn.Parameter(
                torch.eye(b, device=W.device, dtype=torch.float32)
                .expand(K // b, b, b).clone())
        if self._lmix_block > 0 and self.lmix is None:
            b = self._lmix_block
            assert M % b == 0, f"out_dim {M} 不整除 lmix_block {b}"
            self.lmix = nn.Parameter(
                torch.eye(b, device=W.device, dtype=torch.float32)
                .expand(M // b, b, b).clone())

    # ---- E61 擴桶:塊混合 ----

    def apply_mix(self, W2d):
        """(M,K) → (M,K):W @ blockdiag(mix);mix=None 恆等(逐位)。"""
        if self.mix is None:
            return W2d
        Mo, K = W2d.shape
        k, b, _ = self.mix.shape
        out = torch.einsum("mkb,kbc->mkc", W2d.reshape(Mo, k, b).float(),
                           self.mix)
        return out.reshape(Mo, K)

    def mix_dense(self):
        """blockdiag(mix) 密集 (K,K) fp32(comp_event/材化用);None=無混合。"""
        if self.mix is None:
            return None
        return torch.block_diag(*self.mix.detach().float().unbind(0))

    def mix_dense_inv(self):
        if self.mix is None:
            return None
        return torch.block_diag(
            *torch.linalg.inv(self.mix.detach().float()).unbind(0))

    # ---- E62b 輸出側桶:列塊混合 ----

    def apply_lmix(self, W2d):
        """(M,K) → (M,K):blockdiag(lmix) @ W;lmix=None 恆等(逐位)。"""
        if self.lmix is None:
            return W2d
        Mo, K = W2d.shape
        k, b, _ = self.lmix.shape
        out = torch.einsum("kij,kjn->kin", self.lmix,
                           W2d.reshape(k, b, K).float())
        return out.reshape(Mo, K)

    def lmix_dense(self):
        if self.lmix is None:
            return None
        return torch.block_diag(*self.lmix.detach().float().unbind(0))

    def lmix_dense_inv(self):
        if self.lmix is None:
            return None
        return torch.block_diag(
            *torch.linalg.inv(self.lmix.detach().float()).unbind(0))

    @torch.no_grad()
    def load_reservoir(self, B, A, r_res, gamma=1.0):
        """E37:載入預訓水庫(蓄水畢即凍結;排水唯一自由度 = gamma)。"""
        self.res_B = B.detach().float().to(self.a0.device)
        self.res_A = A.detach().float().to(self.a0.device)
        self.res_B.requires_grad_(False)
        self.res_A.requires_grad_(False)
        self._r_res = r_res
        self.gamma = float(gamma)

    # ---- E63 S1:非對稱負尺度 ----

    @torch.no_grad()
    def set_rneg(self, r, cb=None):
        """r (M,nB,1) fp32 → buffer(首次註冊,之後 copy_);cb 碼書(K,)|None。"""
        r = r.detach().float().to(self.a0.device).reshape(self.a0.shape)
        if self.rneg is None:
            self.rneg = r.clone()          # 已註冊 None buffer → 賦值入 _buffers
        else:
            self.rneg.copy_(r)
        self.rneg_cb = None if cb is None else cb.detach().float().to(self.a0.device)

    def _apply_rneg(self, v):
        """v (M,nB,B) 碼值 → 負側乘 r(V = T⁺ − r·T⁻);rneg=None 逐位恆等。"""
        if self.rneg is None:
            return v
        return v * torch.where(v.detach() < 0, self.rneg, torch.ones_like(self.rneg))

    def _res_term(self):
        return self.gamma * (self.res_B @ self.res_A) / self._r_res

    def _u(self, W):
        M, K = W.shape
        w_lat = W.float() + (self.lora_B @ self.lora_A) / self._r
        if self.comp is not None:
            w_lat = w_lat + self.comp
        alpha = 2.0 * torch.sigmoid(self.logit) * self.a0        # (M,nB,1)
        nB = self.a0.shape[1]
        u = w_lat.reshape(M, nB, K // nB) / alpha
        return u, alpha

    def _hard_ste(self, u):
        if self._morph:                     # 走廊中途的硬投影(防禦路徑)
            cv = float(self._c_now().detach())
            vals, t1, t2 = grid_tensors(self._morph_from, cv, device=u.device)
            theta = (vals[1:] + vals[:-1]) / 2
            if self.dtheta is not None:
                h = vals[1:] - vals[:-1]
                theta = theta + THETA_RANGE * h * torch.tanh(
                    self.dtheta.detach())
            _, _, v = round_with_mids(u, theta, vals, t1, t2)
            return u + (v - u).detach()
        theta = self.theta_eff()
        _, _, v = round_with_mids(u, theta.detach(), self.grid.vals,
                                  self.grid.t1, self.grid.t2)
        return u + (v - u).detach()

    def _quant_forward(self, W):
        """量化前向全段(_u+階梯/STE+縮放),無內層 checkpoint——
        僅 full_ckpt 路徑用(整段一個 checkpoint,殘留圖 ~2 B/param)。"""
        M, K = W.shape
        u, alpha = self._u(W)
        if self.mode == "soft":
            if self._morph:
                cm = self._c_now()
                dts = None
                if self.dtheta is not None:
                    _, _, h, _ = grid_from_c(self._morph_from, cm)
                    dts = THETA_RANGE * h * torch.tanh(self.dtheta)
                v = soft_staircase_c(u, self.s, self._morph_from, cm, dts)
            else:
                v = soft_staircase(u, self.s, self.grid, self.theta_eff())
            rate = self._rate_impl(v) if (self.rate_bits is not None and not self._morph) else None
        else:
            v = self._hard_ste(u)
            rate = (self._rate_impl(soft_staircase(u, self.rate_s, self.grid, self.theta_eff()))
                    if (self.rate_bits is not None and not self._morph) else None)
        v = self._apply_rneg(v)
        out = self.apply_lmix(self.apply_mix((alpha * v).reshape(M, K))).to(W.dtype)
        # full_ckpt 整段重算:rate 必須作為輸出回傳才帶 grad_fn(在函式內存 self.last_rate 只會存到 no-grad 前向的值)
        return out, (rate if rate is not None else out.new_zeros(()))

    def forward(self, W):
        res = (self._res_term().to(W.dtype)
               if self.res_B is not None and self.gamma != 0.0 else None)
        if self.mode == "off":
            return W if res is None else W + res
        if self.lora_A is None:
            self._lazy_init(W)
        if self.full_ckpt and torch.is_grad_enabled():
            out, rate = checkpoint(self._quant_forward, W, use_reentrant=False)
            if self.rate_bits is not None and not self._morph:
                self.last_rate = rate
            return out if res is None else out + res
        M, K = W.shape
        u, alpha = self._u(W)
        if self.mode == "soft":
            if self._morph:                  # 態二:θ/h 由 c(m) 閉式生成
                cm = self._c_now()
                dts = None
                if self.dtheta is not None:
                    _, _, h, _ = grid_from_c(self._morph_from, cm)
                    dts = THETA_RANGE * h * torch.tanh(self.dtheta)
                v = checkpoint(soft_staircase_c, u, self.s, self._morph_from,
                               cm, dts, use_reentrant=False)
            else:
                v = checkpoint(soft_staircase, u, self.s, self.grid,
                               self.theta_eff(), use_reentrant=False)
            if self.rate_bits is not None and not self._morph:
                self.last_rate = self._rate_of(v)
        else:
            v = self._hard_ste(u)
            if self.rate_bits is not None and not self._morph and torch.is_grad_enabled():
                self.last_rate = self._rate_of(soft_staircase(u, self.rate_s, self.grid, self.theta_eff()))
        v = self._apply_rneg(v)
        out = self.apply_lmix(self.apply_mix((alpha * v).reshape(M, K))).to(W.dtype)
        return out if res is None else out + res

    @torch.no_grad()
    def extract(self, W):
        """→ (T1 int8 (M,nB,B), T2 int8, alpha fp32 (M,nB,1), w_hard (M,K) fp32)"""
        assert not self._morph, "extract 不得在走廊中途呼叫(先收斂 m=1)"
        assert self.res_B is None or self.gamma == 0.0, \
            "extract 前水庫須排空(gamma=0);殘水不入碼即丟失"
        if self.lora_A is None:
            self._lazy_init(W)
        u, alpha = self._u(W)
        t1, t2, v = round_with_mids(u, self.theta_eff(), self.grid.vals,
                                    self.grid.t1, self.grid.t2)
        w_hard = (alpha * self._apply_rneg(v)).reshape(self._shape)
        return t1.to(torch.int8), t2.to(torch.int8), alpha.clone(), w_hard


class AlphaOnlyParam(nn.Module):
    """E68 P1-alpha:碼 T1/T2 凍結、只學 α(logit)的 parametrize——weight = α ⊙ (T1 + c·T2)。
    與 STQParam 共用 e2e_soft 介面(mode/s/full_ckpt/rate_bits/last_rate/lora_A/lora_B/logit/dtheta/theta_eff/extract),
    翻碼恆 0、熵帳恆定(E67-B stage1 +9.76 pp 全在 α 軸的正式化)。W(模型原權重)不參與。"""

    def __init__(self, T1, T2, a0, grid: SoftGrid, c=C_K2):
        super().__init__()
        self.register_buffer("T1", T1.to(torch.int8))
        self.register_buffer("T2", T2.to(torch.int8))
        self.register_buffer("a0", a0.float())
        self.grid, self.c = grid, c
        self.logit = nn.Parameter(torch.zeros_like(self.a0))
        self.lora_A = nn.Parameter(torch.zeros(0, device=a0.device))      # 介面占位(lr 群組),無梯度
        self.lora_B = nn.Parameter(torch.zeros(0, device=a0.device))
        self.dtheta = nn.Parameter(torch.zeros(0, device=a0.device))
        self.mode, self.s, self.full_ckpt = "hard", 0.0, False
        self.rate_bits = self.last_rate = None
        self._morph, self.res_B, self.gamma = False, None, 0.0
        M, nB, B = self.T1.shape
        self._shape = (M, nB * B)
        self._V = None

    def theta_eff(self):
        return self.grid.theta

    def alpha(self):
        return 2.0 * torch.sigmoid(self.logit) * self.a0

    def forward(self, W):
        if self._V is None:
            self._V = (self.T1.float() + self.c * self.T2.float())
        return (self.alpha() * self._V).reshape(self._shape).to(W.dtype)

    @torch.no_grad()
    def extract(self, W):
        al = self.alpha().clone()
        V = self.T1.float() + self.c * self.T2.float()
        return self.T1.clone(), self.T2.clone(), al, (al * V).reshape(self._shape)

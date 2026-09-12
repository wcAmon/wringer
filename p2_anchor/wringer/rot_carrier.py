"""E60 S1:旋轉載體共用件(全局正交 Q 施於激活側;與 rot_learn v2 RotQ 語義逐位對齊)。

  讀者 y = W_q·(Qᵀx̂)   ← forward_pre_hook:x ← x @ Q
  寫者 r += Q·(W_q z)   ← forward_hook:   y ← y @ Qᵀ
碼 W_q 須在「折 γ 基底」解出(g3rot*/g9f:rot_fold 把 (1+γ) 折入讀者、norm γ←0);
本模組 surgery() 對原模型做同一手術(非目標讀者折 (1+γ)、norm 歸零、lm_head 解綁),
目標讀者權重另由 apply_k2 覆寫。殘差流留原基底,Q 不折入權重(共適應載體,可與碼一起退火)。
部署帳:三元碼 + Q(2560² bf16 = 6.5MB);評測用 export_rot 材化 dense。
"""
import torch
import torch.nn as nn

from p2_anchor.wringer.modelio import LAYER_PREFIX, N_LAYERS, get_module
from p2_anchor.wringer.rot_fold import (READERS_ATT, READERS_LIN,
                                          READERS_MLP, WRITERS)


def use_base_dir(d):
    """把 load_model 指向另一個 HF 目錄(如 exports/a1_fold_bf16);None=釘定原模型。"""
    if d:
        import p1_grouping.modelio as mio
        mio.MODEL_ID = str(d)
        mio.MODEL_REVISION = None


def cayley(P):
    A = P - P.transpose(-1, -2)
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return torch.linalg.solve(I + A, I - A)


def is_writer(name):
    return name.endswith(WRITERS)


class RotCarrier(nn.Module):
    """Q = Q0·Cayley(P);learn=False 時 Q≡Q0(凍結旋轉臂)。每步 refresh() 一次入圖。"""

    def __init__(self, Q0, learn=True):
        super().__init__()
        self.register_buffer("Q0", Q0.detach().float().clone())
        self.P = nn.Parameter(torch.zeros_like(self.Q0), requires_grad=learn)
        self.learn = learn
        self._Q = None

    def refresh(self):
        if self.learn:
            self._Q = (self.Q0 @ cayley(self.P)).to(torch.bfloat16)
        elif self._Q is None:
            self._Q = self.Q0.to(torch.bfloat16)
        return self._Q

    @property
    def Q(self):
        if self._Q is None:
            self.refresh()
        return self._Q

    @torch.no_grad()
    def Q_float(self):
        return (self.Q0 @ cayley(self.P)).float() if self.learn else self.Q0.float()

    @torch.no_grad()
    def angle_rms(self):
        """Q0ᵀQ 主角 RMS(rad);凍結時 0。"""
        ev = torch.linalg.eigvals((self.Q0.T @ self.Q_float()).double())
        ang = torch.atan2(ev.imag, ev.real).abs()
        return float((ang ** 2).mean().sqrt())


def attach(model, targets, carrier):
    """掛旋轉 hook;回傳 handles。單槽記憶:同一子層多個讀者共用同一輸入張量 → 只旋轉一次
    (讀者呼叫在子層內連續;單槽不需清空,與 fwd_chain 直呼層相容)。"""
    slot = [None, None]

    def pre(mod, args):
        x = args[0]
        if slot[0] is x:
            return (slot[1],) + tuple(args[1:])
        y = x @ carrier.Q
        slot[0], slot[1] = x, y
        return (y,) + tuple(args[1:])

    def post(mod, args, out):
        return out @ carrier.Q.T

    hs = []
    for n in targets:
        m = get_module(model, n)
        hs.append(m.register_forward_hook(post) if is_writer(n)
                  else m.register_forward_pre_hook(pre))
    return hs


@torch.no_grad()
def surgery(model, targets):
    """原模型 → 折 γ 基底:非目標讀者折 (1+γ),norm γ←0,lm_head 解綁 E·diag(1+γ_f)。
    目標讀者不動(由 apply_k2 覆寫為折 γ 基底解出的碼)。"""
    lm = model.model.language_model
    tset = set(targets)
    n_fold = 0
    for li in range(N_LAYERS):
        layer = lm.layers[li]
        names = READERS_LIN if hasattr(layer, "linear_attn") else READERS_ATT
        for grp, gname in ((names, "input_layernorm"),
                           (READERS_MLP, "post_attention_layernorm")):
            g = getattr(layer, gname).weight.float()
            for n in grp:
                full = f"{LAYER_PREFIX}.{li}.{n}"
                if full in tset:
                    continue
                m = model.get_submodule(full)
                m.weight.data.copy_((m.weight.float() * (1.0 + g)[None, :]).to(m.weight.dtype))
                n_fold += 1
            getattr(layer, gname).weight.data.zero_()
    E0 = lm.embed_tokens.weight.float()
    gf = lm.norm.weight.float()
    H = E0.shape[1]
    head = nn.Linear(H, E0.shape[0], bias=False, device=E0.device, dtype=torch.bfloat16)
    head.weight.data.copy_((E0 * (1.0 + gf)[None, :]).to(torch.bfloat16))
    head.weight.requires_grad_(False)
    model.lm_head = head
    lm.norm.weight.data.zero_()
    model.config.tie_word_embeddings = False
    model.config.text_config.tie_word_embeddings = False
    return n_fold


def load_Q(path):
    d = torch.load(path, weights_only=False)
    return d["Q"].cuda().float(), d

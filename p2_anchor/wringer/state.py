"""qs 狀態(T1/T2/inv/α/c/grid 逐層 .pt)讀寫與套用 — guava qs_ 家族同格式。

.pt 不進 git;sha256 與超參記於 evidence JSON。
"""
from pathlib import Path

import torch

from p2_anchor.wringer.ktier import k2_weight
from p2_anchor.wringer.modelio import N_LAYERS, get_module

DATA = Path("data")


def qs_dir(tag):
    return DATA / f"qs_{tag}"


def load_state(tag, names):
    st = {}
    for li in range(N_LAYERS):
        p = qs_dir(tag) / f"layer{li:02d}.pt"
        if p.exists():
            st.update(torch.load(p, map_location="cpu", weights_only=False))
    missing = [n for n in names if n not in st]
    assert not missing, f"qs_{tag} 缺 {missing[:3]}"
    return st


def mix_dense(mix):
    """E61:(k,b,b) → blockdiag (K,K) fp32(模型欄序,inv 之後右乘)。"""
    return torch.block_diag(*mix.float().unbind(0))


def dense_weight(c, a=None):
    """qs 條目 → 部署形態 (M,K) fp32 cuda:k2_weight[:, inv](@ blockdiag(mix))。

    E61 之前的條目無 'mix' 鍵,逐位還原舊行為。"""
    a = (c["a0"] if a is None else torch.as_tensor(a)).cuda().float()
    if c.get("t2beta") is not None:          # E63 T2:殘差位面 w = α·T1 (負側×r) + β·T2,β 逐 block (M,nB)
        Mo, nB, B = c["T1"].shape
        T1 = c["T1"].cuda().float()
        w = a * T1
        if c.get("rneg") is not None:
            r = c["rneg"].cuda().float().reshape(Mo, -1, 1).expand(Mo, nB, B)
            w = torch.where(T1 < 0, w * r, w)
        w = (w + c["t2beta"].cuda().float().reshape(Mo, nB, 1) * c["T2"].cuda().float()
             ).reshape(Mo, nB * B)
    else:
        w = k2_weight(c["T1"].cuda(), c["T2"].cuda(), a, c["c"])
        if c.get("rneg") is not None:        # E63:非對稱三元,負尺度 r 逐列 (M,1) 或逐 block (M,nB)(儲存欄序)
            Mo, nB, B = c["T1"].shape
            r = c["rneg"].cuda().float().reshape(Mo, -1, 1).expand(Mo, nB, B).reshape(Mo, nB * B)
            neg = (c["T1"].cuda() < 0).reshape(Mo, -1)
            w = torch.where(neg, w * r, w)
    w = w[:, c["inv"].cuda()]
    if c.get("mix") is not None:
        w = w @ mix_dense(c["mix"].cuda())
    if c.get("lmix") is not None:            # E62b:輸出側列塊 L 左乘
        w = mix_dense(c["lmix"].cuda()) @ w
    return w


def apply_k2(model, st, alphas=None):
    """把 qs 狀態材化回模型權重(fake-quant bf16)。alphas 覆寫 a0(npz dict)。
    E61:條目含 mix 時右乘 blockdiag(mix)(稠密材化,對 vLLM 精確等價)。"""
    for n, c in st.items():
        m = get_module(model, n)
        w = dense_weight(c, None if alphas is None else alphas[n])
        m.weight.data.copy_(w.to(m.weight.dtype))

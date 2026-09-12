"""Corkscrew 模型接口(circus/CUDA 端)。

load_model 沿用 p1_grouping.modelio(釘定 revision);target 枚舉與 Σ 來源
代表移植自 guava-qat modelio(spec §2.1 凍結,同一模型同一 200 linears)。
"""
from p1_grouping.modelio import load_model  # noqa: F401(re-export)

N_LAYERS = 32
LAYER_PREFIX = "model.language_model.layers"

# 排除 in_proj_b / in_proj_a(<0.3%,decay/gate 敏感參數)
TARGET_SUFFIXES = (
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)

# Σ 來源代表:同一輸入點的 targets 共用一份 Σ
_COV_REP = {
    "linear_attn.in_proj_z": "linear_attn.in_proj_qkv",
    "self_attn.k_proj": "self_attn.q_proj",
    "self_attn.v_proj": "self_attn.q_proj",
    "mlp.up_proj": "mlp.gate_proj",
}


def get_module(model, name):
    return model.get_submodule(name)


def set_module(model, name, new):
    parts = name.split(".")
    parent = model.get_submodule(".".join(parts[:-1]))
    setattr(parent, parts[-1], new)


def layer_index(name):
    return int(name.split(f"{LAYER_PREFIX}.")[1].split(".")[0])


def enumerate_targets(model):
    """依序枚舉 target linears;順序凍結(層序 × TARGET_SUFFIXES 序)。"""
    out = []
    for li in range(N_LAYERS):
        for suf in TARGET_SUFFIXES:
            name = f"{LAYER_PREFIX}.{li}.{suf}"
            try:
                m = model.get_submodule(name)
            except AttributeError:
                continue
            if m.__class__.__name__ == "Linear":
                out.append(name)
    return out


def cov_source_for(name):
    li = layer_index(name)
    suf = ".".join(name.split(".")[-2:])
    rep = _COV_REP.get(suf, suf)
    return f"{LAYER_PREFIX}.{li}.{rep}"

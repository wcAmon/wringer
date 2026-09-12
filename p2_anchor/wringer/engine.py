"""壓榨引擎:入口激活快取 + 逐窗直呼 + 等價斷言閘(#24 訓練側)。

語義:X 在第 0 層入口快取一次(kwargs 同批共用——calib 定長無 padding,
position/mask 全批同構);視窗訓練只前推視窗內 4 層;定案層把 X 前進。
相對「每步從第 0 層全前推」平均省 ~18/32 層(~2.75×,晚期窗 ~8×)。

等價閘(鐵律驗收):快取鏈 vs 全模型 forward 的同層出口,rel diff ≤ 1e-6。
"""
import torch

from p2_anchor.wringer.modelio import LAYER_PREFIX


class _Stop(Exception):
    pass


@torch.no_grad()
def capture_layer0(model, calib, batch=8, to_cpu=False):
    """回傳 (X list[(B,S,H) bf16], kwargs dict)。to_cpu=X 駐 CPU 逐批搬運
    (E58R1b OOM 修:384 列 16k 下 X+Y 同駐 GPU 爆 95G;fwd_chain 自動上搬)。"""
    feats, cap = [], {}

    def pre(mod, args, kwargs):
        feats.append(args[0].detach().cpu() if to_cpu else args[0].detach())
        cap["kwargs"] = kwargs
        raise _Stop

    h = model.get_submodule(f"{LAYER_PREFIX}.0").register_forward_pre_hook(
        pre, with_kwargs=True)
    try:
        for i in range(0, calib.shape[0], batch):
            try:
                model(input_ids=calib[i:i + batch].cuda(), use_cache=False)
            except _Stop:
                pass
    finally:
        h.remove()
    return feats, cap["kwargs"]


def layer_fwd(layer, x, kw):
    out = layer(x, **kw)
    return out[0] if isinstance(out, tuple) else out


def fwd_chain(layers, x, kw):
    if x.device.type == "cpu":                # X 駐 CPU 時逐批上搬
        x = x.cuda(non_blocking=True)
    for ly in layers:
        x = layer_fwd(ly, x, kw)
    return x


@torch.no_grad()
def run_layers(layers, X, kw, to_cpu=False):
    if to_cpu:            # 就地逐批置換:避免新舊兩份 X 同駐 RAM(60G 機)
        for i in range(len(X)):
            X[i] = fwd_chain(layers, X[i], kw).cpu()
        return X
    return [fwd_chain(layers, x, kw) for x in X]


@torch.no_grad()
def capture_exit(model, layer_name, batch):
    """全模型 forward 到 layer 出口即停(等價閘的參照路徑)。"""
    box = {}

    def hook(mod, args, out):
        box["h"] = out[0] if isinstance(out, tuple) else out
        raise _Stop

    h = model.get_submodule(layer_name).register_forward_hook(hook)
    try:
        model(input_ids=batch, use_cache=False)
    except _Stop:
        pass
    finally:
        h.remove()
    return box["h"]


@torch.no_grad()
def equivalence_gate(model, layers, X, kw, calib, batch=8, upto=4, tol=1e-6):
    """斷言:X[0] 過快取鏈 vs 全 forward,layer[upto-1] 出口 rel diff ≤ tol。

    引擎變更的驗收閘;FAIL 直接 raise(不得靜默降級)。回傳實測 rel diff。
    """
    ref = capture_exit(model, f"{LAYER_PREFIX}.{upto - 1}",
                       calib[:batch].cuda()).float()
    got = fwd_chain(layers[:upto], X[0], kw).float()
    rel = float((got - ref).norm() / ref.norm().clamp_min(1e-12))
    if not (rel <= tol):
        raise RuntimeError(f"等價閘 FAIL:rel diff {rel:.3e} > {tol:.0e}")
    return rel

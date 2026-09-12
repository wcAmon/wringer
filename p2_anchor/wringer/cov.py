"""Σ = E[xxᵀ] 收集 — 順序傳播語義:呼叫時模型的前置層已量化。

移植自 guava-qat src/gqat/cov.py。hook 收集第 li 層各 Σ 來源模組的輸入;
該層 forward 完成即截斷(不必跑完整模型)。
"""
import torch


class _StopForward(Exception):
    pass


@torch.no_grad()
def collect_cov(model, rep_names, layer_name, calib, batch=8, device="cuda"):
    """回傳 {rep_name: Σ (K,K) fp32 cuda}。calib: (N, S) int64。"""
    acc = {}
    counts = {}
    hooks = []

    def make_pre_hook(rn):
        def pre_hook(module, args, kwargs=None):
            x = args[0] if args else kwargs["input"]
            x = x.reshape(-1, x.shape[-1]).float()
            if rn not in acc:
                acc[rn] = torch.zeros(x.shape[-1], x.shape[-1],
                                      dtype=torch.float32, device=device)
                counts[rn] = 0
            acc[rn].addmm_(x.T, x)
            counts[rn] += x.shape[0]
        return pre_hook

    for rn in rep_names:
        hooks.append(model.get_submodule(rn).register_forward_pre_hook(
            make_pre_hook(rn)))

    def stop_hook(module, args, output):
        raise _StopForward

    hooks.append(model.get_submodule(layer_name).register_forward_hook(stop_hook))

    try:
        for i in range(0, calib.shape[0], batch):
            b = calib[i:i + batch].to(device)
            try:
                model(input_ids=b, use_cache=False)
            except _StopForward:
                pass
    finally:
        for h in hooks:
            h.remove()

    return {rn: acc[rn] / counts[rn] for rn in acc}

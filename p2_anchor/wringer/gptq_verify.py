"""2c 版面驗證(CPU):GPTQ checkpoint 解包 → W = (u - 2^(b-1))·scale,對 bf16 材化 export 逐元素比。
    python -m p2_anchor.wringer.gptq_verify --gptq exports/e69_p3b_w2_gptq4 --bf16 exports/e69_p3b_w2_pack
印每模組摘要與總表:bf16 逐位相同比例、max|Δ|/max|W|。印 GPTQ_VERIFY_DONE <json>。
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]


def unpack_stream(p, bits, dim, n):
    """pack_stream 的逆:int32 沿 dim 的連續 LSB-first 位元流 → n 個 [0,2^bits) 值。"""
    p = np.moveaxis(p, dim, -1).astype(np.int64) & 0xFFFFFFFF
    per = 32 // math.gcd(32, bits)
    words = per * bits // 32
    g = p.reshape(*p.shape[:-1], p.shape[-1] // words, words)
    out = np.zeros((*g.shape[:-1], per), dtype=np.int64)
    mask = (1 << bits) - 1
    for i in range(per):
        w, off = divmod(i * bits, 32)
        v = g[..., w] >> off
        if off + bits > 32:
            v |= g[..., w + 1] << (32 - off)
        out[..., i] = v & mask
    out = out.reshape(*p.shape[:-1], -1)[..., :n]
    return np.moveaxis(out, -1, dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gptq", required=True)
    ap.add_argument("--bf16", required=True)
    args = ap.parse_args()
    g = safe_open(str(ROOT / args.gptq / "model.safetensors"), "np")
    cfg = json.load(open(ROOT / args.gptq / "config.json"))["quantization_config"]
    leaf_bits = cfg["wringer"]["per_module_bits"]
    b16 = safe_open(str(ROOT / args.bf16 / "model.safetensors"), "pt")
    mods = sorted({k[: -len(".qweight")] for k in g.keys() if k.endswith(".qweight")})
    tot = same = 0
    worst = 0.0
    per_leaf = {}
    for m in mods:
        bits = leaf_bits[m.rsplit(".", 1)[-1]]
        qw = g.get_tensor(f"{m}.qweight"); sc = g.get_tensor(f"{m}.scales").astype(np.float32)   # (K*bits/32, N), (K/128, N)
        K = sc.shape[0] * 128
        u = unpack_stream(qw, bits, 0, K)                                                          # (K, N)
        w = ((u - (1 << (bits - 1))).astype(np.float32).reshape(K // 128, 128, -1) * sc[:, None, :]).reshape(K, -1)
        w = torch.from_numpy(w).to(torch.bfloat16).T.contiguous()                                  # (N, K) = HF (out, in)
        ref = b16.get_tensor(f"{m}.weight")
        assert ref.shape == w.shape, (m, ref.shape, w.shape)
        eq = (ref == w).sum().item(); n = ref.numel()
        rel = ((ref.float() - w.float()).abs().max() / ref.float().abs().max()).item()
        tot += n; same += eq; worst = max(worst, rel)
        pl = per_leaf.setdefault(m.rsplit(".", 1)[-1], [0, 0, 0.0]); pl[0] += n; pl[1] += eq; pl[2] = max(pl[2], rel)
    rep = {"gptq": args.gptq, "modules": len(mods), "weights": tot, "bf16_identical_frac": same / tot, "max_rel_dev": worst,
           "per_leaf": {k: {"identical_frac": round(v[1] / v[0], 6), "max_rel_dev": v[2]} for k, v in per_leaf.items()}}
    print(json.dumps(rep, indent=1))
    print("GPTQ_VERIFY_DONE", json.dumps({"identical_frac": round(same / tot, 6), "max_rel_dev": worst}), flush=True)


if __name__ == "__main__":
    main()

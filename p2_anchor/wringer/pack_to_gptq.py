"""2c:Wringer 容器 → GPTQ 版面 checkpoint(vLLM 可載),權重數值與容器材化完全相同(升位保值)。

    python -m p2_anchor.wringer.pack_to_gptq --container hub_stage/wringer-p3b_w2/wringer_p3b_w2.safetensors \
        --src-export hub_stage/wringer-p3b_w2/bf16 --out exports/e69_p3b_w2_gptq4 --mode marlin4

模式(融合層內各分片須同位元,低位元碼原樣放進高位元格 = 數值不變):
  marlin4  2/3/4-bit → 4-bit,8-bit 留 8(q_proj 3→8 配 k/v);原生 vLLM 0.26 Marlin 直接跑
  native   in_proj_qkv/z/out_proj 2、gate 2→3(配 up 3)、down 3、q 3→8(配 k/v 8)、o 4;需 vLLM 2/3-bit 補丁
群大小一律 128;k/v(B=2560 逐列 α)複製到每個 g128 群。
GPTQ 版面:qweight int32 (K·bits/32, N) 連續 LSB-first 位元流、qzeros int32 (K/128, N·bits/32) 填 2^(b-1)-1(v1 慣例)、
scales fp16 (K/128, N)、g_idx int32 (K)。輸出 model.safetensors + config.json(含 quantization_config)+ quantize_config.json + tokenizer 等。
印 GPTQ_PACK_DONE <out> <bytes>。
"""
import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from p2_anchor.wringer.pack_verify import unpack_module

ROOT = Path(__file__).resolve().parents[2]
G = 128
COPY_FILES = ["chat_template.jinja", "generation_config.json", "preprocessor_config.json", "processor_config.json",
              "tokenizer_config.json", "tokenizer.json", "wringer_materialize.json"]


def target_bits(name, bits, mode):
    leaf = name.rsplit(".", 1)[-1]
    if mode == "marlin4":
        return 8 if (bits == 8 or leaf == "q_proj") else 4
    if mode == "native":
        return {"gate_proj": 3, "q_proj": 8}.get(leaf, bits)
    raise ValueError(mode)


def pack_stream(v, bits, dim):
    """沿 dim 把 [0,2^bits) 整數打成連續 LSB-first 位元流 int32(AutoGPTQ/exllama/Marlin 載入版面)。"""
    v = np.moveaxis(v, dim, -1).astype(np.int64)
    n = v.shape[-1]
    assert (n * bits) % 32 == 0, (n, bits)
    per = 32 // math.gcd(32, bits)
    words = per * bits // 32
    g = v.reshape(*v.shape[:-1], n // per, per)
    acc = np.zeros((*g.shape[:-1], words), dtype=np.int64)
    for i in range(per):
        w, off = divmod(i * bits, 32)
        acc[..., w] |= (g[..., i] << off) & 0xFFFFFFFF
        if off + bits > 32:
            acc[..., w + 1] |= g[..., i] >> (32 - off)
    out = acc.reshape(*v.shape[:-1], n // per * words)
    out = np.where(out >= 2**31, out - 2**32, out).astype(np.int32)
    return np.ascontiguousarray(np.moveaxis(out, -1, dim))


def convert_module(name, t, meta, mode):
    T1, a, inv = unpack_module(name, t, meta)          # T1 (M,nB,B) int8;a (M,nB,1) fp32(帳面精度)
    assert inv is None
    M, nB, B = T1.shape
    K = nB * B
    tb = target_bits(name, meta["bits"], mode)
    assert tb >= meta["bits"]
    u = (T1.astype(np.int16) + (1 << (tb - 1))).astype(np.int32).reshape(M, K).T      # (K, M) 無號碼
    assert u.min() >= 0 and u.max() < (1 << tb)
    sc = a.reshape(M, nB).T.astype(np.float32)                                          # (nB, M)
    if B != G:
        assert B % G == 0
        sc = np.repeat(sc, B // G, axis=0)                                              # 逐列 α 複製到每 g128 群
    sc16 = np.ascontiguousarray(sc.astype(np.float16))
    dev = float(np.abs(sc16.astype(np.float32) - sc).max() / np.abs(sc).max())          # α fp16 化偏差
    qweight = pack_stream(u, tb, 0)
    qzeros = pack_stream(np.full((K // G, M), (1 << (tb - 1)) - 1, dtype=np.int32), tb, 1)
    g_idx = np.ascontiguousarray(np.arange(K, dtype=np.int32) // G)
    out = {f"{name}.qweight": torch.from_numpy(qweight), f"{name}.qzeros": torch.from_numpy(qzeros),
           f"{name}.scales": torch.from_numpy(sc16), f"{name}.g_idx": torch.from_numpy(g_idx)}
    return out, tb, dev, M * K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--src-export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["marlin4", "native"], default="marlin4")
    args = ap.parse_args()
    t0 = time.time()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    f = safe_open(str(ROOT / args.container), "np")
    metas = json.loads(f.metadata()["wringer_meta"])
    t = {k: f.get_tensor(k) for k in f.keys()}
    tensors, per_bits, devs, wsum = {}, {}, [], 0
    bits_total = 0
    for name, meta in metas.items():
        o, tb, dev, n = convert_module(name, t, meta, args.mode)
        tensors.update(o)
        per_bits[name] = tb
        devs.append(dev)
        wsum += n
        bits_total += n * tb + 16 * (n // G)
    src = safe_open(str(ROOT / args.src_export / "model.safetensors"), "pt")
    skip = {f"{n}.weight" for n in metas}
    copied = 0
    for k in src.keys():
        if k in skip:
            continue
        tensors[k] = src.get_tensor(k)
        copied += 1
    save_file(tensors, str(out_dir / "model.safetensors"), metadata={"format": "pt", "wringer_gptq_mode": args.mode})
    # 量化設定:base bits + 顯式模組清單 + dynamic 規則(vLLM prefix regex;負規則先)
    base = 4 if args.mode == "marlin4" else 2
    dynamic = {"-:.*visual\\..*": {}, "-:.*in_proj_ba.*": {}, "-:.*mtp\\..*": {}}
    if args.mode == "marlin4":
        dynamic["+:.*self_attn\\.qkv_proj"] = {"bits": 8}
    else:
        dynamic.update({"+:.*mlp\\.gate_up_proj": {"bits": 3}, "+:.*mlp\\.down_proj": {"bits": 3},
                        "+:.*self_attn\\.qkv_proj": {"bits": 8}, "+:.*self_attn\\.o_proj": {"bits": 4}})
    qcfg = {"quant_method": "gptq", "bits": base, "group_size": G, "desc_act": False, "sym": True, "true_sequential": True,
            "checkpoint_format": "gptq", "lm_head": False, "dynamic": dynamic,
            "modules_in_block_to_quantize": ["linear_attn.in_proj", "linear_attn.out_proj", "mlp.", "self_attn."],
            "wringer": {"mode": args.mode, "source": args.container, "per_module_bits": {k.rsplit(".", 1)[-1]: v for k, v in per_bits.items()}}}
    cfg = json.load(open(ROOT / args.src_export / "config.json"))
    cfg["quantization_config"] = qcfg
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=1)
    json.dump(qcfg, open(out_dir / "quantize_config.json", "w"), indent=1)
    for fn in COPY_FILES:
        p = ROOT / args.src_export / fn
        if p.exists():
            shutil.copy(p, out_dir / fn)
    nbytes = (out_dir / "model.safetensors").stat().st_size
    rep = {"mode": args.mode, "modules": len(metas), "copied_tensors": copied, "quantized_weights": wsum,
           "bpw_body_gptq": bits_total / wsum, "alpha_fp16_max_rel_dev": max(devs), "bytes": nbytes,
           "per_leaf_bits": sorted({(k.rsplit(".", 1)[-1], v) for k, v in per_bits.items()}), "seconds": round(time.time() - t0, 1)}
    json.dump(rep, open(out_dir / "wringer_gptq.json", "w"), indent=1)
    print(json.dumps(rep, ensure_ascii=False))
    print(f"GPTQ_PACK_DONE {out_dir} {nbytes}", flush=True)


if __name__ == "__main__":
    main()

"""Standalone unpacker for a Wringer container (no dependency on the research repo).

    python wringer_unpack.py --container wringer_p3b_w2.safetensors --src-export <bf16 export dir> --out <dir>

Container layout (per quantized module `<name>`):
  <name>.code     uint8 bit-stream, log2(grid) bits per weight, LSB-first, little-endian packbits
  <name>.alpha    fp16 (M, nB)                    when a0_bits == 16
  <name>.alpha_q  int8 (M, nB) + <name>.alpha_s fp16 (M,)   when a0_bits == 8   (alpha = q * s)
  <name>.inv      int32 column permutation (absent when identity)
Metadata `wringer_meta` (JSON) holds per-module {M, nB, B, grid, bits, a0_bits, n}.
Weight = (code - grid/2) * alpha[block], then bf16. Every other tensor (embeddings, norms, lm_head,
vision tower) is copied from the bf16 source export, which is why --src-export is required.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def unpack_bits(buf, b, n):
    bits = np.unpackbits(buf, bitorder="little")[: n * b].reshape(n, b)
    return (bits.astype(np.uint16) << np.arange(b, dtype=np.uint16)).sum(1).astype(np.uint8)


def module_weight(cont, name, meta):
    M, nB, B, g, b, n = (meta[k] for k in ("M", "nB", "B", "grid", "bits", "n"))
    u = unpack_bits(cont.get_tensor(f"{name}.code"), b, n)
    T = (u.astype(np.int16) - g // 2).astype(np.float32).reshape(M, nB, B)
    if meta["a0_bits"] == 16:
        a = cont.get_tensor(f"{name}.alpha").astype(np.float32)
    else:
        a = cont.get_tensor(f"{name}.alpha_q").astype(np.float32) * cont.get_tensor(f"{name}.alpha_s").astype(np.float32)[:, None]
    w = torch.from_numpy(T * a.reshape(M, nB, 1)).reshape(M, nB * B)
    if f"{name}.inv" in cont.keys():
        w = w[:, torch.from_numpy(cont.get_tensor(f"{name}.inv")).long()]
    return w.to(torch.bfloat16).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--src-export", required=True, help="bf16 HF export dir providing non-quantized tensors and config")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src, out = Path(args.src_export), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name != "model.safetensors":
            shutil.copy2(f, out / f.name)
    cont = safe_open(args.container, "np")
    metas = json.loads(cont.metadata()["wringer_meta"])
    tensors = {}
    with safe_open(str(src / "model.safetensors"), "pt") as exp:
        for k in exp.keys():
            name = k[: -len(".weight")] if k.endswith(".weight") else None
            tensors[k] = module_weight(cont, name, metas[name]) if name in metas else exp.get_tensor(k).contiguous()
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    print(f"wrote {out/'model.safetensors'} ({len(metas)} modules materialized, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()

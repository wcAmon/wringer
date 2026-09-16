"""2c:讓 vLLM 0.26 的 GPTQ 路徑收 2/3-bit(對稱格)——核心 gptq_gemm/gptq_shuffle 本就支援,只補 Python 層。
    .venv-vllm/bin/python -m p2_anchor.wringer.vllm_patch_23bit apply|revert|status
改動(各檔先備份 *.orig_wringer):
  auto_gptq.py      TYPE_MAP 加 (2,True)/(3,True);pack_factor 對 3-bit 用 Fraction(32,3);LinearMethod 對 <4-bit 跳過 verify_marlin_supported
  gptq_utils.py     override_config 同上 pack_factor
  exllama.py        SUPPORTED_QUANT_TYPES 加 uint2b2/uint3b4;輸出維檢查改 (N·bits)%32;零點打包改連續位元流(3-bit 32 值→3 字)
"""
import shutil
import sys
from pathlib import Path

import vllm

V = Path(vllm.__file__).parent
FILES = {
    "auto_gptq": V / "model_executor/layers/quantization/auto_gptq.py",
    "gptq_utils": V / "model_executor/layers/quantization/utils/gptq_utils.py",
    "exllama": V / "model_executor/kernels/linear/mixed_precision/exllama.py",
}
PACK_FN = '''

def _wringer_pack_stream(v, bits, dim):
    """連續 LSB-first 位元流打包(AutoGPTQ/exllama 版面;3-bit 為 32 值→3 int32)。"""
    import math
    v = v.movedim(dim, -1).to(torch.int64)
    n = v.shape[-1]
    assert (n * bits) % 32 == 0, (n, bits)
    per = 32 // math.gcd(32, bits)
    words = per * bits // 32
    g = v.reshape(*v.shape[:-1], n // per, per)
    acc = torch.zeros(*g.shape[:-1], words, dtype=torch.int64, device=v.device)
    for i in range(per):
        w, off = divmod(i * bits, 32)
        acc[..., w] |= (g[..., i] << off) & 0xFFFFFFFF
        if off + bits > 32:
            acc[..., w + 1] |= g[..., i] >> (32 - off)
    out = acc.reshape(*v.shape[:-1], n // per * words)
    out = torch.where(out >= 2**31, out - 2**32, out).to(torch.int32)
    return out.movedim(-1, dim).contiguous()
'''
EDITS = {
    "auto_gptq": [
        ("        (4, True): scalar_types.uint4b8,\n        (8, True): scalar_types.uint8b128,\n    }",
         "        (2, True): scalar_types.uint2b2,   # wringer patch\n        (3, True): scalar_types.uint3b4,   # wringer patch\n        (4, True): scalar_types.uint4b8,\n        (8, True): scalar_types.uint8b128,\n    }"),
        ("        # Verify supported on platform.\n        verify_marlin_supported(\n            quant_type=self.quant_config.quant_type,\n            group_size=self.quant_config.group_size,\n        )",
         "        # wringer patch: 2/3-bit 走 exllama,跳過 Marlin 檢查\n        if self.quant_config.quant_type.size_bits >= 4:\n            verify_marlin_supported(\n                quant_type=self.quant_config.quant_type,\n                group_size=self.quant_config.group_size,\n            )"),
        ("        self.pack_factor = 32 // weight_bits  # packed into int32",
         "        from fractions import Fraction  # wringer patch\n        self.pack_factor = 32 // weight_bits if 32 % weight_bits == 0 else Fraction(32, weight_bits)  # wringer patch"),
    ],
    "gptq_utils": [
        ("    config.pack_factor = 32 // config.weight_bits  # packed into int32",
         "    from fractions import Fraction  # wringer patch\n    config.pack_factor = 32 // config.weight_bits if 32 % config.weight_bits == 0 else Fraction(32, config.weight_bits)  # wringer patch"),
    ],
    "exllama": [
        ("    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint8b128]",
         "    SUPPORTED_QUANT_TYPES = [scalar_types.uint2b2, scalar_types.uint3b4, scalar_types.uint4b8, scalar_types.uint8b128]  # wringer patch"),
        ("        if c.partition_weight_shape[1] % (32 // c.weight_type.size_bits) != 0:",
         "        if (c.partition_weight_shape[1] * c.weight_type.size_bits) % 32 != 0:  # wringer patch"),
        ("            zeros = pack_quantized_values_into_int32(zeros, c.weight_type, packed_dim=1)",
         "            zeros = _wringer_pack_stream(zeros, c.weight_type.size_bits, 1)  # wringer patch"),
        ("from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig\n",
         "from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig\n" + PACK_FN),
    ],
}


def status():
    return {k: ("patched" if "wringer patch" in p.read_text() else "clean") for k, p in FILES.items()}


def apply():
    for k, p in FILES.items():
        s = p.read_text()
        if "wringer patch" in s:
            continue
        bak = p.with_suffix(p.suffix + ".orig_wringer")
        if not bak.exists():
            shutil.copy(p, bak)
        for old, new in EDITS[k]:
            assert s.count(old) == 1, (k, old[:60], s.count(old))
            s = s.replace(old, new)
        p.write_text(s)
    for p in FILES.values():   # 清掉 pyc
        for c in (p.parent / "__pycache__").glob(p.stem + ".*.pyc"):
            c.unlink()


def revert():
    for p in FILES.values():
        bak = p.with_suffix(p.suffix + ".orig_wringer")
        if bak.exists():
            shutil.copy(bak, p)
        for c in (p.parent / "__pycache__").glob(p.stem + ".*.pyc"):
            c.unlink()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"apply": apply, "revert": revert, "status": lambda: None}[cmd]()
    print("VLLM_PATCH", cmd, status(), flush=True)

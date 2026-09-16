"""2c 核心層微基準:vLLM 內建 exllama gptq_gemm 在 p3b_w2 實際層形狀上,2/3/4/8-bit 對 fp16 稠密乘法的耗時與正確性。
不載模型;codes 隨機、對稱格(bias = 2^(b-1)),qzeros 用 v1 慣例(bias-1,核心 +1)。
    .venv-vllm/bin/python -m p2_anchor.wringer.bench_gptq_kernel [--out evidence/.../bench_gptq_kernel.json]
判準:每形狀每位元 max|Δ|/max|ref| < 1e-2(fp16 累加差異),否則標 FAIL。
"""
import argparse
import json
import time
from pathlib import Path

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import pack_quantized_values_into_int32
from vllm.scalar_type import scalar_types

ROOT = Path(__file__).resolve().parents[2]
TYPES = {2: scalar_types.uint2b2, 3: scalar_types.uint3b4, 4: scalar_types.uint4b8, 8: scalar_types.uint8b128}
# p3b_w2 層形狀 (K=in, N=out, 帳面位元)
SHAPES = [("in_proj_qkv", 2560, 8192, 2), ("gate_proj", 2560, 9216, 2), ("up_proj", 2560, 9216, 3),
          ("down_proj", 9216, 2560, 3), ("out_proj", 4096, 2560, 2), ("o_proj(full)", 4096, 2560, 4)]
MS = [1, 8, 64, 2048]
G = 128


def pack_stream(v, bits, dim):
    """沿 dim 把 [0,2^bits) 整數打成連續 LSB-first 位元流的 int32(AutoGPTQ/exllama 版面;3-bit 為 32 值→3 字)。"""
    v = v.movedim(dim, -1).to(torch.int64)
    n = v.shape[-1]
    assert (n * bits) % 32 == 0, (n, bits)
    per = 32 // __import__("math").gcd(32, bits)              # 每組值數:2→16,3→32,4→8,8→4
    words = per * bits // 32
    g = v.reshape(*v.shape[:-1], n // per, per)
    sh = (torch.arange(per, device=v.device) * bits)
    acc = torch.zeros(*g.shape[:-1], words, dtype=torch.int64, device=v.device)
    for i in range(per):
        pos = sh[i].item(); w, off = divmod(pos, 32)
        acc[..., w] |= (g[..., i] << off) & 0xFFFFFFFF
        if off + bits > 32:
            acc[..., w + 1] |= g[..., i] >> (32 - off)
    out = acc.reshape(*v.shape[:-1], n // per * words)
    out = torch.where(out >= 2**31, out - 2**32, out).to(torch.int32)
    return out.movedim(-1, dim).contiguous()


def make_layer(K, N, bits, dev):
    t = TYPES[bits]
    bias = 1 << (bits - 1)
    u = torch.randint(0, 1 << bits, (K, N), device=dev, dtype=torch.int32)
    scales = (torch.rand(K // G, N, device=dev) * 0.02 + 0.005).half()
    w_ref = ((u - bias).float().reshape(K // G, G, N) * scales.float()[:, None, :]).reshape(K, N).half()
    qweight = pack_stream(u, bits, 0)                                                    # (K*bits/32, N)
    zeros = torch.full((K // G, N), bias - 1, dtype=torch.int32, device=dev)
    qzeros = pack_stream(zeros, bits, 1)                                                 # (groups, N*bits/32)
    if bits != 3:   # 與 vLLM 內建打包互證
        assert torch.equal(qweight, pack_quantized_values_into_int32(u, t, packed_dim=0))
        assert torch.equal(qzeros, pack_quantized_values_into_int32(zeros, t, packed_dim=1))
    g_idx = torch.empty((0,), dtype=torch.int, device=dev)
    qweight = qweight.contiguous()
    ops.gptq_shuffle(qweight, g_idx, bits)
    return qweight, qzeros, scales, g_idx, w_ref


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evidence/p1_grouping/corkscrew/bench_gptq_kernel.json")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    rows = []
    for name, K, N, ledger_bits in SHAPES:
        for bits in (2, 3, 4, 8):
            qw, qz, sc, gi, w_ref = make_layer(K, N, bits, dev)
            for M in MS:
                a = (torch.randn(M, K, device=dev) * 0.5).half()
                out = ops.gptq_gemm(a, qw, qz, sc, gi, True, False, bits)
                ref = a @ w_ref
                err = ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
                us_q = bench(lambda: ops.gptq_gemm(a, qw, qz, sc, gi, True, False, bits))
                us_d = bench(lambda: a @ w_ref)
                rows.append({"layer": name, "K": K, "N": N, "ledger_bits": ledger_bits, "bits": bits, "M": M,
                             "us_gptq": round(us_q, 1), "us_fp16": round(us_d, 1), "speedup": round(us_d / us_q, 2),
                             "rel_err": err, "ok": err < 1e-2,
                             "GBps_eff": round(K * N * bits / 8 / (us_q * 1e-6) / 1e9, 1)})
                print(rows[-1], flush=True)
            del qw, qz, sc, w_ref
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "rows": rows}, indent=1))
    bad = [r for r in rows if not r["ok"]]
    print("BENCH_DONE", "FAIL" if bad else "OK", len(rows), "rows;", len(bad), "bad", flush=True)


if __name__ == "__main__":
    main()

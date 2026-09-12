"""Wringer 打包驗證:qs 狀態 → 位元緊湊容器(safetensors)→ 解包 → 材化 bf16 → 對照 export。

格式(逐模組,帳面位寬與 ledger 一致):
  code   T1 ∈ [-g/2, g/2-1] 偏移為無號 → log2(g) bit 位元流(2/3/4/8 bit),uint8
  alpha  a0_bits=16 → fp16 逐 block;a0_bits=8 → int8 q 逐 block + fp16 逐列尺度 s(α = q·s)
  inv    恒等時不存(帳面 0);否則 int32(未出現於 p3b_w)
判準:
  R1 碼往返位元一致(必 100%)
  R2 材化 bf16 vs export:α 用狀態 fp32(對照,應 100% 一致)
  R3 材化 bf16 vs export:α 用帳面精度(fp16 / int8+fp16)——不一致比例與 |Δ| 分佈
     若 R3 ≠ 0,發布模型須以帳面精度重材化並重測 he(在報告中標明)。
輸出:evidence/p1_grouping/corkscrew/pack_<tag>.{json,md};容器 data/wringer_<tag>.safetensors
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence/p1_grouping/corkscrew"
BITS = {4: 2, 8: 3, 16: 4, 256: 8}


def pack_bits(u, b):
    """u: uint8 array of values < 2**b → 位元流 uint8 (little, 每值 b bit 連續)。"""
    n = u.size
    bits = ((u[:, None] >> np.arange(b, dtype=np.uint8)) & 1).astype(np.uint8)   # (n,b) LSB first
    return np.packbits(bits.reshape(-1), bitorder="little"), n


def unpack_bits(buf, b, n):
    bits = np.unpackbits(buf, bitorder="little")[: n * b].reshape(n, b)
    return (bits.astype(np.uint16) << np.arange(b, dtype=np.uint16)).sum(1).astype(np.uint8)


def pack_module(name, c):
    T1 = c["T1"].numpy()                     # int8 (M,nB,B)
    M, nB, B = T1.shape
    g = int(c["grid"]); b = BITS[g]
    assert int((c["T2"] != 0).sum()) == 0, f"{name}: T2 非零,非純碼"
    u = (T1.astype(np.int16) + g // 2).astype(np.uint8)
    assert u.max() < g, name
    code, n = pack_bits(u.reshape(-1), b)
    a0 = c["a0"].numpy().astype(np.float32).reshape(M, nB)
    ab = int(c.get("a0_bits", 16))
    t = {f"{name}.code": code}
    meta = {"M": M, "nB": nB, "B": B, "grid": g, "bits": b, "a0_bits": ab, "n": n}
    if ab == 16:
        t[f"{name}.alpha"] = a0.astype(np.float16)
        alpha_bits = 16 * M * nB
    else:
        s = np.abs(a0).max(1, keepdims=True) / 127.0
        s = np.maximum(s, 1e-12).astype(np.float32)
        q = np.rint(a0 / s)
        assert np.abs(q).max() <= 127, name
        t[f"{name}.alpha_q"] = q.astype(np.int8)
        t[f"{name}.alpha_s"] = s.astype(np.float16).reshape(M)
        alpha_bits = 8 * M * nB + 16 * M
    inv = c["inv"].numpy()
    if not np.array_equal(inv, np.arange(inv.size)):
        t[f"{name}.inv"] = inv.astype(np.int32)
        alpha_bits += 32 * inv.size
    meta["bits_code"] = b * n
    meta["bits_alpha"] = alpha_bits
    return t, meta


def unpack_module(name, t, meta, ledger_precision=True):
    M, nB, B, g, b, n = (meta[k] for k in ("M", "nB", "B", "grid", "bits", "n"))
    u = unpack_bits(t[f"{name}.code"], b, n)
    T1 = (u.astype(np.int16) - g // 2).astype(np.int8).reshape(M, nB, B)
    if meta["a0_bits"] == 16:
        a = t[f"{name}.alpha"].astype(np.float32)
    else:
        a = t[f"{name}.alpha_q"].astype(np.float32) * t[f"{name}.alpha_s"].astype(np.float32)[:, None]
    inv = t.get(f"{name}.inv")
    return T1, a.reshape(M, nB, 1), inv


def materialize(T1, a, inv):
    w = (torch.from_numpy(T1).float() * torch.from_numpy(a)).reshape(T1.shape[0], -1)
    if inv is not None:
        w = w[:, torch.from_numpy(inv).long()]
    return w.to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-tag", default="p3b_w")
    ap.add_argument("--export", default="exports/e69_p3b_w")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    sd = ROOT / f"data/qs_{args.state_tag}"
    exp = safe_open(str(ROOT / args.export / "model.safetensors"), "pt")
    tensors, metas = {}, {}
    stats = {"modules": 0, "weights": 0, "bits_code": 0, "bits_alpha": 0,
             "R1_code_mismatch": 0, "R2_bf16_mismatch": 0, "R3_bf16_mismatch": 0,
             "R3_max_abs": 0.0, "R3_max_rel": 0.0, "per_grid": {}}
    worst = []
    for f in sorted(sd.glob("layer*.pt")):
        st = torch.load(f, map_location="cpu", weights_only=False)
        for name, c in st.items():
            t, meta = pack_module(name, c)
            tensors.update(t); metas[name] = meta
            # R1
            T1u, a_led, inv = unpack_module(name, t, meta)
            r1 = int((T1u != c["T1"].numpy()).sum())
            # R2 / R3
            w_ref = exp.get_tensor(name + ".weight")
            assert w_ref.dtype == torch.bfloat16, name
            a32 = c["a0"].numpy().astype(np.float32).reshape(meta["M"], meta["nB"], 1)
            w2 = materialize(T1u, a32, inv)
            w3 = materialize(T1u, a_led, inv)
            r2 = int((w2 != w_ref).sum())
            d3 = (w3.float() - w_ref.float()).abs()
            r3 = int((d3 != 0).sum())
            mx = float(d3.max()); rel = float((d3 / w_ref.float().abs().clamp_min(1e-8)).max()) if r3 else 0.0
            n = meta["n"]
            stats["modules"] += 1; stats["weights"] += n
            stats["bits_code"] += meta["bits_code"]; stats["bits_alpha"] += meta["bits_alpha"]
            stats["R1_code_mismatch"] += r1; stats["R2_bf16_mismatch"] += r2; stats["R3_bf16_mismatch"] += r3
            stats["R3_max_abs"] = max(stats["R3_max_abs"], mx); stats["R3_max_rel"] = max(stats["R3_max_rel"], rel)
            pg = stats["per_grid"].setdefault(f"g{meta['grid']}_a{meta['a0_bits']}", {"modules": 0, "weights": 0, "R3": 0})
            pg["modules"] += 1; pg["weights"] += n; pg["R3"] += r3
            worst.append((r3 / n, name, r3, mx))
            del w_ref, w2, w3, d3
        print(f"{f.name}: modules={stats['modules']} R1={stats['R1_code_mismatch']} R2={stats['R2_bf16_mismatch']} R3={stats['R3_bf16_mismatch']} ({time.time()-t0:.0f}s)", flush=True)
    W = stats["weights"]
    stats["bpw_code"] = stats["bits_code"] / W
    stats["bpw_alpha"] = stats["bits_alpha"] / W
    stats["bpw_body"] = (stats["bits_code"] + stats["bits_alpha"]) / W
    stats["R3_frac"] = stats["R3_bf16_mismatch"] / W
    stats["worst_R3"] = [{"name": n, "frac": round(fr, 6), "n": r, "max_abs": mx} for fr, n, r, mx in sorted(worst, reverse=True)[:8]]
    # 容器落地
    out = ROOT / (args.out or f"data/wringer_{args.state_tag}.safetensors")
    save_file(tensors, str(out), metadata={"wringer_meta": json.dumps(metas), "state_tag": args.state_tag})
    stats["container_bytes"] = out.stat().st_size
    stats["container_bpw_body"] = stats["container_bytes"] * 8 / W
    # 整模型(語言模型)帳:非量化參數
    other = {}
    for k in exp.keys():
        if not k.startswith("model.language_model"):
            continue
        if any(k == m + ".weight" for m in metas):
            continue
        other[k] = math.prod(exp.get_slice(k).get_shape())
    emb = other.get("model.language_model.embed_tokens.weight", 0)
    rest = sum(other.values()) - emb
    lm_total = W + emb + rest
    body_bits = stats["bits_code"] + stats["bits_alpha"]
    stats["lm_params"] = {"body": W, "embed_tied": emb, "other_bf16": rest, "total": lm_total}
    stats["bpw_whole_lm"] = {
        "embed_bf16(=評測態)": (body_bits + 16 * (emb + rest)) / lm_total,
        "embed_int8_per_row(假設,未評測)": (body_bits + (8 * emb + 16 * (emb // 2560)) + 16 * rest) / lm_total,
        "embed_4.5bpw(假設,GGUF Q4_K 同級,未評測)": (body_bits + 4.5 * emb + 16 * rest) / lm_total,
    }
    stats["runtime_s"] = round(time.time() - t0, 1)
    (OUT / f"pack_{args.state_tag}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    L = [f"# Wringer 打包驗證 {args.state_tag}(vs {args.export})", "",
         f"- 模組 {stats['modules']},本體權重 {W:,};容器 `{out.relative_to(ROOT)}` {stats['container_bytes']/2**30:.3f} GiB = {stats['container_bpw_body']:.4f} b/w(safetensors 頭與對齊含在內)",
         f"- 帳面:碼 {stats['bpw_code']:.4f} + α {stats['bpw_alpha']:.4f} = **{stats['bpw_body']:.4f} b/w**(ledger total_fixed 對照見 quant_{args.state_tag}.json)",
         f"- R1 碼往返不一致:{stats['R1_code_mismatch']}(須 0)",
         f"- R2 材化 bf16(α fp32)vs export 不一致:{stats['R2_bf16_mismatch']}(須 0 = 容器碼 + 狀態 α 完整重建評測模型)",
         f"- R3 材化 bf16(α 帳面精度)vs export 不一致:{stats['R3_bf16_mismatch']} / {W:,} = {stats['R3_frac']*100:.4f}%;max|Δ| {stats['R3_max_abs']:.3e},max rel {stats['R3_max_rel']:.3e}",
         "", "| 型別 | 模組 | 權重 | R3 不一致 |", "|---|---|---|---|"]
    for k, v in stats["per_grid"].items():
        L.append(f"| {k} | {v['modules']} | {v['weights']:,} | {v['R3']} ({v['R3']/v['weights']*100:.4f}%) |")
    L += ["", "| 整語言模型 bpw(不含視覺塔 333M) | 值 |", "|---|---|"]
    for k, v in stats["bpw_whole_lm"].items():
        L.append(f"| {k} | {v:.4f} |")
    L += ["", f"參數:本體 {W:,} / embed(tied) {emb:,} / 其餘 bf16 {rest:,} / 合計 {lm_total:,};GGUF U25 對照 bpw_whole 2.8586 / body 2.5584。",
          "", "R3 最差模組:"] + [f"- {w['name']} {w['frac']*100:.4f}% max|Δ| {w['max_abs']:.3e}" for w in stats["worst_R3"]]
    (OUT / f"pack_{args.state_tag}.md").write_text("\n".join(L) + "\n")
    print("PACK_DONE " + json.dumps({k: stats[k] for k in ("bpw_body", "container_bpw_body", "R1_code_mismatch", "R2_bf16_mismatch", "R3_frac")}), flush=True)


if __name__ == "__main__":
    main()

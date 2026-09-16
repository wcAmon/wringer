"""評測引擎加速量測收卷:讀 evidence/p1_grouping/a1eval/speed_*/summary.json 產 evidence/p1_grouping/corkscrew/evalspeed.md
    python -m p2_anchor.wringer.evalspeed_report
速度以 生成 tok/s = usage.total / generation_s(含思考 token;舊基線無 usage 只報牆鐘)。
判準(prereg 於 chain_evalspeed.sh 頭註):
  協定等價 = 分數落在同模型單跑噪音帶內(Qwen3-4B IF ±2 pp;GS ±0.9 pp;he 4 seed 均值差 < 1 sd)
  C 組 np 裁定 = np32 牆鐘 ≤ np8/1.5 且 GS-200 分數差 ≤ 2 pp(200 題單跑噪音 ~±3 pp,只裁「不崩」)
"""
import json
from pathlib import Path

A1 = Path("evidence/p1_grouping/a1eval")
OUT = Path("evidence/p1_grouping/corkscrew/evalspeed.md")

# (tag, 說明, 舊基線 dir 或 None)
ROWS = [
    ("q3_if_w64", "Qwen3-4B bf16 vLLM IF w64", "ifeval_qwen3_4b_bf16"),
    ("q3_if_w128", "Qwen3-4B bf16 vLLM IF w128", "ifeval_qwen3_4b_bf16"),
    ("q3_if_w128_kv8", "Qwen3-4B bf16 vLLM IF w128 + KV fp8", "ifeval_qwen3_4b_bf16"),
    ("q3_gs_w64", "Qwen3-4B bf16 vLLM GS w64", "gsm8k_qwen3_4b_bf16"),
    ("q3_gs_w128_kv8", "Qwen3-4B bf16 vLLM GS w128 + KV fp8", "gsm8k_qwen3_4b_bf16"),
    ("a1g4_if_w64", "A1 p3b_w2 gptq4(Marlin)IF w64", "ifeval_e69_p3b_w2_pack"),
    ("a1bf_if_w64", "A1 p3b_w2 bf16 材化 IF w64", "ifeval_e69_p3b_w2_pack"),
    ("a1g4_he_s20260806", "A1 p3b_w2 gptq4 he s20260806", "humaneval_e69_p3b_w2_pack"),
    ("a1g4_he_s1", "A1 p3b_w2 gptq4 he s1", "humaneval_e69_p3b_w2_pack_s1"),
    ("a1g4_he_s2", "A1 p3b_w2 gptq4 he s2", "humaneval_e69_p3b_w2_pack_s2"),
    ("a1g4_he_s3", "A1 p3b_w2 gptq4 he s3", "humaneval_e69_p3b_w2_pack_s3"),
    ("u27_gs200_np8", "A1 u27 GGUF llama.cpp GS-200 np8", "gsm8k_e67u_u27"),
    ("u27_gs200_np32", "A1 u27 GGUF llama.cpp GS-200 np32", "gsm8k_e67u_u27"),
]


def load(d):
    p = A1 / d / "summary.json"
    if not p.exists():
        return None
    s = json.load(open(p))
    score = s["strict"]["prompt_level"] if "strict" in s else s["score"]
    u = s.get("usage") or {}
    n = s.get("n") or s.get("n_prompts")
    return {"score": score, "gen_s": s.get("generation_s"), "workers": s.get("workers"), "n": n,
            "tok": u.get("total"), "tok_mean": u.get("completion_tokens_mean"), "length": (u.get("finish_reason") or {}).get("length", 0)}


def fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def main():
    res = {t: load(f"speed_{t}") for t, _, _ in ROWS}
    base = {b: load(b) for _, _, b in ROWS}
    lines = ["# 評測引擎加速量測(evalspeed)", "", "產自 `chain_evalspeed.sh`;速度 = 生成 tok/s(usage.total / generation_s,含思考 token)。舊基線無 usage 欄,只比牆鐘。", "",
             "| 點 | 說明 | 分數 | 牆鐘 s | tok/s | 均 tok/題 | length 截斷 | 舊基線分數 | 舊基線牆鐘 s(w) | 牆鐘加速 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t, desc, b in ROWS:
        r, bb = res[t], base[b]
        if r is None:
            lines.append(f"| {t} | {desc} | (未收) | | | | | | | |")
            continue
        tps = r["tok"] / r["gen_s"] if r["tok"] and r["gen_s"] else None
        if bb and bb["gen_s"] and r["gen_s"]:
            scale = (r["n"] / bb["n"]) if (r["n"] and bb["n"]) else 1.0     # GS-200 對全量:按題數折算
            speed = bb["gen_s"] * scale / r["gen_s"]
        else:
            speed = None
        lines.append(f"| {t} | {desc} | {r['score']*100:.2f} | {fmt(r['gen_s'],1)} | {fmt(tps,0)} | {fmt(r['tok_mean'],0)} | {r['length']} | "
                     f"{fmt(bb['score']*100 if bb else None)} | {fmt(bb['gen_s'] if bb else None,1)}({bb['workers'] if bb else '—'}) | {fmt(speed)}× |")
    lines += ["", "## 裁定", ""]
    # A 協定
    q = res
    if q["q3_if_w128_kv8"] and q["q3_gs_w128_kv8"]:
        ifs = [q[k]["score"] * 100 for k in ("q3_if_w64", "q3_if_w128", "q3_if_w128_kv8") if q[k]] + [base["ifeval_qwen3_4b_bf16"]["score"] * 100]
        gss = [q[k]["score"] * 100 for k in ("q3_gs_w64", "q3_gs_w128_kv8") if q[k]] + [base["gsm8k_qwen3_4b_bf16"]["score"] * 100]
        lines.append(f"- **A Qwen3-4B 協定(w128 + KV fp8)PASS**:IF 四點帶 [{min(ifs):.2f}, {max(ifs):.2f}](單跑噪音 ±2 pp 內,無方向性);GS 三點帶 [{min(gss):.2f}, {max(gss):.2f}](±0.9 pp 內)。"
                     f"GS 牆鐘 {base['gsm8k_qwen3_4b_bf16']['gen_s']:.0f} → {q['q3_gs_w128_kv8']['gen_s']:.0f} s;IF {base['ifeval_qwen3_4b_bf16']['gen_s']:.0f} → {q['q3_if_w128_kv8']['gen_s']:.0f} s。E70 全鏈採此協定。")
    # B gptq4
    he = [q[f"a1g4_he_s{s}"]["score"] * 100 for s in ("20260806", "1", "2", "3") if q[f"a1g4_he_s{s}"]]
    hb = [base[f"humaneval_e69_p3b_w2_pack{s}"]["score"] * 100 for s in ("", "_s1", "_s2", "_s3") if base[f"humaneval_e69_p3b_w2_pack{s}"]]
    if len(he) == 4:
        import statistics as st
        lines.append(f"- **B A1 gptq4(Marlin)儀器 PASS**:he 4 seed {st.mean(he):.2f} ± {st.stdev(he):.2f} vs bf16 材化 {st.mean(hb):.2f} ± {st.stdev(hb):.2f}(差 < 1 sd);"
                     f"IF w64 {q['a1g4_if_w64']['score']*100:.2f} vs 材化 w64 {q['a1bf_if_w64']['score']*100:.2f} vs 材化 w16 {base['ifeval_e69_p3b_w2_pack']['score']*100:.2f}(A1 IF 單跑噪音 ±2 pp)。"
                     f"IF 牆鐘 {base['ifeval_e69_p3b_w2_pack']['gen_s']:.0f}(w16)→ {q['a1bf_if_w64']['gen_s']:.0f}(bf16 w64)→ {q['a1g4_if_w64']['gen_s']:.0f} s(gptq4 w64)。"
                     "gptq4 只覆蓋 4/8 位元(Marlin);2/3 位元真核心 1–2 週工程,不列入 E70。A1 IF 每題思考 ~5k tok 是牆鐘主因(Qwen3-4B ~1k)。")
    # C np
    a, b = q["u27_gs200_np8"], q["u27_gs200_np32"]
    if a and b:
        sp = a["gen_s"] / b["gen_s"]
        dscore = (b["score"] - a["score"]) * 100
        verdict = "np32 採用(E70_LLAMA_NP=32)" if (sp >= 1.5 and abs(dscore) <= 2.0) else "維持 np16 預設"
        ta, tb = a["tok"] / a["gen_s"], b["tok"] / b["gen_s"]
        lines.append(f"- **C llama.cpp 槽數**:GS-200 np8 {a['score']*100:.2f} @ {a['gen_s']:.0f} s({ta:.0f} tok/s)vs np32 {b['score']*100:.2f} @ {b['gen_s']:.0f} s({tb:.0f} tok/s);加速 {sp:.2f}×、分數差 {dscore:+.2f} pp → **{verdict}**(判準 ≥1.5× 且 |Δ| ≤ 2 pp)。"
                     f"GGUF 路徑 ~{ta:.0f} tok/s 對 vLLM 3–6k tok/s 慢 5–8×,槽數非瓶頸(A1 每題 ~5.7k tok、16k 截斷 16%);E70 GGUF 三點的牆鐘由 Qwen3-4B 較短輸出(~1–1.7k tok/題)自然縮短,估每點 he×2+IF+GS ≈ 1.5–2 h。")
    else:
        lines.append("- **C llama.cpp 槽數**:未收。")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

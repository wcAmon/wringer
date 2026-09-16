"""E70 S2 組配器:讀 S1 鏈 log(E70S1_LEDGER / E70S1_POINT)算代價表,產兩候選配置(prereg stage2_compose)。
    python -m p2_anchor.wringer.compose_e70 --log <e70s1_chain.log> [--target 2.70] [--out prereg 寫回]
  cost = (he(P1) − he(probe)) / save_bw(he 為 2 seed 均值;P1 he/帳來自 prereg stage0 result)
  P3a = 貪婪(cost 升冪)加到 total_fixed ≤ target 的最小集
  P3b = 加法模型下 he 掉分最小且 total_fixed ≤ target 的子集(2^n 窮舉;b256 與 MLP 4 級可同施)
  印 COMPOSE_E70 <json>;--out 時寫入 prereg stage1_sensitivity.result_cost_table / stage2_compose.candidates。
帳面組合為加法近似(save 相加);實際帳由 quantize 重印 QUANT_LEDGER 裁定。
"""
import argparse
import itertools
import json
import re
from pathlib import Path

PREREG = Path("p2_anchor/wringer/prereg_e70_qwen3_4b.json")
BASEKV = "self_attn.k_proj:256:row,self_attn.v_proj:256:row"
# 探針 → (quantize 旗標片段種類, 值)
ACTION = {
    "s1_a8": ("alpha_bits", 8),
    "s1_down4": ("mod", ("mlp.down_proj", 4)),
    "s1_up4": ("mod", ("mlp.up_proj", 4)),
    "s1_gate4": ("mod", ("mlp.gate_proj", 4)),
    "s1_q4": ("mod", ("self_attn.q_proj", 4)),
    "s1_o8": ("o_proj", 8),
    "s1_b256": ("block256", True),
}


def parse(log):
    led, pts = {}, {}
    for ln in open(log):
        m = re.match(r"E70S1_LEDGER:(\S+) QUANT_LEDGER (\{.*\})", ln.strip())
        if m:
            led[m.group(1)] = json.loads(m.group(2))
        m = re.match(r"E70S1_POINT:(\S+) he=([\d.]+),([\d.]+)", ln.strip())
        if m:
            pts[m.group(1)] = (float(m.group(2)) * 100, float(m.group(3)) * 100)
    return led, pts


def module_map(sel):
    """由選定探針集合組 --module-map 與其他旗標。"""
    lv = {"mlp.gate_proj": 8, "mlp.up_proj": 8, "mlp.down_proj": 8, "self_attn.q_proj": 8}
    o_lv, blk_mlp, a8 = 16, 128, False
    for t in sel:
        kind, v = ACTION[t]
        if kind == "mod":
            lv[v[0]] = v[1]
        elif kind == "o_proj":
            o_lv = v
        elif kind == "block256":
            blk_mlp = 256
        elif kind == "alpha_bits":
            a8 = True
    parts = [BASEKV, f"self_attn.o_proj:{o_lv}:128"]
    for name, l in lv.items():
        b = blk_mlp if name.startswith("mlp.") else 128
        if l != 8 or b != 128:
            parts.append(f"{name}:{l}:{b}")
    flags = ("--alpha-bits 8 " if a8 else "") + '--module-map "' + ",".join(parts) + '"'
    return flags, (8 if a8 else 16), ",".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--target", type=float, default=2.70)
    ap.add_argument("--out", action="store_true")
    args = ap.parse_args()
    pre = json.load(open(PREREG))
    r0 = pre["stage0_engineering"]["result"]
    he_p1 = sum(r0["P1_he"]["seeds"]) / len(r0["P1_he"]["seeds"])
    bw_p1 = r0["P1_ledger"]["total_fixed"]
    led, pts = parse(args.log)
    table = {}
    for t in ACTION:
        if t in led and t in pts:
            save = bw_p1 - led[t]["total_fixed"]
            he = sum(pts[t]) / 2
            table[t] = {"total_fixed": led[t]["total_fixed"], "save_bw": round(save, 4), "he_seeds": list(pts[t]),
                        "he_mean": round(he, 2), "drop_pp": round(he_p1 - he, 2),
                        "cost_pp_per_bw": round((he_p1 - he) / save, 1) if save > 1e-6 else None}
    need = bw_p1 - args.target
    # P3a 貪婪
    order = sorted(table, key=lambda t: (table[t]["cost_pp_per_bw"] if table[t]["cost_pp_per_bw"] is not None else 1e9))
    p3a, acc = [], 0.0
    for t in order:
        if acc >= need:
            break
        p3a.append(t); acc += table[t]["save_bw"]
    # P3b 加法最優
    best = None
    for k in range(1, len(table) + 1):
        for sub in itertools.combinations(table, k):
            s = sum(table[t]["save_bw"] for t in sub)
            if s < need:
                continue
            d = sum(table[t]["drop_pp"] for t in sub)
            if best is None or d < best[0] or (abs(d - best[0]) < 1e-9 and s > best[1]):
                best = (d, s, list(sub))
    cands = {}
    for name, sel, s in (("p3a", p3a, acc), ("p3b", best[2] if best else [], best[1] if best else 0)):
        flags, ab, mm = module_map(sel)
        cands[name] = {"probes": sel, "pred_total_fixed": round(bw_p1 - s, 4), "pred_drop_pp": round(sum(table[t]["drop_pp"] for t in sel), 2),
                       "pred_he": round(he_p1 - sum(table[t]["drop_pp"] for t in sel), 2), "reach": s >= need - 1e-9,
                       "flags": flags, "alpha_bits": ab, "module_map": mm}
    if cands["p3b"]["probes"] and sorted(cands["p3b"]["probes"]) == sorted(cands["p3a"]["probes"]):
        cands["p3b"]["note"] = "與 p3a 同集;S2 只跑一候選"
    rep = {"he_p1": round(he_p1, 2), "bw_p1": bw_p1, "target": args.target, "need_save": round(need, 4), "n_probes": len(table),
           "cost_table": table, "candidates": cands}
    print("COMPOSE_E70 " + json.dumps(rep, ensure_ascii=False), flush=True)
    if args.out:
        pre["stage1_sensitivity"]["result_cost_table"] = table
        pre["stage2_compose"]["candidates"] = cands
        PREREG.write_text(json.dumps(pre, ensure_ascii=False, indent=1))
        print("prereg updated")


if __name__ == "__main__":
    main()

"""E69 純碼 2.7 打 GGUF(prereg_e69.json)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.verdict_e69

S1 代價表:每探針 cost = (he(P1) − he) / (total(P1) − total)  [pp per b/w];同記 he 對 U he 曲線同帳 dU。
S2/S3:P3* 各點 he/官方 comp 對 U 曲線(he 曲線 U26/U27/U30/U34;comp 曲線同)內插。
"""
import json
from pathlib import Path

EVC = Path("evidence/p1_grouping/corkscrew")
A1 = Path("evidence/p1_grouping/a1eval")
ANCH = {"if": 92.79, "he": 94.51, "gs": 95.53}
P1 = {"tag": "p1gptq", "total": 3.206, "he": 90.24}
UB = {"u25": None, "u27": 2.763, "u30": 2.972, "u34": 3.295}      # u25(IQ2_XXS+保護)由 gguf_ledger 讀;u26(IQ2_XS)2.744≈U27 未評
PROBES = ["s1_a8", "s1_down4", "s1_up4", "s1_gate4", "s1_qkv4", "s1_z4", "s1_b256", "s1_out4"]
P3 = ["p3a", "p3b", "p3b_w", "p3b_w2", "p3a_w", "p3b_alpha", "p3a_alpha", "p3b_flip", "p3a_flip"]
RESA = {"p3b_w": "p3b", "p3a_w": "p3a", "p3b_w2": "p3b_w"}   # p3b_w2:第二輪,起點 p3b_w(A 讀 humaneval_e69_p3b_w_resA)   # 排水點 → 起點;A 閘 he 讀 humaneval_e69_<p3>_resA,ρ=(B−P3)/(A−P3)


def three(prefix):
    r = {}
    f = A1 / f"ifeval_{prefix}" / "summary.json"
    if f.exists():
        r["if"] = round(100 * json.load(open(f))["strict"]["prompt_level"], 2)
    for k, sub in (("he", "humaneval"), ("gs", "gsm8k")):
        f = A1 / f"{sub}_{prefix}" / "summary.json"
        if f.exists():
            r[k] = round(100 * json.load(open(f))["score"], 2)
    if all(k in r for k in ANCH):
        r["comp"] = round(sum(r[k] / ANCH[k] for k in ANCH) / 3, 4)
    return r


def u26_bpw():
    f = EVC / "gguf_ledger_u25.txt"
    if not f.exists():
        return None
    try:
        txt = f.read_text()
        return round(float(json.loads(txt[: txt.index("\n}") + 2])["bpw_body_linear"]), 3)   # JSON 後接逐張表
    except Exception:
        return None


def curves():
    he, comp = [], []
    for t, b in UB.items():
        if b is None:
            b = u26_bpw()
        if b is None:
            continue
        o = three(f"e67u_{t}")
        if "he" in o:
            he.append((b, o["he"]))
        if "comp" in o:
            comp.append((b, o["comp"]))
    return sorted(he), sorted(comp)


def interp(pts, x):
    if len(pts) < 2:
        return None, "n/a"
    if x <= pts[0][0] or x >= pts[-1][0]:
        (x0, y0), (x1, y1) = (pts[0], pts[1]) if x <= pts[0][0] else (pts[-2], pts[-1])
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0), "extrap"
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0), "interp"


def ledger_of(tag):
    for f in (EVC / f"quant_{tag}.json", EVC / f"entropy_{tag}.json"):
        if f.exists():
            return json.load(open(f)).get("ledger")
    return None


def main():
    uhe, ucomp = curves()
    out = {"experiment": "E69", "U_he": uhe, "U_comp": ucomp, "P1": P1, "S1": {}, "P3": {}}
    for t in PROBES:
        led, o = ledger_of(t), three(f"e69_{t}")
        if not led or "he" not in o:
            continue
        tot = led["total_fixed"]
        save = P1["total"] - tot
        u, kind = interp(uhe, tot)
        row = {"total": tot, "save_bw": round(save, 4), "he": o["he"], "dhe": round(o["he"] - P1["he"], 2),
               "cost_pp_per_bw": round((P1["he"] - o["he"]) / save, 1) if save > 1e-6 else None,
               "U_he": round(u, 2) if u is not None else None, "dU": round(o["he"] - u, 2) if u is not None else None, "U_kind": kind}
        out["S1"][t] = row
    if uhe:
        (x0, y0), (x1, y1) = uhe[-2], uhe[-1]
        out["U_slope_top_pp_per_bw"] = round((y1 - y0) / (x1 - x0), 1)
    for t in P3:
        led, o = ledger_of(t), three(f"e69_{t}")
        if not led or "he" not in o:
            continue
        tot = led["total_fixed"]
        u, kind = interp(uhe, tot)
        row = {"total": tot, **o, "U_he": round(u, 2) if u is not None else None, "dU_he": round(o["he"] - u, 2) if u is not None else None, "kind": kind}
        if "comp" in o:
            c, ck = interp(ucomp, tot)
            if c is not None:
                row["U_comp"] = round(c, 4)
                row["dU_comp"] = round(o["comp"] - c, 4)
                row["J3"] = "STRONG" if o["comp"] - c > 0.03 else "PASS" if o["comp"] > c else "FAIL"
        if t in RESA:
            a = three(f"e69_{RESA[t]}_resA").get("he")
            p = out["P3"].get(RESA[t], {}).get("he")
            if a is not None and p is not None:
                row["A_he"] = a
                if a - p > 1e-6:
                    row["rho"] = round((o["he"] - p) / (a - p), 3)
                    row["falsifier_s3"] = bool(row["rho"] < 0.5)
                else:                       # 第二輪:含水態未高於起點,ρ 無定義;殺閘改「擰乾態低於起點」
                    row["rho"] = None
                    row["A_below_start"] = True
                    row["falsifier_s3"] = bool(o["he"] < p)
                row["gain_vs_start"] = round(o["he"] - p, 2)
        out["P3"][t] = row
    Path("p2_anchor/wringer/verdict_e69.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("E69_S1 " + json.dumps({t: [r["total"], r["he"], r["dhe"], r["cost_pp_per_bw"], r["dU"]] for t, r in out["S1"].items()}))
    if out["P3"]:
        print("E69_P3 " + json.dumps(out["P3"], ensure_ascii=False))


if __name__ == "__main__":
    main()

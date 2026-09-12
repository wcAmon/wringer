"""E67 收卷:U 曲線錨 + A 求解器對決(vs E66-RD R 線)+ B(若有)離曲線判。缺點位跳過。"""
import json, re
from pathlib import Path
A1 = Path("evidence/p1_grouping/a1eval"); EVC = Path("evidence/p1_grouping/corkscrew")
R = {"g9": (3.101, 88.41), "R1": (3.003, 82.93), "R2": (2.908, 76.83), "R3": (2.809, 73.78)}

def score(sub, sfx):
    p = A1 / f"{sub}_{sfx}" / "summary.json"
    if not p.exists(): return None
    d = json.load(open(p)); return round((d["strict"]["prompt_level"] if sub == "ifeval" else d["score"]) * 100, 2)

rows = {"U": {}, "A": {}, "B": {}}
for tag in ("u27", "u30", "u34"):
    lp = EVC / f"gguf_ledger_{tag}.txt"
    if not lp.exists(): continue
    m = re.search(r'"bpw_body_linear": ([\d.]+)', lp.read_text()); mw = re.search(r'"bpw_whole": ([\d.]+)', lp.read_text())
    r = {"bpw_body": round(float(m.group(1)), 3) if m else None, "bpw_whole": round(float(mw.group(1)), 3) if mw else None}
    for sub in ("humaneval", "ifeval", "gsm8k"):
        s = score(sub, f"e67u_{tag}")
        if s is not None: r[sub] = s
    if "humaneval" in r: rows["U"][tag] = r
for tag in ("gsq_a30", "gsq_a29"):
    gp = EVC / f"gsq_{tag}.json"; he = score("humaneval", f"e67a_{tag}")
    if not gp.exists() or he is None: continue
    d = json.load(open(gp)); led = d.get("ledger", {})
    fl = [m["flips"] for m in d["modules"].values()]
    rows["A"][tag] = {"he": he, "H": led.get("code_entropy_joint"), "total_entropy_joint": led.get("total_entropy_joint"),
                      "flips_mean": round(sum(fl) / max(len(fl), 1), 4), "lam": d["hparams"].get("lam")}
J = {}
def he_R_at(H):
    pts = sorted(R.values())
    lo = max([p for p in pts if p[0] <= H], default=None); hi = min([p for p in pts if p[0] >= H], default=None)
    if lo and hi and hi[0] > lo[0]: return lo[1] + (hi[1] - lo[1]) * (H - lo[0]) / (hi[0] - lo[0])
    return (lo or hi)[1]
deltas = {}
for tag, r in rows["A"].items():
    if r["H"] is not None:
        d = r["he"] - he_R_at(r["H"]); deltas[tag] = round(d, 2)
if deltas:
    n_pass = sum(1 for d in deltas.values() if d > 3)
    J["J_A"] = ("PASS" if n_pass == len(deltas) and len(deltas) >= 2 else ("GREY" if n_pass >= 1 else "FAIL(GSQ 機制不轉移)")) + f" Δhe_vs_R={deltas}"
if rows["U"]:
    J["U_curve"] = {t: (r["bpw_body"], r.get("humaneval")) for t, r in rows["U"].items()}
    # 我們的點對 U 曲線(主幹 bpw 口徑):pa64 2.70 熵/2.78 固定 he 64.02;R 點碼熵 + α 0.5
    def u_at(b):
        pts = sorted([(r["bpw_body"], r["humaneval"]) for r in rows["U"].values() if r.get("humaneval") is not None])
        if not pts: return None
        lo = max([p for p in pts if p[0] <= b], default=None); hi = min([p for p in pts if p[0] >= b], default=None)
        if lo and hi and hi[0] > lo[0]: return lo[1] + (hi[1] - lo[1]) * (b - lo[0]) / (hi[0] - lo[0])
        return (lo or hi)[1]
    ours = {"pa64_fixed2.78": (2.78, 64.02), "g9_fixed3.70": (3.70, 88.41), "R1_ent3.50": (3.503, 82.93), "R3_ent3.31": (3.309, 73.78)}
    J["ours_vs_U"] = {k: round(v[1] - u_at(v[0]), 2) for k, v in ours.items() if u_at(v[0]) is not None}
print("E67_ROWS " + json.dumps(rows, ensure_ascii=False)); print("E67_VERDICT " + json.dumps(J, ensure_ascii=False))
json.dump({"experiment": "E67", "rows": rows, "judges": J, "R_line": R}, open("p2_anchor/wringer/verdict_e67.json", "w"), ensure_ascii=False, indent=1)

"""E65-X 收卷:he–位元曲線表 + 判準(prereg_e65_x.json)。缺點位跳過(可在鏈中途執行)。"""
import json
from pathlib import Path

A1 = Path("evidence/p1_grouping/a1eval")
EVC = Path("evidence/p1_grouping/corkscrew")
OLD = {"ifeval": 92.79, "humaneval": 94.51, "gsm8k": 95.53}
REF = {"pa64": {"ifeval": 83.55, "humaneval": 64.02, "gsm8k": 85.29, "comp": 0.8235, "code_fixed": 2.07, "total_fixed": 2.78},
       "g9": {"ifeval": 78.56, "humaneval": 88.41, "gsm8k": 94.16, "comp": 0.9226, "code_fixed": 3.20, "total_fixed": 3.70}}
POINTS = [("ux_u0", "U"), ("ux_u1", "U"), ("ux_u1w", "U"), ("ux_u2", "U"), ("ux_u3", "U"), ("ux_u4", "U"),
          ("ux_d1", "D"), ("ux_d2", "D"), ("ux_d3", "D"),
          ("ux_u1g128", "A"), ("ux_u2g128", "A")]
HE_TARGET = 86.0          # he ≥ 86 ⇒ 該點若 IF/GS 持平 pa64 即 comp ≈ 0.90
BUDGET = 2.75             # 固定率總帳上限(IQ2_M 帶 + 噪音)


def score(sub, sfx):
    p = A1 / f"{sub}_{sfx}" / "summary.json"
    if not p.exists():
        return None, None
    d = json.load(open(p))
    s = (d["strict"]["prompt_level"] if sub == "ifeval" else d["score"]) * 100
    emp = sum(1 for l in open(A1 / f"{sub}_{sfx}" / "responses.jsonl") if not json.loads(l)["response"].strip())
    return round(s, 2), emp


rows = {}
for tag, arm in POINTS:
    lp = EVC / f"ladder_{tag}.json"
    if not lp.exists():
        continue
    led = json.load(open(lp)).get("ledger", {})
    he, e_he = score("humaneval", f"e65x_{tag}")
    if he is None:
        continue
    r = {"arm": arm, "he": he, "empties_he": e_he, **{k: led.get(k) for k in ("code_fixed", "code_entropy", "alpha", "beta", "r", "total_fixed", "total_entropy")}}
    rows[tag] = r

J = {}
he_u1 = max([r["he"] for t, r in rows.items() if t in ("ux_u1", "ux_u1w")], default=None)
if "ux_u1" in rows and "ux_u1w" in rows:
    J["J_TARGET"] = f"cross−w9 Δhe={rows['ux_u1']['he']-rows['ux_u1w']['he']:+.2f}(>3 交叉補償有效 / <−3 有害 / 帶內 NULL)"
he_d3 = rows.get("ux_d3", {}).get("he")
if he_u1 is not None:
    J["J_U1"] = "FALSIFIER_U(淺層 T2 不可閉式分離)" if he_u1 < 70 else ("U1_PASS(≥78)" if he_u1 >= 78 else "U1_GREY[70,78)")
if he_d3 is not None:
    J["J_D3"] = "FALSIFIER_D(三元血統 he 天花板=盆地)" if he_d3 < 70 else ("D_POS" if he_d3 - REF["pa64"]["humaneval"] > 3 else "D_NULL")
for k in ("u1", "u2"):
    g, u = rows.get(f"ux_{k}g128"), rows.get(f"ux_{k}")
    if g and u:
        d = g["he"] - u["he"]
        J[f"J_G128_{k}"] = ("G128_OK(折層 α 粗化無價;帳 −0.23)" if d > -3 else "G128_NEG(α 粒度有價)") + f" Δhe={d:+.2f}"
cands = sorted([(t, r) for t, r in rows.items() if r["he"] >= HE_TARGET and (r.get("total_fixed") or 9) <= BUDGET],
               key=lambda x: x[1]["total_fixed"])
J["collect"] = [t for t, _ in cands[:2]] or "NONE(帶內無 he≥86 點 ⇒ 位元配置無零訓練解,轉訓練)"
# 會合差:同碼熵帶 U vs D
J["meeting_note"] = "he–code_fixed 表見 rows;U 曲線(從上折)與 D 曲線(從下加)在相近 code_fixed 的 he 差 = 血統盆地差"
print("E65X_ROWS " + json.dumps(rows, ensure_ascii=False))
print("E65X_VERDICT " + json.dumps(J, ensure_ascii=False))
json.dump({"experiment": "E65-X 兩端夾擊階梯(零訓練;全部 he 單科裁)", "refs": REF, "rows": rows, "judges": J,
           "he_target": HE_TARGET, "budget_total_fixed": BUDGET},
          open("p2_anchor/wringer/verdict_e65_x.json", "w"), ensure_ascii=False, indent=1)

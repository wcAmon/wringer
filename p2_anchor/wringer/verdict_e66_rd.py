"""E66-RD 收卷:率失真台階 he–熵表 + 判準(prereg_e66_rd.json)。缺點位跳過(可在鏈中途執行)。"""
import json
from pathlib import Path

A1 = Path("evidence/p1_grouping/a1eval")
EVC = Path("evidence/p1_grouping/corkscrew")
REF = {"g9": {"he": 88.41, "ifeval": 78.56, "gsm8k": 94.16, "comp": 0.9226, "code_entropy_joint": 3.101, "code_fixed": 3.20}}
POINTS = [("rd_r1", "R"), ("rd_r2", "R"), ("rd_r3", "R"), ("rd_s1g", "S")]


def he_of(tag):
    p = A1 / f"humaneval_e66rd_{tag}" / "summary.json"
    if not p.exists():
        return None, None
    d = json.load(open(p))
    emp = sum(1 for l in open(p.parent / "responses.jsonl") if not json.loads(l)["response"].strip())
    return round(d["score"] * 100, 2), emp


rows = {}
for tag, arm in POINTS:
    lp = EVC / f"ladder_{tag}.json"
    if not lp.exists():
        continue
    d = json.load(open(lp))
    led = d.get("ledger", {})
    mods = d.get("modules", {})
    fl = [m["flips"] for m in mods.values() if "flips" in m]
    he, emp = he_of(tag)
    if he is None:
        continue
    rows[tag] = {"arm": arm, "he": he, "empties": emp, "lam": d["hparams"].get("lam"), "flips_mean": round(sum(fl) / max(len(fl), 1), 4),
                 "rel_rms_max": round(max([m.get("rel_rms", 0) for m in mods.values()] or [0]), 4),
                 **{k: led.get(k) for k in ("code_entropy_joint", "code_fixed", "alpha", "total_entropy_joint", "total_fixed")}}

J = {}
r1 = rows.get("rd_r1")
if r1:
    J["J_R1"] = ("PASS(接 P2)" if r1["he"] >= 84 else ("GREY[80,84)" if r1["he"] >= 80 else "FALSIFIER_RD(8% 換碼即出盆地,率台階封盤)")) + f" he={r1['he']} Δ={r1['he']-REF['g9']['he']:+.2f}"
rs = [rows[t] for t in ("rd_r1", "rd_r2", "rd_r3") if t in rows]
if len(rs) >= 2:
    slopes = []
    for a, b in zip(rs[:-1], rs[1:]):
        dH = a["code_entropy_joint"] - b["code_entropy_joint"]
        slopes.append(round((a["he"] - b["he"]) / max(dH, 1e-6) * 0.1, 2))
    J["J_SLOPE"] = {"he_drop_per_0.1b": slopes, "note": "後段掉幅 > 前段 2× ⇒ 非線性加速,台階上限記在該點"}
    if len(slopes) >= 2 and slopes[1] > 2 * max(slopes[0], 1.0):
        J["J_SLOPE"]["accel_at"] = "R2→R3"
s1 = rows.get("rd_s1g")
if s1 and len(rs) >= 2:
    # 以 R 曲線線性內插到 S1g 的熵,比 he
    pts = sorted([(r["code_entropy_joint"], r["he"]) for r in rs])
    H = s1["code_entropy_joint"]
    lo = max([p for p in pts if p[0] <= H], default=None)
    hi = min([p for p in pts if p[0] >= H], default=None)
    if lo and hi and hi[0] > lo[0]:
        he_R = lo[1] + (hi[1] - lo[1]) * (H - lo[0]) / (hi[0] - lo[0])
    else:
        he_R = (lo or hi)[1]
    d = he_R - s1["he"]
    J["J_AB"] = ("RATE_WINS" if d > 3 else ("GRID_WINS(改走 §3.3 減格點台階)" if d < -3 else "TIE(以帳與工程裁)")) + f" he_R@H={H:.3f}≈{he_R:.2f} vs S1g {s1['he']} Δ={d:+.2f}"
for t, r in rows.items():
    pass
J["J_TYPE11_note"] = "rel_rms 與 he 排序不一致即記第 11 型;局部誤差不作裁判"
print("E66RD_ROWS " + json.dumps(rows, ensure_ascii=False))
print("E66RD_VERDICT " + json.dumps(J, ensure_ascii=False))
json.dump({"experiment": "E66-RD 率失真台階(零訓練;he 單科裁)", "refs": REF, "rows": rows, "judges": J},
          open("p2_anchor/wringer/verdict_e66_rd.json", "w"), ensure_ascii=False, indent=1)

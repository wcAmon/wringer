"""E68-P 帳面對齊臂裁決(prereg_e68_parity.json FROZEN @8054fc8)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.verdict_e68

每點:total_fixed(quant_<tag>.json ledger)+ 官方 he(a1eval/humaneval_e68_<tag>)→ 對 U 曲線(body bpw)同帳線性內插。
J_P:he(P1-gptq | P1-alpha) − U > 0 PASS / > +3 STRONG;J_S:he(P0-gptq) − he(P0-rtn) > +3;falsifier:he(P0-gptq) < 79.9。
"""
import json
from pathlib import Path

EVC = Path("evidence/p1_grouping/corkscrew")
A1 = Path("evidence/p1_grouping/a1eval")
U = [(2.763, 63.41), (2.972, 77.44), (3.295, 92.68)]
R1 = (3.003, 82.93)
POINTS = ["p0rtn", "p0gptq", "p1gptq", "p1alpha", "p2champ"]
ANCH = {"if": 92.79, "he": 94.51, "gs": 95.53}
UB = {"u27": 2.763, "u30": 2.972, "u34": 3.295}


def three(prefix):
    """官方三科(若齊)→ dict(if/he/gs/comp);缺科則只回已有科。"""
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


def comp_curve():
    return sorted((b, three(f"e67u_{t}")["comp"]) for t, b in UB.items() if "comp" in three(f"e67u_{t}"))


def interp(pts, x):
    if len(pts) < 2:
        return None, "n/a"
    if x <= pts[0][0] or x >= pts[-1][0]:
        (x0, y0), (x1, y1) = (pts[0], pts[1]) if x <= pts[0][0] else (pts[-2], pts[-1])
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0), "extrap"
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0), "interp"


def u_at(x):
    pts = sorted(U)
    if x <= pts[0][0] or x >= pts[-1][0]:
        (x0, y0), (x1, y1) = (pts[0], pts[1]) if x <= pts[0][0] else (pts[-2], pts[-1])
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0), "extrap"
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0), "interp"


def load_point(tag):
    q = EVC / f"quant_{tag}.json"
    he = A1 / f"humaneval_e68_{tag}" / "summary.json"
    ent = EVC / f"entropy_{tag}.json"
    r = {}
    if q.exists():
        r["ledger"] = json.load(open(q)).get("ledger")
    elif ent.exists():
        r["ledger"] = json.load(open(ent)).get("ledger")
    if he.exists():
        r["he"] = round(100 * json.load(open(he))["score"], 2)
    if ent.exists():
        e = json.load(open(ent))
        r["H_joint"] = e.get("H_joint")
    if "ledger" in r and r["ledger"] and "he" in r:
        t = r["ledger"]["total_fixed"]
        u, kind = u_at(t)
        r["total_fixed"] = t
        r["U_at"] = round(u, 2)
        r["dU"] = round(r["he"] - u, 2)
        r["U_kind"] = kind
    return r


def main():
    rows = {t: load_point(t) for t in POINTS}
    rows = {t: r for t, r in rows.items() if r}
    J = {}
    main_pts = [t for t in ("p1alpha", "p1gptq") if rows.get(t, {}).get("dU") is not None]
    if main_pts:
        best = max(main_pts, key=lambda t: rows[t]["dU"])
        d = rows[best]["dU"]
        J["J_P"] = f"{'STRONG' if d > 3 else 'PASS' if d > 0 else 'FAIL'}({best} he {rows[best]['he']} vs U {rows[best]['U_at']} @ {rows[best]['total_fixed']} ⇒ {d:+.2f})"
    if rows.get("p0gptq", {}).get("he") is not None and rows.get("p0rtn", {}).get("he") is not None:
        ds = rows["p0gptq"]["he"] - rows["p0rtn"]["he"]
        J["J_S"] = f"{'SOLVER_WINS' if ds > 3 else 'NO_EDGE' if ds <= 0 else 'GREY'}(gptq {rows['p0gptq']['he']} − rtn {rows['p0rtn']['he']} = {ds:+.2f})"
    if rows.get("p0gptq", {}).get("he") is not None:
        J["falsifier"] = f"{'TRIGGERED(八元/g128 路線封盤)' if rows['p0gptq']['he'] < 79.9 else 'clear'}(p0gptq {rows['p0gptq']['he']} vs 79.9)"
    for t in ("p0rtn", "p0gptq", "p2champ", "p1alpha"):
        if rows.get(t, {}).get("dU") is not None:
            J[f"vs_U_{t}"] = rows[t]["dU"]
    if rows.get("p1alpha", {}).get("he") is not None and rows.get("p1gptq", {}).get("he") is not None:
        J["alpha_gain_pp"] = round(rows["p1alpha"]["he"] - rows["p1gptq"]["he"], 2)
    # 官方三科(chain_e68c):comp 對 U comp 曲線同帳內插;北極星 0.90
    uc = comp_curve()
    U3 = {t: three(f"e67u_{t}") for t in UB}
    for t in ("p1gptq", "p1alpha"):
        o = three(f"e68_{t}")
        if "comp" in o and t in rows:
            rows[t]["official"] = o
            c, kind = interp(uc, rows[t]["total_fixed"])
            if c is not None and kind == "extrap":
                J[f"J_C_{t}"] = f"PENDING(U comp 曲線僅 {len(uc)} 點,{rows[t]['total_fixed']} 落在外推段;等 U34 三科)"
            elif c is not None:
                rows[t]["U_comp_at"] = round(c, 4)
                rows[t]["dU_comp"] = round(o["comp"] - c, 4)
                J[f"J_C_{t}"] = f"{'PASS' if o['comp'] > c else 'FAIL'}(comp {o['comp']} vs U {c:.4f} @ {rows[t]['total_fixed']} {kind} ⇒ {o['comp'] - c:+.4f}; northstar {'✓' if o['comp'] >= 0.90 else '✗'} 0.90)"
    out = {"experiment": "E68-P", "rows": rows, "judges": J, "anchors": {"U": U, "R1": R1, "U_three": U3, "U_comp": uc}}
    Path("p2_anchor/wringer/verdict_e68.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("E68_VERDICT " + json.dumps(J, ensure_ascii=False))


if __name__ == "__main__":
    main()

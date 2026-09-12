"""態的碼熵帳(E67-B 率項保熵儀器):對 9 格模組(無 t2beta)算權重數加權的聯合熵 H(T1+0.6T2)與逐模組 H;
保護模組(linear_attn.out_proj / self_attn.v_proj)另列。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.state_entropy --state-tag gsq_b0 [--out x.json]
輸出行:STATE_ENTROPY {"state":..,"H_joint":..,"H_unprot":..,"weights":..}
"""
import argparse
import glob
import json
import math

import torch

from p2_anchor.wringer.ktier import INT_LEVELS
from p2_anchor.wringer.state import qs_dir

PROT = ("linear_attn.out_proj", "self_attn.v_proj")


def sym9(c):
    return (c["T1"].reshape(-1).to(torch.int64) + 1) * 3 + (c["T2"].reshape(-1).to(torch.int64) + 1)


def ent(cnt):
    p = cnt / cnt.sum()
    p = p[p > 0]
    return float(-(p * p.log2()).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-tag", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    hist = {}                                  # 逐格族直方圖:族 → [tot, unprot](E68 混合格 8/16/9)
    mods = {}
    st_all = {}
    n_other = 0
    for f in sorted(glob.glob(str(qs_dir(args.state_tag) / "layer*.pt"))):
        st = torch.load(f, map_location="cpu", weights_only=False)
        for n, c in st.items():
            if not (isinstance(c, dict) and "T1" in c):
                continue
            g = int(c.get("grid", 9))
            st_all[n] = c
            if g in INT_LEVELS:                     # 整數格:T1 直方圖(256 級 int8 視為固定 8 bit,不入熵)
                if g == 256:
                    n_other += 1
                    continue
                lo, hi = INT_LEVELS[g]
                cnt = torch.bincount(c["T1"].reshape(-1).to(torch.int64) - lo, minlength=hi - lo + 1).double()
            elif g == 9 and c.get("t2beta") is None:
                cnt = torch.bincount(sym9(c), minlength=9).double()
            else:
                n_other += 1
                continue
            mods[n] = {"H": ent(cnt), "n": int(cnt.sum()), "grid": g}
            h = hist.setdefault(g, [torch.zeros_like(cnt), torch.zeros_like(cnt)])
            h[0] += cnt
            if not any(n.endswith(p) for p in PROT):
                h[1] += cnt
    gmain = max(hist, key=lambda g: float(hist[g][0].sum())) if hist else None
    tot, unp = hist[gmain] if gmain is not None else (torch.zeros(1), torch.zeros(1))
    out = {"state": args.state_tag, "H_joint": ent(tot) if tot.sum() > 0 else math.nan,
           "H_unprot": ent(unp) if unp.sum() > 0 else math.nan,
           "weights": int(tot.sum()), "main_grid": gmain, "n_modules": len(mods), "n_non9": n_other,
           "families": {int(g): {"weights": int(h[0].sum()), "H": ent(h[0])} for g, h in hist.items()},
           "H_by_module_mean": sum(m["H"] * m["n"] for m in mods.values()) / max(sum(m["n"] for m in mods.values()), 1)}
    try:
        from p2_anchor.wringer.ladder_e65x import ledger
        out["ledger"] = ledger(st_all)
    except Exception as e:                       # 帳不影響熵輸出
        out["ledger_error"] = str(e)
    print("STATE_ENTROPY " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)
    if args.out:
        out["modules"] = mods
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

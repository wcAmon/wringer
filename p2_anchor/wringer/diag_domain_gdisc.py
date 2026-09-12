"""分域 gdisc:排水殘差按域拆帳——吸力方向假說的直接量測(用戶 2026-08-17)。

參照 = 水庫理想態(冠軍連續態材化 + 滿水庫旁路 B@A/r,即 train_res 的
off+γ=1 目標函數);受測 = 排水後定案態(st42 / st42r / st42c)材化。
兩態各全鏈前傳一次,末層輸出逐列相對殘差,按域標籤聚合:

  rel_d = mean_{row∈d} ‖y_S − y_ref‖ / ‖y_ref‖

吸力傾斜若真:rel_ifshape ≫ rel_code(排水把容量花在 code/he 軸)。
量測面用分層 calib(st42 訓練時未見過)= 泛化吸收而非記憶。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.diag_domain_gdisc \
      --state-tag st42 --ref-state-tag st41g_e2e --res-tag res42 \
      --calib evidence/p1_grouping/calib_drain_e42b.pt \
      --domains evidence/p1_grouping/calib_drain_e42b.domains.json
"""
import argparse
import datetime
import json
from collections import defaultdict
from pathlib import Path

import torch

from p2_anchor.wringer.engine import capture_layer0, run_layers
from p2_anchor.wringer.ktier import k2_weight
from p2_anchor.wringer.modelio import (LAYER_PREFIX, N_LAYERS,
                                         enumerate_targets, get_module,
                                         layer_index, load_model)
from p2_anchor.wringer.state import load_state

EV = Path("evidence/p1_grouping")


def materialize(model, targets, st, res=None):
    """狀態材化;res 給定時疊加滿水庫旁路(參照態)。"""
    for n in targets:
        m = get_module(model, n)
        s = st[n]
        w = k2_weight(s["T1"].cuda(), s["T2"].cuda(),
                      s["a0"].cuda().float(), s["c"])[:, s["inv"].cuda()]
        if res is not None:
            r = res[n]
            w = w + (r["B"].cuda().float() @ r["A"].cuda().float()) / r["r"]
        m.weight.data.copy_(w.to(m.weight.dtype))


def full_forward(model, X, kw):
    layers = [model.get_submodule(f"{LAYER_PREFIX}.{li}")
              for li in range(N_LAYERS)]
    with torch.no_grad():
        return [y.float().cpu() for y in run_layers(layers, X, kw)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-tag", required=True)
    ap.add_argument("--ref-state-tag", default="st41g_e2e")
    ap.add_argument("--res-tag", default="res42")
    ap.add_argument("--calib", default=str(EV / "calib_drain_e42b.pt"))
    ap.add_argument("--domains",
                    default=str(EV / "calib_drain_e42b.domains.json"))
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    doms = json.load(open(args.domains))[:args.n]
    calib = torch.load(args.calib, weights_only=False)[:args.n]
    assert len(doms) == calib.shape[0], "域標籤與 calib 列數不齊"

    model, _ = load_model()
    model.requires_grad_(False)
    targets = enumerate_targets(model)
    st_ref = load_state(args.ref_state_tag, targets)
    st_s = load_state(args.state_tag, targets)
    res_dir = Path("data") / f"res_{args.res_tag}"
    res = {}
    for li in range(N_LAYERS):
        p = res_dir / f"layer{li:02d}.pt"
        if p.exists():
            res.update(torch.load(p, map_location="cpu", weights_only=False))
    assert all(n in res for n in targets), f"res_{args.res_tag} 不齊"

    # 參照態先材化再捕捉 X(embedding 兩態共用,X 只捕一次)
    materialize(model, targets, st_ref, res=res)
    X, kw = capture_layer0(model, calib, batch=args.batch)
    y_ref = full_forward(model, X, kw)

    materialize(model, targets, st_s)
    y_s = full_forward(model, X, kw)

    per_dom = defaultdict(list)
    row = 0
    for br, bs in zip(y_ref, y_s):
        for i in range(br.shape[0]):
            rel = float((bs[i] - br[i]).norm()
                        / br[i].norm().clamp_min(1e-12))
            per_dom[doms[row]].append(rel)
            row += 1
    import statistics as stat
    rep = {d: {"n": len(v), "mean": round(stat.mean(v), 6),
               "median": round(stat.median(v), 6),
               "p90": round(sorted(v)[int(0.9 * len(v))], 6)}
           for d, v in sorted(per_dom.items())}
    allv = [x for v in per_dom.values() for x in v]
    out = {"state": args.state_tag, "ref": args.ref_state_tag,
           "res": args.res_tag, "calib": args.calib, "n": len(allv),
           "overall_mean": round(stat.mean(allv), 6),
           "per_domain": rep,
           "tilt_ratio_if_vs_code": round(
               rep["ifshape"]["mean"] / max(rep["code"]["mean"], 1e-12), 4)
           if {"ifshape", "code"} <= per_dom.keys() else None,
           "timestamp": datetime.datetime.now().isoformat()}
    path = args.out or (EV / f"diag_domgdisc_{args.state_tag}.json")
    Path(path).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["per_domain"], indent=1),
          "\ntilt_if/code =", out["tilt_ratio_if_vs_code"])


if __name__ == "__main__":
    main()

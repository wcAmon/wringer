"""E64 S1:r 上冠軍台 零訓練探針(混格版;冠軍 st51c_e2e 碼 3/5/9 分級 + res62s2 水)。

問題:可交付 2.70 bpw 態(碼 + α + 每 block 2 bit r)在冠軍台零訓練能拿多少?
臂(每模組,同一 W_t = dense(碼) + BA/r、同一 Σ):
  ctrl        碼固定,α 閉式重擬合(對稱)—— 直投基線
  asym        碼固定,(α, r_row) 交替閉式(逐列 r,診斷)
  asymblk     碼固定,逐 block r 全精度(診斷上界,+0.5 b/w 不可交付)
  asymblkq{b} 逐 block r 量化 b 位元(每模組 log 碼書)+ α 重擬合 ← 可交付主臂(b=2)
  asymrefitq2 逐元非對稱重投影(對本模組網格) ↔ (α, r_blk) 交替,末端 r 量化 2 bit + α 重擬合 ← 盆地風險探針(換碼)
判讀:ratio = e_arm/e_ctrl;absorb = 1 − e/e0;flips vs 冠軍碼;局部只診斷,官方三科才裁(J1–J3)。
Σ 於冠軍部署形態單趟量測,各臂不前傳(與 E63 S0 同法)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.probe_asym_mix --smoke
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.probe_asym_mix \
      --state-tag st51c_e2e --res-tag res62s2 --save-arms ctrl,asymblkq2,asymrefitq2 --save-prefix pa64
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from p2_anchor.wringer import ktier
from p2_anchor.wringer.asym import err_h, quantize_r, r_stats
from p2_anchor.wringer.asym_mix import (asym_Vm, code_V0, dense_asym_m, fit_alpha_r_m,
                                          neg_mask, round_grid_asym)
from p2_anchor.wringer.comp import HAccum
from p2_anchor.wringer.engine import capture_layer0, fwd_chain
from p2_anchor.wringer.ktier import scale_joint
from p2_anchor.wringer.modelio import (LAYER_PREFIX, N_LAYERS, cov_source_for,
                                         enumerate_targets, get_module,
                                         layer_index, load_model)
from p2_anchor.wringer.state import dense_weight, load_state

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"
ALL_ARMS = ("ctrl", "asym", "asymblk", "asymblkq", "asymrefitq2")
Q_BITS = (2,)


def fam(n):
    return n.split(".")[-1]


@torch.no_grad()
def probe_module(W_t, W_cur, T1, T2, c, level, inv, Sig, alt, arms):
    """W_t/W_cur (M,K) 模型欄序;T1/T2 (M,nB,B) float;Sig (K,K) 模型欄序。回傳 (rec, states)。"""
    M, K = W_t.shape
    nB, B = T1.shape[1], T1.shape[2]
    q = torch.argsort(inv)
    Wt_s = W_t[:, q]
    Sig_s = Sig[q][:, q] if not torch.equal(inv, torch.arange(K, device=inv.device)) else Sig
    W3 = Wt_s.reshape(M, nB, B)
    V0, neg = code_V0(T1, T2, c), neg_mask(T1)
    e0 = err_h(W_t - W_cur, Sig)
    rec = {"e0": e0, "grid": level, "arms": {}}
    states = {}

    def put(name, T1n, T2n, alpha, r, extra=None, cb=None, deliverable=True):
        Vn, negn = code_V0(T1n, T2n, c), neg_mask(T1n)
        e = err_h(Wt_s - dense_asym_m(Vn, negn, alpha, r), Sig_s)
        d = {"e": e, "ratio": None, "absorb": 1 - e / max(e0, 1e-30),
             "flips": float(((T1n != T1) | (T2n != T2)).float().mean()),
             "zero_frac": float((Vn == 0).float().mean())}
        if r is not None:
            d["r"] = r_stats(r)
        if extra:
            d.update(extra)
        rec["arms"][name] = d
        if deliverable:
            states[name] = {"T1": T1n.to(torch.int8).cpu(), "T2": T2n.to(torch.int8).cpu(),
                            "a0": alpha.cpu(),
                            "rneg": None if r is None else r.reshape(M, -1).cpu(),
                            "rneg_cb": None if cb is None else cb.cpu()}

    al_c, _, _ = fit_alpha_r_m(W3, V0, neg, Sig_s, asym=False)
    put("ctrl", T1, T2, al_c, None)
    e_c = rec["arms"]["ctrl"]["e"]
    if "asym" in arms:
        al_a, r_a, tr_a = fit_alpha_r_m(W3, V0, neg, Sig_s, alt=alt)
        put("asym", T1, T2, al_a, r_a, {"trace_rel": [round(t / max(e_c, 1e-30), 5) for t in tr_a]},
            deliverable=False)
    r_b = None
    if "asymblk" in arms or "asymblkq" in arms or "asymrefitq2" in arms:
        al_b, r_b, tr_b = fit_alpha_r_m(W3, V0, neg, Sig_s, alt=alt, blk=True)
        if "asymblk" in arms:
            put("asymblk", T1, T2, al_b, r_b, {"trace_rel": [round(t / max(e_c, 1e-30), 5) for t in tr_b]},
                deliverable=False)
    if "asymblkq" in arms:
        for b in Q_BITS:
            rq, cb = quantize_r(r_b, b)
            al_q = scale_joint(W3, asym_Vm(V0, neg, rq), Sig_s)
            put(f"asymblkq{b}", T1, T2, al_q, rq, {"codebook": [round(float(v), 4) for v in cb]}, cb=cb)
    if "asymrefitq2" in arms:
        T1n, T2n, al, r = T1, T2, al_b, r_b
        for _ in range(alt):
            T1n, T2n = round_grid_asym(W3, al, r, level, c)
            al, r, _ = fit_alpha_r_m(W3, code_V0(T1n, T2n, c), neg_mask(T1n), Sig_s, alt=1, r0=r, blk=True)
        rq, cb = quantize_r(r, 2)
        Vn, negn = code_V0(T1n, T2n, c), neg_mask(T1n)
        al_q = scale_joint(W3, asym_Vm(Vn, negn, rq), Sig_s)
        put("asymrefitq2", T1n, T2n, al_q, rq, {"codebook": [round(float(v), 4) for v in cb]}, cb=cb)
    for a in rec["arms"].values():
        a["ratio"] = a["e"] / max(e_c, 1e-30)
    return rec, states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-tag", default="st51c_e2e")
    ap.add_argument("--res-tag", default="res62s2")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--calib", default=str(EV / "calib_e58r1b_traj.pt"))
    ap.add_argument("--lengths", default=str(EV / "calib_e58r1b_len.pt"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--alt", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--layers", default="all")
    ap.add_argument("--save-arms", default="")
    ap.add_argument("--save-prefix", default="pa64")
    ap.add_argument("--out", default=str(EVC / "probe_asym_e64_champ.json"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_calib, args.alt, args.layers = 16, 2, "0,20,30"
        args.out = str(EVC / "probe_asym_e64_smoke.json")
        args.save_prefix = "pa64smk"
    arms = [a for a in args.arms.split(",") if a]
    save_arms = [a for a in args.save_arms.split(",") if a]
    layers_sel = (list(range(N_LAYERS)) if args.layers == "all"
                  else [int(x) for x in args.layers.split(",")])
    t0 = time.time()
    torch.manual_seed(0)

    model, _ = load_model()
    model.requires_grad_(False)
    targets = enumerate_targets(model)
    st = load_state(args.state_tag, targets)
    res = {}
    for li in range(N_LAYERS):
        p = Path("data") / f"res_{args.res_tag}" / f"layer{li:02d}.pt"
        if p.exists():
            res.update(torch.load(p, map_location="cpu", weights_only=False))
    assert all(n in res for n in targets), "水庫模組不齊"
    for n in targets:
        get_module(model, n).weight.data.copy_(dense_weight(st[n]).to(torch.bfloat16))
    calib = torch.load(args.calib, weights_only=False)[:args.n_calib]
    lens = torch.load(args.lengths, weights_only=False)[:calib.shape[0]]
    S = calib.shape[1]
    ar = torch.arange(S)
    Mk = [(ar[None] < lens[i:i + args.batch, None]).cuda()
          for i in range(0, calib.shape[0], args.batch)]
    X, kw = capture_layer0(model, calib, batch=args.batch, to_cpu=True)
    layers = [model.get_submodule(f"{LAYER_PREFIX}.{li}") for li in range(N_LAYERS)]
    print(f"load+capture {time.time()-t0:.0f}s  batches {len(X)} seq {S}", flush=True)

    out = {"state": args.state_tag, "res": args.res_tag, "hparams": vars(args),
           "calib": {"rows": int(calib.shape[0]), "seq": int(S)}, "modules": {}}
    save_dirs = {a: Path("data") / f"qs_{args.save_prefix}_{a}" for a in save_arms}
    for d in save_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    for li in range(max(layers_sel) + 1):
        layer = layers[li]
        mods = [n for n in targets if layer_index(n) == li]
        do = li in layers_sel
        acc = HAccum()
        if do:
            srcs = sorted({cov_source_for(n) for n in mods})
            acc.attach([(s, get_module(model, s)) for s in srcs])
        Xn = []
        for bi in range(len(X)):
            acc.mask = Mk[bi]
            Xn.append(fwd_chain([layer], X[bi], kw).to("cpu", torch.bfloat16))
        acc.mask = None
        acc.detach()
        X = Xn
        if not do:
            continue
        per_layer_states = {a: {} for a in save_arms}
        line = []
        for n in mods:
            Sig = acc.H[cov_source_for(n)]
            c = st[n]
            assert c.get("mix") is None and c.get("lmix") is None and c.get("t2beta") is None, \
                f"{n}: 含載體/T2β 的態不入本探針"
            W_cur = dense_weight(c)
            water = (res[n]["B"].cuda().float() @ res[n]["A"].cuda().float()) / float(res[n]["r"])
            W_t = W_cur + water
            ktier.set_block(int(c["T1"].shape[2]))
            rec, states = probe_module(W_t, W_cur, c["T1"].cuda().float(), c["T2"].cuda().float(),
                                       float(c["c"]), int(c["grid"]), c["inv"].cuda(), Sig, args.alt, arms)
            rec["family"] = fam(n)
            rec["layer"] = li
            rec["block"] = int(c["T1"].shape[2])
            out["modules"][n] = rec
            for a in save_arms:
                if a in states:
                    s = states[a]
                    ent = {"T1": s["T1"], "T2": s["T2"], "inv": c["inv"].clone(),
                           "a0": s["a0"], "c": c["c"], "grid": c["grid"]}
                    if s["rneg"] is not None:
                        ent["rneg"] = s["rneg"]
                    if s.get("rneg_cb") is not None:
                        ent["rneg_cb"] = s["rneg_cb"]
                    per_layer_states[a][n] = ent
            line.append(f"{fam(n)[:9]:9s} g{rec['grid']} b{rec['block']} " + " ".join(
                f"{a[:7]}:{rec['arms'][a]['ratio']:.3f}" for a in rec["arms"] if a != "ctrl")
                + (f" | flips(refit)={rec['arms']['asymrefitq2']['flips']:.4f}" if 'asymrefitq2' in rec['arms'] else ""))
            del Sig, W_cur, water, W_t, states
        acc.H.clear()
        torch.cuda.empty_cache()
        for a in save_arms:
            torch.save(per_layer_states[a], save_dirs[a] / f"layer{li:02d}.pt")
        print(f"PROBE_ASYM_LAYER {li} ({time.time()-t0:.0f}s)\n   " + "\n   ".join(line), flush=True)
        Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))

    summ = {}
    arm_names = sorted({a for m in out["modules"].values() for a in m["arms"]} - {"ctrl"})
    for a in arm_names:
        rows = [(m, m["arms"][a]) for m in out["modules"].values() if a in m["arms"]]

        def agg(sel):
            rs = [d["ratio"] for m, d in rows if sel(m)]
            if not rs:
                return None
            return {"n": len(rs), "ratio_mean": round(statistics.mean(rs), 4),
                    "ratio_median": round(statistics.median(rs), 4),
                    "absorb_mean": round(statistics.mean(d["absorb"] for m, d in rows if sel(m)), 4),
                    "flips_mean": round(statistics.mean(d["flips"] for m, d in rows if sel(m)), 5)}
        summ[a] = {"all": agg(lambda m: True),
                   "grid3": agg(lambda m: m["grid"] == 3), "grid5": agg(lambda m: m["grid"] == 5),
                   "grid9": agg(lambda m: m["grid"] == 9),
                   "down_proj": agg(lambda m: m["family"] == "down_proj")}
    summ["ctrl"] = {"absorb_mean": round(statistics.mean(m["arms"]["ctrl"]["absorb"] for m in out["modules"].values()), 4)}
    out["summary"] = summ
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("PROBE_ASYM_SUMMARY " + json.dumps(summ, ensure_ascii=False), flush=True)
    print(f"PROBE_ASYM_DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

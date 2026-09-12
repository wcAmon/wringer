"""E63 步驟 3:down_proj 開第二三元位面(T2)零訓練探針(A4 平台:st54r 碼 + res58r1 水)。

粒度定律推到元素粒度:每權重多一個三元位面 ⇒ 該模組 9 元(vals = T1 + c·T2,c=0.6)。
位元帳:T2 只開在 down_proj(每層線性參數 20.9%)⇒ 全模型 +0.331 b/w;位面格式(idx 4 bit 包)本就 9 元相容。

臂(每模組,同一 W_t、同一 Σ;所有模組都算,交付時只取 down_proj):
  ctrl    T1 固定、T2=0、α 閉式(對稱直投基線;= probe_asym ctrl)
  t2fix   T1 固定,T2 = 對 (W_t/α − T1)/c 逐元最近 {−1,0,1} ↔ α 閉式 交替 ×alt(留在盆地一)
  t2gptq  9 元 GPTQ 直解 W_t(ktier.gptq_grid level 9;T1 亦重解 = 出盆地)+ α 閉式
判讀:ratio = e_arm/e_ctrl;absorb;flips_t1 vs st54r;t2_nz(T2 非零率 = 位元實際使用)。
--compose:組混合 qs 態 data/qs_{prefix}_{arm}_dp[_asym]:down_proj ← 該臂;其他 ← qs_pa63_ctrl / qs_pa63_asymblkq2。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.probe_t2 --smoke
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.probe_t2 --n-calib 128 --batch 4 --save-arms ctrl,t2fix,t2gptq
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.probe_t2 --compose
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from p2_anchor.wringer import ktier
from p2_anchor.wringer.asym import err_h
from p2_anchor.wringer.comp import HAccum
from p2_anchor.wringer.engine import capture_layer0, fwd_chain
from p2_anchor.wringer.ktier import gptq_grid, scale_joint
from p2_anchor.wringer.modelio import (LAYER_PREFIX, N_LAYERS, cov_source_for,
                                         enumerate_targets, get_module,
                                         layer_index, load_model)
from p2_anchor.wringer.state import dense_weight, load_state

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"
ALL_ARMS = ("ctrl", "t2fix", "t2beta", "t2gptq")


def fam(n):
    return n.split(".")[-1]


def round_t2(W3, T1, alpha, c):
    """T1 固定:T2 = argmin_{t∈{−1,0,1}} |W3/α − (T1 + c·t)|(逐元最近)。"""
    res = (W3 / alpha - T1) / c
    return res.clamp(-1, 1).round()


def fit_t2beta(W3, T1, al0, Sig, alt):
    """殘差位面:W ≈ α·T1 + β·T2,β 逐 (row,block) Σ-加權 LS;交替 T2(逐元最近)↔ β ↔ α。
    (amendment_1:固定 c=0.6 的 T2 步距 0.6α ≫ 水殘差 ⇒ T2 全零;β 自適應步距)"""
    al = al0
    Rr = W3 - al * T1
    thr = 0.7 * Rr.abs().mean(-1, keepdim=True)
    T2 = torch.sign(Rr) * (Rr.abs() > thr).float()
    beta = scale_joint(Rr, T2, Sig)
    tr = []
    for _ in range(alt):
        ok = beta.abs() > 1e-12
        T2 = torch.where(ok, (Rr / torch.where(ok, beta, torch.ones_like(beta))).clamp(-1, 1).round(),
                         torch.zeros_like(T2))
        beta = scale_joint(Rr, T2, Sig)
        al = scale_joint(W3 - beta * T2, T1, Sig)
        Rr = W3 - al * T1
        tr.append(err_h((W3 - al * T1 - beta * T2).reshape(W3.shape[0], -1), Sig))
    return T2, al, beta, tr


def probe_module(W_t, W_cur, T1, inv, Sig, c, alt, arms):
    M, K = W_t.shape
    nB, B = T1.shape[1], T1.shape[2]
    q = torch.argsort(inv)
    Wt_s = W_t[:, q]
    Sig_s = Sig[q][:, q] if not torch.equal(inv, torch.arange(K, device=inv.device)) else Sig
    W3 = Wt_s.reshape(M, nB, B)
    e0 = err_h(W_t - W_cur, Sig)
    rec = {"e0": e0, "arms": {}}
    states = {}

    def put(name, Ta, Tb, alpha, extra=None, beta=None):
        V = Ta + c * Tb
        Wq = (alpha * Ta + beta * Tb) if beta is not None else (alpha * V)
        e = err_h(Wt_s - Wq.reshape(M, K), Sig_s)
        d = {"e": e, "ratio": None, "absorb": 1 - e / max(e0, 1e-30),
             "flips_t1": float((Ta != T1).float().mean()),
             "t2_nz": float((Tb != 0).float().mean()),
             "zero_frac": float((V == 0).float().mean())}
        if extra:
            d.update(extra)
        rec["arms"][name] = d
        states[name] = {"T1": Ta.to(torch.int8).cpu(), "T2": Tb.to(torch.int8).cpu(), "a0": alpha.cpu(),
                        "t2beta": None if beta is None else beta.reshape(M, nB).cpu()}

    Z = torch.zeros_like(T1)
    al_c = scale_joint(W3, T1, Sig_s)
    put("ctrl", T1, Z, al_c)
    e_c = rec["arms"]["ctrl"]["e"]
    if "t2fix" in arms:
        al, T2 = al_c, Z
        tr = []
        for _ in range(alt):
            T2 = round_t2(W3, T1, al, c)
            al = scale_joint(W3, T1 + c * T2, Sig_s)
            tr.append(round(err_h(Wt_s - (al * (T1 + c * T2)).reshape(M, K), Sig_s) / max(e_c, 1e-30), 5))
        put("t2fix", T1, T2, al, {"trace_rel": tr})
    if "t2beta" in arms:
        T2b, al_b, beta, tr = fit_t2beta(W3, T1, al_c, Sig_s, alt)
        put("t2beta", T1, T2b, al_b, {"trace_rel": [round(t / max(e_c, 1e-30), 5) for t in tr],
                                     "beta_over_alpha_p50": round(float((beta / al_b).abs().median()), 4)}, beta=beta)
    if "t2gptq" in arms:
        _, Tg1, Tg2 = gptq_grid(Wt_s, Sig_s, 9, c=c)
        Tg1 = Tg1.reshape(M, nB, B)
        Tg2 = Tg2.reshape(M, nB, B)
        put("t2gptq", Tg1, Tg2, scale_joint(W3, Tg1 + c * Tg2, Sig_s))
    for a in rec["arms"].values():
        a["ratio"] = a["e"] / max(e_c, 1e-30)
    return rec, states


def compose_sel(prefix, arm, others_tag, name, families, layer_min):
    """E64 S2 通用重組:模組型態 ∈ families 且層 ≥ layer_min ← qs_{prefix}_{arm};其餘 ← others_tag。
    印所選權重占比(位元帳:所選 × 2.08 b/w,減去所選 r 0.0625)。"""
    dst = Path("data") / f"qs_{name}"
    dst.mkdir(parents=True, exist_ok=True)
    n_sel = n_ot = 0
    w_sel = w_tot = 0
    for li in range(N_LAYERS):
        a = torch.load(Path("data") / f"qs_{prefix}_{arm}" / f"layer{li:02d}.pt", weights_only=False)
        o = torch.load(Path("data") / others_tag / f"layer{li:02d}.pt", weights_only=False)
        ent = {}
        for n in o:
            w = o[n]["T1"].numel()
            w_tot += w
            if fam(n) in families and li >= layer_min:
                ent[n] = a[n]; n_sel += 1; w_sel += w
            else:
                ent[n] = o[n]; n_ot += 1
        torch.save(ent, dst / f"layer{li:02d}.pt")
    share = w_sel / w_tot
    print(f"COMPOSE_SEL {dst.name}: sel {n_sel} ← {arm} (families={families}, layer>={layer_min}), "
          f"others {n_ot} ← {others_tag}; weight_share={share:.4f} bits≈+{share*2.08:.3f}−{share*0.0625:.3f} b/w", flush=True)
    return share


def compose(prefix, arms, others):
    """混合 qs 態:down_proj ← qs_{prefix}_{arm};其他模組 ← others(qs 目錄名)。"""
    for arm in arms:
        for oth_tag, suffix in others.items():
            dst = Path("data") / f"qs_{prefix}_{arm}_dp{suffix}"
            dst.mkdir(parents=True, exist_ok=True)
            n_dp = n_ot = 0
            for li in range(N_LAYERS):
                a = torch.load(Path("data") / f"qs_{prefix}_{arm}" / f"layer{li:02d}.pt", weights_only=False)
                o = torch.load(Path("data") / oth_tag / f"layer{li:02d}.pt", weights_only=False)
                ent = {}
                for n in o:
                    if fam(n) == "down_proj":
                        ent[n] = a[n]; n_dp += 1
                    else:
                        ent[n] = o[n]; n_ot += 1
                torch.save(ent, dst / f"layer{li:02d}.pt")
            print(f"COMPOSE {dst.name}: down_proj {n_dp} ← {arm}, others {n_ot} ← {oth_tag}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-tag", default="st54r")
    ap.add_argument("--res-tag", default="res58r1")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--calib", default=str(EV / "calib_e58r1b_traj.pt"))
    ap.add_argument("--lengths", default=str(EV / "calib_e58r1b_len.pt"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--alt", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--layers", default="all")
    ap.add_argument("--save-arms", default="")
    ap.add_argument("--save-prefix", default="pt63")
    ap.add_argument("--out", default=str(EVC / "probe_t2.json"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--compose", action="store_true")
    ap.add_argument("--compose-families", default="",
                    help="E64 S2:通用重組——這些模組型態(逗號)且層 ≥ --compose-layer-min 取 t2 臂,其餘取 --compose-others")
    ap.add_argument("--compose-layer-min", type=int, default=0)
    ap.add_argument("--compose-arm", default="t2beta")
    ap.add_argument("--compose-others", default="qs_pa63_asymblkq2")
    ap.add_argument("--compose-name", default="pt64_t2beta_readers_asym")
    args = ap.parse_args()
    if args.compose:
        if args.compose_families:
            compose_sel(args.save_prefix, args.compose_arm, args.compose_others, args.compose_name,
                        [f for f in args.compose_families.split(",") if f], args.compose_layer_min)
        else:
            compose(args.save_prefix, ["t2beta", "t2gptq"],
                    {"qs_pa63_ctrl": "", "qs_pa63_asymblkq2": "_asym"})
        print("COMPOSE_DONE", flush=True)
        return
    if args.smoke:
        args.n_calib, args.alt, args.layers = 16, 2, "0,1"
        args.out = str(EVC / "probe_t2_smoke.json")
        args.save_prefix = "pt63smk"
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
            W_cur = dense_weight(c)
            water = (res[n]["B"].cuda().float() @ res[n]["A"].cuda().float()) / float(res[n]["r"])
            W_t = W_cur + water
            ktier.set_block(int(c["T1"].shape[2]))
            rec, states = probe_module(W_t, W_cur, c["T1"].cuda().float(), c["inv"].cuda(),
                                       Sig, float(c["c"]), args.alt, arms)
            rec["family"] = fam(n)
            rec["layer"] = li
            out["modules"][n] = rec
            for a in save_arms:
                if a in states:
                    s = states[a]
                    ent = {"T1": s["T1"], "T2": s["T2"], "inv": c["inv"].clone(),
                           "a0": s["a0"], "c": c["c"], "grid": 3 if a == "ctrl" else 9}
                    if s.get("t2beta") is not None:
                        ent["t2beta"] = s["t2beta"]
                    per_layer_states[a][n] = ent
            line.append(f"{fam(n)[:9]:9s} " + " ".join(
                f"{a}:{rec['arms'][a]['ratio']:.3f}" for a in arms if a in rec["arms"] and a != "ctrl")
                + (f" | fix_nz={rec['arms']['t2fix']['t2_nz']:.3f}" if 't2fix' in rec['arms'] else "")
                + (f" beta_nz={rec['arms']['t2beta']['t2_nz']:.3f} b/a={rec['arms']['t2beta']['beta_over_alpha_p50']}" if 't2beta' in rec['arms'] else "")
                + (f" gptq_flip1={rec['arms']['t2gptq']['flips_t1']:.3f}" if 't2gptq' in rec['arms'] else ""))
            del Sig, W_cur, water, W_t, states
        acc.H.clear()
        torch.cuda.empty_cache()
        for a in save_arms:
            torch.save(per_layer_states[a], save_dirs[a] / f"layer{li:02d}.pt")
        print(f"PROBE_T2_LAYER {li} ({time.time()-t0:.0f}s)\n   " + "\n   ".join(line), flush=True)
        Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))

    summ = {}
    for a in ALL_ARMS[1:]:
        rows = [(m["family"], m["arms"][a]) for m in out["modules"].values() if a in m["arms"]]
        if not rows:
            continue

        def agg(sel):
            rs = [d["ratio"] for f, d in rows if sel(f)]
            return {"n": len(rs), "ratio_mean": round(statistics.mean(rs), 4),
                    "ratio_median": round(statistics.median(rs), 4),
                    "absorb_mean": round(statistics.mean([d["absorb"] for f, d in rows if sel(f)]), 4),
                    "flips_t1_mean": round(statistics.mean([d["flips_t1"] for f, d in rows if sel(f)]), 5),
                    "t2_nz_mean": round(statistics.mean([d["t2_nz"] for f, d in rows if sel(f)]), 4)}
        summ[a] = {"all": agg(lambda f: True), "down_proj": agg(lambda f: f == "down_proj"),
                   "others": agg(lambda f: f != "down_proj")}
    summ["ctrl"] = {"absorb_mean": round(statistics.mean(
        [m["arms"]["ctrl"]["absorb"] for m in out["modules"].values()]), 4)}
    out["summary"] = summ
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("PROBE_T2_SUMMARY " + json.dumps(summ, ensure_ascii=False), flush=True)
    print(f"PROBE_T2_DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

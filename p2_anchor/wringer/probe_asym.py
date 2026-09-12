"""E63 S0:非對稱三元 零訓練探針(A4 平台:st54r 碼 + res58r1 水)。

問題:排水段要把水(W_t = W(st54r) + BA/r)壓回三元碼。多一個逐列自由度 r
(w = α(T⁺ − r T⁻),推論仍 3 符號)能多吸多少水?零訓練、Σ-加權配對比較。

臂(每模組,同一 W_t、同一 Σ):
  base       W_cur = dense(st54r),不動                          e0 = ‖water‖²_Σ
  ctrl       T 固定,α 閉式重擬合(對稱;排水起點的直投基線)
  asym       T 固定,(α, r) 交替閉式                                ← 主臂:純 DOF
  symrefit   T 逐元重投影 ↔ α 重擬合 交替(對稱,無誤差回饋)
  asymrefit  T 逐元非對稱重投影 ↔ (α, r) 交替                       ← DOF + 局部換碼
  symgptq    GPTQ 列序直解(對稱;E59 解法器)+ α 重擬合
  asymgptq   GPTQ 列序直解(非對稱門檻,r 取自 asym)+ (α, r) 交替      ← DOF + 搜索
  asymblk    T 固定,逐 (row,block) r_g 交替閉式                    ← 診斷上界(+0.5 b/w,不可交付)
  asymblkrefit 逐 block r + 逐元非對稱重投影                        ← 診斷上界
  asymblkq{1..4} 逐 block r 量化 b 位元(每模組 log 碼書)+ α 重擬合   ← 位元帳 b/32 b/w 的可交付候選
判讀:ratio = e_arm / e_ctrl(<1 好);absorb = 1 − e_arm/e0;flips vs st54r;r 分佈。
局部指標僅診斷(第 11 型);官方三科(export 各臂 qs 態)才是判官,點火需用戶核准。
Σ 於 st54r 部署形態單趟量測,各臂不前傳(純局部配對;與逐窗鮮 H 的差異入卷)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.probe_asym --smoke
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.probe_asym \
      --n-calib 128 --batch 4 --save-arms ctrl,asym,asymrefit --save-prefix pa63
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from p2_anchor.wringer import ktier
from p2_anchor.wringer.asym import (asym_V, dense_asym, err_h, fit_alpha_r,
                                      gptq_ternary, quantize_r, r_stats, round_ternary)
from p2_anchor.wringer.comp import HAccum
from p2_anchor.wringer.engine import capture_layer0, fwd_chain
from p2_anchor.wringer.ktier import init_alpha, scale_joint
from p2_anchor.wringer.modelio import (LAYER_PREFIX, N_LAYERS, cov_source_for,
                                         enumerate_targets, get_module,
                                         layer_index, load_model)
from p2_anchor.wringer.state import dense_weight, load_state

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"
ALL_ARMS = ("ctrl", "asym", "symrefit", "asymrefit", "symgptq", "asymgptq", "asymblk", "asymblkrefit", "asymblkq")
Q_BITS = (1, 2, 3, 4)


def fam(n):
    return n.split(".")[-1]


@torch.no_grad()
def probe_module(W_t, W_cur, T3, inv, Sig, alt, arms):
    """W_t/W_cur (M,K) 模型欄序;T3 (M,nB,B) float;Sig (K,K) 模型欄序。回傳 (rec, states)。"""
    M, K = W_t.shape
    nB, B = T3.shape[1], T3.shape[2]
    q = torch.argsort(inv)                        # 模型欄序 → 儲存欄序
    Wt_s = W_t[:, q]
    Sig_s = Sig[q][:, q] if not torch.equal(inv, torch.arange(K, device=inv.device)) else Sig
    W3 = Wt_s.reshape(M, nB, B)
    e0 = err_h(W_t - W_cur, Sig)
    rec = {"e0": e0, "arms": {}}
    states = {}

    def put(name, T, alpha, r, extra=None, cb=None):
        e = err_h(Wt_s - dense_asym(T, alpha, r), Sig_s)
        d = {"e": e, "ratio": None, "absorb": 1 - e / max(e0, 1e-30),
             "flips": float((T != T3).float().mean()),
             "zero_frac": float((T == 0).float().mean())}
        if r is not None:
            d["r"] = r_stats(r)
        if extra:
            d.update(extra)
        rec["arms"][name] = d
        if r is None or r.shape[1] == 1 or cb is not None:   # 全精度逐 block r 為診斷臂,不成 qs 態
            states[name] = {"T1": T.to(torch.int8).cpu(), "a0": alpha.cpu(),
                            "rneg": None if r is None else r.reshape(M, -1).cpu(),
                            "rneg_cb": None if cb is None else cb.cpu()}

    al_c, _, tr_c = fit_alpha_r(W3, T3, Sig_s, asym=False)
    put("ctrl", T3, al_c, None)
    e_c = rec["arms"]["ctrl"]["e"]
    if "asym" in arms:
        al_a, r_a, tr_a = fit_alpha_r(W3, T3, Sig_s, alt=alt)
        put("asym", T3, al_a, r_a, {"trace_rel": [round(t / max(e_c, 1e-30), 5) for t in tr_a]})
    else:
        al_a, r_a = al_c, None
    if "symrefit" in arms:
        T, al = T3, al_c
        for _ in range(alt):
            T = round_ternary(W3, al)
            al = scale_joint(W3, T, Sig_s)
        put("symrefit", T, al, None)
    if "asymrefit" in arms and r_a is not None:
        T, al, r = T3, al_a, r_a
        for _ in range(alt):
            T = round_ternary(W3, al, r)
            al, r, _ = fit_alpha_r(W3, T, Sig_s, alt=1, r0=r)
        put("asymrefit", T, al, r)
    if "symgptq" in arms or "asymgptq" in arms:
        a0 = init_alpha(Wt_s)
        if "symgptq" in arms:
            Tg = gptq_ternary(Wt_s, Sig_s, a0, None).reshape(M, nB, B)
            put("symgptq", Tg, scale_joint(W3, Tg, Sig_s), None)
        if "asymgptq" in arms and r_a is not None:
            Tg = gptq_ternary(Wt_s, Sig_s, a0, r_a).reshape(M, nB, B)
            al, r, _ = fit_alpha_r(W3, Tg, Sig_s, alt=alt, r0=r_a)
            put("asymgptq", Tg, al, r)
    if "asymblk" in arms:                      # 診斷上界:逐 block r(+0.5 b/w,不可交付)
        al_b, r_b, tr_b = fit_alpha_r(W3, T3, Sig_s, alt=alt, blk=True)
        put("asymblk", T3, al_b, r_b, {"trace_rel": [round(t / max(e_c, 1e-30), 5) for t in tr_b]})
        if "asymblkrefit" in arms:
            T, al, r = T3, al_b, r_b
            for _ in range(alt):
                T = round_ternary(W3, al, r)
                al, r, _ = fit_alpha_r(W3, T, Sig_s, alt=1, r0=r, blk=True)
            put("asymblkrefit", T, al, r)
        if "asymblkq" in arms:                 # 逐 block r 量化到 b 位元(碼書每模組)+ α 閉式重擬合
            for b in Q_BITS:
                rq, cb = quantize_r(r_b, b)
                al_q = scale_joint(W3, asym_V(T3, rq), Sig_s)
                put(f"asymblkq{b}", T3, al_q, rq, {"codebook": [round(float(v), 4) for v in cb]}, cb=cb)
    for a in rec["arms"].values():
        a["ratio"] = a["e"] / max(e_c, 1e-30)
    return rec, states


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
    ap.add_argument("--save-prefix", default="pa63")
    ap.add_argument("--out", default=str(EVC / "probe_asym.json"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_calib, args.alt, args.layers = 16, 2, "0,1"
        args.out = str(EVC / "probe_asym_smoke.json")
        args.save_prefix = "pa63smk"
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
    for n in targets:                                   # 部署形態 = 排水起點
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
                                       Sig, args.alt, arms)
            rec["family"] = fam(n)
            rec["layer"] = li
            out["modules"][n] = rec
            for a in save_arms:
                if a in states:
                    s = states[a]
                    ent = {"T1": s["T1"], "T2": torch.zeros_like(s["T1"]), "inv": c["inv"].clone(),
                           "a0": s["a0"], "c": c["c"], "grid": 3}
                    if s["rneg"] is not None:
                        ent["rneg"] = s["rneg"]
                    if s.get("rneg_cb") is not None:
                        ent["rneg_cb"] = s["rneg_cb"]
                    per_layer_states[a][n] = ent
            line.append(f"{fam(n)[:9]:9s} " + " ".join(
                f"{a[:5]}:{rec['arms'][a]['ratio']:.3f}" for a in arms if a in rec["arms"] and a != "ctrl")
                + (f" | r p10/p90={rec['arms']['asym']['r']['p10']}/{rec['arms']['asym']['r']['p90']}" if 'asym' in rec['arms'] else "")
                + (f" blk={rec['arms']['asymblk']['r']['p10']}/{rec['arms']['asymblk']['r']['p90']}" if 'asymblk' in rec['arms'] else "")
                + (" q1-4=" + "/".join(f"{rec['arms'][f'asymblkq{b}']['ratio']:.3f}" for b in Q_BITS) if 'asymblkq1' in rec['arms'] else ""))
            del Sig, W_cur, water, W_t, states
        acc.H.clear()
        torch.cuda.empty_cache()
        for a in save_arms:
            torch.save(per_layer_states[a], save_dirs[a] / f"layer{li:02d}.pt")
        print(f"PROBE_ASYM_LAYER {li} ({time.time()-t0:.0f}s)\n   " + "\n   ".join(line), flush=True)
        Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))

    # ---- 摘要:各臂對 ctrl 比值(全體 / down_proj / 其他),吸水率 ----
    summ = {}
    arm_names = sorted({a for m in out["modules"].values() for a in m["arms"]} - {"ctrl"},
                       key=lambda a: (ALL_ARMS.index(a.rstrip("1234")) if a.rstrip("1234") in ALL_ARMS else 99, a))
    for a in arm_names:
        rows = [(m["family"], m["arms"][a]) for m in out["modules"].values() if a in m["arms"]]
        if not rows:
            continue

        def agg(sel):
            rs = [d["ratio"] for f, d in rows if sel(f)]
            ab = [d["absorb"] for f, d in rows if sel(f)]
            fl = [d["flips"] for f, d in rows if sel(f)]
            return {"n": len(rs), "ratio_mean": round(statistics.mean(rs), 4),
                    "ratio_median": round(statistics.median(rs), 4),
                    "absorb_mean": round(statistics.mean(ab), 4),
                    "flips_mean": round(statistics.mean(fl), 5)}
        summ[a] = {"all": agg(lambda f: True), "down_proj": agg(lambda f: f == "down_proj"),
                   "others": agg(lambda f: f != "down_proj")}
    ctrl_abs = [m["arms"]["ctrl"]["absorb"] for m in out["modules"].values()]
    summ["ctrl"] = {"absorb_mean": round(statistics.mean(ctrl_abs), 4)}
    out["summary"] = summ
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("PROBE_ASYM_SUMMARY " + json.dumps(summ, ensure_ascii=False), flush=True)
    print(f"PROBE_ASYM_DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

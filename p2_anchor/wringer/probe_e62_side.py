"""E62 第二探針:M 放哪一側 + L22 殘水>原水 解剖 + 全層激活漂移(污染判定)。

(a) 激活漂移:同一 calib,量每層輸入 X_drained(st61c 全材化)vs X_ideal(st54r 碼 + 滿水 BA/r 稠密)
    rel = ‖X_d − X_i‖/‖X_i‖,32 層曲線。排水的順序語義下,深層碼要補償上游漂移,
    這部分不是「水」,局部 W 殘差看不到 → 殘水>原水 的合法來源。
(b) 殘水解剖(逐模組):drift = C·X_cur − W0q(碼實際位移),water = BA/r。
    ‖resid‖² = ‖water‖² + ‖drift‖² − 2⟨water,drift⟩_H。報 ‖drift‖/‖water‖、cos_H(water,drift)、
    重解 M 後同三量 → 把「殘水>原水」拆成 M 滯後(重解可收)/ 碼漂移非水向(補償或污染)。
(c) 輸出側混合:W_t ≈ L·C·X,L 為列塊對角(塊 128,Mr/128 塊)。解 L(固定 Z=C X):
    mask⊙(L Z H Zᵀ) = mask⊙(W_t H Zᵀ)。比較 X 重解 / L 單獨 / L+X 交替 2 輪 / Monarch。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.probe_e62_side --layers 2,12,16,20,22,24,28,30
"""
import argparse
import json
import time
from pathlib import Path

import torch

from p2_anchor.wringer.cov import collect_cov
from p2_anchor.wringer.ktier import k2_weight
from p2_anchor.wringer.modelio import (LAYER_PREFIX, N_LAYERS, cov_source_for,
                                          enumerate_targets, get_module, load_model)
from p2_anchor.wringer.probe_e62_cap import (bd_mask, obj, pos_mask, quantize_delta,
                                               solve_masked)
from p2_anchor.wringer.state import apply_k2, dense_weight, load_state, mix_dense

EV = Path("evidence/p1_grouping")


@torch.no_grad()
def solve_left(Z, W_t, H, L0, mask, iters=60, tol=1e-4):
    """min_L ‖W_t − L Z‖_H²,L∈mask。G = Z H Zᵀ (Mr,Mr);A(L) = (L G)⊙mask。"""
    G = Z @ H @ Z.T
    R = (W_t @ H) @ Z.T
    mk = mask.float()
    A = lambda L: (L @ G) * mk
    rhs = R * mk
    L = L0 * mk
    r = rhs - A(L)
    p = r.clone()
    rs = float((r * r).sum())
    r0 = rs ** 0.5
    it = 0
    for it in range(iters):
        Ap = A(p)
        a = rs / max(float((p * Ap).sum()), 1e-30)
        L = L + a * p
        r = r - a * Ap
        rs_new = float((r * r).sum())
        if rs_new ** 0.5 <= tol * max(r0, 1e-30):
            break
        p = r + (rs_new / max(rs, 1e-30)) * p
        rs = rs_new
    return L, {"cg_iters": it + 1, "cg_rel": round((rs ** 0.5) / max(r0, 1e-30), 6)}


@torch.no_grad()
def act_drift(model, st_ideal_w, st_drained_w, targets, calib, batch=2):
    """兩趟前向:ideal 權重 → 存各層輸入(CPU bf16);drained 權重 → 對比。回傳 32 層 rel 曲線。"""
    store = {}
    cur = {"mode": None, "num": torch.zeros(N_LAYERS), "den": torch.zeros(N_LAYERS)}

    def mk(li):
        def hook(module, args, kwargs=None):
            x = args[0] if args else kwargs["hidden_states"]
            if cur["mode"] == "ideal":
                store[(li, cur["bi"])] = x.detach().to("cpu", torch.bfloat16)
            else:
                xi = store[(li, cur["bi"])].cuda().float()
                d = (x.float() - xi)
                cur["num"][li] += float((d * d).sum())
                cur["den"][li] += float((xi * xi).sum())
        return hook
    hooks = [model.get_submodule(f"{LAYER_PREFIX}.{li}").register_forward_pre_hook(mk(li), with_kwargs=True)
             for li in range(N_LAYERS)]
    try:
        for mode, weights in (("ideal", st_ideal_w), ("drained", st_drained_w)):
            for n in targets:
                get_module(model, n).weight.data.copy_(weights[n])
            cur["mode"] = mode
            for bi in range(0, calib.shape[0], batch):
                cur["bi"] = bi
                model(input_ids=calib[bi:bi + batch].cuda(), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return [round(float((cur["num"][li] / cur["den"][li].clamp_min(1e-30)).sqrt()), 5) for li in range(N_LAYERS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-tag", default="st61c")
    ap.add_argument("--start-tag", default="st54r")
    ap.add_argument("--res-tag", default="res58r1")
    ap.add_argument("--layers", default="2,12,16,20,22,24,28,30")
    ap.add_argument("--n-calib", type=int, default=32)
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--skip-drift", action="store_true")
    ap.add_argument("--out", default=str(EV / "corkscrew/probe_e62_side.json"))
    args = ap.parse_args()
    t0 = time.time()

    model, _ = load_model()
    targets = enumerate_targets(model)
    st = load_state(args.state_tag, targets)
    st0 = load_state(args.start_tag, targets)
    res = {}
    for li in range(N_LAYERS):
        p = Path("data") / f"res_{args.res_tag}" / f"layer{li:02d}.pt"
        if p.exists():
            res.update(torch.load(p, map_location="cpu", weights_only=False))
    calib = torch.load(EV / "calib_e58r1b_traj.pt", weights_only=False)
    lens = torch.load(EV / "calib_e58r1b_len.pt", weights_only=False)
    rows = [i for i in range(calib.shape[0]) if int(lens[i]) >= args.seq][:args.n_calib]
    calib = calib[rows, :args.seq]
    out = {"state": args.state_tag, "calib": {"rows": len(rows), "seq": args.seq}, "modules": {}}
    print(f"calib {calib.shape}  load {time.time()-t0:.0f}s", flush=True)

    def water_of(n):
        return (res[n]["B"].cuda().float() @ res[n]["A"].cuda().float()) / float(res[n]["r"])

    # ---- (a) 激活漂移 ----
    if not args.skip_drift:
        ta = time.time()
        w_ideal = {n: (dense_weight(st0[n]) + water_of(n)).to(torch.bfloat16) for n in targets}
        w_drain = {n: dense_weight(st[n]).to(torch.bfloat16) for n in targets}
        drift = act_drift(model, w_ideal, w_drain, targets, calib[:8], batch=2)
        del w_ideal
        out["act_drift_rel"] = drift
        print("PROBE_SIDE_DRIFT " + " ".join(f"L{li}:{v:.4f}" for li, v in enumerate(drift)), flush=True)
        print(f"drift pass {time.time()-ta:.0f}s", flush=True)
        for n in targets:                        # 恢復排水態
            get_module(model, n).weight.data.copy_(w_drain[n])
        del w_drain
    else:
        apply_k2(model, st)
    torch.cuda.empty_cache()

    # ---- (b)(c) 逐模組 ----
    for li in [int(x) for x in args.layers.split(",")]:
        mods = [f"{LAYER_PREFIX}.{li}.mlp.down_proj"]
        if li in (12, 22):
            mods.append(f"{LAYER_PREFIX}.{li}.linear_attn.in_proj_qkv")
        mods = [n for n in mods if n in targets]
        reps = sorted({cov_source_for(n) for n in mods})
        Sig = collect_cov(model, reps, f"{LAYER_PREFIX}.{li}", calib, batch=2)
        for n in mods:
            tm = time.time()
            H = Sig[cov_source_for(n)].float()
            H = H / H.diagonal().mean()
            e = st[n]
            C = k2_weight(e["T1"].cuda(), e["T2"].cuda(), e["a0"].cuda().float(), e["c"])[:, e["inv"].cuda()].float()
            X_cur = mix_dense(e["mix"].cuda())
            W0q = dense_weight(st0[n])
            water = water_of(n)
            W_t = W0q + water
            Mr, K = C.shape
            I = torch.eye(K, device="cuda")
            IL = torch.eye(Mr, device="cuda")
            hn = lambda A: float(((A @ H) * A).sum())
            hin = lambda A, B: float(((A @ H) * B).sum())
            wH = hn(water)
            r = {"shape": [Mr, K], "water_H": wH}

            def anatomy(tag, Xs, Ls=None):
                Zd = C @ Xs if Ls is None else Ls @ (C @ Xs)
                drift = Zd - W0q
                resid = W_t - Zd
                dH = hn(drift)
                r[tag] = {"resid_over_water": round(hn(resid) / wH, 4),
                          "drift_over_water": round((dH / wH) ** 0.5, 4),
                          "cos_water_drift": round(hin(water, drift) / max((wH * dH) ** 0.5, 1e-30), 4)}
                return hn(resid)
            o_cur = anatomy("cur", X_cur)
            o_I = hn(W_t - C)
            r["cur_gain_vs_I"] = round(1 - o_cur / o_I, 4)
            m128 = bd_mask(K, 128)
            mL = bd_mask(Mr, 128)
            # X 重解
            Xr, sx = solve_masked(C, W_t, H, X_cur, m128, 0.0, iters=args.iters)
            o_x = anatomy("resolve_X", Xr)
            # L 單獨(X=X_cur)
            L1, sl = solve_left(C @ X_cur, W_t, H, IL, mL, iters=args.iters)
            o_l = anatomy("L_only", X_cur, L1)
            # L 單獨(X=I,無輸入側混合)
            L0, _ = solve_left(C, W_t, H, IL, mL, iters=args.iters)
            o_l0 = anatomy("L_only_XI", I, L0)
            # L + X 交替 2 輪(起點 X 重解)
            Xa, La = Xr.clone(), IL.clone()
            for _ in range(2):
                La, _ = solve_left(C @ Xa, W_t, H, La, mL, iters=args.iters)
                # 固定 L:min ‖W_t − L C X‖_H = 以 C' = L C 解 X
                Xa, _ = solve_masked(La @ C, W_t, H, Xa, m128, 0.0, iters=args.iters)
            o_lx = anatomy("L_plus_X", Xa, La)
            # Monarch 對照(輸入側)
            mF = pos_mask(K, 128)
            X1, F = X_cur.clone(), I.clone()
            for _ in range(2):
                F, _ = solve_masked(C @ X1, W_t, H, F, mF, 0.0, iters=args.iters)
                Wf = torch.linalg.solve(F.T, W_t.T).T
                X1, _ = solve_masked(C, Wf, F @ H @ F.T, X1, m128, 0.0, iters=args.iters)
            o_mo = anatomy("monarch", X1 @ F)
            # int8 量化保留(L 與 X 各自)
            Lq = quantize_delta(La, mL, "int8"); Xq = quantize_delta(Xa, m128, "int8")
            o_lxq = hn(W_t - Lq @ (C @ Xq))
            g = lambda o: round(1 - o / o_cur, 4)
            r["gain_vs_cur"] = {"resolve_X": g(o_x), "L_only": g(o_l), "L_only_XI": g(o_l0),
                                "L_plus_X": g(o_lx), "monarch": g(o_mo), "L_plus_X_int8": g(o_lxq)}
            r["params"] = {"X_bd128": int(m128.sum()), "L_bd128": int(mL.sum()), "monarch_F": int(mF.sum())}
            r["bw_bf16"] = {"X": round(int(m128.sum()) * 16 / (Mr * K), 4), "L": round(int(mL.sum()) * 16 / (Mr * K), 4)}
            r["cg"] = {"X": sx, "L": sl}
            out["modules"][n] = r
            k = n.split('layers.')[1]
            print(f"PROBE_SIDE {k} K={K} resid/water cur={r['cur']['resid_over_water']} drift/water={r['cur']['drift_over_water']} cos={r['cur']['cos_water_drift']} "
                  f"| after X-resolve resid/water={r['resolve_X']['resid_over_water']} cos={r['resolve_X']['cos_water_drift']} "
                  f"| gain: X={r['gain_vs_cur']['resolve_X']} L={r['gain_vs_cur']['L_only']} L(XI)={r['gain_vs_cur']['L_only_XI']} L+X={r['gain_vs_cur']['L_plus_X']} monarch={r['gain_vs_cur']['monarch']} L+X_int8={r['gain_vs_cur']['L_plus_X_int8']} "
                  f"| bw L={r['bw_bf16']['L']} X={r['bw_bf16']['X']} ({time.time()-tm:.0f}s)", flush=True)
            json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
            del C, X_cur, W0q, water, W_t, Xr, L1, L0, Xa, La, X1, F, Lq, Xq
            torch.cuda.empty_cache()
    out["elapsed_s"] = round(time.time() - t0)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("PROBE_SIDE_DONE", flush=True)


if __name__ == "__main__":
    main()

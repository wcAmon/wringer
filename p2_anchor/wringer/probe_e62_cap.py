"""E62 零訓練容量階梯探針:M 的桶是「結構限制」還是「求解器自限」?

固定 st61c 終態碼 C=α⊙V(模型欄序),錨定目標 W_t = W0q(st54r) + BA/r(res58r1,γ=0 滿水),
度量 H = E[xxᵀ](st61c 全模型材化後、順序語義收集)。對 X 求
    min_X ‖W_t − C X‖_H² + λ s ‖X − X0‖²,  X ∈ 遮罩子空間(mask ⊙ X)
共軛梯度解法方程 mask⊙(CᵀC X H) + λ s X = mask⊙(CᵀW_t H) + λ s X0。

軸一(求解器):bd128,從 X_cur 出發,ridge ∈ {1, 0.3, 0.1, 0.01, 0}。
軸二(結構,ridge 0 = 容量上界):bd128 / bd256 / bd512 / dense。
雙層形式(乘積 M = Bd128(M1)·F,交替 LS 2 輪):
    2L-bfly:F = 256 塊內「同位耦合」(i%128==j%128 且同 256 塊)——蝶形一級,參數 2K。
    2L-monarch:F = 跨所有塊同位耦合(i%128==j%128)——Monarch 第二因子,參數 K²/128。
壓回三元:每個解 X = I + Δ,Δ 逐列 TWN 三元化 / int8,量 gain 保留率;位元帳 b/w = params×bits/(M·K)。
旋轉份額:極分解 Q 能量 / ‖X−I‖²(bd 逐塊 SVD;dense/2L 全 SVD)。

判讀規則(先寫死,見 RULES):
  R1 結構綁定:gain_dense(r0) − gain_bd128(r0) ≥ 0.25 → 擴結構;2L 值得 = gain_2L ≥ 0.8·gain_bd256 且成本 < bd256。
  R2 求解器自限:gain_bd128(r0) − gain_bd128(r1.0) ≥ 0.15 → ridge 是免費增益,E62 先調 ridge。
  R3 壓回三元:三元化保留率 ≥ 0.7 → 可壓回;否則 M 需 int8/bf16(入位元帳)。
  R4 桶非瓶頸:gain_dense(r0) < 0.4 → 碼軸主導,不擴桶。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.probe_e62_cap --layers 2,12,22,30
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from p2_anchor.wringer.cov import collect_cov
from p2_anchor.wringer.ktier import k2_weight
from p2_anchor.wringer.modelio import (LAYER_PREFIX, cov_source_for,
                                          enumerate_targets, load_model)
from p2_anchor.wringer.state import apply_k2, dense_weight, load_state, mix_dense

EV = Path("evidence/p1_grouping")
RULES = {"R1_struct_bound": "gain_dense_r0 - gain_bd128_r0 >= 0.25",
         "R1_2L_worth": "gain_2L >= 0.8*gain_bd256_r0 and params_2L < params_bd256",
         "R2_solver_bound": "gain_bd128_r0 - gain_bd128_r1 >= 0.15",
         "R3_ternary_ok": "ternary retention >= 0.7",
         "R4_not_bucket": "gain_dense_r0 < 0.4"}


def bd_mask(K, b):
    idx = torch.arange(K, device="cuda") // b
    return idx[:, None] == idx[None, :]


def pos_mask(K, b, within=None):
    """同位耦合 i%b==j%b;within=w 時再限同 w 塊。"""
    p = torch.arange(K, device="cuda")
    m = (p[:, None] % b) == (p[None, :] % b)
    if within:
        m &= (p[:, None] // within) == (p[None, :] // within)
    return m


@torch.no_grad()
def solve_masked(C, W_t, H, X0, mask, ridge, iters=60, tol=1e-4):
    CtC = C.T @ C
    lam = ridge * float(CtC.diagonal().mean() * H.diagonal().mean())
    mk = mask.float()
    CtW_H = (C.T @ W_t) @ H

    def A(X):
        return (CtC @ X @ H) * mk + lam * X

    rhs = CtW_H * mk + lam * X0
    X = X0 * mk
    r = rhs - A(X)
    p = r.clone()
    rs = float((r * r).sum())
    r0 = rs ** 0.5
    it = 0
    for it in range(iters):
        Ap = A(p)
        a = rs / max(float((p * Ap).sum()), 1e-30)
        X = X + a * p
        r = r - a * Ap
        rs_new = float((r * r).sum())
        if rs_new ** 0.5 <= tol * max(r0, 1e-30):
            break
        p = r + (rs_new / max(rs, 1e-30)) * p
        rs = rs_new
    return X, {"cg_iters": it + 1, "cg_rel": round((rs ** 0.5) / max(r0, 1e-30), 6)}


def obj(C, W_t, H, X):
    E = W_t - C @ X
    return float(((E @ H) * E).sum())


@torch.no_grad()
def rot_share(X, b=None):
    """極分解 X=QP;回傳 (‖Q−I‖²/‖X−I‖², 平均旋轉角 deg, sv_min)。"""
    K = X.shape[0]
    I = torch.eye(K, device=X.device)
    if b:
        Xb = X.reshape(K // b, b, K // b, b)
        idx = torch.arange(K // b, device=X.device)
        Xb = Xb[idx, :, idx, :]
        U, S, Vh = torch.linalg.svd(Xb)
        Q = torch.block_diag(*(U @ Vh).unbind(0))
    else:
        U, S, Vh = torch.linalg.svd(X)
        Q = U @ Vh
    dI = float((X - I).norm() ** 2)
    dQ = float((Q - I).norm() ** 2)
    ang = math.degrees(2 * math.asin(min(1.0, (dQ / (2 * K)) ** 0.5 / 1.0)))
    return round(dQ / max(dI, 1e-30), 4), round(ang, 3), round(float(S.min()), 4)


@torch.no_grad()
def quantize_delta(X, mask, mode):
    """X = I + Δ;Δ 只在 mask 上。ternary:逐列 TWN(thr 0.7·mean|Δ|,α=mean|Δ|>thr);int8:逐列 absmax。"""
    K = X.shape[0]
    I = torch.eye(K, device=X.device)
    D = (X - I) * mask.float()
    if mode in ("int8", "int4"):
        s = D.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / (127.0 if mode == "int8" else 7.0)
        Dq = torch.round(D / s) * s
    else:
        cnt = mask.float().sum(1, keepdim=True).clamp_min(1)
        mean_abs = D.abs().sum(1, keepdim=True) / cnt
        thr = 0.7 * mean_abs
        sel = (D.abs() > thr) & mask
        alpha = (D.abs() * sel).sum(1, keepdim=True) / sel.float().sum(1, keepdim=True).clamp_min(1)
        Dq = alpha * torch.sign(D) * sel.float()
    return I + Dq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-tag", default="st61c")
    ap.add_argument("--start-tag", default="st54r")
    ap.add_argument("--res-tag", default="res58r1")
    ap.add_argument("--layers", default="2,12,22,30")
    ap.add_argument("--n-calib", type=int, default=32)
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--out", default=str(EV / "corkscrew/probe_e62_cap.json"))
    args = ap.parse_args()
    t0 = time.time()

    model, _ = load_model()
    targets = enumerate_targets(model)
    st = load_state(args.state_tag, targets)
    st0 = load_state(args.start_tag, targets)
    apply_k2(model, st)                           # 全模型 st61c 部署態(順序語義)
    res = {}
    for li in range(32):
        p = Path("data") / f"res_{args.res_tag}" / f"layer{li:02d}.pt"
        if p.exists():
            res.update(torch.load(p, map_location="cpu", weights_only=False))
    calib = torch.load(EV / "calib_e58r1b_traj.pt", weights_only=False)
    lens = torch.load(EV / "calib_e58r1b_len.pt", weights_only=False)
    rows = [i for i in range(calib.shape[0]) if int(lens[i]) >= args.seq][:args.n_calib]
    calib = calib[rows, :args.seq]
    print(f"calib {calib.shape} (rows len≥{args.seq}) targets {len(targets)}  load {time.time()-t0:.0f}s", flush=True)

    pick_suf = ["linear_attn.in_proj_qkv", "self_attn.q_proj", "mlp.down_proj", "mlp.gate_proj"]
    out = {"rules": RULES, "state": args.state_tag, "start": args.start_tag, "res": args.res_tag,
           "calib": {"rows": len(rows), "seq": args.seq}, "modules": {}}
    for li in [int(x) for x in args.layers.split(",")]:
        mods = [f"{LAYER_PREFIX}.{li}.{s}" for s in pick_suf if f"{LAYER_PREFIX}.{li}.{s}" in targets]
        reps = sorted({cov_source_for(n) for n in mods})
        tc = time.time()
        Sig = collect_cov(model, reps, f"{LAYER_PREFIX}.{li}", calib, batch=2)
        print(f"layer {li}: H for {reps} ({time.time()-tc:.0f}s)", flush=True)
        for n in mods:
            tm = time.time()
            H = Sig[cov_source_for(n)].float()
            H = H / H.diagonal().mean()
            e = st[n]
            C = k2_weight(e["T1"].cuda(), e["T2"].cuda(), e["a0"].cuda().float(), e["c"])[:, e["inv"].cuda()].float()
            X_cur = mix_dense(e["mix"].cuda())
            W0q = dense_weight(st0[n])
            water = (res[n]["B"].cuda().float() @ res[n]["A"].cuda().float()) / float(res[n]["r"])
            W_t = W0q + water
            Mr, K = C.shape
            I = torch.eye(K, device="cuda")
            o_I, o_cur = obj(C, W_t, H, I), obj(C, W_t, H, X_cur)
            water_h = float(((water @ H) * water).sum())
            r = {"shape": [Mr, K], "obj_I": o_I, "obj_cur": o_cur, "water_H": water_h,
                 "cur_gain_vs_I": round(1 - o_cur / o_I, 4),
                 "resid_over_water_cur": round(o_cur / water_h, 4), "sol": {}}
            m128 = bd_mask(K, 128)

            def rec(key, X, mask, ridge, stats, params):
                o = obj(C, W_t, H, X)
                d = {"ridge": ridge, "obj": o, "gain_vs_cur": round(1 - o / o_cur, 4),
                     "gain_vs_I": round(1 - o / o_I, 4), "params": params,
                     "bw_bf16": round(params * 16 / (Mr * K), 4),
                     "bw_int8": round(params * 8 / (Mr * K), 4),
                     "bw_ter": round(params * 1.6 / (Mr * K), 4),
                     "dist_I": round(float((X - I).norm() / math.sqrt(K)), 4)}
                d.update(stats)
                for q in ("ternary", "int4", "int8"):
                    oq = obj(C, W_t, H, quantize_delta(X, mask, q))
                    d[f"gain_vs_cur_{q}"] = round(1 - oq / o_cur, 4)
                    d[f"retention_{q}"] = round((1 - oq / o_cur) / max(1 - o / o_cur, 1e-9), 3)
                r["sol"][key] = d
                return d

            # 軸一:bd128 ridge 掃(從 X_cur)
            for rg in (1.0, 0.3, 0.1, 0.01, 0.0):
                X, s_ = solve_masked(C, W_t, H, X_cur, m128, rg, iters=args.iters)
                d = rec(f"bd128_r{rg}", X, m128, rg, s_, int(m128.sum()))
                d["rot_share"], d["rot_deg"], d["sv_min"] = rot_share(X, 128)
            # 軸二:結構階梯(ridge 0,從 X_cur)
            for b in (256, 512):
                mb = bd_mask(K, b)
                X, s_ = solve_masked(C, W_t, H, X_cur, mb, 0.0, iters=args.iters)
                d = rec(f"bd{b}_r0", X, mb, 0.0, s_, int(mb.sum()))
                d["rot_share"], d["rot_deg"], d["sv_min"] = rot_share(X, b)
            md = torch.ones(K, K, dtype=torch.bool, device="cuda")
            X, s_ = solve_masked(C, W_t, H, X_cur, md, 0.0, iters=args.iters)
            d = rec("dense_r0", X, md, 0.0, s_, K * K)
            d["rot_share"], d["rot_deg"], d["sv_min"] = rot_share(X)
            # 雙層乘積:M = X1 · F,交替 2 輪(ridge 0)
            for key, mF in (("2L_bfly", pos_mask(K, 128, within=256)), ("2L_monarch", pos_mask(K, 128))):
                X1, F = X_cur.clone(), I.clone()
                for _ in range(2):
                    Ce = C @ X1
                    F, sF = solve_masked(Ce, W_t, H, F, mF, 0.0, iters=args.iters)
                    Wf = torch.linalg.solve(F.T, W_t.T).T          # W_t F⁻¹
                    Hf = F @ H @ F.T
                    X1, s1 = solve_masked(C, Wf, Hf, X1, m128, 0.0, iters=args.iters)
                Xp = X1 @ F
                params = int(m128.sum()) + int(mF.sum())
                mprod = (Xp - I).abs() > 0                         # 乘積遮罩(量化用):近似取非零
                d = rec(key, Xp, mprod, 0.0, {"cg_iters_F": sF["cg_iters"], "cg_iters_X1": s1["cg_iters"]}, params)
                d["rot_share"], d["rot_deg"], d["sv_min"] = rot_share(Xp)
                d["params_F"] = int(mF.sum())
                # 分件量化:X1 與 F 各自三元化再相乘(部署形態)
                Xq = quantize_delta(X1, m128, "ternary") @ quantize_delta(F, mF, "ternary")
                oq = obj(C, W_t, H, Xq)
                d["gain_vs_cur_ternary_split"] = round(1 - oq / o_cur, 4)
                d["retention_ternary_split"] = round((1 - oq / o_cur) / max(d["gain_vs_cur"], 1e-9), 3)
            out["modules"][n] = r
            s = r["sol"]
            print(f"PROBE_E62 {n.split('layers.')[1]} K={K} cur_gain_vs_I={r['cur_gain_vs_I']} "
                  f"bd128 r1={s['bd128_r1.0']['gain_vs_cur']} r0.1={s['bd128_r0.1']['gain_vs_cur']} r0={s['bd128_r0.0']['gain_vs_cur']} | "
                  f"bd256={s['bd256_r0']['gain_vs_cur']} bd512={s['bd512_r0']['gain_vs_cur']} dense={s['dense_r0']['gain_vs_cur']} | "
                  f"bfly={s['2L_bfly']['gain_vs_cur']}({s['2L_bfly']['params']}) monarch={s['2L_monarch']['gain_vs_cur']}({s['2L_monarch']['params']}) | "
                  f"ter_ret bd128r0={s['bd128_r0.0']['retention_ternary']} dense={s['dense_r0']['retention_ternary']} monarch_split={s['2L_monarch']['retention_ternary_split']} "
                  f"rot bd128={s['bd128_r0.0']['rot_share']} dense={s['dense_r0']['rot_share']} ({time.time()-tm:.0f}s)", flush=True)
            json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
            del C, X_cur, W0q, water, W_t, H, X, Xp, X1, F
            torch.cuda.empty_cache()
    # 彙總
    mods = out["modules"]
    def mean(key, f):
        v = [f(m["sol"][key]) for m in mods.values()]
        return round(sum(v) / len(v), 4)
    g = {k: mean(k, lambda d: d["gain_vs_cur"]) for k in ["bd128_r1.0", "bd128_r0.3", "bd128_r0.1", "bd128_r0.01", "bd128_r0.0", "bd256_r0", "bd512_r0", "dense_r0", "2L_bfly", "2L_monarch"]}
    ret = {k: mean(k, lambda d: d["retention_ternary"]) for k in ["bd128_r0.0", "bd256_r0", "dense_r0"]}
    ret["2L_monarch_split"] = mean("2L_monarch", lambda d: d["retention_ternary_split"])
    ret["2L_bfly_split"] = mean("2L_bfly", lambda d: d["retention_ternary_split"])
    ret_i8 = {k: mean(k, lambda d: d["retention_int8"]) for k in ["bd128_r0.0", "dense_r0"]}
    ret_i8.update({k + "_int4": mean(k, lambda d: d["retention_int4"]) for k in ["bd128_r0.0", "bd256_r0", "dense_r0", "2L_monarch"]})
    bw = {k: mean(k, lambda d: d["bw_bf16"]) for k in ["bd128_r0.0", "bd256_r0", "bd512_r0", "dense_r0", "2L_bfly", "2L_monarch"]}
    verdict = {"R1_struct_bound": g["dense_r0"] - g["bd128_r0.0"] >= 0.25,
               "R1_2L_worth": {k: (g[k] >= 0.8 * g["bd256_r0"] and bw[k] < bw["bd256_r0"]) for k in ("2L_bfly", "2L_monarch")},
               "R2_solver_bound": g["bd128_r0.0"] - g["bd128_r1.0"] >= 0.15,
               "R3_ternary_ok": {k: v >= 0.7 for k, v in ret.items()},
               "R4_not_bucket": g["dense_r0"] < 0.4}
    out["summary"] = {"gain_vs_cur_mean": g, "retention_ternary_mean": ret, "retention_int8_mean": ret_i8, "bw_bf16_mean": bw, "verdict": verdict,
                      "elapsed_s": round(time.time() - t0)}
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("PROBE_E62_SUMMARY gain", g, flush=True)
    print("PROBE_E62_SUMMARY ternary_ret", ret, "int8_ret", ret_i8, flush=True)
    print("PROBE_E62_SUMMARY bw_bf16", bw, flush=True)
    print("PROBE_E62_VERDICT", verdict, flush=True)
    print("PROBE_E62_DONE", flush=True)


if __name__ == "__main__":
    main()

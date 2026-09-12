"""E62b S0:窗級前向裁判探針(零訓練;E62a 27/27 撤回的歸因與 E62b 設計校準)。

對 st61c 終態,在數個窗 [lo, lo+2) 上重演 train_res 的窗語義:
  X_dr = 排水態上游輸入(st61c 層 0..lo−1 前推);Y = 理想窗(st54r+滿水)作用在 X_dr。
對 lo 層各模組同時取三份二階矩:
  H   = Σ x_dr x_drᵀ(排水窗內該模組實際輸入;E62a/st61c LS 用的度量)
  G   = Σ x_id x_drᵀ(理想窗內該模組輸入 × 排水輸入,交叉共變)
  Hid = Σ x_id x_idᵀ
候選(對 lo 層全部模組同時套用,與 tail-LS 相同):
  cur      現況 M(梯度+事件 LS 所得)
  X_H      輸入側 M 重解,目標 W_t·H(=E62a 尾段重解)
  X_G      同上但右端 W_t·G(漂移感知目標)
  L_H/L_G  輸出側 L(列塊 128)重解,H/G 目標
  LX_G     L、X 交替(2 輪),G 目標
  ideal    lo 層權重換成理想 W_t(下界參考)
兩本帳:
  W 空間 ‖W_t − W‖_H²(E62a 探針所量)
  前向逐模組 E_fwd = ‖W_t x_id − W x_dr‖² = tr(W H Wᵀ) − 2 tr(W G W_tᵀ) + tr(W_t Hid W_tᵀ)
裁判:窗前向 huber(遮罩)vs Y,train/hold 批,比值 vs cur(=E62a 的 hold×)。
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from p2_anchor.wringer.engine import capture_layer0, fwd_chain, run_layers
from p2_anchor.wringer.ktier import k2_weight
from p2_anchor.wringer.mixls import solve_left_ls, solve_mix_ls
from p2_anchor.wringer.modelio import (LAYER_PREFIX, N_LAYERS,
                                         enumerate_targets, get_module,
                                         layer_index, load_model)
from p2_anchor.wringer.state import dense_weight, load_state, mix_dense

EV = Path("evidence/p1_grouping")


def bd_from_blocks(X):
    return torch.block_diag(*X.float().unbind(0))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-tag", default="st61c")
    ap.add_argument("--start-tag", default="st54r")
    ap.add_argument("--res-tag", default="res58r1")
    ap.add_argument("--windows", default="0,4,11,18,22,26")
    ap.add_argument("--n-calib", type=int, default=40)
    ap.add_argument("--hold", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--ridge", type=float, default=0.1)
    ap.add_argument("--cap", type=float, default=0.25)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--out", default=str(EV / "corkscrew/probe_e62b_win.json"))
    args = ap.parse_args()
    t0 = time.time()

    model, _ = load_model()
    model.requires_grad_(False)
    targets = enumerate_targets(model)
    st = load_state(args.state_tag, targets)
    st0 = load_state(args.start_tag, targets)
    res = {}
    for li in range(N_LAYERS):
        p = Path("data") / f"res_{args.res_tag}" / f"layer{li:02d}.pt"
        if p.exists():
            res.update(torch.load(p, map_location="cpu", weights_only=False))
    calib = torch.load(EV / "calib_e58r1b_traj.pt", weights_only=False)[:args.n_calib]
    lens = torch.load(EV / "calib_e58r1b_len.pt", weights_only=False)[:args.n_calib]
    S = calib.shape[1]
    ar = torch.arange(S)
    Mk = [(ar[None] < lens[i:i + args.batch, None]).cuda()
          for i in range(0, calib.shape[0], args.batch)]
    nb = len(Mk)
    n_hold = max(1, args.hold // args.batch)
    train_bi = list(range(nb - n_hold))
    hold_bi = list(range(nb - n_hold, nb))

    def water_of(n):
        return (res[n]["B"].cuda().float() @ res[n]["A"].cuda().float()) / float(res[n]["r"])

    w_drain = {n: dense_weight(st[n]) for n in targets}      # fp32 cuda
    for n in targets:
        get_module(model, n).weight.data.copy_(w_drain[n].to(torch.bfloat16))
    layers = [model.get_submodule(f"{LAYER_PREFIX}.{li}") for li in range(N_LAYERS)]
    X, kw = capture_layer0(model, calib, batch=args.batch, to_cpu=True)
    print(f"load+capture {time.time()-t0:.0f}s  batches {nb} (hold {hold_bi})", flush=True)

    wins = [int(x) for x in args.windows.split(",")]
    out = {"state": args.state_tag, "calib": {"rows": int(calib.shape[0]), "seq": int(S)},
           "hparams": vars(args), "windows": {}}
    cur_l = 0
    for lo in wins:
        hi = lo + 2
        if lo > cur_l:                        # 前推 X_dr 到 lo
            X = run_layers(layers[cur_l:lo], X, kw, to_cpu=True)
            cur_l = lo
        wl = layers[lo:hi]
        mods = [n for n in targets if layer_index(n) == lo]
        w_ideal = {n: dense_weight(st0[n]) + water_of(n)
                   for n in targets if lo <= layer_index(n) < hi}

        def set_w(names, wdict):
            for n in names:
                get_module(model, n).weight.data.copy_(wdict[n].to(torch.bfloat16))

        # ---- 兩趟:理想窗(存 x_id、Y)/ 排水窗(H, G, Hid) ----
        store, acc = {}, {n: {} for n in mods}
        mode = {"m": None, "bi": None}

        def mk(n):
            def pre(mod, a):
                x = a[0].detach().reshape(-1, a[0].shape[-1])
                x = x[Mk[mode["bi"]].reshape(-1)]
                if mode["m"] == "ideal":
                    store[n] = x.to(torch.bfloat16)
                else:
                    xd = x.float()
                    dd = store.pop(n).float() - xd            # 漂移 d = x_id − x_dr
                    d = acc[n]
                    # 無相消分帳:H = Σ x_dr x_drᵀ、Cd = Σ x_dr dᵀ、Hd = Σ d dᵀ
                    # (G = Σ x_id x_drᵀ = H + Cdᵀ,Hid = H + Cd + Cdᵀ + Hd)
                    for key, val in (("H", xd.T @ xd), ("Cd", xd.T @ dd), ("Hd", dd.T @ dd)):
                        d[key] = d[key] + val if key in d else val
                    del dd, xd
            return pre
        hooks = [get_module(model, n).register_forward_pre_hook(mk(n)) for n in mods]
        Y = []
        for bi in range(nb):
            mode["bi"] = bi
            set_w(w_ideal, w_ideal)
            mode["m"] = "ideal"
            Y.append(fwd_chain(wl, X[bi], kw).to("cpu", torch.bfloat16))
            set_w(w_ideal, w_drain)
            mode["m"] = "drain"
            fwd_chain(wl, X[bi], kw)
        for h in hooks:
            h.remove()
        torch.cuda.empty_cache()

        def win_loss(bis):
            tot = 0.0
            for bi in bis:
                pe = F.huber_loss(fwd_chain(wl, X[bi], kw).float(), Y[bi].cuda().float(),
                                  reduction="none")
                mk_ = Mk[bi]
                tot += float((pe * mk_[:, :, None]).sum() / (mk_.sum() * pe.shape[-1]))
            return tot / len(bis)

        # ---- 逐模組候選 ----
        per = {}
        cand_w = {c: {} for c in ("cur", "X_H", "X_G", "L_H", "L_G", "LX_G", "ideal")}
        for n in mods:
            c = st[n]
            C = k2_weight(c["T1"].cuda(), c["T2"].cuda(), c["a0"].cuda().float(), c["c"])[:, c["inv"].cuda()]
            Xc = c["mix"].cuda().float()
            H, Cd, Hd = acc[n]["H"], acc[n]["Cd"], acc[n]["Hd"]
            G = H + Cd.T                      # Σ x_id x_drᵀ(法方程右端:W H = W_t G)
            W_t = w_ideal[n]
            Mo, K = C.shape
            H64, Cd64, Wt64 = H.double(), Cd.double(), W_t.double()
            tWH = float(((Wt64 @ Hd.double()) * Wt64).sum())     # 純漂移項 ‖W_t d‖²

            def e_fwd(W):
                # ‖W_t x_id − W x_dr‖² = ‖ΔW x_dr + W_t d‖²,ΔW = W_t − W(三項皆小,無相消)
                D = Wt64 - W.double()
                return float(((D @ H64) * D).sum() + 2.0 * ((D @ Cd64) * Wt64).sum() + tWH)

            def e_w(W):
                E = W_t - W
                return float(((E @ H) * E).sum())

            W_cur = C @ bd_from_blocks(Xc)
            E0f, E0w = e_fwd(W_cur), e_w(W_cur)
            Wi_f = e_fwd(W_t)                       # 理想權重在漂移輸入上的前向誤差
            r = {"K": K, "M": Mo, "E_fwd_cur": E0f, "E_w_cur": E0w,
                 "drift_rel": round((float(Hd.diagonal().sum())
                                     / max(float((H + 2 * Cd + Hd).diagonal().sum()), 1e-30)) ** 0.5, 5),
                 "ideal_w_on_drift": round(Wi_f / max(E0f, 1e-30), 4)}
            cand_w["cur"][n] = W_cur
            cand_w["ideal"][n] = W_t
            XH, sH = solve_mix_ls(C, W_t, H, Xc, ridge=args.ridge, cap=args.cap, iters=args.iters)
            XG, sG = solve_mix_ls(C, W_t, H, Xc, ridge=args.ridge, cap=args.cap, iters=args.iters, G=G)
            cand_w["X_H"][n] = C @ bd_from_blocks(XH)
            cand_w["X_G"][n] = C @ bd_from_blocks(XG)
            L0 = torch.eye(Mo, device="cuda")
            LH, lH = solve_left_ls(W_cur, W_t, H, L0, args.block, ridge=0.0, cap=args.cap, iters=args.iters)
            LG, lG = solve_left_ls(W_cur, W_t, H, L0, args.block, ridge=0.0, cap=args.cap, iters=args.iters, G=G)
            cand_w["L_H"][n] = LH @ W_cur
            cand_w["L_G"][n] = LG @ W_cur
            # L/X 交替(G 目標):L 固定解 X 需目標 L⁻¹W_t … 改用等價:對 Z=C·Bd(X) 解 L,再對 (LC)·Bd(X) 解 X
            Lk, Xk = LG.clone(), XG.clone()
            for _ in range(2):
                Xk, _s = solve_mix_ls(Lk @ C, W_t, H, Xk, ridge=args.ridge, cap=args.cap, iters=args.iters, G=G)
                Zk = C @ bd_from_blocks(Xk)
                Lk, _s = solve_left_ls(Zk, W_t, H, Lk, args.block, ridge=0.0, cap=args.cap, iters=args.iters, G=G)
            cand_w["LX_G"][n] = Lk @ (C @ bd_from_blocks(Xk))
            for cn in ("X_H", "X_G", "L_H", "L_G", "LX_G"):
                W = cand_w[cn][n]
                r[cn] = {"E_fwd_ratio": round(e_fwd(W) / max(E0f, 1e-30), 4),
                         "E_w_ratio": round(e_w(W) / max(E0w, 1e-30), 4)}
            r["X_H"]["t"], r["X_G"]["t"], r["L_G"]["t"] = sH["t"], sG["t"], lG["t"]
            r["X_H"]["sv_min"], r["L_G"]["sv_min"] = sH["sv_min"], lG["sv_min"]
            per[n.split("layers.")[1]] = r
            del C, H, G, Cd, Hd, XH, XG, LH, LG, Lk, Xk, H64, Cd64, Wt64
            torch.cuda.empty_cache()

        # ---- 窗前向裁判 ----
        # 起點:lo 層排水、lo+1 層排水
        set_w(w_ideal, w_drain)
        judge = {}
        base_t, base_h = win_loss(train_bi), win_loss(hold_bi)
        judge["cur"] = {"train": base_t, "hold": base_h}
        nodrift = [n for n in mods if per[n.split("layers.")[1]]["drift_rel"] < 0.02]
        drift = [n for n in mods if n not in nodrift]
        variants = [(cn, mods) for cn in ("X_H", "X_G", "L_H", "L_G", "LX_G", "ideal")]
        variants += [("X_H@nodrift", nodrift), ("X_H@drift", drift),
                     ("X_G@drift", drift), ("LX_G@drift", drift)]
        # 夥伴層變體:lo+1 換成理想(滿水)權重——重演 E62a 窗 lo 時第二層 γ=1 的情境
        partner = [n for n in w_ideal if layer_index(n) == lo + 1]
        set_w(w_ideal, w_drain)
        set_w(partner, w_ideal)
        pb_t, pb_h = win_loss(train_bi), win_loss(hold_bi)
        judge["cur|p=ideal"] = {"train": pb_t, "hold": pb_h, "vs_drainpartner": round(pb_h / max(base_h, 1e-30), 4)}
        variants += [("X_H|p=ideal", mods), ("X_G|p=ideal", mods), ("L_H|p=ideal", mods), ("LX_G|p=ideal", mods)]
        for tag, names in variants:
            cn = tag.split("@")[0].split("|")[0]
            pi = "|p=ideal" in tag
            if not names:
                continue
            set_w(w_ideal, w_drain)
            if pi:
                set_w(partner, w_ideal)
            for n in names:
                get_module(model, n).weight.data.copy_(cand_w[cn][n].to(torch.bfloat16))
            lt, lh = win_loss(train_bi), win_loss(hold_bi)
            bt, bh = (pb_t, pb_h) if pi else (base_t, base_h)
            judge[tag] = {"train": lt, "hold": lh, "train_ratio": round(lt / max(bt, 1e-30), 4),
                          "hold_ratio": round(lh / max(bh, 1e-30), 4), "n_mods": len(names)}
        set_w(w_ideal, w_drain)
        out["windows"][str(lo)] = {"modules": per, "judge": judge,
                                   "nodrift_mods": [n.split("layers.")[1] for n in nodrift]}
        summ = " ".join(f"{k}:{v['hold_ratio']:.3f}/{v['train_ratio']:.3f}" for k, v in judge.items() if "hold_ratio" in v)
        print(f"PROBE_WIN lo={lo} hold {summ}  ({time.time()-t0:.0f}s)", flush=True)
        for k, v in per.items():
            print(f"   {k:28s} drift {v['drift_rel']:.3f} idealW/cur {v['ideal_w_on_drift']:.2f} | "
                  + " ".join(f"{cn}: fwd {v[cn]['E_fwd_ratio']:.3f} w {v[cn]['E_w_ratio']:.3f}"
                             for cn in ("X_H", "X_G", "L_H", "L_G", "LX_G")), flush=True)
        del Y, cand_w, acc, store, w_ideal
        torch.cuda.empty_cache()
        Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"PROBE_WIN_DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

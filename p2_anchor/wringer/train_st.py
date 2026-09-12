"""ST 滑動視窗重建(Corkscrew 第二段;移植 guava run_ste9 + circus 擴充)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.train_st \
      --grid 3 --state-tag g3 [--learn-theta] [--resume] [--smoke]

視窗 4 層滑 2;輸入 = 已定案量化前段傳播的活化(入口快取,#24 壓榨形);
目標 = 同輸入過 fp 視窗。可訓 = LoRA + δ_α logit [+ dθ]。
epoch 前 γ 軟退火 s→s₀,尾段 STE 硬收尾(訓練終態 = 部署形態)。

circus 擴充 vs guava 原版:
  - 等價閘 1e-6(引擎驗收,跑前斷言)
  - --learn-theta(門檻逐模組可學;部署自由)
  - --resume(逐層 .pt 即天然斷點:定案檔齊的窗直接套用並前進 X)
  - G-disc 記錄(gdisc1 出口軟硬差 / gdisc2 格點距 / gdisc3 佔用)
"""
import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from p2_anchor.wringer.comp import HAccum, greedy_comp
from p2_anchor.wringer.engine import (capture_layer0, equivalence_gate,
                                        fwd_chain, run_layers)
from p2_anchor.wringer.ktier import (C_K2, grid_tensors, k2_weight,
                                       round_with_mids)
from p2_anchor.wringer.modelio import (LAYER_PREFIX, N_LAYERS,
                                         enumerate_targets, get_module,
                                         layer_index, load_model)
from p2_anchor.wringer.state import load_state, qs_dir
from p2_anchor.wringer.stq import STQParam, SoftGrid, THETA_RANGE

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"
SEED = 20260806


def set_mode(params, mode, s=0.0):
    for p in params:
        p.mode = mode
        p.s = s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=9, choices=(9, 7, 5, 3))
    ap.add_argument("--tag", default=None, help="預設 st<grid>")
    ap.add_argument("--state-tag", default=None, help="α₀/T 初始(預設 g<grid>)")
    ap.add_argument("--n-calib", type=int, default=512)
    ap.add_argument("--calib", default=str(EV / "calib_train.pt"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--win", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--s0", type=float, default=30.0)
    ap.add_argument("--plateau-frac", type=float, default=0.0,
                    help="AOQ 探索平台:前此比例 epoch 鎖 s=--s-plateau"
                         "(0=關,行為與舊版全同)")
    ap.add_argument("--s-plateau", type=float, default=2.0,
                    help="探索平台的 s 值(低 s=高翻轉自由度)")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-r-deep", type=int, default=None,
                    help="深層 LoRA rank(E32 階梯;None=全域同 --lora-r)")
    ap.add_argument("--deep-from", type=int, default=22,
                    help="--lora-r-deep 生效的起始層號")
    ap.add_argument("--lora-lr", type=float, default=6e-4)
    ap.add_argument("--factor-lr", type=float, default=3e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--learn-theta", action="store_true")
    ap.add_argument("--theta-lr", type=float, default=None,
                    help="預設同 --factor-lr")
    ap.add_argument("--morph-to", type=int, default=None, choices=(7, 3),
                    help="態二走廊:7=E34(9元 c→0.5)、3=E35(7元 c→0,"
                         "c 收歸 α);--grid 須為來源檔位,落盤為目標檔位")
    ap.add_argument("--morph-lo", type=float, default=2.0,
                    help="走廊起始 epoch(m=0)")
    ap.add_argument("--morph-hi", type=float, default=8.0,
                    help="走廊收斂 epoch(m=1;須 < gamma·epochs)")
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="cal 尾端保留比例,逐 epoch 記 holdout 重建 loss"
                         "(E35 走廊過擬合偵測;0=關,行為與舊版全同)")
    ap.add_argument("--comp", action="store_true",
                    help="E36 合併即補償+移動老師(僅 --morph-to 3 生效;"
                         "關=行為與 E35 逐位相同)")
    ap.add_argument("--comp-every", type=int, default=5,
                    help="走廊內補償事件間隔(epoch)")
    ap.add_argument("--comp-rounds", type=int, default=3,
                    help="貪婪殘差回饋輪數")
    ap.add_argument("--comp-cap", type=float, default=0.03,
                    help="每模組每事件翻轉總封頂(占權重比;每輪 cap/rounds)")
    ap.add_argument("--teacher-lambda", type=float, default=0.6,
                    help="移動老師混合損失 λ 平台值")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="2 窗×2 epoch×32 calib,驗接線與 resume")
    args = ap.parse_args()
    tag = args.tag or f"st{args.grid}"
    state_tag = args.state_tag or f"g{args.grid}"
    if args.smoke:
        args.n_calib, args.epochs, args.batch = 32, 2, 4
        if args.morph_to:
            args.morph_lo, args.morph_hi = 0.0, 1.0   # 2 epoch 內走完走廊
            if args.comp:   # E36 smoke:需 ≥1 個 Δc>0 補償事件 → 4ep 走廊[0,3]
                args.epochs, args.morph_lo, args.morph_hi = 4, 0.0, 3.0
    MORPH_SRC = {7: 9, 3: 7}          # 目標檔位 → 來源檔位
    MORPH_CTGT = {7: 0.5, 3: 0.0}     # 目標檔位 → 走廊終點 c
    if args.morph_to:
        assert args.grid == MORPH_SRC[args.morph_to], \
            f"morph→{args.morph_to} 起點檔位須為 {MORPH_SRC[args.morph_to]}元"
        assert args.morph_hi <= args.gamma * args.epochs, \
            "走廊須在 STE 硬收尾前收斂"

    t0 = time.time()
    torch.manual_seed(SEED)
    QS = qs_dir(tag)
    QS.mkdir(parents=True, exist_ok=True)
    EVC.mkdir(parents=True, exist_ok=True)
    model, tok = load_model()
    model.requires_grad_(False)
    targets = enumerate_targets(model)
    g0 = load_state(state_tag, targets)          # α₀ = joint α;T 供翻轉率
    calib = torch.load(args.calib, weights_only=False)[:args.n_calib]
    val = torch.load(EV / "calib_val.pt", weights_only=False)
    grids = {}                      # 逐模組 grid 從 state 讀(混合多元傳播)

    from p1_grouping.e3_runner import eval_ce

    X, kw = capture_layer0(model, calib, batch=args.batch)
    layers = [model.get_submodule(f"{LAYER_PREFIX}.{li}")
              for li in range(N_LAYERS)]
    rel = equivalence_gate(model, layers, X, kw, calib, batch=args.batch,
                           upto=args.win)
    print(f"captured X: {len(X)}×{tuple(X[0].shape)}  kwargs={list(kw)}  "
          f"等價閘 PASS rel={rel:.2e}", flush=True)

    starts = list(range(0, N_LAYERS - args.win + 1, args.stride))
    windows = [(s, s + args.win) for s in starts]
    if args.smoke:
        windows = windows[:2]

    n_hold = int(round(args.holdout_frac * len(X)))
    train_bi = list(range(len(X) - n_hold))
    hold_bi = list(range(len(X) - n_hold, len(X)))
    if n_hold:
        print(f"holdout 偵測器:train {len(train_bi)} 批 / hold {n_hold} 批",
              flush=True)

    stq = {}
    null3 = {}        # M3-v2①:morph→3 的投影 null 基準碼(int8, CPU)
    ridge_gpu = {}    # ridge 遮罩(g0 碼 T1==0 ∧ T2≠0;GPU bool,窗後釋放)
    rec = {"tag": tag, "grid": args.grid, "state_tag": state_tag,
           "seed": SEED, "equiv_gate_rel": rel, "hparams": vars(args),
           "windows": []}
    flips, ntot = 0, 0

    for wi, (lo, hi) in enumerate(windows):
        fin_hi = hi if wi == len(windows) - 1 else lo + args.stride
        fin_lis = list(range(lo, fin_hi))
        wlis = list(range(lo, hi))

        # resume:定案檔齊 → 套硬權重、前進 X、跳過訓練
        if args.resume and all((QS / f"layer{li:02d}.pt").exists()
                               for li in fin_lis):
            for li in fin_lis:
                for n, c in torch.load(QS / f"layer{li:02d}.pt",
                                       weights_only=False).items():
                    m = get_module(model, n)
                    w = k2_weight(c["T1"].cuda(), c["T2"].cuda(),
                                  c["a0"].cuda().float(), c["c"])
                    m.weight.data.copy_(w.to(m.weight.dtype))
            with torch.no_grad():
                X = run_layers([layers[li] for li in fin_lis], X, kw)
            print(f"window [{lo},{hi}) resume-skip(fin={fin_lis})", flush=True)
            continue

        wtgt = [n for n in targets if layer_index(n) in wlis]
        for n in wtgt:                            # 重疊層帶熱啟動,不重掛
            if n not in stq:
                m = get_module(model, n)
                r_n = (args.lora_r_deep if args.lora_r_deep is not None
                       and layer_index(n) >= args.deep_from else args.lora_r)
                gn = int(g0[n].get("grid", args.grid))
                cn = float(g0[n].get("c", C_K2))  # c 從 state 讀,禁默認洩漏
                if (gn, cn) not in grids:
                    grids[(gn, cn)] = SoftGrid(c=cn, n=gn)
                p = STQParam(g0[n]["a0"].cuda(), grids[(gn, cn)], r=r_n,
                             c=cn, learn_theta=args.learn_theta,
                             use_comp=(args.comp and args.morph_to == 3
                                       and gn == MORPH_SRC[args.morph_to]))
                if args.morph_to and gn == MORPH_SRC[args.morph_to]:
                    p.enable_morph(c_tgt=MORPH_CTGT[args.morph_to],
                                   to=args.morph_to)
                    if args.morph_to == 3:   # M3-v2①:null=起始 u 直投 3元
                        a0g = g0[n]["a0"].cuda().float()
                        u0 = (m.weight.detach().float().reshape(
                            a0g.shape[0], a0g.shape[1], -1) / a0g)
                        null3[n] = torch.where(
                            u0.abs() < 0.5, torch.zeros_like(u0),
                            torch.sign(u0)).to(torch.int8).cpu()
                        ridge_gpu[n] = ((g0[n]["T1"].int() == 0) &
                                        (g0[n]["T2"].int() != 0)).cuda()
                        del u0, a0g
                parametrize.register_parametrization(m, "weight", p)
                stq[n] = p
        wp = [stq[n] for n in wtgt]
        wlayers = [layers[li] for li in wlis]

        set_mode(wp, "off")
        with torch.no_grad():
            Y = run_layers(wlayers, X, kw)        # fp 目標(逐窗錨定)

        # lazy LoRA:先跑一步 soft 前向觸發 init,再建 optimizer
        set_mode(wp, "soft", s=args.s0 / (args.gamma * args.epochs))
        with torch.no_grad():
            _ = fwd_chain(wlayers, X[0], kw)
        groups = [
            {"params": [q for n in wtgt for q in
                        (stq[n].lora_A, stq[n].lora_B)], "lr": args.lora_lr},
            {"params": [stq[n].logit for n in wtgt]
             + [stq[n].logit_c for n in wtgt if hasattr(stq[n], "logit_c")],
             "lr": args.factor_lr}]
        if args.learn_theta:
            groups.append({"params": [stq[n].dtheta for n in wtgt],
                           "lr": args.theta_lr or args.factor_lr})
        opt = torch.optim.Adam(groups)

        g_rng = torch.Generator().manual_seed(SEED + lo)
        ep_log, ep_hold, ridge_traj = [], [], []
        comp_on = args.comp and args.morph_to == 3
        last_c, teacher_Y, comp_events = {}, None, []
        ev_eps = set()
        if comp_on:                       # 事件:每 --comp-every ∪ 末步前
            lo_i, hi_i = int(round(args.morph_lo)), int(round(args.morph_hi))
            ev_eps = set(range(lo_i + args.comp_every, hi_i, args.comp_every))
            ev_eps.add(hi_i - 1)          # 末事件必在 collapse 前(T₂ 尚存)
            ev_eps.add(lo_i)              # 老師初始化(Δc=0,無 comp)

        def _c_of(n):
            q = stq[n]
            return float(q._c_now().detach()) if q._morph else float(q.c)

        def lam_of(ep):                   # λ:走廊內平台,collapse 後重錨 fp
            if not comp_on:
                return 0.0
            lmx = args.teacher_lambda
            ramp = min(args.comp_every, max(args.morph_hi - args.morph_lo, 1))
            if ep < args.morph_lo:
                return 0.0
            if ep < args.morph_lo + ramp:
                return lmx * (ep - args.morph_lo) / ramp
            if ep <= args.morph_hi:
                return lmx
            return max(lmx * (1.0 - (ep - args.morph_hi) / 7.0), 0.0)

        for ep in range(args.epochs):
            if args.morph_to:                     # 走廊排程 m(ep) 線性
                m_sched = min(max((ep - args.morph_lo)
                                  / max(args.morph_hi - args.morph_lo, 1e-9),
                                  0.0), 1.0)
                for p_ in wp:
                    p_.set_m(m_sched)             # 已收斂者 no-op
            if comp_on and ep in ev_eps:          # E36 事件:老師刷新+補償
                acc = HAccum()
                acc.attach([(n, get_module(model, n)) for n in wtgt])
                set_mode(wp, "hard")              # 部署形態老師(c_now)
                with torch.no_grad():
                    teacher_Y = {bi: fwd_chain(wlayers, X[bi], kw).detach()
                                 .clone() for bi in train_bi}
                acc.detach()
                with torch.no_grad():
                    drift = sum(float(F.huber_loss(
                        teacher_Y[bi].float(), Y[bi].float()))
                        for bi in train_bi) / len(train_bi)
                ev = {"ep": ep, "teacher_drift": round(drift, 6),
                      "flips": 0, "absorb": [], "dc": []}
                for n in wtgt:
                    q = stq[n]
                    cv = _c_of(n)
                    dc = last_c.get(n, cv) - cv
                    last_c[n] = cv
                    Hn = acc.pop(n)
                    if q.comp is None or dc <= 1e-4 or Hn is None:
                        continue
                    mq = get_module(model, n)
                    W0q = mq.parametrizations.weight.original
                    with torch.no_grad():   # R = W_fp − W_hard(總功能虧欠,
                        # 直接量測;Δc 恆等式為走廊增量理論+跳過閘)
                        u_, al_ = q._u(W0q)
                        vals, t1g, t2g = grid_tensors(7, cv, device=u_.device)
                        th_ = (vals[1:] + vals[:-1]) / 2
                        if q.dtheta is not None:
                            hg_ = vals[1:] - vals[:-1]
                            th_ = th_ + THETA_RANGE * hg_ * torch.tanh(
                                q.dtheta.detach())
                        _, _, v_h = round_with_mids(u_, th_, vals, t1g, t2g)
                        R = W0q.float() - (al_ * v_h).reshape(W0q.shape)
                        clog = greedy_comp(q, W0q, R, Hn, cv,
                                           rounds=args.comp_rounds,
                                           cap=args.comp_cap)
                    ev["flips"] += clog["flips"]
                    ev["absorb"].append(clog["absorb"])
                    ev["dc"].append(round(dc, 4))
                    del R, Hn, u_, al_
                acc.H.clear()
                comp_events.append(ev)
            t_soft = (ep + 1) / args.epochs
            if t_soft <= args.gamma:
                if args.plateau_frac > 0 and t_soft <= args.plateau_frac:
                    s = args.s_plateau            # AOQ 探索平台:低 s 高翻轉
                elif args.plateau_frac > 0:
                    s = args.s_plateau + (
                        (t_soft - args.plateau_frac)
                        / max(args.gamma - args.plateau_frac, 1e-9)
                        * (args.s0 - args.s_plateau))
                else:
                    s = (t_soft / args.gamma) * args.s0
                set_mode(wp, "soft", s=s)
            else:
                set_mode(wp, "hard")
            tot = 0.0
            lam = lam_of(ep) if comp_on else 0.0
            for oi in torch.randperm(len(train_bi), generator=g_rng).tolist():
                bi = train_bi[oi]
                opt.zero_grad(set_to_none=True)
                yh = fwd_chain(wlayers, X[bi], kw)
                loss_fp = F.huber_loss(yh.float(), Y[bi].float())
                loss = loss_fp
                if lam > 0.0 and teacher_Y is not None:
                    loss = (1.0 - lam) * loss_fp + lam * F.huber_loss(
                        yh.float(), teacher_Y[bi].float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [q for g in opt.param_groups for q in g["params"]],
                    args.clip)
                opt.step()
                tot += float(loss_fp)     # ep_log 記 fp 項(V5/曲線可比性)
            m_loss = tot / len(train_bi)
            ep_log.append(round(m_loss, 6))
            if not np.isfinite(m_loss):
                raise RuntimeError(f"V5 警報:窗 {lo} ep {ep} loss {m_loss}")
            if hold_bi:                       # holdout gap(同模式同 s,無梯度)
                with torch.no_grad():
                    hl = sum(float(F.huber_loss(
                        fwd_chain(wlayers, X[bi], kw).float(),
                        Y[bi].float())) for bi in hold_bi) / len(hold_bi)
                ep_hold.append(round(hl, 6))
            if args.morph_to == 3 and any(p_._morph for p_ in wp):
                with torch.no_grad():         # M3-v2③:ridge 在 ±1 簇的佔比軌跡
                    num = den = 0
                    for n in wtgt:
                        if not stq[n]._morph or n not in ridge_gpu:
                            continue
                        W0 = get_module(model, n).parametrizations.weight.original
                        u, _ = stq[n]._u(W0)
                        b = (1.0 + float(stq[n]._c_now().detach())) / 2
                        rm = ridge_gpu[n]
                        num += int((u.abs()[rm] > b).sum())
                        den += int(rm.sum())
                    if den:
                        ridge_traj.append(round(num / den, 4))
        if ep_log[-1] > ep_log[0] * 1.5:
            print(f"  ⚠ V5 帶化註記:窗 {lo} 末 {ep_log[-1]} > 首×1.5",
                  flush=True)

        # G-disc1:出口軟硬輸出差(訓練終態 s)
        with torch.no_grad():
            set_mode(wp, "hard")
            y_hard = fwd_chain(wlayers, X[0], kw).float()
            set_mode(wp, "soft", s=args.s0)
            y_soft = fwd_chain(wlayers, X[0], kw).float()
            gdisc1 = float((y_soft - y_hard).norm()
                           / y_hard.norm().clamp_min(1e-12))
            set_mode(wp, "hard")

        # 定案:extract → 存層檔 → 材化硬權重 → 前進 X
        gds, occ_inner = [], []
        flow = {"n": 0, "to_half": 0, "to_zero": 0, "to_other": 0}  # M3(E34)
        flow3 = {"ridge_n": 0, "ridge_to_zero": 0, "ridge_to_one": 0,
                 "outer_n": 0, "outer_to_one": 0, "outer_to_zero": 0,
                 "null_n": 0, "null_flip": 0, "ridge_null_flip": 0}  # M3-v2
        lcf = {}                                   # logit_c 終值(量測義務②)
        state_by_layer = {li: {} for li in fin_lis}
        thetas = {}
        for n in list(wtgt):
            li = layer_index(n)
            if li not in state_by_layer:
                continue
            m = get_module(model, n)
            W0 = m.parametrizations.weight.original
            T1, T2, al, w_hard = stq[n].extract(W0)
            f = int((T1.cpu() != g0[n]["T1"]).sum()
                    + (T2.cpu() != g0[n]["T2"]).sum())
            flips += f
            ntot += T1.numel() * 2
            M, K = W0.shape
            nB = stq[n].a0.shape[1]
            u = (W0.float() + (stq[n].lora_B @ stq[n].lora_A) / stq[n]._r
                 ).reshape(M, nB, K // nB) / (
                2.0 * torch.sigmoid(stq[n].logit) * stq[n].a0)
            v = (T1.float() + stq[n].c * T2.float())
            gds.append(float((u - v).abs().mean()))
            occ_inner.append(float(((T1 != 0) & (T2 != 0) &
                                    (T1.int() * T2.int() < 0)).float().mean()))
            if args.morph_to == 7:            # M3(E34):±(1−c) 盆地質量流向
                basin = (g0[n]["T1"].int() * g0[n]["T2"].int() == -1)
                if basin.any():
                    va = (T1.cpu().float()
                          + stq[n].c * T2.cpu().float())[basin].abs()
                    flow["n"] += int(basin.sum())
                    flow["to_half"] += int((va == 0.5).sum())
                    flow["to_zero"] += int((va == 0.0).sum())
                    flow["to_other"] += int(((va != 0.5) & (va != 0.0)).sum())
            elif args.morph_to == 3:          # M3-v2:ridge/outer 流向+null 翻轉
                T1c = T1.cpu().to(torch.int8)   # c=0:終值 = T1
                ridge = (g0[n]["T1"].int() == 0) & (g0[n]["T2"].int() != 0)
                outer = (g0[n]["T1"].int() != 0) & \
                        (g0[n]["T2"].int() == g0[n]["T1"].int())
                flow3["ridge_n"] += int(ridge.sum())
                flow3["ridge_to_zero"] += int((T1c[ridge] == 0).sum())
                flow3["ridge_to_one"] += int((T1c[ridge] != 0).sum())
                flow3["outer_n"] += int(outer.sum())
                flow3["outer_to_one"] += int((T1c[outer] != 0).sum())
                flow3["outer_to_zero"] += int((T1c[outer] == 0).sum())
                if n in null3:
                    flip = (T1c != null3[n])
                    flow3["null_n"] += flip.numel()
                    flow3["null_flip"] += int(flip.sum())
                    flow3["ridge_null_flip"] += int(flip[ridge].sum())
                    del null3[n]
                ridge_gpu.pop(n, None)
                if hasattr(stq[n], "logit_c_final"):
                    lcf[n] = round(stq[n].logit_c_final, 4)
            if args.learn_theta:
                thetas[n] = stq[n].theta_eff().detach().cpu().tolist()
            state_by_layer[li][n] = {
                "T1": T1.cpu(), "T2": T2.cpu(),
                "inv": torch.arange(K), "a0": al.cpu(), "c": stq[n].c,
                "grid": stq[n].grid.n}
            parametrize.remove_parametrizations(m, "weight",
                                                leave_parametrized=False)
            m.weight.data.copy_(w_hard.to(m.weight.dtype))
            del stq[n]
        for li in fin_lis:
            torch.save(state_by_layer[li], QS / f"layer{li:02d}.pt")
        with torch.no_grad():
            X = run_layers([layers[li] for li in fin_lis], X, kw)
        del Y
        teacher_Y = None
        torch.cuda.empty_cache()
        w_rec = {"window": [lo, hi], "finalized": fin_lis, "loss": ep_log,
                 "gdisc1_exit_rel": round(gdisc1, 6),
                 "gdisc2_grid_dist": round(sum(gds) / max(len(gds), 1), 6),
                 "gdisc3_opposed_frac": round(
                     sum(occ_inner) / max(len(occ_inner), 1), 6)}
        if thetas:
            w_rec["theta_eff"] = thetas
        if ep_hold:
            w_rec["loss_holdout"] = ep_hold
        if args.morph_to == 7 and flow["n"]:
            w_rec["morph_flow"] = {
                "basin_n": flow["n"],
                "to_half": round(flow["to_half"] / flow["n"], 4),
                "to_zero": round(flow["to_zero"] / flow["n"], 4),
                "to_other": round(flow["to_other"] / flow["n"], 4)}
        elif args.morph_to == 3 and flow3["ridge_n"]:
            w_rec["morph3_flow"] = {
                "ridge_n": flow3["ridge_n"],
                "ridge_to_zero": round(
                    flow3["ridge_to_zero"] / flow3["ridge_n"], 4),
                "ridge_to_one": round(
                    flow3["ridge_to_one"] / flow3["ridge_n"], 4),
                "outer_n": flow3["outer_n"],
                "outer_to_one": round(
                    flow3["outer_to_one"] / max(flow3["outer_n"], 1), 4),
                "outer_to_zero": round(
                    flow3["outer_to_zero"] / max(flow3["outer_n"], 1), 4),
                "null_flip_frac": round(
                    flow3["null_flip"] / max(flow3["null_n"], 1), 6),
                "ridge_null_flip_frac": round(
                    flow3["ridge_null_flip"] / max(flow3["ridge_n"], 1), 6)}
            if ridge_traj:
                w_rec["ridge_traj_to_one"] = ridge_traj
            if lcf:
                w_rec["logit_c_final"] = lcf
            if comp_events:
                w_rec["comp_events"] = comp_events
        rec["windows"].append(w_rec)
        print(f"window {lo:2d}-{hi - 1:2d}  loss {ep_log[0]:.5f}→{ep_log[-1]:.5f}"
              f"  gdisc1={gdisc1:.4f}  fin={fin_lis}  ({time.time()-t0:.0f}s)",
              flush=True)

    del X
    torch.cuda.empty_cache()
    ce = eval_ce(model, val)
    rec["zerotrain_val_ce"] = ce
    rec["t_flip_rate_vs_init"] = round(flips / max(ntot, 1), 6)
    rec["runtime_s"] = round(time.time() - t0, 1)
    rec["timestamp"] = datetime.datetime.now().isoformat()
    st = load_state(tag, targets if not args.smoke else [])
    np.savez(EVC / f"alphas_{tag}.npz",
             **{n: st[n]["a0"].numpy() for n in st})
    (EVC / f"train_{tag}.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"{tag} zero-train val CE {ce:.6f}  "
          f"flip {rec['t_flip_rate_vs_init']:.4f}  ({rec['runtime_s']}s)",
          flush=True)


if __name__ == "__main__":
    main()

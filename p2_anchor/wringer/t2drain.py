"""E59 S1:9→3 轉換單矛——T2 平面逐窗排水(prereg_e59 FROZEN @cb1c2bb)。

與 train_res 的本質差異:碼決策不走 γ 階梯+greedy 翻碼(E58R1b 定罪的
盆地一行走機制),而是逐窗 GPTQ 對 W₉ 藍圖直解 grid3(解法器瞬移)+
α/θ/LoRA 逐窗吸收(訓練期自由度,extract 折進碼+α,部署純三元)。

逐窗流程(win2/stride1,窗目標 Y = 9元 藍圖理想函數):
  1) 解碼事件(層首入窗):鮮 H(部署形態前傳)→ gptq_grid(W₉, H, 3)
     → scale_joint α 重擬合;殺閘=H 度量誤差 err_s > 1.05×err_ctrl
     (ctrl=直投:9元 T1 原樣+α 重擬合)⇒ 模組回退直投,照實記數。
  2) 藍圖殘差 R = W₉ − α·T1' 掛為凍結 dense 旁路(γ=1 復現 W₉ 算 Y;
     γ=0 純學生訓練/extract)。
  3) 吸收:soft 退火數 epoch + STE 尾(lora/logit/dθ);定案 extract。
  4) 落點探針:窗 4 累計 T1 對 g3b32 一致率 ≥0.99 ⇒ 沒跳出盆地一,
     E59_ABORT_LANDING 判死(prereg kill gate)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.t2drain \
      --tag st59 --n-calib 384 --calib evidence/p1_grouping/calib_e58r1b_traj.pt \
      --lengths evidence/p1_grouping/calib_e58r1b_len.pt --batch 4 \
      --learn-theta --holdout-frac 0.05 --x-cpu --resume
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from p2_anchor.wringer.comp import HAccum
from p2_anchor.wringer.engine import (capture_layer0, equivalence_gate,
                                        fwd_chain, run_layers)
from p2_anchor.wringer.ktier import BLOCK, gptq_grid, k2_weight, scale_joint
from p2_anchor.wringer.modelio import (LAYER_PREFIX, N_LAYERS,
                                         enumerate_targets, get_module,
                                         layer_index, load_model)
from p2_anchor.wringer.state import load_state, qs_dir
from p2_anchor.wringer.stq import STQParam, SoftGrid
from p2_anchor.wringer.train_st import set_mode

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"
SEED = 20260806
C3 = 0.6           # grid3 下 c 無作用(T2≡0),存卷沿用家族常數
LAND_TH = 0.99     # 落點探針殺閘(prereg)
LAND_WIN = 4       # 探針裁決窗


class BlueprintRes(STQParam):
    """dense 藍圖殘差旁路:γ=1 → W + R = W₉(算窗目標 Y);γ=0 → 純學生。"""

    @torch.no_grad()
    def load_dense(self, R):
        self.res_B = R.detach().float().to(self.a0.device)
        self.res_B.requires_grad_(False)
        self.res_A = None
        self._r_res = 1
        self.gamma = 0.0

    def _res_term(self):
        return self.gamma * self.res_B


def err_h(dW, Sig):
    """H 度量誤差 tr(ΔW Σ ΔWᵀ)(GPTQ 目標函數,殺閘同度量)。"""
    return float(((dW @ Sig) * dW).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="st59")
    ap.add_argument("--blueprint-tag", default="g9")
    ap.add_argument("--probe-tag", default="g3b32",
                    help="落點探針參照(盆地一標記=bf16 直解 3元)")
    ap.add_argument("--n-calib", type=int, default=384)
    ap.add_argument("--calib", default=str(EV / "calib_e58r1b_traj.pt"))
    ap.add_argument("--lengths", default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--win", type=int, default=2)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--absorb-ep", type=int, default=10)
    ap.add_argument("--tail-ste", type=int, default=2)
    ap.add_argument("--s-plateau", type=float, default=2.0)
    ap.add_argument("--s0", type=float, default=30.0)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-lr", type=float, default=6e-4)
    ap.add_argument("--factor-lr", type=float, default=3e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--kill-mult", type=float, default=1.05)
    ap.add_argument("--learn-theta", action="store_true")
    ap.add_argument("--holdout-frac", type=float, default=0.0)
    ap.add_argument("--x-cpu", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--grid", type=int, default=3, choices=(5, 3),
                    help="E60 S3 分段:9→5(grid 5)或 →3;ctrl 直投=藍圖碼巢狀投影")
    ap.add_argument("--base-dir", default=None,
                    help="底模目錄(折 γ 基底 exports/a1_fold_bf16;藍圖須同基底 g9f)")
    ap.add_argument("--rot-q", default=None,
                    help="E60:凍結旋轉載體 Q .pt(窗階段 Q 不動;段間 e2e 再學)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 窗×32 calib×2ep,驗接線/殺閘/探針/extract/resume")
    args = ap.parse_args()
    if args.smoke:
        args.n_calib, args.batch = 32, 4
        args.absorb_ep, args.tail_ste = 2, 1

    t0 = time.time()
    torch.manual_seed(SEED)
    QS = qs_dir(args.tag)
    QS.mkdir(parents=True, exist_ok=True)
    from p2_anchor.wringer.rot_carrier import use_base_dir
    use_base_dir(args.base_dir)
    model, tok = load_model()
    model.requires_grad_(False)
    targets = enumerate_targets(model)
    G = args.grid
    if args.rot_q:
        from p2_anchor.wringer.rot_carrier import RotCarrier, attach, load_Q
        Q0, qmeta = load_Q(args.rot_q)
        carrier = RotCarrier(Q0, learn=False).cuda()
        attach(model, targets, carrier)
        carrier.refresh()
        print(f"E60 凍結旋轉載體 {args.rot_q}(codes {qmeta.get('codes')})", flush=True)
    st9 = load_state(args.blueprint_tag, targets)   # 9元 藍圖(凍結)
    st3p = load_state(args.probe_tag, targets)      # 盆地一探針參照(CPU)
    for n in targets:                               # 全網材化為 W₉
        m = get_module(model, n)
        w = k2_weight(st9[n]["T1"].cuda(), st9[n]["T2"].cuda(),
                      st9[n]["a0"].cuda().float(), st9[n]["c"])
        m.weight.data.copy_(w[:, st9[n]["inv"].cuda()].to(m.weight.dtype))

    calib = torch.load(args.calib, weights_only=False)[:args.n_calib]
    M = None
    if args.lengths:
        lens = torch.load(args.lengths, weights_only=False)[:calib.shape[0]]
        assert lens.shape[0] == calib.shape[0], "lengths 與 calib 列數不齊"
        S = calib.shape[1]
        ar = torch.arange(S)
        M = [(ar[None] < lens[i:i + args.batch, None]).cuda()
             for i in range(0, calib.shape[0], args.batch)]
        print(f"lengths 遮罩:有效率 "
              f"{float(lens.sum()) / (lens.shape[0] * S):.3f}", flush=True)
    val = torch.load(EV / "calib_val.pt", weights_only=False)
    from p1_grouping.e3_runner import eval_ce

    X, kw = capture_layer0(model, calib, batch=args.batch, to_cpu=args.x_cpu)
    layers = [model.get_submodule(f"{LAYER_PREFIX}.{li}")
              for li in range(N_LAYERS)]
    rel = equivalence_gate(model, layers, X, kw, calib, batch=args.batch,
                           upto=args.win)
    print(f"captured X: {len(X)}×{tuple(X[0].shape)}  等價閘 PASS "
          f"rel={rel:.2e}", flush=True)

    starts = list(range(0, N_LAYERS - args.win + 1, args.stride))
    windows = [(s, s + args.win) for s in starts]
    if args.smoke:
        windows = windows[:2]

    n_hold = int(round(args.holdout_frac * len(X)))
    train_bi = list(range(len(X) - n_hold))
    hold_bi = list(range(len(X) - n_hold, len(X)))

    grid3 = SoftGrid(c=C3, n=G)
    stq, solved_t1 = {}, {}                 # solved_t1:解碼快照(flips 帳)
    land_eq, land_n = 0, 0                  # 落點探針累計(vs g3b32)
    rec = {"tag": args.tag, "grid": G, "blueprint": args.blueprint_tag,
           "probe": args.probe_tag, "seed": SEED, "equiv_gate_rel": rel,
           "hparams": vars(args), "windows": []}
    flips_vs_solved, ntot = 0, 0

    def solve_module(n, Sig):
        """GPTQ 直解(grid G)+α 重擬合+直投對照殺閘;回傳 (T1,T2 (M,nB,B) int8, α, 統計)。
        ctrl 直投 = 藍圖碼巢狀投影:G=3 → T1 原樣;G=5 → T1 原樣、T2 僅留 T1=0 處。"""
        m = get_module(model, n)
        Weff = m.weight.detach().float()            # 當前=藍圖 W
        Mo, K = Weff.shape
        nB = K // BLOCK
        a0s, T1s, T2s = gptq_grid(Weff, Sig, G, c=C3)
        T1s3 = T1s.reshape(Mo, nB, BLOCK)
        T2s3 = T2s.reshape(Mo, nB, BLOCK)
        Vs3 = T1s3 + C3 * T2s3
        al_s = scale_joint(Weff.reshape(Mo, nB, BLOCK), Vs3, Sig)
        if not torch.isfinite(al_s).all():
            al_s = a0s
        e_s = err_h(Weff - (al_s * Vs3).reshape(Mo, K), Sig)
        T1c3 = st9[n]["T1"].cuda().float()          # 直投 ctrl:藍圖 T1 原樣
        T2c3 = (st9[n]["T2"].cuda().float() * (T1c3 == 0) if G == 5
                else torch.zeros_like(T1c3))
        Vc3 = T1c3 + C3 * T2c3
        al_c = scale_joint(Weff.reshape(Mo, nB, BLOCK), Vc3, Sig)
        if not torch.isfinite(al_c).all():
            al_c = st9[n]["a0"].cuda().float()
        e_c = err_h(Weff - (al_c * Vc3).reshape(Mo, K), Sig)
        revert = e_s > args.kill_mult * e_c
        T1f, T2f, alf, e_f = ((T1c3, T2c3, al_c, e_c) if revert
                              else (T1s3, T2s3, al_s, e_s))
        agree3 = float((T1f.to(torch.int8).cpu() == st3p[n]["T1"])
                       .float().mean())
        agree9 = float((T1f.to(torch.int8).cpu() == st9[n]["T1"])
                       .float().mean())
        stats = {"err_s": e_s, "err_c": e_c, "revert": bool(revert),
                 "agree_g3": agree3, "agree_g9t1": agree9,
                 "t2_nonzero": float((T2f != 0).float().mean()),
                 "numel": T1f.numel()}
        return (T1f.to(torch.int8), T2f.to(torch.int8), alf, Weff, e_f,
                stats)

    for wi, (lo, hi) in enumerate(windows):
        fin_hi = hi if wi == len(windows) - 1 else lo + args.stride
        fin_lis = list(range(lo, fin_hi))
        wlis = list(range(lo, hi))

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
                X = run_layers([layers[li] for li in fin_lis], X, kw,
                               to_cpu=args.x_cpu)
            print(f"window [{lo},{hi}) resume-skip", flush=True)
            continue

        wtgt = [n for n in targets if layer_index(n) in wlis]
        ftgt = [n for n in wtgt if layer_index(n) in fin_lis]
        wlayers = [layers[li] for li in wlis]
        to_solve = [n for n in wtgt if n not in stq]

        # ---- 解碼事件:鮮 H(現部署形態前傳)→ 逐模組直解+殺閘 ----
        w_stats = []
        if to_solve:
            acc = HAccum()
            acc.attach([(n, get_module(model, n)) for n in to_solve])
            set_mode([stq[n] for n in wtgt if n in stq], "hard")
            with torch.no_grad():
                for bi in train_bi:
                    acc.mask = None if M is None else M[bi]
                    fwd_chain(wlayers, X[bi], kw)
            acc.mask = None
            acc.detach()
            for n in to_solve:
                Sig = acc.pop(n)
                T1f, T2f, alf, Weff, e_f, stats = solve_module(n, Sig)
                m = get_module(model, n)
                w_sol = (alf * (T1f.cuda().float()
                                + C3 * T2f.cuda().float())).reshape(Weff.shape)
                m.weight.data.copy_(w_sol.to(m.weight.dtype))   # 落格起點
                p = BlueprintRes(alf, grid3, r=args.lora_r, c=C3,
                                 learn_theta=args.learn_theta)
                p.load_dense(Weff - w_sol)          # 藍圖殘差旁路(凍結)
                parametrize.register_parametrization(m, "weight", p)
                stq[n] = p
                solved_t1[n] = T1f.cpu()
                land_eq += stats["agree_g3"] * stats["numel"]
                land_n += stats["numel"]
                w_stats.append({"n": n.split(".")[-3] + "." + n.split(".")[-1]
                                if n.count(".") > 2 else n, **{
                                    k: (round(v, 4) if isinstance(v, float)
                                        else v) for k, v in stats.items()
                                    if k != "numel"}})
                del Sig, Weff, w_sol
            acc.H.clear()
            torch.cuda.empty_cache()

        wp = [stq[n] for n in wtgt]

        # ---- 窗目標 Y = 藍圖 W₉ 理想函數(γ=1 off) ----
        for n in wtgt:
            stq[n].gamma = 1.0
        set_mode(wp, "off")
        with torch.no_grad():
            Y = run_layers(wlayers, X, kw)
        for n in wtgt:
            stq[n].gamma = 0.0

        # ---- 吸收:soft + STE 尾(lora/logit/dθ) ----
        set_mode(wp, "soft", s=args.s_plateau)
        with torch.no_grad():
            _ = fwd_chain(wlayers, X[0], kw)        # lazy LoRA init
        groups = [
            {"params": [q for n in wtgt for q in
                        (stq[n].lora_A, stq[n].lora_B)], "lr": args.lora_lr},
            {"params": [stq[n].logit for n in wtgt], "lr": args.factor_lr}]
        if args.learn_theta:
            groups.append({"params": [stq[n].dtheta for n in wtgt],
                           "lr": args.factor_lr})
        opt = torch.optim.Adam(groups)
        g_rng = torch.Generator().manual_seed(SEED + lo)
        ep_log, ep_hold = [], []

        def train_epoch(s, hard=False):
            set_mode(wp, "hard") if hard else set_mode(wp, "soft", s=s)
            tot = 0.0
            for oi in torch.randperm(len(train_bi),
                                     generator=g_rng).tolist():
                bi = train_bi[oi]
                opt.zero_grad(set_to_none=True)
                yh = fwd_chain(wlayers, X[bi], kw)
                if M is None:
                    loss = F.huber_loss(yh.float(), Y[bi].float())
                else:
                    mk = M[bi]
                    per = F.huber_loss(yh.float(), Y[bi].float(),
                                       reduction="none")
                    loss = ((per * mk[:, :, None]).sum()
                            / (mk.sum() * per.shape[-1]))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [q for g in opt.param_groups for q in g["params"]],
                    args.clip)
                opt.step()
                tot += float(loss.detach())
            m_loss = tot / len(train_bi)
            ep_log.append(round(m_loss, 6))
            if not np.isfinite(m_loss):
                raise RuntimeError(f"警報:窗 {lo} loss {m_loss}")
            if hold_bi:
                with torch.no_grad():
                    hl = 0.0
                    for bi in hold_bi:
                        pe = F.huber_loss(
                            fwd_chain(wlayers, X[bi], kw).float(),
                            Y[bi].float(), reduction="none")
                        if M is None:
                            hl += float(pe.mean())
                        else:
                            mk = M[bi]
                            hl += float((pe * mk[:, :, None]).sum()
                                        / (mk.sum() * pe.shape[-1]))
                    ep_hold.append(round(hl / len(hold_bi), 6))

        for _ in range(args.absorb_ep):
            train_epoch(args.s_plateau)
        for _ in range(args.tail_ste):
            train_epoch(None, hard=True)

        with torch.no_grad():
            set_mode(wp, "hard")
            y_hard = fwd_chain(wlayers, X[0], kw).float()
            set_mode(wp, "soft", s=args.s0)
            y_soft = fwd_chain(wlayers, X[0], kw).float()
            gdisc1 = float((y_soft - y_hard).norm()
                           / y_hard.norm().clamp_min(1e-12))
            set_mode(wp, "hard")

        # ---- 定案:extract → 落盤 → 材化 → 前進 X ----
        state_by_layer = {li: {} for li in fin_lis}
        w_flips = 0
        for n in ftgt:
            m = get_module(model, n)
            W0 = m.parametrizations.weight.original
            T1, T2, al, w_hard = stq[n].extract(W0)
            f = int((T1.cpu() != solved_t1[n]).sum())
            w_flips += f
            flips_vs_solved += f
            ntot += T1.numel()
            state_by_layer[layer_index(n)][n] = {
                "T1": T1.cpu(), "T2": T2.cpu(),
                "inv": torch.arange(W0.shape[1]), "a0": al.cpu(),
                "c": C3, "grid": G}
            parametrize.remove_parametrizations(m, "weight",
                                                leave_parametrized=False)
            m.weight.data.copy_(w_hard.to(m.weight.dtype))
            del stq[n], solved_t1[n]
        for li in fin_lis:
            torch.save(state_by_layer[li], QS / f"layer{li:02d}.pt")
        with torch.no_grad():
            X = run_layers([layers[li] for li in fin_lis], X, kw,
                           to_cpu=args.x_cpu)
        del Y
        torch.cuda.empty_cache()

        agree_cum = land_eq / max(land_n, 1)
        n_rev = sum(1 for s in w_stats if s["revert"])
        rec["windows"].append({
            "window": [lo, hi], "finalized": fin_lis,
            "loss": ep_log, **({"loss_holdout": ep_hold} if ep_hold else {}),
            "gdisc1_exit_rel": round(gdisc1, 6), "flips_vs_solved": w_flips,
            "reverted": n_rev, "agree_g3_cum": round(agree_cum, 4),
            "solve_stats": w_stats})
        print(f"window {lo:2d}-{hi - 1:2d}  absorb {len(ep_log)}ep  "
              f"loss {ep_log[0]:.5f}→{ep_log[-1]:.5f}  gdisc1={gdisc1:.4f}  "
              f"flips={w_flips}  rev={n_rev}  agree_g3={agree_cum:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

        if wi == LAND_WIN and not args.smoke and agree_cum >= LAND_TH:
            rec["abort"] = {"landing_probe": round(agree_cum, 4)}
            (EVC / f"train_{args.tag}.json").write_text(
                json.dumps(rec, indent=1, ensure_ascii=False))
            print(f"E59_ABORT_LANDING:agree_g3={agree_cum:.4f}≥{LAND_TH}"
                  f"(沒跳出盆地一,臂判死)", flush=True)
            sys.exit(3)

    del X
    torch.cuda.empty_cache()
    ce = eval_ce(model, val)
    rec["post_convert_val_ce"] = ce
    rec["flip_vs_solved_rate"] = round(flips_vs_solved / max(ntot, 1), 6)
    rec["agree_g3_final"] = round(land_eq / max(land_n, 1), 4)
    rec["runtime_s"] = round(time.time() - t0, 1)
    rec["timestamp"] = datetime.datetime.now().isoformat()
    if not args.smoke:
        st = load_state(args.tag, targets)
        np.savez(EVC / f"alphas_{args.tag}.npz",
                 **{n: st[n]["a0"].numpy() for n in st})
    (EVC / f"train_{args.tag}.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"{args.tag} post-convert val CE {ce:.6f}  "
          f"flip_vs_solved {rec['flip_vs_solved_rate']:.4f}  "
          f"agree_g3 {rec['agree_g3_final']:.4f}  "
          f"({rec['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main()

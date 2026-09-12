"""E37 phase2:凍結計量排水 sweep(prereg_e37_reservoir D4–D8)。

起點 = 冠軍態(qs_stmix_e2e 材化)+ 預訓水庫(res_fill 產物,凍結)。
逐窗滑動(win2/stride1);窗目標 Y = off 模式 + γ=1 = 「冠軍硬碼+滿水庫」
理想函數。窗內對本窗定案模組執行 γ 五格階梯(1→0,Δγ=0.2):

  每格 = 釋放事件(γ 下調)→ 逐 epoch:鮮 H(部署形態 no-grad 前傳)
  → R_eff = W₀ + (1−γ)·BA/r − W_hard(未吸收自動累積,免記帳)
  → greedy_comp_grid 翻碼 + 梯度(LoRA/logit/dθ)吸收連續部分
  → 水位閉環:‖R_eff‖_H ≤ τ×放水後基準 或 等滿 W_max epoch → 下一格。

γ≡0 斷言先於 STE 尾(extract 殘水即丟失,D6 流向④);尾段 s 退火
s_plateau→s0 + STE 硬收尾。無走廊、無 morph、無老師(D7)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.train_res \
      --tag st37r --state-tag stmix_e2e --res-tag res37 \
      --learn-theta --holdout-frac 0.05 --resume
"""
import argparse
import datetime
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from p2_anchor.wringer.comp import HAccum, greedy_comp_grid
from p2_anchor.wringer.comp_joint import build_tuple_codebook, joint_comp_grid
from p2_anchor.wringer.engine import (capture_layer0, equivalence_gate,
                                        fwd_chain, run_layers)
from p2_anchor.wringer.ktier import k2_weight, round_with_mids
from p2_anchor.wringer.modelio import (LAYER_PREFIX, N_LAYERS,
                                         enumerate_targets, get_module,
                                         layer_index, load_model)
from p2_anchor.wringer.state import dense_weight, load_state, qs_dir
from p2_anchor.wringer.mixls import refit_alpha, solve_left_ls, solve_mix_ls
from p2_anchor.wringer.asym import (asym_V, err_h, quantize_r, r_stats,
                                      solve_r_blk)
from p2_anchor.wringer.ktier import scale_joint
from p2_anchor.wringer.stq import STQParam, SoftGrid
from p2_anchor.wringer.train_st import set_mode

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"
SEED = 20260806

# E41-2:6 層帶零逃逸率 P(3元非零|9元零)(diag_zero_overlap_e40_vs_9e @40be701)
ESCAPE_BAND = [0.0014, 0.0021, 0.0036, 0.0100, 0.0306, 0.0673]


def strain_wmax(li, gsteps):
    """逐層 W_max:目標 epoch 帶 25·(1+2·s_l) / gsteps − 1(s_l=帶內插值 1 歸一)。

    淺層還原基線 4(25·1.04/5−1);末帶 L30-31 → 14(上限 75 epoch,τ 閉環可早退)。
    """
    s = ESCAPE_BAND[min(li // 6, len(ESCAPE_BAND) - 1)] / ESCAPE_BAND[-1]
    return max(1, round(25 * (1 + 2 * s) / gsteps) - 1)


def split_groups(s, g_new):
    """g32→g16 組細分:a0 逐組複製、T1/T2 reshape——起點權重逐位保值。"""
    M, nB, B = s["T1"].shape
    assert B % g_new == 0, f"block {B} 不整除 g{g_new}"
    f = B // g_new
    if f == 1:
        return s
    return dict(s, T1=s["T1"].reshape(M, nB * f, g_new),
                T2=s["T2"].reshape(M, nB * f, g_new),
                a0=s["a0"].repeat_interleave(f, dim=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="st37r")
    ap.add_argument("--state-tag", default="stmix_e2e")
    ap.add_argument("--res-tag", default="res37")
    ap.add_argument("--n-calib", type=int, default=512)
    ap.add_argument("--calib", default=str(EV / "calib_train.pt"))
    ap.add_argument("--lengths", default=None,
                    help="calib 逐列有效長度 .pt(E54 8k pad 遮罩:"
                         "H 累積/窗 loss/holdout 只計有效位置;"
                         "None=舊行為逐位不變)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--win", type=int, default=2)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--gsteps", type=int, default=5, help="γ 格數(Δγ=1/格數)")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="水位閉環:‖R_eff‖_H ≤ τ×放水後基準 → 下一格")
    ap.add_argument("--w-max", type=int, default=4,
                    help="每格最大等待 epoch(超時強制下一格,殘量自動累積)")
    ap.add_argument("--comp-rounds", type=int, default=3)
    ap.add_argument("--comp-cap", type=float, default=0.03)
    ap.add_argument("--tail-soft", type=int, default=10,
                    help="排空後 s 退火 epoch 數(s_plateau→s0)")
    ap.add_argument("--tail-ste", type=int, default=4)
    ap.add_argument("--s0", type=float, default=30.0)
    ap.add_argument("--s-plateau", type=float, default=2.0)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-lr", type=float, default=6e-4)
    ap.add_argument("--factor-lr", type=float, default=3e-3)
    ap.add_argument("--theta-lr", type=float, default=None)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--learn-theta", action="store_true")
    ap.add_argument("--holdout-frac", type=float, default=0.0)
    ap.add_argument("--x-cpu", action="store_true",
                    help="X 激活快取駐 CPU 逐批搬運(Y 照舊駐 GPU;"
                         "E58R1b:n-calib 384@16k X+Y 同駐 GPU 爆 95G 之修)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--res-zero", action="store_true",
                    help="O1 排空對照:載入水庫後 B 歸零(空水走全鏈,"
                         "裁排水機器稅是否與水內容無關)")
    ap.add_argument("--group-override", default=None,
                    help="E41-1 深層細粒:'30-35:16' = 該層帶 α 組細分至 g16"
                         "(a0 複製、T1/T2 reshape,起點權重逐位保值)")
    ap.add_argument("--strain-wmax", action="store_true",
                    help="E41-2 應變導引:逐窗 W_max 按零逃逸剖面加權"
                         "(diag_zero_overlap @40be701;τ 不動)")
    ap.add_argument("--calib-domains", default=None,
                    help="E43 桿2/3:calib 逐列域標籤 json(與 --calib 列序對齊)")
    ap.add_argument("--domain-weights", default=None,
                    help="梯度路吸力再平衡:'ifshape:2.0,code:1.0'(未列=1;"
                         "solver 路 H 由 calib 成分掌管,兩桿正交)")
    ap.add_argument("--dom-exit", default=None,
                    help="分域水位退出:'ifshape'=該域 epoch loss 未達 "
                         "τ_dom×放格基準前不進下一格(τ 閉環 AND 條件)")
    ap.add_argument("--tau-dom", type=float, default=None,
                    help="分域退出比(預設沿用 --tau)")
    ap.add_argument("--beta-denoise", type=float, default=0.0,
                    help="E44 去噪排水:窗目標 (1−β)·T_w(X)+β·T_w(XT),"
                         "XT=乾淨理想軌跡(冠軍+滿水庫逐窗前進)。"
                         "0=現行(目標僅條件於學生漂移流);1=全去噪"
                         "(每窗須同時實現理想變換並收斂上游三元誤差)")
    ap.add_argument("--gamma-field", action="store_true",
                    help="E44 培養排水:棄 γ 階梯,ftgt 逐批隨機 γ~U(0,1) "
                         "訓練(學整個去噪場而非單一路徑點);部署基態 γ=0,"
                         "comp 以 cap 階梯自主吸收滿水;路徑湧現非設計")
    ap.add_argument("--field-epochs", type=int, default=0,
                    help="γ-場模式排水 epoch 預算(0=自動 gsteps×(w_max+1),"
                         "對齊階梯模式典型 72ep)")
    ap.add_argument("--stop-after-layer", type=int, default=None,
                    help="E46 蓄排一體化:定案至此層(含)後寫 partial rec、"
                         "印 STOP_AFTER_LAYER 哨兵並退出;配 res_refill 交錯,"
                         "之後同 tag --resume 續排")
    ap.add_argument("--smoke", action="store_true",
                    help="2 窗×32 calib×2 格,驗接線/事件/γ=0 斷言/resume")
    ap.add_argument("--mix-block", type=int, default=0,
                    help="E61 擴桶:輸入維逐塊可學混合 M(b×b,初始 I),"
                         "部署形態 W_hard@blockdiag(M),不吸收不進碼;"
                         "0=關(逐位舊行為)")
    ap.add_argument("--mix-lr", type=float, default=1e-4)
    ap.add_argument("--mix-ls", action="store_true",
                    help="E61 st61c:每 comp 事件 greedy 後對錨定目標 W_t 做 "
                         "(α,M) 閉式重解(mixls.solve_mix_ls + refit_alpha)")
    ap.add_argument("--mix-ls-ridge", type=float, default=1.0)
    ap.add_argument("--mix-ls-cap", type=float, default=0.05)
    ap.add_argument("--mix-ls-iters", type=int, default=25)
    ap.add_argument("--mix-ls-sv-floor", type=float, default=0.3)
    ap.add_argument("--no-alpha-ls", action="store_true",
                    help="只解 M 不重解 α(消融)")
    ap.add_argument("--lmix-block", type=int, default=0,
                    help="E62b:輸出側列塊 L 桶大小(0=關);只掛在 --lmix-targets 後綴模組")
    ap.add_argument("--lmix-targets", default="mlp.down_proj",
                    help="逗號分隔模組後綴;探針 @probe_e62b_win:down_proj L 0.11 b/w(int8)")
    ap.add_argument("--lmix-ls-ridge", type=float, default=0.0)
    ap.add_argument("--lmix-ls-cap", type=float, default=0.25)
    ap.add_argument("--tail-ls", action="store_true",
                    help="E62a:尾段(退火+STE)後對終態碼重解 M(碼不動;"
                         "鮮 H、錨定目標 W_t)+ holdout(hard)回退閘")
    ap.add_argument("--tail-ls-ridge", type=float, default=0.1)
    ap.add_argument("--tail-ls-cap", type=float, default=0.25)
    ap.add_argument("--tail-ls-iters", type=int, default=40)
    ap.add_argument("--tail-ls-tol", type=float, default=0.02,
                    help="回退閘:holdout(hard) after > before×(1+tol) 則撤回")
    ap.add_argument("--comp-solver", choices=("greedy", "joint", "vq"), default="greedy",
                    help="E64 S3:comp 事件求解器——greedy(現行)/ joint(d 欄聯合打分,全格)/ "
                         "vq(聯合 + 元組碼書,窗首事件依現行碼頻率建 --vq-k 種)")
    ap.add_argument("--vq-d", type=int, default=4)
    ap.add_argument("--vq-k", type=int, default=64)
    ap.add_argument("--asym-r-bits", type=int, default=0,
                    help="E63 S1:comp 事件 greedy 後 (α, r) 交替閉式 → r 逐 block 量化 "
                         "bits 位元碼書 → α 重擬合(V=T⁺−r_q T⁻);0=關(逐位舊行為);"
                         "只支援 grid 3、無 M/L")
    ap.add_argument("--asym-alt", type=int, default=2,
                    help="(α, r) 交替次數(量化前)")
    ap.add_argument("--tail-alpha-ls", action="store_true",
                    help="尾段重解亦重解 α(會改碼;預設關,M-only 保碼不動)")
    args = ap.parse_args()
    if args.smoke:
        args.n_calib, args.batch = 32, 4
        args.gsteps, args.w_max = 2, 1
        args.tail_soft, args.tail_ste = 2, 1

    t0 = time.time()
    torch.manual_seed(SEED)
    QS = qs_dir(args.tag)
    QS.mkdir(parents=True, exist_ok=True)
    EVC.mkdir(parents=True, exist_ok=True)
    model, tok = load_model()
    model.requires_grad_(False)
    targets = enumerate_targets(model)
    st0 = load_state(args.state_tag, targets)     # 冠軍態(P0;混元 grids)
    if args.group_override:                       # E41-1:載入即細分(保值)
        band, g_new = args.group_override.split(":")
        b_lo, b_hi = (int(x) for x in band.split("-"))
        g_lis = set(range(b_lo, b_hi + 1))
        n_split = 0
        for n in targets:
            if layer_index(n) in g_lis:
                st0[n] = split_groups(st0[n], int(g_new))
                n_split += 1
        print(f"group-override L{b_lo}-{b_hi} → g{g_new}"
              f"({n_split} tensors 細分,起點保值)", flush=True)
    for n in targets:                             # 材化(res_fill 已過 P0 gate)
        m = get_module(model, n)
        m.weight.data.copy_(dense_weight(st0[n]).to(m.weight.dtype))
    res_dir = Path("data") / f"res_{args.res_tag}"
    res_st = {}
    for li in range(N_LAYERS):
        p = res_dir / f"layer{li:02d}.pt"
        if p.exists():
            res_st.update(torch.load(p, map_location="cpu",
                                     weights_only=False))
    assert all(n in res_st for n in targets), f"res_{args.res_tag} 不齊"
    if args.res_zero:                             # O1:BA/r ≡ 0,結構照舊
        for n in targets:
            res_st[n] = dict(res_st[n], B=torch.zeros_like(res_st[n]["B"]))
        print("res-zero:水庫 B 已歸零(O1 排空對照)", flush=True)

    calib = torch.load(args.calib, weights_only=False)[:args.n_calib]
    M = None                                      # E54:逐批 (B,S) 有效遮罩
    if args.lengths:
        lens = torch.load(args.lengths,
                          weights_only=False)[:calib.shape[0]]
        assert lens.shape[0] == calib.shape[0], "lengths 與 calib 列數不齊"
        S = calib.shape[1]
        ar = torch.arange(S)
        M = [(ar[None] < lens[i:i + args.batch, None]).cuda()
             for i in range(0, calib.shape[0], args.batch)]
        print(f"lengths 遮罩:有效率 "
              f"{float(lens.sum()) / (lens.shape[0] * S):.3f}"
              f" / 滿窗列 {(lens == S).sum().item()}", flush=True)
    doms = wvec = None                            # E43 桿2:逐列域標籤+權重
    if args.calib_domains:
        doms = json.load(open(args.calib_domains))[:calib.shape[0]]
        assert len(doms) == calib.shape[0], "域標籤與 calib 列數不齊"
        wmap = {}
        if args.domain_weights:
            wmap = {k: float(v) for k, v in
                    (kv.split(":") for kv in args.domain_weights.split(","))}
        wvec = torch.tensor([wmap.get(d, 1.0) for d in doms]).cuda()
    val = torch.load(EV / "calib_val.pt", weights_only=False)
    from p1_grouping.e3_runner import eval_ce

    if args.x_cpu:
        assert args.beta_denoise == 0.0, "--x-cpu 未支援去噪 XT/YT 路徑"
    X, kw = capture_layer0(model, calib, batch=args.batch,
                           to_cpu=args.x_cpu)
    layers = [model.get_submodule(f"{LAYER_PREFIX}.{li}")
              for li in range(N_LAYERS)]
    rel = equivalence_gate(model, layers, X, kw, calib, batch=args.batch,
                           upto=args.win)
    print(f"captured X: {len(X)}×{tuple(X[0].shape)}  等價閘 PASS "
          f"rel={rel:.2e}", flush=True)

    assert 0.0 <= args.beta_denoise <= 1.0, "--beta-denoise 須在 [0,1]"
    XT = ([x.clone() for x in X] if args.beta_denoise > 0 else None)
    if XT is not None:
        print(f"E44 去噪排水:β={args.beta_denoise}(乾淨流 XT 已建)",
              flush=True)

    starts = list(range(0, N_LAYERS - args.win + 1, args.stride))
    windows = [(s, s + args.win) for s in starts]
    if args.smoke:
        windows = windows[:2]

    n_hold = int(round(args.holdout_frac * len(X)))
    train_bi = list(range(len(X) - n_hold))
    hold_bi = list(range(len(X) - n_hold, len(X)))

    gammas = [1.0 - (i + 1) / args.gsteps for i in range(args.gsteps)]
    grids, stq, ba_cache = {}, {}, {}
    rec = {"tag": args.tag, "grid": "mix(state)", "state_tag": args.state_tag,
           "res_tag": args.res_tag, "seed": SEED, "equiv_gate_rel": rel,
           "hparams": vars(args), "windows": []}
    flips_vs_p0, ntot = 0, 0

    for wi, (lo, hi) in enumerate(windows):
        fin_hi = hi if wi == len(windows) - 1 else lo + args.stride
        fin_lis = list(range(lo, fin_hi))
        wlis = list(range(lo, hi))

        if args.resume and all((QS / f"layer{li:02d}.pt").exists()
                               for li in fin_lis):
            if XT is not None:      # 乾淨流理想前進(須在硬碼拷貝前:
                with torch.no_grad():  # 此刻 m.weight 仍是冠軍態)
                    for n in [t for t in targets
                              if layer_index(t) in fin_lis]:
                        m = get_module(model, n)
                        m.weight.data.add_(
                            (res_st[n]["B"].float().cuda()
                             @ res_st[n]["A"].float().cuda()
                             / res_st[n]["r"]).to(m.weight.dtype))
                    XT = run_layers([layers[li] for li in fin_lis], XT, kw)
            for li in fin_lis:
                for n, c in torch.load(QS / f"layer{li:02d}.pt",
                                       weights_only=False).items():
                    m = get_module(model, n)
                    m.weight.data.copy_(dense_weight(c).to(m.weight.dtype))
            with torch.no_grad():
                X = run_layers([layers[li] for li in fin_lis], X, kw,
                               to_cpu=args.x_cpu)
            print(f"window [{lo},{hi}) resume-skip(fin={fin_lis})",
                  flush=True)
            continue

        wtgt = [n for n in targets if layer_index(n) in wlis]
        ftgt = [n for n in wtgt if layer_index(n) in fin_lis]  # 本窗排水+定案
        for n in wtgt:
            if n not in stq:
                m = get_module(model, n)
                gn = int(st0[n]["grid"])
                cn = float(st0[n]["c"])
                if (gn, cn) not in grids:
                    grids[(gn, cn)] = SoftGrid(c=cn, n=gn)
                lb = (args.lmix_block if any(n.endswith(sfx) for sfx in
                                             args.lmix_targets.split(","))
                      else 0)
                p = STQParam(st0[n]["a0"].cuda(), grids[(gn, cn)],
                             r=args.lora_r, c=cn,
                             learn_theta=args.learn_theta, use_comp=True,
                             mix_block=args.mix_block, lmix_block=lb)
                p.load_reservoir(res_st[n]["B"], res_st[n]["A"],
                                 res_st[n]["r"], gamma=1.0)
                parametrize.register_parametrization(m, "weight", p)
                stq[n] = p
            if n not in ba_cache:             # BA 凍結 → 窗內快取
                q = stq[n]
                ba_cache[n] = (q.res_B @ q.res_A) / q._r_res
        wp = [stq[n] for n in wtgt]
        wlayers = [layers[li] for li in wlis]

        set_mode(wp, "off")                   # off + γ=1 = 冠軍+滿水庫理想
        with torch.no_grad():
            Y = run_layers(wlayers, X, kw)

        drift_rel = None
        if XT is not None:                    # E44:乾淨軌跡目標(去噪排水)
            with torch.no_grad():
                YT = run_layers(wlayers, XT, kw)
                num = sum(float((yt.float() - y.float()).norm() ** 2)
                          for yt, y in zip(YT, Y))
                den = sum(float(yt.float().norm() ** 2) for yt in YT)
                drift_rel = (num / max(den, 1e-24)) ** 0.5
                b = args.beta_denoise         # 就地混合,省一份顯存
                for bi in range(len(Y)):
                    Y[bi].mul_(1.0 - b).add_(YT[bi].to(Y[bi].dtype), alpha=b)
                del YT

        set_mode(wp, "soft", s=args.s_plateau)
        with torch.no_grad():
            _ = fwd_chain(wlayers, X[0], kw)  # lazy LoRA/comp init
        groups = [
            {"params": [q for n in wtgt for q in
                        (stq[n].lora_A, stq[n].lora_B)], "lr": args.lora_lr},
            {"params": [stq[n].logit for n in wtgt], "lr": args.factor_lr}]
        if args.learn_theta:
            groups.append({"params": [stq[n].dtheta for n in wtgt],
                           "lr": args.theta_lr or args.factor_lr})
        if args.mix_block > 0:                # E61 擴桶:M 梯度吸水載體
            groups.append({"params": [stq[n].mix for n in wtgt],
                           "lr": args.mix_lr})
        if args.lmix_block > 0:               # E62b:輸出側 L(同 mix_lr)
            lp = [stq[n].lmix for n in wtgt if stq[n].lmix is not None]
            if lp:
                groups.append({"params": lp, "lr": args.mix_lr})
        opt = torch.optim.Adam(groups)

        w_max_here = (strain_wmax(lo, args.gsteps) if args.strain_wmax
                      else args.w_max)
        g_rng = torch.Generator().manual_seed(SEED + lo)
        ep_log, ep_hold, drain_events = [], [], []

        dom_ep = {}                               # E43:本 epoch 分域 loss

        def train_epoch(s, hard=False, field=False):
            if hard:
                set_mode(wp, "hard")
            else:
                set_mode(wp, "soft", s=s)
            tot = 0.0
            dsum, dcnt = defaultdict(float), defaultdict(int)
            for oi in torch.randperm(len(train_bi),
                                     generator=g_rng).tolist():
                bi = train_bi[oi]
                if field:                     # γ-場:逐批隨機噪聲位準
                    gb = float(torch.rand((), generator=g_rng))
                    for n in ftgt:
                        stq[n].gamma = gb
                opt.zero_grad(set_to_none=True)
                yh = fwd_chain(wlayers, X[bi], kw)
                if doms is None:
                    if M is None:
                        loss = F.huber_loss(yh.float(), Y[bi].float())
                    else:                         # E54:遮罩均值(同尺度)
                        mk = M[bi]
                        per = F.huber_loss(yh.float(), Y[bi].float(),
                                           reduction="none")
                        loss = ((per * mk[:, :, None]).sum()
                                / (mk.sum() * per.shape[-1]))
                else:                             # 桿2:梯度路逐列加權
                    pe = F.huber_loss(yh.float(), Y[bi].float(),
                                      reduction="none")
                    if M is None:
                        per = pe.mean(dim=(1, 2))
                    else:
                        mk = M[bi]
                        per = ((pe * mk[:, :, None]).sum(dim=(1, 2))
                               / (mk.sum(dim=1) * pe.shape[-1]))
                    r0 = bi * args.batch
                    wb = wvec[r0:r0 + per.shape[0]]
                    loss = (per * wb).sum() / wb.sum()
                    for j in range(per.shape[0]):
                        dsum[doms[r0 + j]] += float(per[j])
                        dcnt[doms[r0 + j]] += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [q for g in opt.param_groups for q in g["params"]],
                    args.clip)
                opt.step()
                tot += float(loss)
            if field:                         # 部署基態還原(holdout/comp
                for n in ftgt:                # 一律在 γ=0 部署形態量測)
                    stq[n].gamma = 0.0
            m_loss = tot / len(train_bi)
            ep_log.append(round(m_loss, 6))
            if doms is not None:
                dom_ep.clear()
                dom_ep.update({d: dsum[d] / dcnt[d] for d in dsum})
            if not np.isfinite(m_loss):
                raise RuntimeError(f"警報:窗 {lo} loss {m_loss}")
            if hold_bi:
                with torch.no_grad():
                    if M is None:
                        hl = sum(float(F.huber_loss(
                            fwd_chain(wlayers, X[bi], kw).float(),
                            Y[bi].float())) for bi in hold_bi) / len(hold_bi)
                    else:
                        hl = 0.0
                        for bi in hold_bi:
                            pe = F.huber_loss(
                                fwd_chain(wlayers, X[bi], kw).float(),
                                Y[bi].float(), reduction="none")
                            mk = M[bi]
                            hl += float((pe * mk[:, :, None]).sum()
                                        / (mk.sum() * pe.shape[-1]))
                        hl /= len(hold_bi)
                ep_hold.append(round(hl, 6))

        vq_allowed = {}                        # E64 S3:每窗每模組元組碼書(首事件建)

        def comp_event(gstep_i, released):
            """鮮 H → R_eff → solver;回傳窗級水位 Σ‖R_eff‖_H。"""
            acc = HAccum()
            acc.attach([(n, get_module(model, n)) for n in ftgt])
            set_mode(wp, "hard")              # 部署形態(含當前 γ)
            with torch.no_grad():
                for bi in train_bi:
                    acc.mask = None if M is None else M[bi]
                    fwd_chain(wlayers, X[bi], kw)
            acc.mask = None
            acc.detach()
            level = 0.0
            ev = {"gamma": round(1.0 - (gstep_i + 1) / args.gsteps, 2)
                  if released else None, "flips": 0, "absorb": [],
                  "obj": [], "ev_flips": 0, "revisit": 0}
            for n in ftgt:
                q = stq[n]
                Hn = acc.pop(n)
                mq = get_module(model, n)
                W0q = mq.parametrizations.weight.original
                with torch.no_grad():
                    u_, al_ = q._u(W0q)
                    _, _, v_h = round_with_mids(
                        u_, q.theta_eff().detach(), q.grid.vals,
                        q.grid.t1, q.grid.t2)
                    w_hard = (al_ * q._apply_rneg(v_h)).reshape(W0q.shape)
                    W_t = W0q.float() + (1.0 - q.gamma) * ba_cache[n]
                    if q.mix is None:
                        R = W_t - w_hard
                    else:
                        # E61:部署 W_eff = w_hard@Bd;虧欠拉回碼座標
                        #   ‖W_t − w_hard Bd‖_H = ‖W_t Bd⁻¹ − w_hard‖_{Bd H Bdᵀ}
                        # greedy 在碼座標對 H' 打分,水位語義不變
                        # amendment_1(S1 排水後發現):W0q 為碼座標(前向
                        # 先量化再右乘 Bd),只有水 BA 在部署座標(mix 之外
                        # 相加)。舊式 W_t·Bd⁻¹ 把 W0q 也拉回,製造偽殘差
                        # W0q(Bd⁻¹−I)(量級≈水本身);S1 st61a 帶此偏差跑完。
                        Bd = q.mix_dense()
                        Bdi = q.mix_dense_inv()
                        water = (1.0 - q.gamma) * ba_cache[n]
                        Ld = q.lmix_dense()          # E62b:None=無 L
                        if Ld is None:
                            Rd = (W0q.float() - w_hard) @ Bd + water
                        else:                        # 部署 W_eff = L w_hard Bd
                            Rd = Ld @ ((W0q.float() - w_hard) @ Bd) + water
                            water = q.lmix_dense_inv() @ water   # 拉回 L 內側
                        obj_mix = float(((Rd @ Hn) * Rd).sum())
                        Rn = W_t - w_hard
                        obj_nomix = float(((Rn @ Hn) * Rn).sum())
                        ev.setdefault("mix_gain", []).append(
                            round(1.0 - obj_mix / max(obj_nomix, 1e-30), 4))
                        ev.setdefault("mix_dist", []).append(round(float(
                            (q.mix.detach() - torch.eye(
                                q.mix.shape[1], device=q.mix.device)
                             ).norm() / q.mix.shape[0] ** 0.5), 5))
                        R = W0q.float() + water @ Bdi - w_hard
                        H0 = Hn if args.mix_ls else None   # LS 用原度量
                        Hn = Bd @ Hn @ Bd.T
                        del Rd, Rn, Bd, Bdi, water, Ld
                    if args.comp_solver == "greedy":
                        clog = greedy_comp_grid(q, W0q, R, Hn,
                                                rounds=args.comp_rounds,
                                                cap=args.comp_cap, rneg=q.rneg)
                    else:                        # E64 S3:聯合求解器 / 碼書
                        allowed = None
                        if args.comp_solver == "vq":
                            if n not in vq_allowed:
                                with torch.no_grad():
                                    u0, _ = q._u(W0q)
                                    idx0 = torch.bucketize(
                                        u0.reshape(W0q.shape).float(),
                                        q.theta_eff().detach().float())
                                vq_allowed[n] = build_tuple_codebook(
                                    idx0, args.vq_d, args.vq_k, q.grid.n)
                                del u0, idx0
                            allowed = vq_allowed[n]
                        clog = joint_comp_grid(q, W0q, R, Hn,
                                               rounds=args.comp_rounds,
                                               cap=args.comp_cap, rneg=q.rneg,
                                               d=args.vq_d, allowed=allowed)
                        ev.setdefault("conform", []).append(clog["conform_flips"])
                    if args.asym_r_bits > 0:
                      # E63 S1:greedy 後對錨定目標 W_t 做 (α, r) 交替閉式 →
                      # r 逐 block 量化(每模組 log 域 Lloyd 碼書)→ α 重擬合;
                      # r 進 buffer(STE 不學);水位改記閉式後殘差
                      assert q.mix is None and q.lmix is None and q.grid.n == 3
                      with torch.no_grad():
                        u_, al_ = q._u(W0q)
                        _, _, v_h = round_with_mids(
                            u_, q.theta_eff().detach(), q.grid.vals,
                            q.grid.t1, q.grid.t2)
                        T3 = v_h.float()
                        W3 = W_t.reshape(T3.shape)
                        alpha = al_.detach().float()
                        e_g = err_h(W_t - (alpha * q._apply_rneg(T3)
                                           ).reshape(W_t.shape), Hn)
                        r = q.rneg
                        for _ in range(args.asym_alt):
                            r = solve_r_blk(W3, T3, alpha, Hn)
                            alpha = scale_joint(W3, asym_V(T3, r), Hn)
                        r_q, cb = quantize_r(r, args.asym_r_bits)
                        _, ast = refit_alpha(q, W_t, asym_V(T3, r_q), Hn)
                        q.set_rneg(r_q, cb)
                        u_n, al_n = q._u(W0q)
                        _, _, v_n = round_with_mids(
                            u_n, q.theta_eff().detach(), q.grid.vals,
                            q.grid.t1, q.grid.t2)
                        e_a = err_h(W_t - (al_n * q._apply_rneg(T3)
                                           ).reshape(W_t.shape), Hn)
                        e_d = err_h(W_t - (al_n * q._apply_rneg(v_n)
                                           ).reshape(W_t.shape), Hn)
                        ev.setdefault("asym_ls", []).append({
                            "gain": round(1.0 - e_a / max(e_g, 1e-30), 4),
                            "gain_dep": round(1.0 - e_d / max(e_g, 1e-30), 4),
                            "refit_flip": round(float((v_n != v_h).float().mean()), 6),
                            "r": r_stats(r_q), "cb": [round(float(x), 4)
                                                     for x in cb],
                            **ast})
                        clog["obj_after"] = e_a
                        del v_h, T3, W3, alpha, r, r_q, cb, al_n, u_n, v_n   # u_/al_ 由外層 del
                    if q.mix is not None and args.mix_ls:
                      # st61c:greedy 後以錨定目標 W_t(部署座標)閉式重解 (α,M)
                      with torch.no_grad():
                        u_, al_ = q._u(W0q)
                        _, _, v_h = round_with_mids(
                            u_, q.theta_eff().detach(), q.grid.vals,
                            q.grid.t1, q.grid.t2)
                        C = (al_ * v_h).reshape(W0q.shape).float()
                        Ld = q.lmix_dense()
                        # E62b:X 解的載體含 L(部署 W_eff = L C Bd(X))
                        X_new, lst = solve_mix_ls(
                            C if Ld is None else Ld @ C, W_t, H0,
                            q.mix.detach().float(),
                            ridge=args.mix_ls_ridge, cap=args.mix_ls_cap,
                            iters=args.mix_ls_iters,
                            sv_floor=args.mix_ls_sv_floor)
                        q.mix.copy_(X_new)
                        if Ld is not None:           # E62b:L 解(固定碼與 X)
                            Z = C @ q.mix_dense()
                            L_new, llst = solve_left_ls(
                                Z, W_t, H0, Ld, q.lmix.shape[1],
                                ridge=args.lmix_ls_ridge,
                                cap=args.lmix_ls_cap, iters=args.mix_ls_iters,
                                sv_floor=args.mix_ls_sv_floor)
                            kb = q.lmix.shape[0]; bb = q.lmix.shape[1]
                            idx = torch.arange(kb, device=L_new.device)
                            q.lmix.copy_(L_new.reshape(kb, bb, kb, bb)[idx, :, idx, :])
                            ev.setdefault("lmix_ls", []).append(
                                {k_: llst[k_] for k_ in ("ls_gain", "step_max",
                                                         "t", "cg_iters",
                                                         "sv_min", "sv_floor_hits")})
                            del Z, L_new
                        if not args.no_alpha_ls:
                            Bd = q.mix_dense()
                            tgt = W_t @ q.mix_dense_inv()
                            if Ld is not None:       # 碼座標目標 L⁻¹ W_t Bd⁻¹
                                tgt = q.lmix_dense_inv() @ tgt
                            _, ast = refit_alpha(q, tgt, v_h, Bd @ H0 @ Bd.T)
                            lst.update(ast)
                            del Bd, tgt
                        keys = {"ls_gain", "step_max", "t", "cg_iters",
                                "sv_min", "sv_floor_hits"}
                        if not args.no_alpha_ls:
                            keys |= {"alpha_rel_change", "alpha_clip_frac"}
                        ev.setdefault("mix_ls", []).append(
                            {k_: lst[k_] for k_ in sorted(keys)})
                        del C, X_new, H0, Ld
                    # 事件後部署碼快照 → 全量事件翻碼 + revisit 計數
                    # (裁「多權重各翻一次 vs 少數權重往返震盪」;monitor-only)
                    u2, _ = q._u(W0q)
                    _, _, v2 = round_with_mids(
                        u2, q.theta_eff().detach(), q.grid.vals,
                        q.grid.t1, q.grid.t2)
                    v2i = torch.searchsorted(
                        q.grid.vals, v2.detach().reshape(-1).contiguous()
                    ).reshape(v2.shape).to(torch.int8)
                    del v2, u2
                    if n not in vh_p0:
                        vh_p0[n] = v2i.clone()
                    else:
                        ch = v2i != vh_prev[n]
                        ev["ev_flips"] += int(ch.sum())
                        ev["revisit"] += int((ch & (v2i == vh_p0[n])).sum())
                    vh_prev[n] = v2i
                level += clog["obj_after"]
                ev["flips"] += clog["flips"]
                ev["absorb"].append(clog["absorb"])
                ev["obj"].append(round(clog["obj_after"], 4))
                del R, Hn, u_, al_, w_hard
            acc.H.clear()
            return level, ev

        # ---- 排水段:γ 階梯 + 水位閉環(每 epoch 一事件 + 一訓練輪) ----
        vh_p0, vh_prev = {}, {}                # 事件碼快照(revisit 儀器)
        exit_doms = ([d for d in args.dom_exit.split(",") if d]
                     if (args.dom_exit and doms is not None) else [])
        tau_dom = args.tau_dom if args.tau_dom is not None else args.tau
        dom_post = {}
        gi, wait, n_post = 0, 0, None
        ep = 0
        if args.gamma_field:
            # E44 培養排水:部署基態 γ=0,水不按階梯放——訓練逐批隨機
            # γ~U(0,1)(窗學整個去噪場),comp 以 cap 階梯對滿水殘差
            # 自主分期吸收;排水路徑=湧現,非設計
            F_EP = args.field_epochs or args.gsteps * (args.w_max + 1)
            cadence = max(1, F_EP // args.gsteps)
            for n in ftgt:
                stq[n].gamma = 0.0
            for ep in range(F_EP):
                if ep % cadence == 0:
                    level, ev = comp_event(-1, False)
                    ev.update({"ep": ep, "level": round(level, 4),
                               "wait": 0, "gamma_now": 0.0,
                               "field": True})
                    drain_events.append(ev)
                train_epoch(args.s_plateau, field=True)
                if doms is not None:
                    drain_events[-1]["dom_loss"] = {
                        d: round(v, 6) for d, v in dom_ep.items()}
            level, ev = comp_event(-1, False)  # 收官事件(定案前水位)
            ev.update({"ep": F_EP, "level": round(level, 4), "wait": 0,
                       "gamma_now": 0.0, "field": True})
            drain_events.append(ev)
            ep = F_EP
        while (not args.gamma_field) and gi < args.gsteps:
            released = (n_post is None)
            if released:                       # 放下一格
                for n in ftgt:
                    stq[n].gamma = gammas[gi]
            level, ev = comp_event(gi, released)
            if released:
                n_post, wait = max(level, 1e-30), 0
            else:
                wait += 1
            ev.update({"ep": ep, "level": round(level, 4),
                       "wait": wait,
                       "gamma_now": round(stq[ftgt[0]].gamma, 2)})
            drain_events.append(ev)
            train_epoch(args.s_plateau)
            if doms is not None:               # 分域儀表入事件流
                ev["dom_loss"] = {d: round(v, 6) for d, v in dom_ep.items()}
            if released and exit_doms:         # 桿3:放格基準(同 n_post 語義)
                dom_post = {d: max(dom_ep.get(d, 0.0), 1e-30)
                            for d in exit_doms}
            ep += 1
            dom_ok = (all(dom_ep.get(d, 0.0) <= tau_dom * dom_post[d]
                          for d in exit_doms) if exit_doms else True)
            if (level <= args.tau * n_post and dom_ok) \
                    or wait >= w_max_here:
                gi += 1
                n_post = None                  # 觸發下一格釋放
        for n in ftgt:                         # γ≡0 斷言(D6④ 前哨)
            assert stq[n].gamma == 0.0, f"{n} γ={stq[n].gamma}≠0"
        drain_eps = ep

        # ---- 尾段:s 退火 + STE 硬收尾 ----
        for te in range(args.tail_soft):
            s = args.s_plateau + (te + 1) / args.tail_soft * (
                args.s0 - args.s_plateau)
            train_epoch(s)
        for _ in range(args.tail_ste):
            train_epoch(None, hard=True)

        # ---- E62a:窗尾 M 重解(終態碼固定)+ holdout(hard)回退閘 ----
        def hold_loss_hard():
            bis = hold_bi if hold_bi else train_bi[:2]
            with torch.no_grad():
                set_mode(wp, "hard")
                tot = 0.0
                for bi in bis:
                    pe = F.huber_loss(fwd_chain(wlayers, X[bi], kw).float(),
                                      Y[bi].float(), reduction="none")
                    if M is None:
                        tot += float(pe.mean())
                    else:
                        mk = M[bi]
                        tot += float((pe * mk[:, :, None]).sum()
                                     / (mk.sum() * pe.shape[-1]))
            return tot / len(bis)

        tail_ls = None
        if args.tail_ls and any(stq[n].mix is not None for n in ftgt):
            tl0 = time.time()
            before = hold_loss_hard()
            saved = {n: (stq[n].mix.detach().clone(),
                         stq[n].logit.detach().clone())
                     for n in ftgt if stq[n].mix is not None}
            acc = HAccum()
            acc.attach([(n, get_module(model, n)) for n in saved])
            set_mode(wp, "hard")
            with torch.no_grad():
                for bi in train_bi:
                    acc.mask = None if M is None else M[bi]
                    fwd_chain(wlayers, X[bi], kw)
            acc.mask = None
            acc.detach()
            per = {}
            with torch.no_grad():
                for n in saved:
                    q = stq[n]
                    Hn = acc.pop(n)
                    W0q = get_module(model, n).parametrizations.weight.original
                    u_, al_ = q._u(W0q)
                    _, _, v_h = round_with_mids(
                        u_, q.theta_eff().detach(), q.grid.vals,
                        q.grid.t1, q.grid.t2)
                    C = (al_ * v_h).reshape(W0q.shape).float()
                    if q.lmix is not None:
                        C = q.lmix_dense() @ C
                    W_t = W0q.float() + (1.0 - q.gamma) * ba_cache[n]
                    X_new, lst = solve_mix_ls(
                        C, W_t, Hn, q.mix.detach().float(),
                        ridge=args.tail_ls_ridge, cap=args.tail_ls_cap,
                        iters=args.tail_ls_iters,
                        sv_floor=args.mix_ls_sv_floor)
                    q.mix.copy_(X_new)
                    if args.tail_alpha_ls:
                        Bd = q.mix_dense()
                        _, ast = refit_alpha(
                            q, W_t @ q.mix_dense_inv(), v_h,
                            Bd @ Hn @ Bd.T)
                        lst.update(ast)
                        del Bd
                    per[n.split("layers.")[1]] = {
                        k_: lst[k_] for k_ in ("ls_gain", "step_max", "t",
                                               "sv_min", "cg_iters")}
                    del C, X_new, Hn, u_, al_, v_h, W_t
            acc.H.clear()
            after = hold_loss_hard()
            reverted = (not np.isfinite(after)
                        or after > before * (1.0 + args.tail_ls_tol))
            if reverted:
                with torch.no_grad():
                    for n, (mx, lg) in saved.items():
                        stq[n].mix.copy_(mx)
                        stq[n].logit.copy_(lg)
            gains = [v["ls_gain"] for v in per.values()]
            tail_ls = {"hold_before": round(before, 7),
                       "hold_after": round(after, 7),
                       "hold_ratio": round(after / max(before, 1e-30), 4),
                       "reverted": reverted,
                       "gain_mean": round(sum(gains) / len(gains), 4),
                       "gain_min": round(min(gains), 4),
                       "t_min": min(v["t"] for v in per.values()),
                       "per": per, "secs": round(time.time() - tl0, 1)}
            del saved
            torch.cuda.empty_cache()

        with torch.no_grad():                  # G-disc1(出口軟硬差)
            set_mode(wp, "hard")
            y_hard = fwd_chain(wlayers, X[0], kw).float()
            set_mode(wp, "soft", s=args.s0)
            y_soft = fwd_chain(wlayers, X[0], kw).float()
            gdisc1 = float((y_soft - y_hard).norm()
                           / y_hard.norm().clamp_min(1e-12))
            set_mode(wp, "hard")

        if XT is not None:                     # 乾淨流理想前進(γ 已排空,
            with torch.no_grad():              # 暫拉回 1 走滿水庫再還原)
                set_mode(wp, "off")
                for n in ftgt:
                    stq[n].gamma = 1.0
                XT = run_layers([layers[li] for li in fin_lis], XT, kw)
                for n in ftgt:
                    stq[n].gamma = 0.0
                set_mode(wp, "hard")

        # 定案:extract(γ=0 斷言在內)→ 落盤 → 材化 → 前進 X
        state_by_layer = {li: {} for li in fin_lis}
        w_flips = 0
        mix_stats = {}
        for n in ftgt:
            m = get_module(model, n)
            W0 = m.parametrizations.weight.original
            T1, T2, al, w_hard = stq[n].extract(W0)
            f = int((T1.cpu() != st0[n]["T1"]).sum()
                    + (T2.cpu() != st0[n]["T2"]).sum())
            w_flips += f
            flips_vs_p0 += f
            ntot += T1.numel() * 2
            entry = {
                "T1": T1.cpu(), "T2": T2.cpu(),
                "inv": torch.arange(W0.shape[1]), "a0": al.cpu(),
                "c": stq[n].c, "grid": stq[n].grid.n}
            if stq[n].rneg is not None:       # E63 S1:r_q 逐 block + 碼書隨態落盤
                entry["rneg"] = stq[n].rneg.detach().reshape(
                    T1.shape[0], T1.shape[1]).cpu()
                if stq[n].rneg_cb is not None:
                    entry["rneg_cb"] = stq[n].rneg_cb.detach().cpu()
            if stq[n].mix is not None:        # E61:M 隨態落盤(不吸收)
                mx = stq[n].mix.detach().float()
                entry["mix"] = mx.cpu()
                sv = torch.linalg.svdvals(mx)
                mix_stats[n] = {
                    "sv_min": round(float(sv.min()), 4),
                    "sv_max": round(float(sv.max()), 4),
                    "dist_I": round(float(
                        (mx - torch.eye(mx.shape[1], device=mx.device)
                         ).norm() / mx.shape[0] ** 0.5), 5)}
                w_hard = stq[n].apply_mix(w_hard)
            if stq[n].lmix is not None:       # E62b:L 隨態落盤(不吸收)
                lx = stq[n].lmix.detach().float()
                entry["lmix"] = lx.cpu()
                sv = torch.linalg.svdvals(lx)
                mix_stats[n + "|L"] = {
                    "sv_min": round(float(sv.min()), 4),
                    "sv_max": round(float(sv.max()), 4),
                    "dist_I": round(float(
                        (lx - torch.eye(lx.shape[1], device=lx.device)
                         ).norm() / lx.shape[0] ** 0.5), 5)}
                w_hard = stq[n].apply_lmix(w_hard)
            state_by_layer[layer_index(n)][n] = entry
            parametrize.remove_parametrizations(m, "weight",
                                                leave_parametrized=False)
            m.weight.data.copy_(w_hard.to(m.weight.dtype))
            del stq[n]
            ba_cache.pop(n, None)
        for li in fin_lis:
            torch.save(state_by_layer[li], QS / f"layer{li:02d}.pt")
        with torch.no_grad():
            X = run_layers([layers[li] for li in fin_lis], X, kw,
                               to_cpu=args.x_cpu)
        del Y
        vh_p0.clear()
        vh_prev.clear()
        torch.cuda.empty_cache()

        w_rec = {"window": [lo, hi], "finalized": fin_lis,
                 "w_max": w_max_here,
                 "drain_epochs": drain_eps, "loss": ep_log,
                 "gdisc1_exit_rel": round(gdisc1, 6),
                 **({"drift_rel": round(drift_rel, 6)}
                    if drift_rel is not None else {}),
                 "flips_vs_p0": w_flips,
                 "final_level": drain_events[-1]["level"],
                 **({"mix_stats": mix_stats} if mix_stats else {}),
                 **({"tail_ls": tail_ls} if tail_ls else {}),
                 "drain_events": drain_events}
        if ep_hold:
            w_rec["loss_holdout"] = ep_hold
        rec["windows"].append(w_rec)
        print(f"window {lo:2d}-{hi - 1:2d}  drain {drain_eps}ep  "
              f"loss {ep_log[0]:.5f}→{ep_log[-1]:.5f}  gdisc1={gdisc1:.4f}"
              + (f"  drift={drift_rel:.4f}" if drift_rel is not None else "")
              + f"  flips={w_flips}"
              + (f"  tailLS gain={tail_ls['gain_mean']:.3f} "
                 f"hold×{tail_ls['hold_ratio']:.3f} "
                 f"{'REVERT' if tail_ls['reverted'] else 'keep'}"
                 if tail_ls else "")
              + f"  ({time.time()-t0:.0f}s)", flush=True)

        if (args.stop_after_layer is not None
                and max(fin_lis) >= args.stop_after_layer):
            rec["stopped_after_layer"] = max(fin_lis)
            rec["runtime_s"] = round(time.time() - t0, 1)
            rec["timestamp"] = datetime.datetime.now().isoformat()
            (EVC / f"train_{args.tag}.partial_L{max(fin_lis):02d}.json"
             ).write_text(json.dumps(rec, indent=1, ensure_ascii=False))
            print(f"STOP_AFTER_LAYER:{max(fin_lis)}", flush=True)
            return

    del X
    torch.cuda.empty_cache()
    ce = eval_ce(model, val)
    rec["post_drain_val_ce"] = ce
    rec["t_flip_rate_vs_p0"] = round(flips_vs_p0 / max(ntot, 1), 6)
    rec["runtime_s"] = round(time.time() - t0, 1)
    rec["timestamp"] = datetime.datetime.now().isoformat()
    st = load_state(args.tag, targets if not args.smoke else [])
    np.savez(EVC / f"alphas_{args.tag}.npz",
             **{n: st[n]["a0"].numpy() for n in st})
    (EVC / f"train_{args.tag}.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"{args.tag} post-drain val CE {ce:.6f}  "
          f"flip_vs_p0 {rec['t_flip_rate_vs_p0']:.4f}  "
          f"({rec['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main()

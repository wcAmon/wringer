"""端到端軟打磨(Corkscrew 第三段 v2,E31 起取代 α-only polish)。

逐窗退火收卷後、部署硬化前:全模型掛 STQParam(mode=soft),
可訓 = LoRA r64 + δ_α logit + dθ,T 經軟階梯可重指派;
目標 = 對 teacher(預設 bf16)的全詞表 logit KD。
前 gamma 段線性升 s 至 s0(軟→尖),尾段 STE 硬收尾,extract 落 qs 狀態。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.e2e_soft \
      --tag st3b16_e2e --state-tag st3b16 --grid 3 --steps 600 \
      --data evidence/p1_grouping/calib_train_512.pt

起點恆等性:狀態材化後 u 恰在網格點上,soft 前向 = 硬權重(任意 s),
訓練從零漂移開始——不需 warmup。
"""
import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from p2_anchor.wringer.modelio import (enumerate_targets, get_module,
                                         layer_index, load_model)
from p2_anchor.wringer.polish import (kd_loss, kd_loss_hidchunk,
                                        closure_weights)
from p2_anchor.wringer.state import apply_k2, load_state, qs_dir
from p2_anchor.wringer.stq import AlphaOnlyParam, SoftGrid, STQParam
from p2_anchor.wringer.train_st import set_mode

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"
SEED = 20260806


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--state-tag", required=True)
    ap.add_argument("--grid", type=int, default=None, choices=(9, 7, 5, 3),
                    help="fallback;預設從 state 逐模組讀(混合多元)")
    ap.add_argument("--data", default=str(EV / "calib_train.pt"))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=0.8,
                    help="軟段占比;其後 STE")
    ap.add_argument("--s0", type=float, default=30.0)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--factor-lr", type=float, default=3e-4)
    ap.add_argument("--theta-lr", type=float, default=None)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--teacher-dir", default=None,
                    help="HF 目錄;預設 bf16 原模型")
    ap.add_argument("--kd-dir", choices=("fwd", "rev"), default="fwd",
                    help="KD 方向:fwd=前向 KL(預設);rev=反向 KL"
                         "(E40 互補臂;損失值與 fwd 不可比)")
    ap.add_argument("--cache-teacher-k", type=int, default=0,
                    help="E45 效率桿:老師 top-K logit 快取(0=關;例 256)"
                         "。建畢卸載老師(省 ~9GB 顯存 + 每步老師前向);"
                         "僅支援 fwd;截斷覆蓋審計入 rec")
    ap.add_argument("--mtp-k", type=int, default=0,
                    help="E45 MTP 桿:輔助頭數 k(頭 j 於位置 t 預測 "
                         "t+1+j;目標=快取老師分布;頭為暫時鷹架,"
                         "export 前丟棄;需 --cache-teacher-k)")
    ap.add_argument("--mtp-lam", type=float, default=0.25,
                    help="輔助損失權重(總 aux 平均後乘此)")
    ap.add_argument("--mtp-frac", type=float, default=0.25,
                    help="每樣本輔助損失抽樣位置比例")
    ap.add_argument("--mtp-r", type=int, default=256,
                    help="MTP 頭低秩寬度")
    ap.add_argument("--mtp-lr", type=float, default=1e-3)
    ap.add_argument("--span-m", type=int, default=0,
                    help="E45 桿3 段落路徑對齊:每 m token 一段,對齊段內"
                         "Σlogp(trace) 學生 vs 老師(補償自由度=容量讓步"
                         "方向修正;spec-decode 接受率的訓練目標形式);"
                         "0=關;需 --cache-teacher-k")
    ap.add_argument("--span-lam", type=float, default=0.3,
                    help="換不是加:loss=(1−λ)·逐token FKL + λ·span huber")
    ap.add_argument("--microbatch", type=int, default=1,
                    help="1=舊行為(逐樣本累加);0=整批一次前向;n=微批 n"
                         "(僅非快取路徑;--cache-teacher-k 仍逐樣本)")
    ap.add_argument("--kd-impl", choices=("chunk", "hidchunk"),
                    default="chunk",
                    help="chunk=原 kd_loss;hidchunk=hidden-chunk 版"
                         "(logits 不整條物化,microbatch>1 建議)")
    ap.add_argument("--full-ckpt", action="store_true",
                    help="STQParam 整前向 checkpoint(參數化殘留圖 "
                         "~14→~2 B/param;等效閘 Gate1 驗證後用)")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="骨幹梯度檢查點(16k 列學生活化圖 OOM 修;"
                         "default 關=舊行為)")
    ap.add_argument("--alpha-only", action="store_true",
                    help="E68 P1-alpha:碼凍結只學 α(AlphaOnlyParam);翻碼恆 0、帳不動;lora/dtheta 群組為空")
    ap.add_argument("--rate-lam", type=float, default=0.0,
                    help="E67-B 率項 λ:loss += λ·mean_modules(bits_interp(v_soft));0 = 關(逐位還原)")
    ap.add_argument("--rate-bits", default="",
                    help="9 格碼長 json(含 'bits',升冪格序;如 rd_lambda_pick 的 bits)或 JSON 列表;空且 λ>0 ⇒ 由起點態直方圖")
    ap.add_argument("--rate-protect", default="linear_attn.out_proj,self_attn.v_proj",
                    help="不計率的模組後綴(逗號)")
    ap.add_argument("--close-weight", type=float, default=1.0,
                    help="E51 閉合軸:</think> target 位置 KD 權重 λ_close"
                         "(1.0=關閉,逐位走原路徑)")
    ap.add_argument("--close-post-weight", type=float, default=16.0,
                    help="</think> 後首 K 個 target 位置權重 λ_post")
    ap.add_argument("--close-post-k", type=int, default=8,
                    help="閉合後加權位置數 K")
    ap.add_argument("--lengths", default=None,
                    help="E53 變長整軌:(N,) 有效長度 .pt;pad 位 loss "
                         "權重歸零(w 通道,/w.sum() 歸一);None=舊行為")
    ap.add_argument("--rot-q", default=None,
                    help="E60:旋轉載體 Q .pt(含 'Q');碼須為折 γ 基底解出(g3rot*/g9f)")
    ap.add_argument("--rot-lr", type=float, default=1e-4)
    ap.add_argument("--rot-frozen", action="store_true",
                    help="E60 α0 臂:Q 凍結於輸入值(旋轉開關對照)")
    args = ap.parse_args()
    kd_fn = kd_loss if args.kd_impl == "chunk" else kd_loss_hidchunk
    assert args.span_m == 0 or args.cache_teacher_k > 0, \
        "--span-m 需 --cache-teacher-k(老師 trace logp 取自快取)"
    assert args.mtp_k == 0 or args.cache_teacher_k > 0, \
        "--mtp-k 需 --cache-teacher-k(輔助目標取自快取)"
    assert args.cache_teacher_k == 0 or args.kd_dir == "fwd", \
        "top-K 快取僅支援 fwd KL"

    t0 = time.time()
    torch.manual_seed(SEED)
    EVC.mkdir(parents=True, exist_ok=True)
    model, tok = load_model()
    if args.teacher_dir:
        from transformers import AutoModelForImageTextToText
        teacher = AutoModelForImageTextToText.from_pretrained(
            args.teacher_dir, dtype=torch.bfloat16).to("cuda").eval()
    else:
        teacher, _ = load_model()
    model.requires_grad_(False)
    teacher.requires_grad_(False)
    targets = enumerate_targets(model)
    st = load_state(args.state_tag, targets)
    apply_k2(model, st)                      # 材化輸入態(硬)為 latent
    carrier = None
    if args.rot_q:
        from p2_anchor.wringer.rot_carrier import (RotCarrier, attach,
                                                     load_Q, surgery)
        Q0, qmeta = load_Q(args.rot_q)
        n_fold = surgery(model, targets)
        carrier = RotCarrier(Q0, learn=not args.rot_frozen).cuda()
        attach(model, targets, carrier)
        carrier.refresh()
        print(f"E60 旋轉載體:Q0={args.rot_q}(codes {qmeta.get('codes')} "
              f"val_ce_frozen {qmeta.get('val_ce_frozen')})  "
              f"{'凍結' if args.rot_frozen else f'可學 lr {args.rot_lr}'}  "
              f"手術折 γ 非目標讀者 {n_fold}", flush=True)
    data = torch.load(args.data, weights_only=False)
    val = torch.load(EV / "calib_val.pt", weights_only=False)
    lens = (torch.load(args.lengths, weights_only=False)
            if args.lengths else None)
    if lens is not None:
        assert lens.shape[0] == data.shape[0], "lengths 與 data 列數不符"
        assert args.cache_teacher_k == 0, "--lengths 僅支援非快取路徑"
        print(f"E53 變長整軌:len 四分位 "
              f"{[int(x) for x in lens.float().quantile(torch.tensor([.25, .5, .75])).tolist()]}"
              f" / 滿窗列 {(lens == data.shape[1]).sum().item()}", flush=True)
    close_id = tok.convert_tokens_to_ids("</think>")
    if args.close_weight > 1.0:
        print(f"E51 閉合加權:</think> id={close_id} "
              f"λ_close={args.close_weight} λ_post={args.close_post_weight} "
              f"K={args.close_post_k}", flush=True)

    from p1_grouping.e3_runner import eval_ce
    ce0 = eval_ce(model, val)
    print(f"pre-e2e val CE {ce0:.6f}", flush=True)

    cache = None
    if args.cache_teacher_k > 0:
        from p2_anchor.wringer.polish import build_topk_cache, kd_topk
        tC = time.time()
        cache = build_topk_cache(teacher, data, K=args.cache_teacher_k)
        del teacher
        teacher = None
        torch.cuda.empty_cache()
        print(f"teacher top-{args.cache_teacher_k} 快取畢 "
              f"({time.time()-tC:.0f}s)  覆蓋分位 {cache['cover']}"
              f"  老師已卸載", flush=True)

    grids = {}
    stq = {}
    for n in targets:
        gn = int(st[n].get("grid", args.grid or 3))
        cn = float(st[n].get("c", 0.6))       # E34:c 從 state 讀,禁默認洩漏
        if (gn, cn) not in grids:
            grids[(gn, cn)] = SoftGrid(c=cn, n=gn)
        if args.alpha_only:
            p = AlphaOnlyParam(st[n]["T1"].cuda(), st[n]["T2"].cuda(), st[n]["a0"].cuda().float(),
                               grids[(gn, cn)], c=cn)
        else:
            p = STQParam(st[n]["a0"].cuda().float(), grids[(gn, cn)],
                         r=args.lora_r, c=cn, learn_theta=True)
        parametrize.register_parametrization(get_module(model, n), "weight", p)
        stq[n] = p
    params = list(stq.values())
    rate_params = []
    if args.rate_lam > 0:
        if args.rate_bits:
            rb = json.loads(Path(args.rate_bits).read_text())["bits"] if args.rate_bits.endswith(".json") else json.loads(args.rate_bits)
        else:
            from p2_anchor.wringer.ladder_e65x import g9_bits
            rb = g9_bits(st)
        prot = [x for x in args.rate_protect.split(",") if x]
        for n, p in stq.items():
            if p.grid.n == 9 and not any(n.endswith(x) for x in prot):
                p.rate_bits = torch.tensor(rb, device="cuda", dtype=torch.float32)
                rate_params.append(p)
        print(f"rate: lam={args.rate_lam} modules={len(rate_params)}/{len(params)} bits={[round(b, 3) for b in rb]}", flush=True)
    if args.full_ckpt:
        for p in params:
            p.full_ckpt = True
        print("full_ckpt=ON(STQParam 整前向 checkpoint)", flush=True)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.train()   # HF ckpt 僅 training=True 生效;dropout 全 0 語義不變
        print("grad_ckpt=ON(骨幹活化逐層重算;model.train)", flush=True)
    # lazy LoRA init:一次 soft 前向
    set_mode(params, "soft", s=1e-3)
    with torch.no_grad():
        _ = model(input_ids=data[:1].cuda(), use_cache=False).logits[:, :1]
    heads, h_buf, lm, hook = None, {}, None, None
    if args.mtp_k > 0:
        try:
            lm = model.get_submodule("lm_head")
        except AttributeError:
            lm = model.get_submodule("model.lm_head")
        hook = lm.register_forward_hook(
            lambda mod, inp, out: h_buf.__setitem__("h", inp[0]))

        class MTPHead(nn.Module):
            """Medusa-lite 低秩殘差探針:h → h + U·gelu(V·LN(h)) →
            共享(綁定)lm_head。U 零初始化 → 起點分布 = 主路徑,
            無梯度衝擊;export 前整體丟棄(純訓練鷹架)。"""

            def __init__(self, d, r):
                super().__init__()
                self.ln = nn.LayerNorm(d)
                self.V = nn.Linear(d, r, bias=False)
                self.U = nn.Linear(r, d, bias=False)
                nn.init.zeros_(self.U.weight)

            def forward(self, h):
                z = self.U(F.gelu(self.V(self.ln(h.float()))))
                return h + z.to(h.dtype)

        d_model = lm.weight.shape[1]
        heads = nn.ModuleList([MTPHead(d_model, args.mtp_r)
                               for _ in range(args.mtp_k)]).cuda()
        print(f"MTP 頭 ×{args.mtp_k}(r={args.mtp_r},d={d_model},"
              f"λ={args.mtp_lam},frac={args.mtp_frac})", flush=True)
    if args.alpha_only:
        print(f"alpha_only=ON(碼凍結;可學 α {sum(stq[n].logit.numel() for n in targets)/1e6:.1f}M)", flush=True)
    groups = [
        {"params": [q for n in targets for q in
                    (stq[n].lora_A, stq[n].lora_B)], "lr": args.lora_lr},
        {"params": [stq[n].logit for n in targets], "lr": args.factor_lr},
        {"params": [stq[n].dtheta for n in targets],
         "lr": args.theta_lr or args.factor_lr}]
    if heads is not None:
        groups.append({"params": list(heads.parameters()),
                       "lr": args.mtp_lr})
    if carrier is not None and carrier.learn:
        groups.append({"params": [carrier.P], "lr": args.rot_lr})
    opt = torch.optim.Adam(groups)

    g_rng = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(data), generator=g_rng)
    pos = 0
    log = []
    soft_steps = int(args.gamma * args.steps)
    for step in range(1, args.steps + 1):
        if step <= soft_steps:
            set_mode(params, "soft", s=(step / soft_steps) * args.s0)
        else:
            set_mode(params, "hard")
        if pos + args.batch > len(data):
            order = torch.randperm(len(data), generator=g_rng)
            pos = 0
        rows = order[pos:pos + args.batch]
        b = data[rows].cuda()
        pos += args.batch
        opt.zero_grad(set_to_none=True)
        if carrier is not None:
            carrier.refresh()
        kd_val, aux_val, span_val, rate_val = 0.0, 0.0, 0.0, 0.0
        if cache is None:
            mb = b.shape[0] if args.microbatch == 0 else args.microbatch
            for i in range(0, b.shape[0], mb):
                if carrier is not None:
                    carrier.refresh()   # 每次 micro 前向重建 Q 圖(多次 backward)
                xb = b[i:i + mb]
                xw = None
                if args.close_weight > 1.0:
                    xw = closure_weights(xb, close_id, args.close_weight,
                                         args.close_post_weight,
                                         args.close_post_k)
                if lens is not None:
                    xl = lens[rows[i:i + mb]].to(xb.device)
                    T = xb.shape[1]
                    lim = torch.where(xl < T, xl - 1, xl)
                    mask = (torch.arange(T, device=xb.device)[None]
                            < lim[:, None]).float()
                    xw = mask if xw is None else xw * mask
                loss = kd_fn(model, teacher, xb, direction=args.kd_dir,
                             w=xw) * xb.shape[0] / b.shape[0]
                if rate_params:
                    rate_t = sum(p.last_rate for p in rate_params) / len(rate_params)
                    loss = loss + args.rate_lam * rate_t * xb.shape[0] / b.shape[0]
                    rate_val += float(rate_t.detach()) * xb.shape[0] / b.shape[0]
                loss.backward()
                kd_val += float(loss)
        else:
            for i in range(b.shape[0]):
                if carrier is not None:
                    carrier.refresh()
                sl = model(input_ids=b[i:i + 1], use_cache=False).logits
                r = int(rows[i])
                pk = cache["probs"][r].unsqueeze(0).cuda()
                ik = cache["idx"][r].unsqueeze(0).cuda().long()
                pt = cache["tail"][r].unsqueeze(0).cuda()
                if args.span_m > 0:     # 桿3:段內 Σlogp 對齊(換不是加)
                    tr = b[i:i + 1]
                    tid = torch.cat([tr[:, 1:], tr[:, :1]], dim=1).long()
                    loss_tok, ls_tr = kd_topk(sl, pk, ik, pt,
                                              trace_ids=tid)
                    lt_tr = cache["trace_lp"][r].unsqueeze(0).cuda().float()
                    n_sp = (sl.shape[1] - 1) // args.span_m  # 排除末位
                    ns = n_sp * args.span_m
                    dm = (ls_tr[:, :ns] - lt_tr[:, :ns]).view(
                        1, n_sp, args.span_m).mean(-1)   # 段均 log 比
                    span = F.huber_loss(dm, torch.zeros_like(dm))
                    span_val += float(span) * args.span_lam / b.shape[0]
                    loss = (1.0 - args.span_lam) * loss_tok \
                        + args.span_lam * span
                else:
                    loss = kd_topk(sl, pk, ik, pt)
                if heads is not None:       # MTP:位置 t 的 h 預測 t+1+j
                    T = sl.shape[1]
                    n_sel = max(1, int(args.mtp_frac * (T - args.mtp_k)))
                    sel = torch.randperm(T - args.mtp_k,
                                         generator=g_rng)[:n_sel].cuda()
                    h = h_buf.pop("h")[0]           # (T,d) post-norm
                    aux = 0.0
                    for j in range(1, args.mtp_k + 1):
                        lj = F.linear(heads[j - 1](h[sel]), lm.weight)
                        tj = sel + j                # 目標=老師在 t+j 的分布
                        aux = aux + kd_topk(lj.unsqueeze(0), pk[:, tj],
                                            ik[:, tj], pt[:, tj])
                    aux = aux / args.mtp_k
                    aux_val += float(aux) * args.mtp_lam / b.shape[0]
                    loss = loss + args.mtp_lam * aux
                loss = loss / b.shape[0]
                loss.backward()
                kd_val += float(loss)
        torch.nn.utils.clip_grad_norm_(
            [q for g in opt.param_groups for q in g["params"]], args.clip)
        opt.step()
        if not np.isfinite(kd_val):
            raise RuntimeError(f"警報:step {step} KD loss 非有限 {kd_val}")
        ent = {"step": step, "kd": round(kd_val, 6),
               "phase": "soft" if step <= soft_steps else "ste"}
        if heads is not None:
            ent["aux"] = round(aux_val, 6)
        if args.span_m > 0:
            ent["span"] = round(span_val, 6)
        if rate_params:                      # E67-B 率軌跡入卷(kd 欄含 λ·rate)
            ent["rate"] = round(rate_val, 5)
        log.append(ent)
        if step % 10 == 0:
            print(f"step {step:4d}  kd {kd_val:.5f}  "
                  + (f"rate {rate_val:.4f}  " if rate_params else "")
                  + (f"aux {aux_val:.5f}  " if heads is not None else "")
                  + f"[{'soft' if step <= soft_steps else 'ste'}]  "
                  f"({(time.time()-t0)/step:.1f}s/step)", flush=True)

    if hook is not None:                     # MTP 鷹架拆除(不入 export)
        hook.remove()
        del heads
        heads = None
        torch.cuda.empty_cache()

    # 硬化 extract → 存 qs 狀態 + alphas npz
    out_qs = qs_dir(args.tag)
    out_qs.mkdir(parents=True, exist_ok=True)
    flips = 0
    ntot = 0
    thetas = {}
    state_by_layer = {}
    for n in targets:
        m = get_module(model, n)
        W0 = m.parametrizations.weight.original
        T1, T2, al, w_hard = stq[n].extract(W0)
        flips += int((T1.cpu() != st[n]["T1"]).sum()
                     + (T2.cpu() != st[n]["T2"]).sum())
        ntot += T1.numel() * 2
        thetas[n] = stq[n].theta_eff().detach().cpu().tolist()
        li = layer_index(n)
        state_by_layer.setdefault(li, {})[n] = {
            "T1": T1.cpu(), "T2": T2.cpu(),
            "inv": torch.arange(W0.shape[1]), "a0": al.cpu(),
            "c": stq[n].c, "grid": stq[n].grid.n}
        parametrize.remove_parametrizations(m, "weight",
                                            leave_parametrized=False)
        m.weight.data.copy_(w_hard.to(m.weight.dtype))
        del stq[n]
    for li, ls in state_by_layer.items():
        torch.save(ls, out_qs / f"layer{li:02d}.pt")
    torch.cuda.empty_cache()

    rot_rec = {}
    if carrier is not None:
        with torch.no_grad():
            carrier.refresh()
        ce1 = eval_ce(model, val)
        QL = carrier.Q_float().cpu()
        torch.save({"Q": QL, "from": args.rot_q, "codes": args.tag,
                    "step": args.steps, "val_ce_frozen": ce1},
                   EVC / f"rot_Q_{args.tag}.pt")
        rot_rec = {"rot_q0": args.rot_q, "rot_lr": args.rot_lr,
                   "rot_frozen": args.rot_frozen,
                   "rot_angle_rms_vs_q0": carrier.angle_rms(),
                   "rot_Q_out": str(EVC / f"rot_Q_{args.tag}.pt")}
        print(f"E60 rot_Q_{args.tag}.pt 主角 RMS vs Q0 "
              f"{rot_rec['rot_angle_rms_vs_q0']:.4f} rad", flush=True)
    else:
        ce1 = eval_ce(model, val)
    npz = EVC / f"alphas_{args.tag}.npz"
    np.savez(npz, **{n: state_by_layer[layer_index(n)][n]["a0"].numpy()
                     for n in targets})
    rec = {"tag": args.tag, "state_tag": args.state_tag,
           "grids": sorted({int(st[n].get("grid", 0)) for n in targets}),
           "teacher": args.teacher_dir or "bf16",
           **({"teacher_cache": {"K": cache["K"],
                                 "cover_quantiles": cache["cover"]}}
              if cache is not None else {}),
           "data": str(args.data),
           "data_sha256": hashlib.sha256(
               Path(args.data).read_bytes()).hexdigest(),
           "hparams": vars(args), "seed": SEED, **rot_rec,
           "pre_e2e_val_ce": ce0, "post_e2e_val_ce": ce1,
           "t_flip_rate_vs_input": round(flips / ntot, 6),
           "theta_eff": thetas, "kd_log": log,
           "runtime_s": round(time.time() - t0, 1),
           "timestamp": datetime.datetime.now().isoformat()}
    (EVC / f"train_{args.tag}.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"post-e2e val CE {ce1:.6f}  flip_vs_input {flips/ntot:.4f}  "
          f"→ {npz}", flush=True)
    print("E2E_DONE", flush=True)


if __name__ == "__main__":
    main()

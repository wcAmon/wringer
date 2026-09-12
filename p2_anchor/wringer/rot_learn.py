"""E60-L 探針 v2:凍碼學旋轉(旋轉=連續載體;用戶 09-03「開工」)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.rot_learn \
      --q0 evidence/p1_grouping/corkscrew/rot_Q.pt --codes g3rot --tag L1 [--steps 150 --lr 1e-4]

v1(迴圈內 TWN 假量化)撤案:三元 RTN val CE 8.79 vs GPTQ 2.50=不同 régime。
v2:碼 = GPTQ 已解(qs_<codes>,在 Q0 基底解出)全程凍結;旋轉改在激活側現算——
  量化讀者 y = W_q·(Qᵀx̂)   ← 實作 F.linear(x̂ @ Q, W_q)
  量化寫者 r += Q·(W_q z)   ← 實作 F.linear(z, W_q) @ Qᵀ
殘差流留原基底;非目標讀者(in_proj_a/b)折 (1+γ) 用原基底;norm γ←0;lm_head 解綁 E·diag(1+γ_f)。
Q = Q0·Cayley(A) 只學 A;等價閘:Q=Q0 時 val CE 須≈quant_<codes> 的 GPTQ CE(|Δ|<0.02)。
輸出 rot_Q_<tag>.pt + rot_learn_<tag>.json(ce_frozen_q0 / ce_frozen_best / 軌跡 / 主角 RMS)。
「凍碼下收回多少」= 一樓轉方向接東西的直接量測;重解 GPTQ 後的 CE 由鏈另量。

v3(E60 S0′,用戶 09-03「1」):--kd 模式——目標改 KD 對 bf16 老師(全詞表 fwd KL,E51 閉合加權
λ_close 256/λ_post 16/K 8),量測改 held-out KL-to-bf16(r_KL = 1 − KL/KL_ctl)+ </think> 閉合 logP 儀器。
v2 的 CE 目標/同族 val CE 已裁為第 13 型 proxy 反轉(verdict_e60_s0:L9 9元+Q CE 1.69 < bf16 2.00)。
"""
import argparse
import datetime
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from p2_anchor.wringer.ktier import k2_weight
from p2_anchor.wringer.modelio import (LAYER_PREFIX, N_LAYERS,
                                         enumerate_targets, load_model,
                                         set_module)
from p2_anchor.wringer.rot_fold import (READERS_ATT, READERS_LIN,
                                          READERS_MLP, WRITERS)
from p2_anchor.wringer.state import load_state

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"
SEED = 20260806


class Ctx:
    Q = None        # 全局 (H,H) 或 逐層 (L,H,H)(--per-layer,S1 儀器探針)


class RotQ(nn.Module):
    def __init__(self, Wq, side, li=0):
        super().__init__()
        self.register_buffer("Wq", Wq)
        self.side = side
        self.li = li

    def forward(self, x):
        Q = Ctx.Q if Ctx.Q.dim() == 2 else Ctx.Q[self.li]
        if self.side == "r":
            return F.linear(x @ Q, self.Wq)
        return F.linear(x, self.Wq) @ Q.T


def cayley(P):
    A = P - P.transpose(-1, -2)
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return torch.linalg.solve(I + A, I - A)


def rms_angle(Q0, Q):
    ev = torch.linalg.eigvals((Q0.T @ Q).double())
    ang = torch.atan2(ev.imag, ev.real).abs()
    return float((ang ** 2).mean().sqrt())


@torch.no_grad()
def val_kl(model, teacher, val, bs=4, chunk=512):
    """held-out 全詞表 fwd KL(p_t‖q_s)每 token 均值;bf16 自身=0。"""
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(val), bs):
        b = val[i:i + bs].cuda()
        tl = teacher(input_ids=b, use_cache=False).logits
        sl = model(input_ids=b, use_cache=False).logits
        for c0 in range(0, sl.shape[1], chunk):
            lt = torch.log_softmax(tl[:, c0:c0 + chunk].float(), -1)
            ls = torch.log_softmax(sl[:, c0:c0 + chunk].float(), -1)
            tot += float((lt.exp() * (lt - ls)).sum())
            n += lt.shape[0] * lt.shape[1]
        del tl, sl
    model.train()
    return tot / n


@torch.no_grad()
def closure_probe(model, traj, lens, close_id, n_rows=48, win=4096):
    """真閉合位 </think> 的 logP(mean/median)、rank、前 64 位過早閉合質量(postmortem_e60_s0 同構)。"""
    model.eval()
    lp, rk, pre = [], [], []
    for r in range(n_rows):
        L = min(int(lens[r]), win)
        ids = traj[r, :L].cuda()[None]
        logp = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[0, :-1].float(), -1)
        tgt = ids[0, 1:]
        for p in (tgt == close_id).nonzero().flatten().tolist():
            lp.append(float(logp[p, close_id]))
            rk.append(int((logp[p] > logp[p, close_id]).sum()))
            pre.append(float(logp[max(0, p - 64):p, close_id].exp().sum()))
        del logp
    model.train()
    med = lambda a: sorted(a)[len(a) // 2] if a else None
    return {"n_close": len(lp), "logp_close_mean": sum(lp) / max(len(lp), 1),
            "logp_close_med": med(lp), "rank_med": med(rk),
            "pre64_mass_mean": sum(pre) / max(len(pre), 1)}


@torch.no_grad()
def val_ce(model, val, bs=4):
    model.eval()
    tot = 0.0
    for i in range(0, len(val), bs):
        b = val[i:i + bs].cuda()
        tot += float(model(input_ids=b, labels=b).loss)
    model.train()
    return tot / math.ceil(len(val) / bs)


def build_chunks(traj, lens, win=2048):
    out = []
    for r in range(traj.shape[0]):
        L = int(lens[r])
        for s in range(0, L - win + 1, win):
            out.append(traj[r, s:s + win])
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q0", required=True)
    ap.add_argument("--codes", required=True, help="qs_<tag>(在 q0 基底解出的 GPTQ 碼)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=25)
    ap.add_argument("--gate", type=float, default=0.02)
    ap.add_argument("--traj", default=str(EV / "calib_e58r1b_traj.pt"))
    ap.add_argument("--lens", default=str(EV / "calib_e58r1b_len.pt"))
    ap.add_argument("--per-layer", action="store_true",
                    help="逐層 Q_l(L,H,H;S1 儀器探針:逐層旋角分佈 vs 全局單 Q)")
    ap.add_argument("--kd", action="store_true",
                    help="v3:KD 對 bf16 老師(閉合加權)+ held-out KL 量測(S0′ 正確儀器)")
    ap.add_argument("--close-weight", type=float, default=256.0)
    ap.add_argument("--close-post-weight", type=float, default=16.0)
    ap.add_argument("--close-post-k", type=int, default=8)
    ap.add_argument("--kl-ctl", type=float, default=None,
                    help="r_KL 分母:直投對照 KL(kl_ref.py 量得);None 則只記 KL")
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(SEED)

    model, tok = load_model()
    model.requires_grad_(False)
    teacher = None
    if args.kd:
        teacher, _ = load_model()
        teacher.requires_grad_(False)
        teacher.eval()
        from p2_anchor.wringer.polish import closure_weights, kd_loss
        close_id = tok.convert_tokens_to_ids("</think>")
    val = torch.load(EV / "calib_val.pt", weights_only=False)
    chunks = build_chunks(torch.load(args.traj, weights_only=False),
                          torch.load(args.lens, weights_only=False))
    print(f"chunks {chunks.shape[0]}×2048 = {chunks.numel()/1e6:.1f}M tokens", flush=True)

    Q0 = torch.load(args.q0, weights_only=False)["Q"].cuda().float()
    H = Q0.shape[0]
    targets = enumerate_targets(model)
    st = load_state(args.codes, targets)
    ce_ref = json.load(open(EVC / f"quant_{args.codes}.json"))["zerotrain_val_ce"]
    lm = model.model.language_model

    with torch.no_grad():
        for li in range(N_LAYERS):
            layer = lm.layers[li]
            names = READERS_LIN if hasattr(layer, "linear_attn") else READERS_ATT
            for grp, gname in ((names, "input_layernorm"),
                               (READERS_MLP, "post_attention_layernorm")):
                g = getattr(layer, gname).weight.float()
                for n in grp:
                    full = f"{LAYER_PREFIX}.{li}.{n}"
                    m = model.get_submodule(full)
                    if full in st:
                        c = st[full]
                        Wq = k2_weight(c["T1"].cuda(), c["T2"].cuda(), c["a0"].cuda().float(), c["c"])
                        set_module(model, full, RotQ(Wq[:, c["inv"].cuda()].to(torch.bfloat16), "r", li))
                    else:   # 非目標讀者:折 (1+γ),原基底
                        m.weight.data.copy_((m.weight.float() * (1.0 + g)[None, :]).to(m.weight.dtype))
                getattr(layer, gname).weight.data.zero_()
            for n in WRITERS:
                full = f"{LAYER_PREFIX}.{li}.{n}"
                if full in st:
                    c = st[full]
                    Wq = k2_weight(c["T1"].cuda(), c["T2"].cuda(), c["a0"].cuda().float(), c["c"])
                    set_module(model, full, RotQ(Wq[:, c["inv"].cuda()].to(torch.bfloat16), "w", li))
        E0 = lm.embed_tokens.weight.float()
        gf = lm.norm.weight.float()
        model.lm_head = nn.Linear(H, E0.shape[0], bias=False, device="cuda", dtype=torch.bfloat16)
        model.lm_head.weight.data.copy_((E0 * (1.0 + gf)[None, :]).to(torch.bfloat16))
        model.lm_head.weight.requires_grad_(False)
        lm.norm.weight.data.zero_()
        del E0, st
    n_q = sum(isinstance(m, RotQ) for m in model.modules())
    print(f"RotQ {n_q}(凍碼 qs_{args.codes},GPTQ CE 參考 {ce_ref:.4f})", flush=True)

    P = torch.zeros((N_LAYERS, H, H) if args.per_layer else (H, H),
                    device="cuda", dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([P], lr=args.lr)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    Ctx.Q = Q0.to(torch.bfloat16)
    ce_q0 = val_ce(model, val)
    kl_q0 = val_kl(model, teacher, val) if args.kd else None
    if args.kd:
        print(f"step 0  val KL(凍碼,Q0) {kl_q0:.4f}" + (f"  r_KL {1 - kl_q0/args.kl_ctl:.3f}" if args.kl_ctl else ""), flush=True)
    d = abs(ce_q0 - ce_ref)
    print(f"step 0  val CE(凍碼,Q0) {ce_q0:.4f}  vs GPTQ {ce_ref:.4f}  |Δ|={d:.4f}  gate<{args.gate}", flush=True)
    if d >= args.gate:
        print("ROTL_EQUIV_FAIL")
        raise SystemExit(1)
    print("ROTL_EQUIV_PASS", flush=True)
    traj_log = [{"step": 0, "val_ce": ce_q0, **({"val_kl": kl_q0} if args.kd else {})}]
    traj_full = torch.load(args.traj, weights_only=False) if args.kd else None
    lens_full = torch.load(args.lens, weights_only=False) if args.kd else None

    g = torch.Generator().manual_seed(SEED)
    order = torch.randperm(chunks.shape[0], generator=g)
    ptr = 0
    Q0b = Q0[None].expand(N_LAYERS, -1, -1) if args.per_layer else Q0
    best = ((kl_q0 if args.kd else ce_q0), Q0b.clone(), 0)   # kd 模式以 val KL 選最佳
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        if ptr + args.batch > len(order):
            order = torch.randperm(chunks.shape[0], generator=g)
            ptr = 0
        b = chunks[order[ptr:ptr + args.batch]].cuda()
        ptr += args.batch
        Ctx.Q = (Q0b @ cayley(P)).to(torch.bfloat16)
        if args.kd:
            xw = closure_weights(b, close_id, args.close_weight,
                                 args.close_post_weight, args.close_post_k)
            loss = kd_loss(model, teacher, b, direction="fwd", w=xw)
        else:
            loss = model(input_ids=b, labels=b).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(P.grad.norm())
        opt.step()
        lv = float(loss.detach())
        if step % 10 == 0 or step == 1:
            print(f"step {step:4d}  train CE {lv:.4f}  |g| {gn:.2e}  lr {lr:.1e}  "
                  f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}G  ({time.time()-t0:.0f}s)", flush=True)
            traj_log.append({"step": step, "train_ce": lv, "gnorm": gn})
        if step % args.val_every == 0 or step == args.steps:
            Qc = (Q0b @ cayley(P)).detach().float()
            Ctx.Q = Qc.to(torch.bfloat16)
            ce = val_ce(model, val)
            if args.kd:
                kl = val_kl(model, teacher, val)
                traj_log.append({"step": step, "val_ce": ce, "val_kl": kl})
                print(f"step {step:4d}  val KL(凍碼) {kl:.4f}  CE {ce:.4f}  best KL {min(best[0], kl):.4f}"
                      + (f"  r_KL {1 - kl/args.kl_ctl:.3f}" if args.kl_ctl else ""), flush=True)
                key = kl
            else:
                traj_log.append({"step": step, "val_ce": ce})
                print(f"step {step:4d}  val CE(凍碼) {ce:.4f}  best {min(best[0], ce):.4f}", flush=True)
                key = ce
            if key < best[0]:
                best = (key, Qc.clone(), step)

    key_b, QL, step_b = best
    Ctx.Q = QL.to(torch.bfloat16)
    ce_b = val_ce(model, val) if args.kd else key_b
    kd_rec = {}
    if args.kd:
        kl_b = key_b
        clo = closure_probe(model, traj_full, lens_full, close_id)
        kd_rec = {"kd": True, "kl_ctl": args.kl_ctl, "val_kl_q0": kl_q0, "val_kl_best": kl_b,
                  "r_kl_q0": (1 - kl_q0 / args.kl_ctl) if args.kl_ctl else None,
                  "r_kl_best": (1 - kl_b / args.kl_ctl) if args.kl_ctl else None,
                  "closure_best": clo,
                  "close_weight": args.close_weight}
        print(f"closure(best) logp mean {clo['logp_close_mean']:.4f} med {clo['logp_close_med']:.4f} "
              f"rank_med {clo['rank_med']} pre64 {clo['pre64_mass_mean']:.5f}", flush=True)
    per_layer_angle = None
    if args.per_layer:
        per_layer_angle = [round(rms_angle(Q0, QL[li]), 5) for li in range(N_LAYERS)]
        rms_angle_v = float(torch.tensor(per_layer_angle).pow(2).mean().sqrt())
    else:
        rms_angle_v = rms_angle(Q0, QL)
    torch.save({"Q": QL.cpu(), "from": args.q0, "codes": args.codes, "step": step_b,
                "val_ce_frozen": ce_b, **({"val_kl_frozen": key_b} if args.kd else {})},
               EVC / f"rot_Q_{args.tag}.pt")
    rec = {"tag": args.tag, "q0": args.q0, "codes": args.codes, "steps": args.steps,
           "best_step": step_b, "batch": args.batch, "lr": args.lr,
           "chunks": int(chunks.shape[0]), "gptq_ce_ref": ce_ref,
           "ce_frozen_q0": ce_q0, "ce_frozen_best": ce_b, "delta_frozen": ce_b - ce_q0,
           "rms_principal_angle": rms_angle_v, "per_layer": args.per_layer, **kd_rec,
           "per_layer_angle_rms": per_layer_angle, "trajectory": traj_log,
           "runtime_s": round(time.time() - t0, 1),
           "timestamp": datetime.datetime.now().isoformat()}
    (EVC / f"rot_learn_{args.tag}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"ROTL_DONE:{args.tag} 凍碼 CE Q0 {ce_q0:.4f} → QL {ce_b:.4f}  Δ {ce_b-ce_q0:+.4f} "
          f"@step {step_b}  angle_rms {rms_angle_v:.4f}"
          + (f"  KL {kl_q0:.4f}→{key_b:.4f}" + (f" r_KL {kd_rec['r_kl_best']:.3f}" if args.kl_ctl else "") if args.kd else ""),
          flush=True)


if __name__ == "__main__":
    main()

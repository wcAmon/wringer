"""E67-A 區塊級 GSQ(prereg_e67.json):九元格點不變,逐權重 local-shift logits + δα,Gumbel-Softmax 退火,
率項 λ·E_soft[bits],張量保護(λ=0)。逐層循序:目標 = bf16 老師該層在「學生輸入」上的輸出(局部目標,無交叉補償)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.gsq_block --save-tag gsq_a30 --lam 0.02 --steps 300
  --steps 0 ⇒ 硬指派 = 起點碼(bit-exact 儀器檢查)

參數化:W = α₀·exp(δ) ⊙ Σ_g p_g · vals[idx₀ + shift_g],shift ∈ {−S..S},越界 shift 遮 −inf;
p = softmax((logits + gumbel)/τ);硬化 = argmax logits。起點 logits[shift=0] = +init,其餘 0。
率:bits(格)由全態直方圖 −log₂p(同 E66-RD);模組率 = mean_w Σ_g p_g bits[idx₀+shift_g];保護模組 λ=0。
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from p2_anchor.wringer import ktier
from p2_anchor.wringer.engine import capture_layer0, fwd_chain
from p2_anchor.wringer.ladder_e65x import entropy_joint, g9_bits, ledger, sym9
from p2_anchor.wringer.modelio import (LAYER_PREFIX, N_LAYERS, enumerate_targets,
                                         get_module, layer_index, load_model)
from p2_anchor.wringer.state import dense_weight, load_state

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"


class GSQParam(nn.Module):
    """掛在 Linear.weight 的參數化。idx0 (M,K) 儲存欄序格索引(升冪 vals);inv 部署置換。"""

    def __init__(self, c, vals, bits, shift, init_logit, device="cuda"):
        super().__init__()
        M, nB, B = c["T1"].shape
        K = nB * B
        self.M, self.nB, self.B, self.K = M, nB, B, K
        self.register_buffer("vals", vals.float().to(device))                           # (G,)
        self.register_buffer("bits", bits.float().to(device))                           # (G,)
        G = vals.numel()
        t1 = c["T1"].reshape(M, K).to(torch.int64)
        t2 = c["T2"].reshape(M, K).to(torch.int64)
        sym = (t1 + 1) * 3 + (t2 + 1)
        gv, gt1, gt2 = ktier.grid_tensors(9, float(c["c"]), "cpu")
        order = ((gt1 + 1) * 3 + (gt2 + 1)).long()                                      # g -> sym
        g_of_sym = torch.empty(9, dtype=torch.int64)
        g_of_sym[order] = torch.arange(9)
        self.register_buffer("idx0", g_of_sym[sym].to(device, torch.int64))             # (M,K)
        self.register_buffer("inv", c["inv"].to(device))
        self.register_buffer("a0", c["a0"].reshape(M, nB, 1).to(device).float())
        self.S = int(shift)
        self.register_buffer("shifts", torch.arange(-self.S, self.S + 1, device=device))   # (2S+1,)
        cand = self.idx0.unsqueeze(-1) + self.shifts                                     # (M,K,2S+1)
        self.register_buffer("valid", (cand >= 0) & (cand < G))
        self.register_buffer("cand", cand.clamp(0, G - 1))
        lg = torch.zeros(M, K, 2 * self.S + 1, device=device)
        lg[..., self.S] = init_logit
        lg = lg.masked_fill(~self.valid, -1e4)
        self.logits = nn.Parameter(lg)
        self.dlog = nn.Parameter(torch.zeros(M, nB, 1, device=device))
        self.tau = 1.0
        self.gumbel = True
        self.gumbel_scale = 1.0
        self.hard = False
        self.last_rate = None

    def alpha(self):
        return self.a0 * torch.exp(self.dlog)

    def probs(self):
        lg = self.logits
        if self.gumbel and self.training:
            u = torch.rand_like(lg).clamp_(1e-9, 1 - 1e-9)
            lg = lg - self.gumbel_scale * torch.log(-torch.log(u))
        return F.softmax(lg / self.tau, dim=-1)

    def hard_idx(self):
        return self.cand.gather(-1, self.logits.argmax(-1, keepdim=True)).squeeze(-1)    # (M,K)

    def forward(self, W_orig):
        if self.hard:
            v = self.vals[self.hard_idx()]
            self.last_rate = None
        else:
            p = self.probs()                                                             # (M,K,2S+1)
            v = (p * self.vals[self.cand]).sum(-1)
            self.last_rate = (p * self.bits[self.cand]).sum(-1).mean()
        w = (self.alpha().expand(self.M, self.nB, self.B).reshape(self.M, self.K)) * v
        return w[:, self.inv].to(W_orig.dtype)

    @torch.no_grad()
    def extract(self, c):
        idx = self.hard_idx()
        gv, gt1, gt2 = ktier.grid_tensors(9, float(c["c"]), idx.device)
        T1 = gt1[idx].reshape(self.M, self.nB, self.B)
        T2 = gt2[idx].reshape(self.M, self.nB, self.B)
        return {"T1": T1.to(torch.int8).cpu(), "T2": T2.to(torch.int8).cpu(), "inv": c["inv"].clone(),
                "a0": self.alpha().cpu(), "c": c["c"], "grid": 9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-tag", default="g9")
    ap.add_argument("--save-tag", required=True)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--rd-bits", default="", help="9 格碼長 json 檔(含 'bits')或列表;空 = 起點態直方圖")
    ap.add_argument("--protect", default="linear_attn.out_proj,self_attn.v_proj", help="λ=0 的模組後綴(逗號)")
    ap.add_argument("--shift", type=int, default=2)
    ap.add_argument("--init-logit", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--logit-opt", choices=["adam", "tsgd"], default="adam",
                    help="logits 優化器:adam(逐參數正規化)/ tsgd(逐張量標量 RMS 正規化 SGD,保留逐權重梯度比)")
    ap.add_argument("--alpha-lr", type=float, default=1e-3)
    ap.add_argument("--tau0", type=float, default=1.0)
    ap.add_argument("--tau1", type=float, default=0.1)
    ap.add_argument("--no-gumbel", action="store_true")
    ap.add_argument("--gumbel-scale", type=float, default=1.0, help="Gumbel 噪聲尺度(1 = 標準;<1 減少探索噪聲)")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--calib", default=str(EV / "calib_e58r1b_traj.pt"))
    ap.add_argument("--lengths", default=str(EV / "calib_e58r1b_len.pt"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-layer", type=int, default=N_LAYERS - 1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_calib, args.max_layer = 16, min(args.max_layer, 2)
    args.out = args.out or str(EVC / f"gsq_{args.save_tag}.json")
    protect = [s for s in args.protect.split(",") if s]
    torch.manual_seed(0)
    t0 = time.time()

    model, _ = load_model()
    model.requires_grad_(False)
    model_t, _ = load_model()                                   # bf16 老師(權重不動)
    model_t.requires_grad_(False)
    targets = enumerate_targets(model)
    st = load_state(args.state_tag, targets)
    for n in targets:
        get_module(model, n).weight.data.copy_(dense_weight(st[n]).to(torch.bfloat16))
    if args.rd_bits:
        bits = json.loads(Path(args.rd_bits).read_text())["bits"] if args.rd_bits.endswith(".json") else json.loads(args.rd_bits)
    else:
        bits = g9_bits(st)
    bits_t = torch.tensor(bits)
    vals, _, _ = ktier.grid_tensors(9, 0.6, "cpu")
    print(f"gsq: lam={args.lam} steps={args.steps} shift={args.shift} protect={protect} bits={[round(b,3) for b in bits]}", flush=True)

    calib = torch.load(args.calib, weights_only=False)[:args.n_calib]
    lens = torch.load(args.lengths, weights_only=False)[:calib.shape[0]]
    S = calib.shape[1]
    ar = torch.arange(S)
    Mk = [(ar[None] < lens[i:i + args.batch, None]) for i in range(0, calib.shape[0], args.batch)]
    X, kw = capture_layer0(model, calib, batch=args.batch, to_cpu=True)
    layers = [model.get_submodule(f"{LAYER_PREFIX}.{li}") for li in range(N_LAYERS)]
    layers_t = [model_t.get_submodule(f"{LAYER_PREFIX}.{li}") for li in range(N_LAYERS)]
    print(f"load+capture {time.time()-t0:.0f}s batches {len(X)} seq {S}", flush=True)

    out = {"mode": "gsq_block", "state": args.state_tag, "hparams": vars(args), "bits": bits, "modules": {}}
    save_dir = Path("data") / f"qs_{args.save_tag}"
    save_dir.mkdir(parents=True, exist_ok=True)
    new_state = {}

    for li in range(N_LAYERS):
        mods = [n for n in targets if layer_index(n) == li]
        if li > args.max_layer:
            for n in mods:
                new_state[n] = st[n]
            torch.save({n: new_state[n] for n in mods}, save_dir / f"layer{li:02d}.pt")
            continue
        layer, layer_t = layers[li], layers_t[li]
        # 老師目標(學生輸入)與能量
        with torch.no_grad():
            Y = [fwd_chain([layer_t], X[bi], kw).to("cpu", torch.bfloat16) for bi in range(len(X))]
            e_ref = sum(float((Y[bi].float() ** 2 * Mk[bi].unsqueeze(-1)).sum()) for bi in range(len(X)))
        # 掛參數化
        params = {}
        for n in mods:
            c = st[n]
            p = GSQParam(c, vals, bits_t, args.shift, args.init_logit)
            parametrize.register_parametrization(get_module(model, n), "weight", p, unsafe=True)
            p.gumbel = not args.no_gumbel
            p.gumbel_scale = args.gumbel_scale
            params[n] = p
        lam_mod = {n: (0.0 if any(n.endswith(s) for s in protect) else args.lam) for n in mods}

        def rel_err(hard):
            for p in params.values():
                p.hard = hard
            with torch.no_grad():
                e = 0.0
                for bi in range(len(X)):
                    ys = fwd_chain([layer], X[bi], kw)
                    e += float(((ys.float() - Y[bi].cuda().float()) ** 2 * Mk[bi].cuda().unsqueeze(-1)).sum())
            for p in params.values():
                p.hard = False
            return math.sqrt(e / max(e_ref, 1e-30))

        err_hard0 = rel_err(True)
        if args.steps > 0:
            # logit 優化器:adam=逐參數正規化(只剩符號⇒率項成整模組門檻);
            # tsgd=逐張量單一標量 RMS 正規化(保留權重間梯度大小比⇒逐權重率失真取捨),步幅均值 ≈ lr
            opt = torch.optim.Adam([{"params": [p.logits for p in params.values()], "lr": args.lr if args.logit_opt == "adam" else 0.0},
                                    {"params": [p.dlog for p in params.values()], "lr": args.alpha_lr}])
            rms_ema = {n: None for n in mods}
            for p in params.values():
                p.train()
            for step in range(1, args.steps + 1):
                tau = args.tau0 * (args.tau1 / args.tau0) ** ((step - 1) / max(args.steps - 1, 1))
                for p in params.values():
                    p.tau = tau
                bi = (step - 1) % len(X)
                ys = fwd_chain([layer], X[bi], kw)
                m = Mk[bi].cuda().unsqueeze(-1)
                rec = ((ys.float() - Y[bi].cuda().float()) ** 2 * m).sum() / (e_ref / len(X))
                rate = sum(lam_mod[n] * params[n].last_rate for n in mods if lam_mod[n] > 0)
                loss = rec + (rate if isinstance(rate, torch.Tensor) else 0.0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if args.logit_opt == "tsgd":
                    with torch.no_grad():
                        for n in mods:
                            g = params[n].logits.grad
                            if g is None:
                                continue
                            r = g.float().pow(2).mean().sqrt()
                            rms_ema[n] = r if rms_ema[n] is None else 0.9 * rms_ema[n] + 0.1 * r
                            params[n].logits.sub_(args.lr * g / (rms_ema[n] + 1e-30))
                if step % 50 == 0 or step == args.steps:
                    print(f"  L{li} step {step} tau {tau:.3f} rec {float(rec):.5f} rate {float(rate) if isinstance(rate, torch.Tensor) else 0:.4f} ({time.time()-t0:.0f}s)", flush=True)
            for p in params.values():
                p.eval()
        err_hard = rel_err(True)
        line = []
        for n in mods:
            c, p = st[n], params[n]
            ent = p.extract(c)
            s_old, s_new = sym9(c), sym9(ent)
            rec = {"layer": li, "lam": lam_mod[n], "flips": float((s_old != s_new).float().mean()),
                   "entropy_joint": entropy_joint(ent), "dalpha_med": float(p.dlog.abs().median()),
                   "dense_rel_change": float((dense_weight(ent) - dense_weight(c)).norm() / dense_weight(c).norm())}
            out["modules"][n] = rec
            new_state[n] = ent
            parametrize.remove_parametrizations(get_module(model, n), "weight", leave_parametrized=False)
            get_module(model, n).weight.data.copy_(dense_weight(ent).to(torch.bfloat16))
            line.append(f"{n.split('.')[-1][:9]:9s} flips={rec['flips']:.4f} H={rec['entropy_joint']:.3f} dW={rec['dense_rel_change']:.3f} lam={lam_mod[n]}")
        out.setdefault("layers", {})[li] = {"err_hard0": err_hard0, "err_hard": err_hard}
        del params, Y
        with torch.no_grad():
            for bi in range(len(X)):
                X[bi] = fwd_chain([layer], X[bi], kw).to("cpu", torch.bfloat16)
        torch.cuda.empty_cache()
        torch.save({n: new_state[n] for n in mods}, save_dir / f"layer{li:02d}.pt")
        print(f"GSQ_LAYER {li} err_hard {err_hard0:.4f}->{err_hard:.4f} ({time.time()-t0:.0f}s)\n   " + "\n   ".join(line), flush=True)
        Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))

    out["ledger"] = ledger(new_state)
    out["runtime_s"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("GSQ_LEDGER " + json.dumps(out["ledger"]), flush=True)
    print(f"GSQ_DONE {args.save_tag} {out['runtime_s']}s", flush=True)


if __name__ == "__main__":
    main()

"""E37 phase0+1:起點絕對化 + KD 預訓水庫(prereg_e37_reservoir)。

phase0:load 冠軍態(qs_stmix_e2e)→ 材化 → 對 exports/e32_e2e 逐 tensor
        位元一致斷言(P0 gate;FAIL 即停機重審起點)。
phase1:每 target Linear 插量化後旁路 W_eff = W_hard + γ·(B_rA_r)/r,
        r=128、γ≡1、B 零初始(恆等起步);bf16 老師全詞表 KD 只訓旁路
        (量化側全凍——碼已材化為權重,天然凍結)。val CE 平台判停
        (監控不裁決,上限 --steps)。
落盤:水庫 data/res_<tag>/layerXX.pt + 合併 fp 模型 exports/<out>
      (裁判點 A 直接 vLLM serve 此目錄 = serve_merge 併入本腳本)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.res_fill \
      --tag res37 --state-tag stmix_e2e --ref-export exports/e32_e2e \
      --out exports/e37_resA --steps 1200 --data cal2路徑
"""
import argparse
import datetime
import glob
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from p2_anchor.wringer.export import SNAPSHOT
from p2_anchor.wringer.modelio import (CALIB_VAL, enumerate_targets, get_module,
                                         layer_index, load_model)
from p2_anchor.wringer.polish import (closure_weights, kd_loss,
                                        kd_loss_hidchunk)
from p2_anchor.wringer.state import apply_k2, load_state

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"
SEED = 20260806


class ResBypass(nn.Module):
    """phase1 旁路 parametrization:W + (B@A)/r(γ≡1)。B=0 恆等起步。"""

    def __init__(self, M, K, r=128, device="cuda"):
        super().__init__()
        self.r = r
        A = torch.empty(r, K, device=device, dtype=torch.float32)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(torch.zeros(M, r, device=device,
                                          dtype=torch.float32))

    def forward(self, W):
        return W + ((self.B @ self.A) / self.r).to(W.dtype)


def p0_gate(model, targets, ref_export):
    """材化態 vs 冠軍 export 逐 tensor 位元一致(P0 gate)。"""
    from safetensors import safe_open
    ref = {}
    for f in glob.glob(str(Path(ref_export) / "*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                ref[k] = f
    bad = 0
    for i, n in enumerate(targets):
        key = n + ".weight"
        with safe_open(ref[key], framework="pt") as sf:
            r = sf.get_tensor(key)
        w = get_module(model, n).weight.detach().cpu()
        ok = torch.equal(w, r)
        bad += not ok
        if not ok or i % 50 == 0:
            print(f"P0 [{i + 1}/{len(targets)}] {n} bit-exact={ok}",
                  flush=True)
    assert bad == 0, f"P0 gate FAIL:{bad} tensors 不一致——停機重審起點"
    print(f"P0 gate PASS({len(targets)} tensors 位元一致 vs {ref_export})",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="res37")
    ap.add_argument("--state-tag", default="stmix_e2e")
    ap.add_argument("--ref-export", default="exports/e32_e2e")
    ap.add_argument("--out", default="exports/e37_resA")
    ap.add_argument("--r-res", type=int, default=128)
    ap.add_argument("--data", default=str(EV / "calib_train_512_v2.pt"))
    ap.add_argument("--lengths", default=None,
                    help="data 逐列有效長度 .pt(E54 8k pad 遮罩:KD loss "
                         "位置權重歸零,語義同 e2e_soft --lengths;"
                         "None=舊行為逐位不變)")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--plateau-eps", type=float, default=0.005,
                    help="平台判停線:val CE 改善比例 <eps 即停"
                         "(E57 步數主導時降 0.002 僅保底)")
    ap.add_argument("--res-init", default=None,
                    help="warm-start:載入 data/res_<tag>/layerXX.pt 的 "
                         "A/B 續訓(E57 梯子複利;None=fresh)")
    ap.add_argument("--plateau-every", type=int, default=300,
                    help="每隔此步數量 val CE;改善 <0.5% 即平台判停")
    ap.add_argument("--plateau-min-steps", type=int, default=0,
                    help="此步數前 plateau 不判停(val CE 照量監控;"
                         "0=舊行為;E57 修理:warm-start 首檢即觸發"
                         "=第12型 proxy 鈍化,發散保底另設 +5% 線)")
    ap.add_argument("--smoke", action="store_true",
                    help="20 步、免 P0 全查(抽 8 tensors)、免導出")
    ap.add_argument("--microbatch", type=int, default=1,
                    help="1=舊行為(逐樣本累加);0=整批一次前向;n=微批 n")
    ap.add_argument("--kd-impl", choices=("chunk", "hidchunk"),
                    default="chunk",
                    help="chunk=原 kd_loss;hidchunk=hidden-chunk 版"
                         "(logits 不整條物化,microbatch>1 建議)")
    ap.add_argument("--close-weight", type=float, default=1.0,
                    help="E69 S3:</think> target 位置 KD 權重 λ_close(1.0=關)")
    ap.add_argument("--close-post-weight", type=float, default=16.0)
    ap.add_argument("--close-post-k", type=int, default=8)
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="骨幹梯度檢查點(16k 列學生活化圖 OOM 修;"
                         "default 關=舊行為)")
    args = ap.parse_args()
    kd_fn = kd_loss if args.kd_impl == "chunk" else kd_loss_hidchunk

    t0 = time.time()
    torch.manual_seed(SEED)
    model, tok = load_model()
    teacher, _ = load_model()
    model.requires_grad_(False)
    teacher.requires_grad_(False)
    targets = enumerate_targets(model)
    st = load_state(args.state_tag, targets)
    apply_k2(model, st)                       # 材化冠軍態(硬碼→權重)
    p0_gate(model, targets if not args.smoke else targets[::32],
            args.ref_export)

    data = torch.load(args.data, weights_only=False)
    lens = (torch.load(args.lengths, weights_only=False)
            if args.lengths else None)
    if lens is not None:
        assert lens.shape[0] == data.shape[0], "lengths 與 data 列數不符"
        print(f"lengths 遮罩:有效率 "
              f"{float(lens.sum()) / data.numel():.3f}"
              f" / 滿窗列 {(lens == data.shape[1]).sum().item()}",
              flush=True)
    val = torch.load(CALIB_VAL, weights_only=False)
    from p1_grouping.e3_runner import eval_ce
    close_id = tok.convert_tokens_to_ids("</think>")
    if args.close_weight > 1.0:
        print(f"閉合加權:</think> id={close_id} λ_close={args.close_weight} "
              f"λ_post={args.close_post_weight} K={args.close_post_k}",
              flush=True)

    if args.smoke:
        args.steps, args.plateau_every = 20, 10**9
        data = data[:16]
        lens = lens[:16] if lens is not None else None

    res = {}
    for n in targets:
        m = get_module(model, n)
        M, K = m.weight.shape
        b = ResBypass(M, K, r=args.r_res, device=m.weight.device)
        parametrize.register_parametrization(m, "weight", b)
        res[n] = b
    if args.res_init:
        init_dir = Path("data") / f"res_{args.res_init}"
        n_loaded = 0
        for f in sorted(init_dir.glob("layer*.pt")):
            for n, d in torch.load(f, map_location="cpu",
                                   weights_only=False).items():
                assert d["r"] == args.r_res, f"r 不符:{n} {d['r']}"
                with torch.no_grad():
                    res[n].A.copy_(d["A"].cuda())
                    res[n].B.copy_(d["B"].cuda())
                n_loaded += 1
        assert n_loaded == len(targets), \
            f"res-init 只載到 {n_loaded}/{len(targets)}"
        print(f"res_init={args.res_init}(warm-start {n_loaded} targets)",
              flush=True)
    opt = torch.optim.Adam(
        [q for b in res.values() for q in (b.A, b.B)], lr=args.lr)

    if args.grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.train()   # HF ckpt 僅 training=True 生效;dropout 全 0 語義不變
        print("grad_ckpt=ON(骨幹活化逐層重算;model.train)", flush=True)

    ce0 = eval_ce(model, val)                 # BA=0 → 即冠軍態 CE(監控錨)
    print(f"pre-fill val CE {ce0:.6f}(=冠軍態;BA=0 恆等)", flush=True)

    g_rng = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(data), generator=g_rng)
    pos = 0
    log, ces = [], [(0, ce0)]
    last_ce = ce0
    stop_reason = "steps"
    for step in range(1, args.steps + 1):
        if pos + args.batch > len(data):
            order = torch.randperm(len(data), generator=g_rng)
            pos = 0
        rows = order[pos:pos + args.batch]
        b = data[rows].cuda()
        pos += args.batch
        opt.zero_grad(set_to_none=True)
        kd_val = 0.0
        mb = b.shape[0] if args.microbatch == 0 else args.microbatch
        for i in range(0, b.shape[0], mb):
            xb = b[i:i + mb]
            xw = None
            if args.close_weight > 1.0:       # e2e_soft 同式(polish.closure_weights)
                xw = closure_weights(xb, close_id, args.close_weight,
                                     args.close_post_weight,
                                     args.close_post_k)
            if lens is not None:              # e2e_soft --lengths 同語義
                xl = lens[rows[i:i + mb]].to(xb.device)
                T = xb.shape[1]
                lim = torch.where(xl < T, xl - 1, xl)
                mask = (torch.arange(T, device=xb.device)[None]
                        < lim[:, None]).float()
                xw = mask if xw is None else xw * mask
            loss = kd_fn(model, teacher, xb,
                         w=xw) * xb.shape[0] / b.shape[0]
            loss.backward()
            kd_val += float(loss)
        torch.nn.utils.clip_grad_norm_(
            [q for b_ in res.values() for q in (b_.A, b_.B)], args.clip)
        opt.step()
        if not np.isfinite(kd_val):
            raise RuntimeError(f"警報:step {step} KD loss 非有限 {kd_val}")
        log.append(round(kd_val, 6))
        if step % 10 == 0:
            print(f"step {step:4d}  kd {kd_val:.5f}  "
                  f"({(time.time()-t0)/step:.1f}s/step)", flush=True)
        if step % args.plateau_every == 0 and step < args.steps:
            ce = eval_ce(model, val)
            ces.append((step, ce))
            print(f"  val CE @{step}: {ce:.6f}(前次 {last_ce:.6f})",
                  flush=True)
            if ce > last_ce * 1.05:           # 發散保底(不受鈍化影響)
                stop_reason = f"diverge@{step}"
                last_ce = ce
                break
            if (step >= args.plateau_min_steps
                    and ce > last_ce * (1 - args.plateau_eps)):
                # 改善 <eps → 平台判停(min-steps 前僅監控)
                stop_reason = f"plateau@{step}"
                last_ce = ce
                break
            last_ce = ce

    # 水庫落盤(凍結資產;.pt 不進 git)
    out_res = Path("data") / f"res_{args.tag}"
    out_res.mkdir(parents=True, exist_ok=True)
    by_layer = {}
    for n in targets:
        by_layer.setdefault(layer_index(n), {})[n] = {
            "B": res[n].B.detach().cpu(), "A": res[n].A.detach().cpu(),
            "r": args.r_res}
    for li, ls in by_layer.items():
        torch.save(ls, out_res / f"layer{li:02d}.pt")

    ce1 = eval_ce(model, val)
    ces.append(("final", ce1))
    print(f"post-fill val CE {ce1:.6f}(Δ {ce1 - ce0:+.6f};監控不裁決)",
          flush=True)

    # 合併導出(裁判點 A 的 serve 目錄;fp 模型,無位面包)
    if not args.smoke:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for n in targets:                     # 合併 = 去 parametrization 留值
            m = get_module(model, n)
            w_eff = m.weight.detach().clone()
            parametrize.remove_parametrizations(m, "weight",
                                                leave_parametrized=False)
            m.weight.data.copy_(w_eff)
        model.save_pretrained(out, safe_serialization=True)
        tok.save_pretrained(out)
        for f in ("chat_template.jinja", "preprocessor_config.json",
                  "processor_config.json", "generation_config.json"):
            src = SNAPSHOT / f
            if src.exists() and not (out / f).exists():
                shutil.copy(src, out / f)
        print(f"merged export → {out}", flush=True)

    rec = {"tag": args.tag, "state_tag": args.state_tag,
           "ref_export": args.ref_export, "p0_gate": "PASS",
           "r_res": args.r_res, "data": str(args.data),
           "data_sha256": hashlib.sha256(
               Path(args.data).read_bytes()).hexdigest(),
           "hparams": vars(args), "seed": SEED,
           "pre_fill_val_ce": ce0, "post_fill_val_ce": ce1,
           "val_ce_curve": [[s, round(c, 6)] for s, c in ces],
           "stop_reason": stop_reason, "kd_log": log,
           "runtime_s": round(time.time() - t0, 1),
           "timestamp": datetime.datetime.now().isoformat()}
    (EVC / f"fill_{args.tag}.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"FILL_DONE({stop_reason},{rec['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main()

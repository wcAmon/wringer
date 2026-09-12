"""α-only logit KD 輕打磨(Corkscrew 第三段;移植 guava run_k2 train)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.polish \
      --tag st3_pol50 --state-tag st3 --steps 50 --lr 3e-5 \
      [--teacher-dir exports/t9]   # 預設 teacher = bf16 原模型

teacher 常駐(96GB 同卡雙 bf16 可容);--teacher-dir 換 9 元 teacher 臂
(槓桿 C-1:輸出對齊天花板降為 teacher 保留率,換取部署域老師)。
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
from torch.utils.checkpoint import checkpoint

from p2_anchor.wringer.ktier import AlphaParamK2
from p2_anchor.wringer.modelio import (enumerate_targets, get_module,
                                         load_model)
from p2_anchor.wringer.state import load_state

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"
SEED = 20260806


def closure_weights(b, close_id, lam_close, lam_post=16.0, post_k=8):
    """E51:閉合訊號位置級 loss 權重(</think> 稀有度 1/5428 的解毒劑)。

    位置 t 的分布預測 b[:,t+1]:target==</think> 之位置權重 lam_close、
    其後 post_k 個位置 lam_post(重教開口作答),餘 1;末位(target 出窗)1。
    與 kd_loss 的 w 參數配套:loss = Σ w·KL_t / Σ w(λ=1 時傳 None 走原路徑)。"""
    B, T = b.shape
    tgt = b[:, 1:]
    hit = tgt == close_id                              # (B, T-1)
    post = torch.zeros_like(hit)
    for k in range(1, post_k + 1):
        post[:, k:] |= hit[:, :-k]
    wt = torch.where(hit, lam_close,
                     torch.where(post, lam_post, 1.0))
    w = torch.ones(B, T, device=b.device, dtype=wt.dtype)
    w[:, :-1] = wt
    return w


def kd_loss(model, teacher, b, chunk=512, direction="fwd", w=None):
    """分塊全詞表 KL,塊內 fp32(e13mini/e16 同款)。

    direction="fwd":D_KL(p_t‖q_s),老師加權(mode-covering,全鏈預設)。
    direction="rev":D_KL(q_s‖p_t),學生加權(mode-seeking;E40 e2e 互補臂,
    欠容量端集中容量學主模式)。不同 direction 的損失值不可比。
    w:(B,T) 位置權重(E51 閉合軸);None 時歸約序與歷史逐位一致。"""
    with torch.no_grad():
        tl = teacher(input_ids=b, use_cache=False).logits
    sl = model(input_ids=b, use_cache=False).logits
    n_pos = sl.shape[1]
    loss = 0.0
    for c0 in range(0, n_pos, chunk):
        c1 = c0 + chunk
        with torch.no_grad():
            ltc = F.log_softmax(tl[:, c0:c1].float(), dim=-1)
        lsc = F.log_softmax(sl[:, c0:c1].float(), dim=-1)
        if direction == "fwd":
            with torch.no_grad():
                ptc = ltc.exp()
            if w is None:
                loss = loss + (ptc * (ltc - lsc)).sum()
            else:
                loss = loss + ((ptc * (ltc - lsc)).sum(-1)
                               * w[:, c0:c1]).sum()
        else:
            qsc = lsc.exp()                       # 梯度經 q_s 與 log q_s 雙路
            if w is None:
                loss = loss + (qsc * (lsc - ltc)).sum()
            else:
                loss = loss + ((qsc * (lsc - ltc)).sum(-1)
                               * w[:, c0:c1]).sum()
    return loss / (sl.shape[0] * n_pos) if w is None else loss / w.sum()


class _IdHead(nn.Module):
    """lm_head 佔位:恆等 → model(...).logits 即 final hidden。"""

    def forward(self, x):
        return x


def _backbone_hidden(mdl, b):
    """前向到 final hidden(lm_head 暫換恆等,避免全長 logits 物化)。

    回傳 (hidden (B,T,H), lm_head 模組)。用 HF 標準
    get/set_output_embeddings,對 ImageTextToText 包裝不猜屬性路徑。"""
    lm = mdl.get_output_embeddings()
    mdl.set_output_embeddings(_IdHead())
    try:
        h = mdl(input_ids=b, use_cache=False).logits
    finally:
        mdl.set_output_embeddings(lm)
    return h, lm


def kd_loss_hidchunk(model, teacher, b, chunk=512, direction="fwd", w=None):
    """kd_loss 同式,但 logits 永不整條物化(引擎壓榨 Patch B)。

    骨幹各前向一次到 hidden;lm_head+log_softmax+KL 逐 chunk 在
    checkpoint 內算,反向逐 chunk 重算 → 圖上唯一大張量 = hidden
    (B,T,H)。batch 8 時 logits 側 ~20 GB → <1 GB。與 kd_loss 的數值
    等價性由等效閘量測(GEMM 分塊可能引入 ULP 級差,非逐位保證)。"""
    with torch.no_grad():
        th, tlm = _backbone_hidden(teacher, b)
    sh, slm = _backbone_hidden(model, b)
    n_pos = sh.shape[1]

    def _blk(shc, thc, wc):
        with torch.no_grad():
            ltc = F.log_softmax(tlm(thc).float(), dim=-1)
        lsc = F.log_softmax(slm(shc).float(), dim=-1)
        if direction == "fwd":
            with torch.no_grad():
                ptc = ltc.exp()
            if wc is None:
                return (ptc * (ltc - lsc)).sum()
            return ((ptc * (ltc - lsc)).sum(-1) * wc).sum()
        qsc = lsc.exp()
        if wc is None:
            return (qsc * (lsc - ltc)).sum()
        return ((qsc * (lsc - ltc)).sum(-1) * wc).sum()

    loss = 0.0
    for c0 in range(0, n_pos, chunk):
        wc = None if w is None else w[:, c0:c0 + chunk]
        loss = loss + checkpoint(_blk, sh[:, c0:c0 + chunk],
                                 th[:, c0:c0 + chunk], wc,
                                 use_reentrant=False)
    return loss / (sh.shape[0] * n_pos) if w is None else loss / w.sum()


@torch.no_grad()
def build_topk_cache(teacher, data, K=256, batch=4, chunk=512):
    """E45:老師 top-K 分布一次性快取(teacher trace 固定,1200 步重複
    前向是純浪費)。回傳 CPU dict:probs/idx (N,T,K) fp16/int32、
    tail (N,T) fp16(K 外殘餘質量,FKL 以單桶近似)、cover=1−tail 分位數
    (截斷覆蓋審計入檔)。"""
    N, T = data.shape
    probs = torch.empty(N, T, K, dtype=torch.float16)
    idx = torch.empty(N, T, K, dtype=torch.int32)
    tail = torch.empty(N, T, dtype=torch.float16)
    trace_lp = torch.zeros(N, T, dtype=torch.float16)   # 老師對 trace 下一
    for r0 in range(0, N, batch):                       # token 的 logp(span 用)
        b = data[r0:r0 + batch].cuda()
        tl = teacher(input_ids=b, use_cache=False).logits
        for c0 in range(0, T, chunk):
            p = F.softmax(tl[:, c0:c0 + chunk].float(), dim=-1)
            pk, ik = torch.topk(p, K, dim=-1)
            probs[r0:r0 + b.shape[0], c0:c0 + chunk] = pk.half().cpu()
            idx[r0:r0 + b.shape[0], c0:c0 + chunk] = ik.int().cpu()
            tail[r0:r0 + b.shape[0], c0:c0 + chunk] = (
                1.0 - pk.sum(-1)).clamp_min(0).half().cpu()
            t_hi = min(c0 + chunk, T - 1)               # 末位無目標,保持 0
            if c0 < t_hi:
                tgt = b[:, c0 + 1:t_hi + 1].unsqueeze(-1).long()
                lp = p[:, :t_hi - c0].gather(-1, tgt).clamp_min(
                    1e-12).log()
                trace_lp[r0:r0 + b.shape[0], c0:t_hi] = \
                    lp.squeeze(-1).half().cpu()
        del tl
    cov = 1.0 - tail.float()
    cs, _ = cov.flatten().sort()                    # quantile() 有 2^24 上限
    n = cs.numel()
    cover = {q: round(float(cs[min(int(q * (n - 1)), n - 1)]), 6)
             for q in (0.01, 0.05, 0.5)}
    return {"probs": probs, "idx": idx, "tail": tail, "K": K,
            "cover": cover, "trace_lp": trace_lp}


def kd_topk(sl, pk, ik, pt, chunk=512, trace_ids=None):
    """FKL 對 top-K 快取:Σ_K p(log p − log q) + 尾桶項;逐位置平均,
    與 kd_loss(fwd) 同規約(差 = K 外逐詞項聚成單桶,smoke 等價閘量測)。
    sl (B,T,V) 帶梯度;pk/ik/pt 已在 GPU,ik 為 long。
    trace_ids 非 None(B,T,span 對齊用)→ 另回傳學生對 trace 下一
    token 的逐位置 logp(B,T,帶梯度)。"""
    B, T, _ = sl.shape
    loss = 0.0
    ls_tr = [] if trace_ids is not None else None
    for c0 in range(0, T, chunk):
        c1 = c0 + chunk
        ls = F.log_softmax(sl[:, c0:c1].float(), dim=-1)
        lq = ls.gather(-1, ik[:, c0:c1])
        p = pk[:, c0:c1].float().clamp_min(1e-12)
        q_tail = (1.0 - lq.exp().sum(-1)).clamp_min(1e-12)
        p_tail = pt[:, c0:c1].float().clamp_min(1e-12)
        loss = loss + (p * (p.log() - lq)).sum() \
            + (p_tail * (p_tail.log() - q_tail.log())).sum()
        if ls_tr is not None:
            ls_tr.append(ls.gather(
                -1, trace_ids[:, c0:c1].unsqueeze(-1)).squeeze(-1))
    loss = loss / (B * T)
    if ls_tr is not None:
        return loss, torch.cat(ls_tr, dim=1)
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--state-tag", required=True)
    ap.add_argument("--data", default=str(EV / "calib_train.pt"))
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--accum", choices=("sample", "batched"), default="sample")
    ap.add_argument("--teacher-dir", default=None,
                    help="HF 目錄(如 exports/t9);預設 bf16 原模型")
    args = ap.parse_args()

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
    data = torch.load(args.data, weights_only=False)
    val = torch.load(EV / "calib_val.pt", weights_only=False)

    from p1_grouping.e3_runner import eval_ce

    for n in targets:
        c = st[n]
        parametrize.register_parametrization(
            get_module(model, n), "weight",
            AlphaParamK2(c["T1"].cuda(), c["T2"].cuda(), c["inv"].cuda(),
                         c["a0"].cuda(), c["c"]))
    torch.cuda.empty_cache()
    ce0 = eval_ce(model, val)
    print(f"pre-polish val CE {ce0:.6f}", flush=True)

    alphas = {n: get_module(model, n).parametrizations.weight[0].alpha
              for n in targets}
    opt = torch.optim.Adam(list(alphas.values()), lr=args.lr)
    g_rng = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(data), generator=g_rng)
    pos = 0
    log = []
    for step in range(1, args.steps + 1):
        if pos + args.batch > len(data):
            order = torch.randperm(len(data), generator=g_rng)
            pos = 0
        b = data[order[pos:pos + args.batch]].cuda()
        pos += args.batch
        opt.zero_grad(set_to_none=True)
        if args.accum == "batched":
            loss = kd_loss(model, teacher, b)
            loss.backward()
            kd_val = float(loss)
        else:
            kd_val = 0.0
            for i in range(b.shape[0]):
                loss = kd_loss(model, teacher, b[i:i + 1]) / b.shape[0]
                loss.backward()
                kd_val += float(loss)
        opt.step()
        if not np.isfinite(kd_val):
            raise RuntimeError(f"警報:step {step} KD loss 非有限 {kd_val}")
        log.append({"step": step, "kd": round(kd_val, 6)})
        if step % 10 == 0:
            print(f"step {step:4d}  kd {kd_val:.5f}  "
                  f"({(time.time()-t0)/step:.1f}s/step)", flush=True)

    ce1 = eval_ce(model, val)
    out = EVC / f"alphas_{args.tag}.npz"
    np.savez(out, **{n: a.detach().cpu().numpy() for n, a in alphas.items()})
    rec = {"tag": args.tag, "state_tag": args.state_tag,
           "teacher": args.teacher_dir or "bf16",
           "data": str(args.data),
           "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
           "steps": args.steps, "batch": args.batch, "lr": args.lr,
           "accum": args.accum, "seed": SEED,
           "pre_polish_val_ce": ce0, "post_polish_val_ce": ce1,
           "alpha_npz_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
           "kd_log": log, "runtime_s": round(time.time() - t0, 1),
           "timestamp": datetime.datetime.now().isoformat()}
    (EVC / f"train_{args.tag}.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"post-polish val CE {ce1:.6f}  → {out}")


if __name__ == "__main__":
    main()

"""spec-decode 接受率診斷(E44/E45 共用儀器;monitor-only,嚴禁裁決)。

學生(qs 態材化)當草稿、bf16 老師當驗證:對 val 前綴讓學生採樣續寫,
老師對同一序列逐位評分,接受率 a_t = min(1, p_T(x_t)/q_S(x_t))
(speculative sampling 的逐 token 接受機率)。

這是本專案**首個 off-trace 儀器**:既有 proxy 全是 teacher-forced
on-trace;此處量的是學生自生成路徑上的短段路徑質量——與 E45 桿3
(段落 Σlogp 對齊)量測同一個量的部署形式。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.diag_specaccept \
      --state-tag qs_st44d_e2e --prompts 48 --plen 64 --gen 128
"""
import argparse
import datetime
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from p2_anchor.wringer.modelio import enumerate_targets, load_model
from p2_anchor.wringer.state import apply_k2, load_state

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"
SEED = 20260806


@torch.no_grad()
def token_logp(m, seqs, lo, chunk=8):
    """回傳 (N, L-1):位置 t 對實際 token x_{t+1} 的 logp,取 t≥lo−1。"""
    outs = []
    for r0 in range(0, seqs.shape[0], chunk):
        b = seqs[r0:r0 + chunk].cuda()
        lg = m(input_ids=b, use_cache=False).logits[:, lo - 1:-1]
        lp = F.log_softmax(lg.float(), dim=-1)
        outs.append(lp.gather(-1, b[:, lo:].unsqueeze(-1))
                    .squeeze(-1).cpu())
        del lg, lp
    return torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-tag", required=True)
    ap.add_argument("--prompts", type=int, default=48)
    ap.add_argument("--plen", type=int, default=64)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--block", type=int, default=4,
                    help="段接受長度期望的段長 k")
    ap.add_argument("--temp", type=float, default=1.0)
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(SEED)
    model, tok = load_model()
    model.requires_grad_(False)
    targets = enumerate_targets(model)
    st = load_state(args.state_tag, targets)
    apply_k2(model, st)                       # 學生=部署硬態
    model.eval()

    val = torch.load(EV / "calib_val.pt", weights_only=False)
    prompts = val[:args.prompts, :args.plen]

    seqs = []
    for r0 in range(0, prompts.shape[0], 8):  # 學生採樣續寫(草稿分布 q)
        out = model.generate(prompts[r0:r0 + 8].cuda(),
                             max_new_tokens=args.gen, do_sample=True,
                             temperature=args.temp, top_p=1.0, top_k=0,
                             pad_token_id=(tok.eos_token_id or 0))
        seqs.append(out.cpu())
    seqs = torch.cat(seqs)

    lq = token_logp(model, seqs, args.plen)   # 學生 logq(自生成 token)
    teacher, _ = load_model()                 # 驗證器 p
    teacher.requires_grad_(False)
    lp = token_logp(teacher, seqs, args.plen)
    del teacher
    torch.cuda.empty_cache()

    a = (lp - lq).exp().clamp(max=1.0)        # (N, gen) 逐 token 接受率
    k = args.block
    nb = a.shape[1] // k
    ab = a[:, :nb * k].view(a.shape[0], nb, k)
    cum = ab.cumprod(dim=-1)                  # 段內前綴積
    exp_len = cum.sum(dim=-1)                 # E[段接受長度](上限 k)
    rec = {"state_tag": args.state_tag, "seed": SEED,
           "prompts": args.prompts, "plen": args.plen, "gen": args.gen,
           "temp": args.temp, "block": k,
           "accept_mean": round(float(a.mean()), 6),
           "accept_median": round(float(a.median()), 6),
           "accept_p10": round(float(a.quantile(0.10)), 6),
           "block_exp_len_mean": round(float(exp_len.mean()), 4),
           "per_prompt_accept": [round(float(x), 4)
                                 for x in a.mean(dim=1)],
           "runtime_s": round(time.time() - t0, 1),
           "timestamp": datetime.datetime.now().isoformat(),
           "discipline": "monitor-only;12 起 proxy 失效在案,嚴禁裁決"}
    out = EVC / f"diag_specaccept_{args.state_tag}.json"
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"accept mean {rec['accept_mean']}  median "
          f"{rec['accept_median']}  E[L{k}] {rec['block_exp_len_mean']}"
          f"  → {out}", flush=True)
    print("SPECACCEPT_DONE", flush=True)


if __name__ == "__main__":
    main()

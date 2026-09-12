"""GPTQ 式逐層量化 + joint α(Corkscrew 第一段)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.quantize \
      --grid 3 --tag g3 [--n-calib 128]

Σ 以當前模型狀態順序傳播收集(前置層已量化);狀態存 data/qs_<tag>/,
證據 JSON 存 evidence/p1_grouping/corkscrew/quant_<tag>.json。
"""
import argparse
import datetime
import json
import time
from pathlib import Path

import torch

from p2_anchor.wringer.cov import collect_cov
from p2_anchor.wringer.ktier import C_K2, gptq_grid, scale_joint
from p2_anchor.wringer.modelio import (LAYER_PREFIX, N_LAYERS,
                                         cov_source_for, enumerate_targets,
                                         get_module, layer_index, load_model)
from p2_anchor.wringer.state import load_state, qs_dir

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"
SEED = 20260806


def parse_module_map(s):
    """'suffix:grid:block,...' → {suffix: (grid, block|'row')}"""
    m = {}
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        suf, g, b = part.split(":")
        m[suf] = (int(g), "row" if b == "row" else int(b))
    return m


def parse_grid_map(s):
    m = {}
    for part in s.split(","):
        rng, g = part.split(":")
        a, _, b = rng.partition("-")
        for li in range(int(a), int(b or a) + 1):
            m[li] = int(g)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, required=True, choices=(9, 7, 5, 3, 8, 16, 4))
    ap.add_argument("--solver", choices=("gptq", "rtn"), default="gptq",
                    help="E68:rtn = GGUF 式(逐 block 尺度搜尋 + 最近格點,diag(Σ) 加權,無列補償);gptq = 舊路 + scale_joint")
    ap.add_argument("--alpha-init", choices=("auto", "twn", "search"), default="auto",
                    help="α 初值:twn=舊 init_alpha;search=alpha_search(帶方向位);auto=整數格 search、k2 格 twn")
    ap.add_argument("--prior-lam", type=float, default=0.0,
                    help="scale_joint 先驗正則(拉向 α 初值);0=舊行為(E65-X 教訓:整數格建議 0.01)")
    ap.add_argument("--alpha-bits", type=int, choices=(16, 8), default=16,
                    help="E69:α 存 int8(逐列 fp16 尺度,對稱 round;逐列格 nB==1 保持 16)")
    ap.add_argument("--module-map", default="",
                    help="逐模組 (grid,block) 覆寫:'self_attn.k_proj:256:row,self_attn.v_proj:256:row,self_attn.o_proj:16:128'")
    ap.add_argument("--grid-map", default=None,
                    help="逐層 grid,如 '0-15:3,16-27:5,28-31:9';"
                         "未涵蓋層退回 --grid(E32 混合多元)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--c", type=float, default=C_K2)
    ap.add_argument("--block", type=int, default=32,
                    help="α group 大小(E31 起支援 16)")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--calib", default=str(EV / "calib_train.pt"))
    ap.add_argument("--base-dir", default=None,
                    help="底模目錄(E60:折 γ 基底 exports/a1_fold_bf16 → g9f)")
    args = ap.parse_args()

    t0 = time.time()
    from p2_anchor.wringer import ktier
    from p2_anchor.wringer.ktier import alpha_search, grid_tensors, round_to_grid
    ktier.set_block(args.block)
    B = args.block
    mmap = parse_module_map(args.module_map)
    torch.manual_seed(SEED)
    out = qs_dir(args.tag)
    out.mkdir(parents=True, exist_ok=True)
    EVC.mkdir(parents=True, exist_ok=True)
    from p2_anchor.wringer.rot_carrier import use_base_dir
    use_base_dir(args.base_dir)
    model, tok = load_model()
    model.requires_grad_(False)
    calib = torch.load(args.calib, weights_only=False)[:args.n_calib]
    val = torch.load(EV / "calib_val.pt", weights_only=False)
    targets = enumerate_targets(model)

    from p1_grouping.e3_runner import eval_ce

    gmap = parse_grid_map(args.grid_map) if args.grid_map else {}

    for li in range(N_LAYERS):
        gli = gmap.get(li, args.grid)
        lt = [n for n in targets if layer_index(n) == li]
        reps = sorted({cov_source_for(n) for n in lt})
        cov = collect_cov(model, reps, f"{LAYER_PREFIX}.{li}", calib)
        layer_state = {}
        for n in lt:
            m = get_module(model, n)
            w = m.weight.detach().float().cuda()
            M, K = w.shape
            Sp = cov[cov_source_for(n)]
            gm, Bm = gli, B
            for suf, (g_, b_) in mmap.items():          # E68 逐模組覆寫(k/v int8 逐列、o 4bit)
                if n.endswith(suf):
                    gm, Bm = g_, (K if b_ == "row" else b_)
            ktier.set_block(Bm)
            nB = K // Bm
            vals, gt1, gt2 = grid_tensors(gm, args.c, w.device)
            use_search = (args.alpha_init == "search") or (args.alpha_init == "auto" and gm in ktier.INT_LEVELS)
            a_init = alpha_search(w, vals, wdiag=torch.diagonal(Sp)) if use_search else None
            if args.solver == "rtn":
                A = a_init if a_init is not None else ktier.init_alpha(w)
                T1, T2, _ = round_to_grid(w.reshape(M, nB, Bm) / A, vals, gt1, gt2)
                T1, T2 = T1.reshape(M, K), T2.reshape(M, K)
                a_j = A
            else:
                _, T1, T2 = gptq_grid(w, Sp, gm, c=args.c, a0=a_init)
                V = (T1 + args.c * T2).reshape(M, nB, Bm)
                a_j = scale_joint(w.reshape(M, nB, Bm), V, Sp,
                                  prior=(a_init if (a_init is not None and args.prior_lam > 0) else None), lam=args.prior_lam)
            a_bits = 16
            if args.alpha_bits == 8 and nB > 1:                       # E69:α → int8(逐列尺度),存去量化後的值
                s_row = a_j.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
                s_row = s_row.half().float()                           # 2b:列尺度投影到 fp16(帳面精度),再取碼
                a_j = torch.round(a_j / s_row).clamp_(-127, 127) * s_row
                a_bits = 8
            else:
                a_j = a_j.half().float()                               # 2b:α 投影到 fp16(帳面精度),狀態=容器可重建值
            V = (T1 + args.c * T2).reshape(M, nB, Bm)
            what = (a_j * V).reshape(M, K)
            m.weight.data.copy_(what.to(m.weight.dtype))
            layer_state[n] = {
                "T1": T1.reshape(M, nB, Bm).to(torch.int8).cpu(),
                "T2": T2.reshape(M, nB, Bm).to(torch.int8).cpu(),
                "inv": torch.arange(K), "a0": a_j.cpu(), "c": args.c,
                "grid": gm, "block": Bm, "a0_bits": a_bits, "solver": args.solver}
            ktier.set_block(B)
            del w, Sp, what, T1, T2, V
        torch.save(layer_state, out / f"layer{li:02d}.pt")
        del cov, layer_state
        torch.cuda.empty_cache()
        print(f"layer {li:2d} quantized (grid {gli})  ({time.time()-t0:.0f}s)",
              flush=True)

    ce = eval_ce(model, val)
    st = load_state(args.tag, targets)
    n1 = sum(int((c["T1"] != 0).sum()) for c in st.values())
    n2 = sum(int((c["T2"] != 0).sum()) for c in st.values())
    nt = sum(c["T1"].numel() for c in st.values())
    from p2_anchor.wringer.ladder_e65x import ledger
    led = ledger(st)
    print("QUANT_LEDGER " + json.dumps(led), flush=True)
    rec = {"tag": args.tag, "grid": args.grid, "grid_map": args.grid_map,
           "c": args.c, "solver": args.solver, "alpha_init": args.alpha_init, "prior_lam": args.prior_lam,
           "module_map": args.module_map, "ledger": led,
           "block": args.block, "n_calib": args.n_calib, "seed": SEED,
           "zerotrain_val_ce": ce,
           "t1_nonzero": round(n1 / nt, 4), "t2_nonzero": round(n2 / nt, 4),
           "runtime_s": round(time.time() - t0, 1),
           "timestamp": datetime.datetime.now().isoformat()}
    (EVC / f"quant_{args.tag}.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps(rec, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

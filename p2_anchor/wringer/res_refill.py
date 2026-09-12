"""E46 蓄排一體化:排水中段水庫重蓄(refill)。

機制動機(E45 總終裁的時間差稅結論):水庫在 t=0 的全連續幾何上一次性
學成;排水逐層凍結後,補償的拍攝時刻與使用時刻脫節=排水稅 ρ 的來源。
本模組在排水中停點(train_res --stop-after-layer)重建「當前有效模型」:

  凍結前綴 = qs_<qs-tag>/layerXX.pt 已定案硬碼(逐層覆寫)
  未凍後綴 = 冠軍態材化碼 + ResBypass(由現行 res_st 的 B/A 熱初始化)

只訓未凍後綴的旁路(bf16 老師常駐全詞表 KD),把剩餘補償重新對準
當前幾何,寫回 data/res_<res-tag-out>/(凍結層條目原樣保留——僅為
train_res 載入齊全性斷言,不再被使用)。之後 train_res --resume
--res-tag <res-tag-out> 續排。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.res_refill \
      --tag rf46_L03 --state-tag st41g_e2e --qs-tag st46r \
      --res-tag-in res42 --res-tag-out res46r \
      --steps 500 --data evidence/p1_grouping/calib_e42_train.pt

紀律:val CE monitor-only(16 起 proxy 型態在案);.pt 不進 git。
"""
import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.utils.parametrize as parametrize

from p2_anchor.wringer.ktier import k2_weight
from p2_anchor.wringer.modelio import (N_LAYERS, enumerate_targets,
                                         get_module, layer_index, load_model)
from p2_anchor.wringer.polish import kd_loss
from p2_anchor.wringer.res_fill import ResBypass
from p2_anchor.wringer.state import load_state, qs_dir

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"
SEED = 20260806


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="本次 refill 事件名(入檔)")
    ap.add_argument("--state-tag", required=True, help="冠軍態(未凍層碼源)")
    ap.add_argument("--qs-tag", required=True, help="排水 tag(凍結層碼源)")
    ap.add_argument("--res-tag-in", required=True)
    ap.add_argument("--res-tag-out", required=True,
                    help="可與 in 相同=就地更新")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--data", default=str(EV / "calib_e42_train.pt"))
    ap.add_argument("--smoke", action="store_true",
                    help="20 步×16 條,驗接線/凍結覆寫/寫回")
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(SEED)
    model, _ = load_model()
    teacher, _ = load_model()
    model.requires_grad_(False)
    teacher.requires_grad_(False)
    targets = enumerate_targets(model)

    st0 = load_state(args.state_tag, targets)
    for n in targets:                             # 全模型先材化冠軍態
        m = get_module(model, n)
        w = k2_weight(st0[n]["T1"].cuda(), st0[n]["T2"].cuda(),
                      st0[n]["a0"].cuda().float(), st0[n]["c"])
        m.weight.data.copy_(w[:, st0[n]["inv"].cuda()].to(m.weight.dtype))

    QS = qs_dir(args.qs_tag)                      # 凍結前綴逐層覆寫
    frozen = []
    for li in range(N_LAYERS):
        p = QS / f"layer{li:02d}.pt"
        if not p.exists():
            continue
        frozen.append(li)
        for n, c in torch.load(p, weights_only=False).items():
            m = get_module(model, n)
            w = k2_weight(c["T1"].cuda(), c["T2"].cuda(),
                          c["a0"].cuda().float(), c["c"])
            m.weight.data.copy_(w.to(m.weight.dtype))
    assert frozen, f"qs_{args.qs_tag} 無已定案層——refill 應在中停點後執行"
    open_lis = [li for li in range(N_LAYERS) if li not in frozen]
    assert open_lis, "全層已凍結——無旁路可重蓄,直接續排即可"

    res_dir = Path("data") / f"res_{args.res_tag_in}"
    res_st = {}
    for li in range(N_LAYERS):
        p = res_dir / f"layer{li:02d}.pt"
        if p.exists():
            res_st.update(torch.load(p, map_location="cpu",
                                     weights_only=False))
    assert all(n in res_st for n in targets), f"res_{args.res_tag_in} 不齊"

    otgt = [n for n in targets if layer_index(n) in open_lis]
    res = {}
    for n in otgt:                                # 旁路熱初始化(B/A 續值)
        m = get_module(model, n)
        b = ResBypass(*m.weight.shape, r=int(res_st[n]["r"]),
                      device=m.weight.device)
        with torch.no_grad():
            b.A.copy_(res_st[n]["A"].float())
            b.B.copy_(res_st[n]["B"].float())
        parametrize.register_parametrization(m, "weight", b)
        res[n] = b
    opt = torch.optim.Adam(
        [q for b in res.values() for q in (b.A, b.B)], lr=args.lr)

    data = torch.load(args.data, weights_only=False)
    val = torch.load(EV / "calib_val.pt", weights_only=False)
    from p1_grouping.e3_runner import eval_ce
    if args.smoke:
        args.steps = 20
        data = data[:16]

    ce0 = eval_ce(model, val)                     # 中停點有效模型 CE(監控)
    print(f"pre-refill val CE {ce0:.6f}(frozen={len(frozen)}層 "
          f"open={len(open_lis)}層 旁路 {len(otgt)} targets)", flush=True)

    g_rng = torch.Generator().manual_seed(SEED + len(frozen))
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
        kd_val = 0.0
        for i in range(b.shape[0]):
            loss = kd_loss(model, teacher, b[i:i + 1]) / b.shape[0]
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

    ce1 = eval_ce(model, val)
    print(f"post-refill val CE {ce1:.6f}(Δ {ce1 - ce0:+.6f};監控不裁決)",
          flush=True)

    out_dir = Path("data") / f"res_{args.res_tag_out}"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_layer = {}
    for n in targets:                             # 未凍=新值;凍結=原樣搬運
        e = (dict(res_st[n], B=res[n].B.detach().cpu(),
                  A=res[n].A.detach().cpu())
             if n in res else res_st[n])
        by_layer.setdefault(layer_index(n), {})[n] = e
    for li, ls in by_layer.items():
        torch.save(ls, out_dir / f"layer{li:02d}.pt")

    rec = {"tag": args.tag, "state_tag": args.state_tag,
           "qs_tag": args.qs_tag, "res_in": args.res_tag_in,
           "res_out": args.res_tag_out, "frozen_layers": frozen,
           "open_layers": open_lis, "steps": args.steps,
           "batch": args.batch, "lr": args.lr, "data": str(args.data),
           "data_sha256": hashlib.sha256(
               Path(args.data).read_bytes()).hexdigest(),
           "seed": SEED, "pre_refill_val_ce": ce0,
           "post_refill_val_ce": ce1, "kd_log": log,
           "runtime_s": round(time.time() - t0, 1),
           "timestamp": datetime.datetime.now().isoformat()}
    (EVC / f"refill_{args.tag}.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"REFILL_DONE({args.tag},frozen≤L{max(frozen):02d},"
          f"{rec['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main()

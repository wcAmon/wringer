"""E47-P A′ 軌跡鎖定(trajectory-pinning):排水中停點的旁路局部重擬合。

E46 重蓄兩難的第三路:局部函數匹配=no-op(線性旁路與輸入分佈無關)、
端到端 KD=全局再分配不可吸收(−6.29 實證)。A′ 改學「漂移差值」:

  (W_hard + B′A′/r)·x_cur ≈ (W_hard + BA_t0/r)·x_ref

x_ref 來自 REF(冠軍碼+res42 滿水旁路=t0 連續參考幾何),x_cur 來自
CUR(已排前綴 qs 硬碼+未凍後綴冠軍碼+現行旁路)。零漂移時閉式解
退化為 M=BA/r(恆等改寫);有漂移時 M 額外攜帶 W_hard·(x_ref−x_cur)
的抵銷項——逐 target 局部、低秩可表達=可吸收形狀,不觸再分配地雷。

實作:雙模型常駐同批教師強制前傳,hook 流式累積正規方程
(AA 按 cov_source_for 輸入點共用),ridge 閉式解 → svd_lowrank 截秩
r 寫回 res 目錄;train_res --resume 續排時窗目標 Y(off+γ=1)即吃到
pin 後旁路=機制閉合。R²/截秩能量入檔 monitor-only(17 筆紀律)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.res_pin \
      --tag pin47p_L03 --state-tag st41g_e2e --qs-tag st47p \
      --res-tag-in res42 --res-tag-out res47p
"""
import argparse
import datetime
import json
import time
from pathlib import Path

import torch
import torch.nn.utils.parametrize as parametrize

from p2_anchor.wringer.ktier import k2_weight
from p2_anchor.wringer.modelio import (N_LAYERS, cov_source_for,
                                         enumerate_targets, get_module,
                                         layer_index, load_model)
from p2_anchor.wringer.res_fill import ResBypass
from p2_anchor.wringer.state import load_state, qs_dir

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"
SEED = 20260806


def materialize(model, targets, state_tag):
    st0 = load_state(state_tag, targets)
    for n in targets:
        m = get_module(model, n)
        w = k2_weight(st0[n]["T1"].cuda(), st0[n]["T2"].cuda(),
                      st0[n]["a0"].cuda().float(), st0[n]["c"])
        m.weight.data.copy_(w[:, st0[n]["inv"].cuda()].to(m.weight.dtype))


def load_res(res_tag, targets):
    res_dir = Path("data") / f"res_{res_tag}"
    res_st = {}
    for li in range(N_LAYERS):
        p = res_dir / f"layer{li:02d}.pt"
        if p.exists():
            res_st.update(torch.load(p, map_location="cpu",
                                     weights_only=False))
    assert all(n in res_st for n in targets), f"res_{res_tag} 不齊"
    return res_st


def attach(model, names, res_st):
    out = {}
    for n in names:
        m = get_module(model, n)
        b = ResBypass(*m.weight.shape, r=int(res_st[n]["r"]),
                      device=m.weight.device)
        with torch.no_grad():
            b.A.copy_(res_st[n]["A"].float())
            b.B.copy_(res_st[n]["B"].float())
        parametrize.register_parametrization(m, "weight", b)
        out[n] = b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--state-tag", required=True, help="冠軍態(碼源)")
    ap.add_argument("--qs-tag", required=True, help="排水 tag(凍結前綴碼源)")
    ap.add_argument("--res-tag-in", required=True)
    ap.add_argument("--res-tag-out", required=True)
    ap.add_argument("--res-tag-ref", default="res42",
                    help="t0 參考旁路(恆為滿水 t0 幾何,不隨 pin 更新)")
    ap.add_argument("--calib", default=str(EV / "calib_drain_e42b.pt"))
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--frozen-upto", type=int, default=None,
                    help="僅將 ≤此層號 視為凍結(煙測用;正式跑省略)")
    ap.add_argument("--smoke", action="store_true",
                    help="4 列+寫回等價驗證")
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(SEED)
    if args.smoke:
        args.rows = 4

    data = torch.load(args.calib, weights_only=False)[:args.rows]

    # ---- CUR:已排前綴硬碼 + 未凍後綴冠軍碼 + 現行旁路 ----
    m_cur, _ = load_model()
    m_cur.requires_grad_(False)
    targets = enumerate_targets(m_cur)
    materialize(m_cur, targets, args.state_tag)
    QS = qs_dir(args.qs_tag)
    frozen = []
    for li in range(N_LAYERS):
        p = QS / f"layer{li:02d}.pt"
        if not p.exists():
            continue
        if args.frozen_upto is not None and li > args.frozen_upto:
            continue
        frozen.append(li)
        for n, c in torch.load(p, weights_only=False).items():
            m = get_module(m_cur, n)
            w = k2_weight(c["T1"].cuda(), c["T2"].cuda(),
                          c["a0"].cuda().float(), c["c"])
            m.weight.data.copy_(w.to(m.weight.dtype))
    assert frozen, f"qs_{args.qs_tag} 無已定案層——pin 應在中停點後執行"
    open_lis = [li for li in range(N_LAYERS) if li not in frozen]
    assert open_lis, "全層已凍結——無旁路可 pin"
    res_in = load_res(args.res_tag_in, targets)
    otgt = [n for n in targets if layer_index(n) in open_lis]
    cur_bp = attach(m_cur, otgt, res_in)

    # ---- REF:冠軍碼 + t0 滿水旁路(全 targets)----
    m_ref, _ = load_model()
    m_ref.requires_grad_(False)
    materialize(m_ref, targets, args.state_tag)
    res_ref = load_res(args.res_tag_ref, targets)
    attach(m_ref, targets, res_ref)
    print(f"雙模型就緒 frozen={len(frozen)}層 open={len(open_lis)}層 "
          f"pin {len(otgt)} targets ({time.time()-t0:.0f}s)", flush=True)

    # ---- hook 捕捉:CUR 存 (x, out),REF 存 out ----
    cap_c, cap_r, hooks = {}, {}, []
    for n in otgt:
        def hc(mod, inp, out, n=n):
            cap_c[n] = (inp[0].detach()[0], out.detach()[0])
        def hr(mod, inp, out, n=n):
            cap_r[n] = out.detach()[0]
        hooks.append(get_module(m_cur, n).register_forward_hook(hc))
        hooks.append(get_module(m_ref, n).register_forward_hook(hr))

    groups = sorted({cov_source_for(n) for n in otgt})
    AA = {g: None for g in groups}
    CC, S0 = {}, {}
    NT = 0
    with torch.no_grad():
        for i in range(data.shape[0]):
            ids = data[i:i + 1].cuda()
            m_cur(ids)
            m_ref(ids)
            done_g = set()
            for n in otgt:
                x, oc = cap_c[n]
                yr = cap_r[n]
                x32 = x.float()
                g = cov_source_for(n)
                if g not in done_g:
                    done_g.add(g)
                    a = x32.T @ x32
                    AA[g] = a if AA[g] is None else AA[g] + a
                b = cur_bp[n]
                bp = ((x32 @ b.A.T) @ b.B.T) / b.r
                d = yr.float() - oc.float() + bp
                c = x32.T @ d
                CC[n] = CC[n] + c if n in CC else c
                S0[n] = S0.get(n, 0.0) + float((d.double() ** 2).sum())
            cap_c.clear()
            cap_r.clear()
            NT += ids.shape[1]
            if (i + 1) % 8 == 0:
                print(f"fwd {i+1}/{data.shape[0]}", flush=True)
    for h in hooks:
        h.remove()

    # ---- 逐 target 閉式解 + 截秩寫回 ----
    out_res = dict(res_in)
    stats = {}
    for n in otgt:
        g = cov_source_for(n)
        din = AA[g].shape[0]
        M = torch.linalg.solve(
            AA[g] + args.ridge * NT * torch.eye(din, device="cuda"), CC[n])
        r2 = float((M * CC[n]).sum().double()) / max(S0[n], 1e-30)
        Wt = M.T.contiguous()                      # 目標旁路 = B′A′/r
        r = int(res_in[n]["r"])
        q = min(r + 16, min(Wt.shape))
        U, S, V = torch.svd_lowrank(Wt, q=q)
        k = min(r, S.shape[0])
        energy = float((S[:k] ** 2).sum() / (Wt ** 2).sum().clamp_min(1e-30))
        B = torch.zeros(Wt.shape[0], r, device="cuda")
        A = torch.zeros(r, Wt.shape[1], device="cuda")
        B[:, :k] = U[:, :k] * S[:k] * r             # forward 除 r,這裡乘回
        A[:k] = V[:, :k].T
        out_res[n] = dict(res_in[n], B=B.cpu(), A=A.cpu())
        stats[n] = {"r2": round(r2, 4), "rank_energy": round(energy, 4)}
        if args.smoke and n == otgt[0]:
            rel = float(((B @ A) / r - Wt).norm() / Wt.norm().clamp_min(1e-30))
            print(f"SMOKE 重建檢核 {n}: k={k} rel={rel:.3e} "
                  f"r2={r2:.3f} energy={energy:.3f}", flush=True)

    out_dir = Path("data") / f"res_{args.res_tag_out}"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_layer = {}
    for n in targets:
        by_layer.setdefault(layer_index(n), {})[n] = out_res[n]
    for li, ls in by_layer.items():
        torch.save(ls, out_dir / f"layer{li:02d}.pt")

    med = sorted(s["r2"] for s in stats.values())
    med = med[len(med) // 2]
    med_e = sorted(s["rank_energy"] for s in stats.values())
    med_e = med_e[len(med_e) // 2]
    rec = {"tag": args.tag, "state_tag": args.state_tag,
           "qs_tag": args.qs_tag, "res_in": args.res_tag_in,
           "res_out": args.res_tag_out, "res_ref": args.res_tag_ref,
           "frozen_layers": frozen, "rows": int(data.shape[0]),
           "tokens": NT, "ridge": args.ridge,
           "r2_median": med, "rank_energy_median": med_e,
           "per_target": stats,
           "note": "R²/energy monitor-only;零漂移退化=恆等改寫",
           "runtime_s": round(time.time() - t0, 1),
           "timestamp": datetime.datetime.now().isoformat()}
    (EVC / f"pin_{args.tag}.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"PIN_DONE({args.tag},frozen≤L{max(frozen):02d},"
          f"r2_med={med:.3f},energy_med={med_e:.3f},"
          f"{rec['runtime_s']}s)", flush=True)
    if args.smoke:
        print("PIN_SMOKE_PASS", flush=True)


if __name__ == "__main__":
    main()

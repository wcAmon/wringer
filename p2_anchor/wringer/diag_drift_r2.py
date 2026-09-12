"""E47 A′ 前置閘:排水輸入漂移的逐層線性可預測度(forward-only 診斷)。

REF = 冠軍態材化碼 + res42 滿水旁路(γ=1,t0 參考幾何)
CUR = qs_st46r 全定案硬碼(E46 遺留全排水終態=最壞情況漂移)

雙模型常駐(2×4B bf16 ≈16G,96G 內裕),同批教師強制前傳流式累積,
對每層 l 擬合全秩線性映射(ridge 正規方程):
  d_l = h_ref[l] − h_cur[l] ≈ Mᵀ · h_cur[l−1]
R²_l = 1 − ||d − Mᵀx||² / ||d||²(未中心化)。全秩 R² 是低秩 A′ 的上界:
上界都低 → A′ 撤案;上界高 → 建案有據(rank 上限另在建案時裁)。

設計閘非官方代理,不觸 proxy 紀律。輸出 evidence/.../diag_drift_r2.json。
"""
import argparse
import datetime
import json
import time
from pathlib import Path

import torch
import torch.nn.utils.parametrize as parametrize

from p2_anchor.wringer.ktier import k2_weight
from p2_anchor.wringer.modelio import (N_LAYERS, enumerate_targets,
                                         get_module, layer_index, load_model)
from p2_anchor.wringer.res_fill import ResBypass
from p2_anchor.wringer.state import load_state, qs_dir

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"


def materialize(model, targets, state_tag, qs_tag=None):
    """冠軍態材化;qs_tag 給定時逐層覆寫為排水硬碼。"""
    st0 = load_state(state_tag, targets)
    for n in targets:
        m = get_module(model, n)
        w = k2_weight(st0[n]["T1"].cuda(), st0[n]["T2"].cuda(),
                      st0[n]["a0"].cuda().float(), st0[n]["c"])
        m.weight.data.copy_(w[:, st0[n]["inv"].cuda()].to(m.weight.dtype))
    if qs_tag:
        QS = qs_dir(qs_tag)
        done = 0
        for li in range(N_LAYERS):
            p = QS / f"layer{li:02d}.pt"
            if not p.exists():
                continue
            done += 1
            for n, c in torch.load(p, weights_only=False).items():
                m = get_module(model, n)
                w = k2_weight(c["T1"].cuda(), c["T2"].cuda(),
                              c["a0"].cuda().float(), c["c"])
                m.weight.data.copy_(w.to(m.weight.dtype))
        assert done == N_LAYERS, f"qs_{qs_tag} 僅 {done}/32 層定案"


def attach_res(model, targets, res_tag):
    res_dir = Path("data") / f"res_{res_tag}"
    res_st = {}
    for li in range(N_LAYERS):
        p = res_dir / f"layer{li:02d}.pt"
        if p.exists():
            res_st.update(torch.load(p, map_location="cpu",
                                     weights_only=False))
    assert all(n in res_st for n in targets), f"res_{res_tag} 不齊"
    for n in targets:
        m = get_module(model, n)
        b = ResBypass(*m.weight.shape, r=int(res_st[n]["r"]),
                      device=m.weight.device)
        with torch.no_grad():
            b.A.copy_(res_st[n]["A"].float())
            b.B.copy_(res_st[n]["B"].float())
        parametrize.register_parametrization(m, "weight", b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-tag", default="st41g_e2e")
    ap.add_argument("--qs-tag", default="st46r")
    ap.add_argument("--res-tag", default="res42")
    ap.add_argument("--calib", default=str(EV / "calib_drain_e42b.pt"))
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--ridge", type=float, default=1e-3)
    args = ap.parse_args()
    t0 = time.time()

    data = torch.load(args.calib, weights_only=False)[:args.rows]

    m_cur, _ = load_model()
    m_cur.requires_grad_(False)
    tg = enumerate_targets(m_cur)
    materialize(m_cur, tg, args.state_tag, qs_tag=args.qs_tag)

    m_ref, _ = load_model()
    m_ref.requires_grad_(False)
    materialize(m_ref, enumerate_targets(m_ref), args.state_tag)
    attach_res(m_ref, enumerate_targets(m_ref), args.res_tag)
    print(f"雙模型常駐就緒 ({time.time()-t0:.0f}s)", flush=True)

    D = 2560
    AA = torch.zeros(N_LAYERS, D, D, device="cuda")
    CC = torch.zeros(N_LAYERS, D, D, device="cuda")
    S0 = torch.zeros(N_LAYERS, dtype=torch.float64, device="cuda")
    NT = 0
    with torch.no_grad():
        for i in range(data.shape[0]):
            ids = data[i:i + 1].cuda()
            hc = m_cur(ids, output_hidden_states=True).hidden_states
            hr = m_ref(ids, output_hidden_states=True).hidden_states
            for l in range(N_LAYERS):
                x = hc[l][0].float()              # 層 l+1 輸入(CUR)
                d = (hr[l + 1][0] - hc[l + 1][0]).float()  # 層輸出漂移
                AA[l] += x.T @ x
                CC[l] += x.T @ d
                S0[l] += (d.double() ** 2).sum()
            NT += hc[0].shape[1]
            del hc, hr
            if (i + 1) % 8 == 0:
                print(f"fwd {i+1}/{data.shape[0]}", flush=True)

    r2 = []
    for l in range(N_LAYERS):
        A = AA[l] + args.ridge * NT * torch.eye(D, device="cuda")
        M = torch.linalg.solve(A, CC[l])
        expl = float((M * CC[l]).sum().double())  # trace(MᵀC)
        r2.append(round(expl / max(float(S0[l]), 1e-30), 4))
    deep = sorted(r2[12:])
    med_deep = deep[len(deep) // 2]
    gate = ("BUILD" if med_deep >= 0.4 else
            "KILL" if med_deep < 0.2 else "GRAY")

    rec = {"rows": int(data.shape[0]), "tokens": NT, "ridge": args.ridge,
           "r2_per_layer": r2, "r2_median_L12plus": med_deep,
           "gate": gate, "qs_tag": args.qs_tag, "res_tag": args.res_tag,
           "note": "CUR=全排水終態=最壞情況漂移;交錯執行時逐段更小。"
                   "全秩 R²=低秩上界。設計閘非官方代理。",
           "runtime_s": round(time.time() - t0, 1),
           "timestamp": datetime.datetime.now().isoformat()}
    (EVC / "diag_drift_r2.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print("R2_PROFILE:", " ".join(f"L{l:02d}={v:.2f}"
                                  for l, v in enumerate(r2)), flush=True)
    print(f"R2_GATE:{gate} med_L12+={med_deep:.3f} ({rec['runtime_s']}s)",
          flush=True)


if __name__ == "__main__":
    main()

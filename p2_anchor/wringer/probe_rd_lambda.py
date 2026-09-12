"""E66-RD λ 校準(GPU、無 Σ 回饋):抽樣 g9 模組,u = W9/α 精確落格;對 λ 網格做率罰最近取整,bits 迭代到定點,
量 H(熵)/flips;輸出每 λ 的定點 bits 供 ladder --rd-bits。約 1–2 min。"""
import json, sys, torch
from pathlib import Path
from p2_anchor.wringer import ktier
EVC = Path("evidence/p1_grouping/corkscrew")
LAYERS = [0, 5, 10, 15, 20, 25, 31]
lams = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
vals, t1, t2 = ktier.grid_tensors(9, 0.6, "cuda")
G = 9
us, syms = [], []
for L in LAYERS:
    st = torch.load(f"data/qs_g9/layer{L:02d}.pt", map_location="cpu")
    for n, c in st.items():
        T1, T2 = c["T1"].cuda().float(), c["T2"].cuda().float()
        u = (T1 + 0.6 * T2)                                   # α 固定 ⇒ u 就是格值本身
        idx = torch.bucketize(u.reshape(-1).contiguous(), (vals[1:] + vals[:-1]) / 2)
        sub = torch.randperm(idx.numel(), device="cuda")[: 2_000_000]
        syms.append(idx[sub]); us.append(u.reshape(-1)[sub])
u = torch.cat(us); s0 = torch.cat(syms)
p0 = torch.bincount(s0, minlength=G).double(); p0 /= p0.sum()
H = lambda p: float(-(p[p > 0] * p[p > 0].log2()).sum())
print(f"sample n={u.numel()} H0={H(p0):.4f} hist={[round(float(x),3) for x in p0]}", flush=True)
out = {"layers": LAYERS, "n": int(u.numel()), "H0": H(p0), "grid_vals": [float(v) for v in vals], "points": []}
for lam in lams:
    bits = (-p0.log2()).float().cuda()
    for it in range(8):                                        # 定點迭代:bits ← −log2 p(碼)
        _, _, v = ktier.round_rd(u, vals, t1, t2, bits, lam)
        idx = torch.bucketize(v.contiguous(), (vals[1:] + vals[:-1]) / 2)
        p = torch.bincount(idx, minlength=G).double().clamp_min(1e-9); p /= p.sum()
        nb = (-p.log2()).float().cuda()
        if (nb - bits).abs().max() < 1e-4:
            bits = nb; break
        bits = nb
    flips = float((idx != s0).float().mean())
    rec = {"lam": lam, "H": H(p), "flips": flips, "iters": it + 1, "bits": [round(float(b), 5) for b in bits],
           "hist": [round(float(x), 4) for x in p], "mse_u": float(((v - u) ** 2).mean())}
    out["points"].append(rec)
    print(f"lam={lam:<6} H={rec['H']:.4f} flips={flips:.4f} mse_u={rec['mse_u']:.5f} center={rec['hist'][4]:.3f} it={it+1}", flush=True)
json.dump(out, open(EVC / "rd_lambda_calib.json", "w"), indent=1)
print("RD_CALIB_DONE")

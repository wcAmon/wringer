"""E57 收卷後判別:st54r(8k 教)vs st57r(16k 教)逐位置 KL(用戶 08-29 核准)。

假說(用戶):短窗教學下碼分配傾向「眼前的學習」,FLA 遞歸狀態的長程
轉換失準 → 8k 教的學生在 ~8k 位置後 KL(teacher||student) 上翹;16k 教
的學生持平 → 長窗適配軸獨立成立。若全位置均勻 → 內容軸(閉合示範)
已足以解釋 A₃ 增量,窗長本身非機制。

量測:e57f 語料第 128–143 列(drain 只用前 128 列,本段兩生皆未在
drain 見過;st54r 全程未見 e57f)。teacher bf16 + 兩 export 同機常駐,
hidden 各前向一次,lm_head+KL 逐 512-chunk 計,chunk=bin 對齊。

產出:evidence/p1_grouping/E57_KL_POS.json(逐 bin 曲線+前後半聚合)
+ 印 E57_KL_DONE:{...} 哨兵。零訓練、無 GPU 競爭前提(鏈已收卷)。
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from p1_grouping.modelio import load_model
from p2_anchor.wringer.polish import _backbone_hidden

EV = Path("evidence/p1_grouping")


def load_export(path):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    try:
        m = AutoModelForImageTextToText.from_pretrained(
            path, torch_dtype=torch.bfloat16)
    except ValueError:
        m = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16)
    return m.to("cuda").eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=str(EV / "calib_e57f_traj.pt"))
    ap.add_argument("--lens", default=str(EV / "calib_e57f_len.pt"))
    ap.add_argument("--row0", type=int, default=128,
                    help="drain 用前 128 列;由此起取判別列")
    ap.add_argument("--n-rows", type=int, default=16)
    ap.add_argument("--bin", type=int, default=512)
    args = ap.parse_args()

    students = {"st54r": "exports/e54_r", "st57r": "exports/e57_r"}
    teacher, _ = load_model()
    mdls = {k: load_export(v) for k, v in students.items()}

    t = torch.load(args.traj)[args.row0:args.row0 + args.n_rows]
    L = torch.load(args.lens)[args.row0:args.row0 + args.n_rows]
    T = t.shape[1]
    nbins = T // args.bin
    sums = {k: torch.zeros(nbins, dtype=torch.float64) for k in mdls}
    cnts = torch.zeros(nbins, dtype=torch.float64)

    for i in range(len(t)):
        b = t[i:i + 1].cuda()
        li = int(L[i])
        th, tlm = _backbone_hidden(teacher, b)
        hs = {k: _backbone_hidden(m, b)[0] for k, m in mdls.items()}
        slms = {k: m.get_output_embeddings() for k, m in mdls.items()}
        for c0 in range(0, min(li, T), args.bin):
            c1 = min(c0 + args.bin, li)
            lt = F.log_softmax(tlm(th[:, c0:c1]).float(), dim=-1)
            pt = lt.exp()
            bi = c0 // args.bin
            for k in mdls:
                ls = F.log_softmax(slms[k](hs[k][:, c0:c1]).float(), dim=-1)
                kl = (pt * (lt - ls)).sum(-1)          # (1, c1-c0)
                sums[k][bi] += float(kl.sum())
            cnts[bi] += c1 - c0
        print(f"row {i + 1}/{len(t)}  len={li}", flush=True)

    curve = {k: [round(float(s / c), 5) if c > 0 else None
                 for s, c in zip(sums[k], cnts)] for k in sums}
    half = T // 2
    agg = {}
    for k in sums:
        lo = float(sums[k][:half // args.bin].sum()
                   / max(float(cnts[:half // args.bin].sum()), 1))
        hi_c = float(cnts[half // args.bin:].sum())
        hi = float(sums[k][half // args.bin:].sum() / max(hi_c, 1))
        agg[k] = {"kl_0_8k": round(lo, 5),
                  "kl_8k_16k": round(hi, 5) if hi_c > 0 else None,
                  "ratio_hi_lo": round(hi / lo, 3) if hi_c > 0 else None}
    out = {"rows": f"{args.row0}..{args.row0 + len(t) - 1}",
           "n_rows": len(t), "bin": args.bin,
           "pos_counts": [int(c) for c in cnts],
           "curve_mean_kl_fwd": curve, "aggregate": agg,
           "discriminant": "st54r ratio_hi_lo ≫ st57r ⇒ 長窗適配軸成立;"
                           "兩生同帶均勻 ⇒ 內容軸已足以解釋"}
    (EV / "E57_KL_POS.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("E57_KL_DONE:" + json.dumps(agg))


if __name__ == "__main__":
    main()

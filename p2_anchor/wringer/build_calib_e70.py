"""E70 語料分層抽樣:gen_e58_examroom 產物(calib_<src>_traj/_len + E58_TRAJ_MANIFEST_<src>.json)
→ 反推每列所屬域(產線依 --domains 順序追加列後 randperm(SEED)),分層抽 128 窗(code 66 / ifshape 33 / math 29,
同 calib_e58r1b 甲臂)存 calib_<out>_traj.pt/_len.pt;另抽 16 窗切 32×2048 當 val(calib_<out>_val.pt,同 calib_val.pt 形狀)。
    python -m p2_anchor.wringer.build_calib_e70 --src e70r0 --out e70
    python -m p2_anchor.wringer.build_calib_e70 --src e70r0=code,math --src e70r1=ifshape --out e70   # 多來源分域(E70:Qwen3-4B ifshape 短,180 題只打包 10 列,另補產 1080 題)
每來源的列域由其 manifest per_domain 順序(=產線 --domains 順序)+ randperm(SEED) 反推;來源後接 =域清單 表只取該些域的列。
印 CALIB_E70_DONE <json>。
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

EV = Path("evidence/p1_grouping")
SEED = 20260829          # gen_e58_examroom.SEED(randperm 用)
PICK_SEED = 20260912
QUOTA = {"code": 66, "ifshape": 33, "math": 29}
VAL_QUOTA = {"code": 8, "ifshape": 4, "math": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, action="append", help="來源 tag,可多個;tag=dom1,dom2 只取該些域")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ts, Ls, dom, closure, src_rows = [], [], [], {}, {}
    for spec in args.src:
        tag, _, keep = spec.partition("=")
        keep = keep.split(",") if keep else None
        t1 = torch.load(EV / f"calib_{tag}_traj.pt", weights_only=False)
        L1 = torch.load(EV / f"calib_{tag}_len.pt", weights_only=False)
        man = json.load(open(EV / f"E58_TRAJ_MANIFEST_{tag}.json"))
        order = list(man["per_domain"].keys())                                   # 產線 --domains 順序
        counts = [man["per_domain"][d]["packed_rows"] for d in order]
        n1 = sum(counts)
        assert n1 == t1.shape[0], (tag, n1, t1.shape)
        perm = torch.randperm(n1, generator=torch.Generator().manual_seed(SEED))  # 產線:t = t_orig[perm]
        dom_orig = []
        for d, c in zip(order, counts):
            dom_orig += [d] * c
        dom1 = [dom_orig[int(i)] for i in perm]                                   # 現行列 i 的域
        sel = [i for i, x in enumerate(dom1) if keep is None or x in keep]
        ts.append(t1[sel]); Ls.append(L1[sel]); dom += [dom1[i] for i in sel]
        src_rows[spec] = len(sel)
        for d in (keep or order):
            closure[d] = man["per_domain"][d].get("closure_rate")
    t, L = torch.cat(ts), torch.cat(Ls)
    n = t.shape[0]
    names = list(QUOTA)
    assert set(dom) == set(names), (set(dom), names)
    rng = torch.Generator().manual_seed(PICK_SEED)
    by = {d: [i for i, x in enumerate(dom) if x == d] for d in names}
    pick, val = [], []
    for d in names:
        idx = by[d]
        order = [idx[j] for j in torch.randperm(len(idx), generator=rng).tolist()]
        need = QUOTA[d] + VAL_QUOTA[d]
        assert len(order) >= need, (d, len(order), need)
        pick += order[:QUOTA[d]]
        val += order[QUOTA[d]:need]
    pick = [pick[j] for j in torch.randperm(len(pick), generator=rng).tolist()]  # 分層後洗牌
    tr, lr = t[pick].clone(), L[pick].clone()
    torch.save(tr, EV / f"calib_{args.out}_traj.pt")
    torch.save(lr, EV / f"calib_{args.out}_len.pt")
    chunks = []
    for i in val:
        row = t[i, : int(L[i])]
        for s in range(0, len(row) - 2048 + 1, 2048):
            chunks.append(row[s:s + 2048])
            if len(chunks) >= 32:
                break
        if len(chunks) >= 32:
            break
    assert len(chunks) == 32, len(chunks)
    v = torch.stack(chunks)
    torch.save(v, EV / f"calib_{args.out}_val.pt")
    rep = {"src": args.src, "out": args.out, "src_rows": src_rows, "domain_rows": {d: dom.count(d) for d in names},
           "pick": {d: sum(dom[i] == d for i in pick) for d in names}, "val_windows": len(val), "val_shape": list(v.shape),
           "len_quartiles": [int(x) for x in torch.quantile(lr.float(), torch.tensor([0.25, 0.5, 0.75])).tolist()],
           "sha256_8": hashlib.sha256(tr.numpy().tobytes()).hexdigest()[:8], "pick_indices": pick, "val_indices": val,
           "closure": closure}
    (EV / f"E70_CALIB_MANIFEST_{args.out}.json").write_text(json.dumps(rep, indent=1))
    print("CALIB_E70_DONE", json.dumps({k: rep[k] for k in ("src_rows", "domain_rows", "pick", "val_shape", "len_quartiles", "closure")}), flush=True)


if __name__ == "__main__":
    main()

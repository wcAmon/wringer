"""E52 合裝:hec65 底盤(不動)+ 庫存補集閉合窗 + code 閉合窗(prereg_e52)。

hec65 子採樣的補集以同 seed 重放取得(gen_hecode_mix SEED=20260824),
其中僅取含 </think> 的列;code part 補集為空(hec65 全取)。附加不
替換:冠軍配方普通 KD 質量保留,轉折佔比目標 ≤30%。
"""
import hashlib
import json
from pathlib import Path

import torch

EV = Path("evidence/p1_grouping")
SEED_HEC = 20260824              # gen_hecode_mix 的採樣 seed(重放補集用)
CAL4_TAKE = 609
TAKE = {"agentic": 1831, "ifshape": 2202, "code": 12600, "math": 2143}
SEED = 20260826
THINK_END = 248069


def main():
    hec65 = torch.load(EV / "calib_e42_train_hec65.pt")
    parts, report = [hec65], {"hec65_base": int(hec65.shape[0])}

    for name, n in TAKE.items():
        t = torch.load(EV / f"calib_v6_part_{name}.pt")
        g = torch.Generator().manual_seed(SEED_HEC + len(name))
        idx = torch.randperm(t.shape[0], generator=g)
        comp = t[idx[n:]]
        keep = comp[(comp == THINK_END).any(dim=1)]
        parts.append(keep)
        report[f"comp_{name}"] = int(keep.shape[0])
    cal4 = torch.load(EV / "calib_selfgen_v4.pt")
    g = torch.Generator().manual_seed(SEED_HEC + 99)
    idx = torch.randperm(cal4.shape[0], generator=g)
    keep = cal4[idx[CAL4_TAKE:]][(cal4[idx[CAL4_TAKE:]] == THINK_END)
                                 .any(dim=1)]
    parts.append(keep)
    report["comp_cal4"] = int(keep.shape[0])

    cc = torch.load(EV / "calib_e52_codeclose.pt")
    parts.append(cc)
    report["codeclose"] = int(cc.shape[0])

    full = torch.cat(parts, dim=0)
    perm = torch.randperm(full.shape[0],
                          generator=torch.Generator().manual_seed(SEED))
    full = full[perm]
    out = EV / "calib_e52_fill.pt"
    torch.save(full, out)

    trans = int((full == THINK_END).any(dim=1).sum())
    rec = {"version": "e52_fill(hec65+補集閉合窗+codeclose;附加不替換)",
           "rows": report, "total": int(full.shape[0]),
           "transition_rows": trans,
           "transition_share": round(trans / full.shape[0], 4),
           "expected_exposure_at_9600": round(9600 / full.shape[0], 3),
           "seed": SEED, "shape": list(full.shape),
           "codeclose_sha": json.loads(
               (EV / "E52_CODECLOSE_MANIFEST.json").read_text())["sha256"],
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    (EV / "E52_FILL_MANIFEST.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({"total": rec["total"],
                      "transition_share": rec["transition_share"],
                      "sha256": rec["sha256"][:16]}, indent=1))
    print("E52_FILL_DONE", flush=True)


if __name__ == "__main__":
    main()

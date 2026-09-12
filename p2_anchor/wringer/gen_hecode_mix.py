"""E48b 臂A:he 加重混裝——從 cal6 既有 part 檔零生成重合裝(prereg_e48b)。

code 12,600 全保 + agentic 1831/ifshape 2202/math 2143/cal4 609 子採樣
= 19,385 列(code 65.0%);列級 shuffle;除汙繼承 part 級 8-gram 切除。
"""
import hashlib
import json
from pathlib import Path

import torch

EV = Path("evidence/p1_grouping")
SEED = 20260824
TAKE = {"agentic": 1831, "ifshape": 2202, "code": 12600, "math": 2143}
CAL4_TAKE = 609


def main():
    parts, report = [], {}
    for name, n in TAKE.items():
        t = torch.load(EV / f"calib_v6_part_{name}.pt")
        g = torch.Generator().manual_seed(SEED + len(name))
        idx = torch.randperm(t.shape[0], generator=g)[:n]
        parts.append(t[idx])
        report[name] = n
    cal4 = torch.load(EV / "calib_selfgen_v4.pt")
    g = torch.Generator().manual_seed(SEED + 99)
    parts.append(cal4[torch.randperm(cal4.shape[0], generator=g)[:CAL4_TAKE]])
    report["cal4"] = CAL4_TAKE

    full = torch.cat(parts, dim=0)
    perm = torch.randperm(full.shape[0],
                          generator=torch.Generator().manual_seed(SEED))
    full = full[perm]
    out = EV / "calib_e42_train_hec65.pt"
    torch.save(full, out)
    rec = {"version": "hec65(prereg_e48b:code 65.0%,cal6 part 重合裝零生成)",
           "rows": report, "total": int(full.shape[0]),
           "code_share": round(TAKE["code"] / full.shape[0], 4),
           "seed": SEED, "shape": list(full.shape),
           "source_parts_sha": {
               n: json.loads((EV / f"calib_v6_part_{n}.json")
                             .read_text())["sha256"] for n in TAKE},
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    (EV / "HEC65_MANIFEST.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({"shape": rec["shape"], "code_share": rec["code_share"],
                      "sha256": rec["sha256"][:16]}, indent=1))


if __name__ == "__main__":
    main()

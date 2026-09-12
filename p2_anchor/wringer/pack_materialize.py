"""Wringer 容器 → 材化 bf16 export(發布用評測態)。

讀 data/wringer_<tag>.safetensors(pack_verify 產物),逐模組以帳面精度解包(碼 + fp16 α 或 int8 q·fp16 s),
材化 bf16 權重,其餘張量(embed/norm/lm_head/視覺塔…)自來源 export 原樣複製;config/tokenizer 檔亦複製。
輸出 exports/<out>/model.safetensors。與來源 export 的差異即 pack_verify 的 R3(α 精度 1 ulp)。
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from p2_anchor.wringer.pack_verify import materialize, unpack_module

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--src-export", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    src = ROOT / args.src_export
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.name != "model.safetensors":
            shutil.copy2(f, out / f.name)
    cont = safe_open(str(ROOT / args.container), "np")
    metas = json.loads(cont.metadata()["wringer_meta"])
    tensors = {}
    n_mat = n_copy = 0
    max_abs = 0.0
    with safe_open(str(src / "model.safetensors"), "pt") as exp:
        for k in exp.keys():
            name = k[:-len(".weight")] if k.endswith(".weight") else None
            if name in metas:
                meta = metas[name]
                t = {f"{name}.code": cont.get_tensor(f"{name}.code")}
                if meta["a0_bits"] == 16:
                    t[f"{name}.alpha"] = cont.get_tensor(f"{name}.alpha")
                else:
                    t[f"{name}.alpha_q"] = cont.get_tensor(f"{name}.alpha_q")
                    t[f"{name}.alpha_s"] = cont.get_tensor(f"{name}.alpha_s")
                if f"{name}.inv" in cont.keys():
                    t[f"{name}.inv"] = cont.get_tensor(f"{name}.inv")
                T1, a, inv = unpack_module(name, t, meta)
                w = materialize(T1, a, inv)
                ref = exp.get_tensor(k)
                assert w.shape == ref.shape and ref.dtype == torch.bfloat16, name
                max_abs = max(max_abs, float((w.float() - ref.float()).abs().max()))
                tensors[k] = w.contiguous()
                n_mat += 1
            else:
                tensors[k] = exp.get_tensor(k).contiguous()
                n_copy += 1
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    rec = {"container": args.container, "src_export": args.src_export, "out": args.out,
           "materialized_modules": n_mat, "copied_tensors": n_copy, "max_abs_diff_vs_src": max_abs,
           "runtime_s": round(time.time() - t0, 1)}
    (out / "wringer_materialize.json").write_text(json.dumps(rec, indent=1))
    print("MATERIALIZE_DONE " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()

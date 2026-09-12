"""E60 S0:旋轉載體態材化匯出(凍碼 + 學到的 Q → dense bf16,vLLM 可直接 serve)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export_rot \
      --codes g3rotL1 --Q evidence/p1_grouping/corkscrew/rot_Q_L2.pt \
      --out exports/g3rotL1_QL2_carrier

與 rot_learn v2 的 RotQ 語義逐位對齊(殘差流留原基底,旋轉施於激活側):
  讀者 y = W_q·(Qᵀx̂)  ⇔  dense W = W_q·Qᵀ
  寫者 r += Q·(W_q z)  ⇔  dense W = Q·W_q
非目標讀者折 (1+γ);norm γ←0;lm_head 解綁 E·diag(1+γ_f)。
等價閘:材化模型 val CE 對 rot_Q_<tag>.pt 內 val_ce_frozen |Δ|<gate,FAIL 即停。
部署帳:三元碼 + 一個 2560×2560 bf16 Q(6.5MB,線上乘;dense 材化只為 vLLM 評測)。
"""
import argparse
import datetime
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from p2_anchor.wringer.export import SNAPSHOT
from p2_anchor.wringer.ktier import k2_weight
from p2_anchor.wringer.modelio import (LAYER_PREFIX, N_LAYERS,
                                         enumerate_targets, load_model)
from p2_anchor.wringer.rot_fold import (READERS_ATT, READERS_LIN,
                                          READERS_MLP, WRITERS)
from p2_anchor.wringer.state import load_state

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True)
    ap.add_argument("--Q", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate", type=float, default=0.01)
    ap.add_argument("--rec", default=None)
    args = ap.parse_args()
    out = Path(args.out)

    from p1_grouping.e3_runner import eval_ce
    model, tok = load_model()
    model.requires_grad_(False)
    val = torch.load(EV / "calib_val.pt", weights_only=False)
    qd = torch.load(args.Q, weights_only=False)
    Q = qd["Q"].cuda().float()
    ce_ref = qd.get("val_ce_frozen")
    targets = enumerate_targets(model)
    st = load_state(args.codes, targets)
    lm = model.model.language_model
    n_r = n_w = 0

    def wq(full):
        c = st[full]
        w = k2_weight(c["T1"].cuda(), c["T2"].cuda(), c["a0"].cuda().float(), c["c"])
        return w[:, c["inv"].cuda()].float()

    for li in range(N_LAYERS):
        layer = lm.layers[li]
        names = READERS_LIN if hasattr(layer, "linear_attn") else READERS_ATT
        for grp, gname in ((names, "input_layernorm"),
                           (READERS_MLP, "post_attention_layernorm")):
            g = getattr(layer, gname).weight.float()
            for n in grp:
                full = f"{LAYER_PREFIX}.{li}.{n}"
                m = model.get_submodule(full)
                if full in st:
                    m.weight.data.copy_((wq(full) @ Q.T).to(m.weight.dtype))
                    n_r += 1
                else:
                    m.weight.data.copy_((m.weight.float() * (1.0 + g)[None, :]).to(m.weight.dtype))
            getattr(layer, gname).weight.data.zero_()
        for n in WRITERS:
            full = f"{LAYER_PREFIX}.{li}.{n}"
            if full in st:
                m = model.get_submodule(full)
                m.weight.data.copy_((Q @ wq(full)).to(m.weight.dtype))
                n_w += 1
    E0 = lm.embed_tokens.weight.float()
    gf = lm.norm.weight.float()
    model.lm_head.weight = nn.Parameter(
        (E0 * (1.0 + gf)[None, :]).to(torch.bfloat16), requires_grad=False)
    lm.norm.weight.data.zero_()
    del E0, st
    model.config.tie_word_embeddings = False
    model.config.text_config.tie_word_embeddings = False
    print(f"materialized readers {n_r} writers {n_w}", flush=True)

    ce = eval_ce(model, val)
    print(f"carrier val CE {ce:.6f}  ref(frozen) {ce_ref}", flush=True)
    if ce_ref is not None:
        d = abs(ce - ce_ref)
        print(f"|Δ|={d:.6f} gate<{args.gate}", flush=True)
        if d >= args.gate:
            print("ROTX_EQUIV_FAIL")
            raise SystemExit(1)
    print("ROTX_EQUIV_PASS", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SNAPSHOT, out, dirs_exist_ok=True)
    for f in out.glob("*.safetensors*"):
        f.unlink()
    model.save_pretrained(out, safe_serialization=True)
    rec = {"codes": args.codes, "Q": args.Q, "Q_step": qd.get("step"),
           "ce_ref_frozen": ce_ref, "ce_materialized": ce, "gate": args.gate,
           "out": str(out), "readers": n_r, "writers": n_w,
           "note": "dense=W_q·Qᵀ(讀)/Q·W_q(寫);殘差原基底;γ 折歸零;lm_head 解綁",
           "timestamp": datetime.datetime.now().isoformat()}
    (EVC / (args.rec or f"export_rot_{out.name}.json")).write_text(
        json.dumps(rec, indent=1, ensure_ascii=False))
    print("ROTX_EXPORT_DONE →", out, flush=True)


if __name__ == "__main__":
    main()

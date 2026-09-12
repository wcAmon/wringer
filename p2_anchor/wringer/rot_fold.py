"""E60 前置探針:全局殘差流旋轉折疊(QuaRot R1 型)+ 等價閘。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.rot_fold \
      --out exports/a1_rot_bf16 [--seed 20260806]

慣例 x' = Qᵀx(Q 隨機正交,QR of Gaussian,Haar 分佈):
  讀者(殘差→投影) W ← (W·diag(1+γ))·Q,前置 RMSNorm γ ← 0
  寫者(投影→殘差) W ← Qᵀ·W
  embed  E ← E·Q;lm_head 解綁 = (E·diag(1+γ_f))·Q,final norm γ_f ← 0
注意:本骨幹 RMSNorm 為 (1+weight) 風格(modeling_qwen3_5),折 1+γ 後歸零。
等價閘:折疊後 val CE 對 bf16 錨 |Δ| < 0.01(bf16 重捨入噪音帶),FAIL 即停。
範圍:text-only(vision tower 未旋轉,image 輸入不再有效);機制探針用
隨機正交,部署版(加法路徑)換 ±1 Hadamard 同構替換。
"""
import argparse
import datetime
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from p2_anchor.wringer.modelio import LAYER_PREFIX, N_LAYERS, load_model

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"
SNAPSHOT = (Path.home() / ".cache/huggingface/hub/"
            "models--InternScience--Agents-A1-4B/snapshots/"
            "945c40a4aa6f534d434a353207b8d42ecf7a5293")

READERS_LIN = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
               "linear_attn.in_proj_b", "linear_attn.in_proj_a")
READERS_ATT = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
READERS_MLP = ("mlp.gate_proj", "mlp.up_proj")
WRITERS = ("linear_attn.out_proj", "self_attn.o_proj", "mlp.down_proj")


@torch.no_grad()
def fold_reader(m, gamma, Q):
    w = m.weight.float()
    w = w * (1.0 + gamma.float())[None, :]
    m.weight.data.copy_((w @ Q).to(m.weight.dtype))


@torch.no_grad()
def fold_writer(m, Q):
    assert m.bias is None, "writer bias 需 Qᵀb 處理,本骨幹應無 bias"
    m.weight.data.copy_((Q.T @ m.weight.float()).to(m.weight.dtype))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--gate", type=float, default=0.01)
    ap.add_argument("--Q", default=None, help="載入既有 Q(.pt 含 'Q')取代隨機正交")
    ap.add_argument("--rec", default="rot_fold.json")
    ap.add_argument("--identity", action="store_true",
                    help="Q=I:只折 γ/解綁 lm_head(E60 S1 折 γ 基底 bf16 底模)")
    args = ap.parse_args()
    out = Path(args.out)

    from p1_grouping.e3_runner import eval_ce
    model, tok = load_model()
    model.requires_grad_(False)
    val = torch.load(EV / "calib_val.pt", weights_only=False)

    ce_bf16 = eval_ce(model, val)
    print(f"bf16 val CE {ce_bf16:.6f}", flush=True)

    torch.manual_seed(args.seed)
    H = model.config.text_config.hidden_size
    if args.identity:
        Q = torch.eye(H, device="cuda", dtype=torch.float32)
    elif args.Q:
        Q = torch.load(args.Q, weights_only=False)["Q"].cuda().float()
    else:
        A = torch.randn(H, H, dtype=torch.float32, device="cuda")
        Q, R = torch.linalg.qr(A)
        Q = Q * torch.sign(torch.diagonal(R))[None, :]   # Haar 正交
    assert torch.allclose(Q.T @ Q, torch.eye(H, device="cuda"), atol=1e-4)

    lm = model.model.language_model
    for li in range(N_LAYERS):
        layer = lm.layers[li]
        pre = layer.input_layernorm.weight
        names = READERS_LIN if hasattr(layer, "linear_attn") else READERS_ATT
        for n in names:
            fold_reader(model.get_submodule(f"{LAYER_PREFIX}.{li}.{n}"), pre, Q)
        layer.input_layernorm.weight.data.zero_()
        post = layer.post_attention_layernorm.weight
        for n in READERS_MLP:
            fold_reader(model.get_submodule(f"{LAYER_PREFIX}.{li}.{n}"), post, Q)
        layer.post_attention_layernorm.weight.data.zero_()
        for n in WRITERS:
            try:
                m = model.get_submodule(f"{LAYER_PREFIX}.{li}.{n}")
            except AttributeError:
                continue
            fold_writer(m, Q)
        print(f"layer {li:2d} folded", flush=True)

    # embed / final norm / lm_head(tied → 解綁:兩者折法差 diag(1+γ_f))
    E0 = lm.embed_tokens.weight.float()
    gf = lm.norm.weight.float()
    model.lm_head.weight = nn.Parameter(
        ((E0 * (1.0 + gf)[None, :]) @ Q).to(torch.bfloat16), requires_grad=False)
    lm.embed_tokens.weight.data.copy_((E0 @ Q).to(torch.bfloat16))
    lm.norm.weight.data.zero_()
    del E0
    model.config.tie_word_embeddings = False
    model.config.text_config.tie_word_embeddings = False

    ce_rot = eval_ce(model, val)
    d = abs(ce_rot - ce_bf16)
    print(f"rot  val CE {ce_rot:.6f}  |Δ|={d:.6f}  gate<{args.gate}", flush=True)
    if d >= args.gate:
        print("ROT_EQUIV_FAIL")
        raise SystemExit(1)
    print("ROT_EQUIV_PASS", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SNAPSHOT, out, dirs_exist_ok=True)
    for f in out.glob("*.safetensors*"):
        f.unlink()
    model.save_pretrained(out, safe_serialization=True)
    EVC.mkdir(parents=True, exist_ok=True)
    if args.identity:
        torch.save({"Q": Q.cpu(), "seed": None, "note": "identity"}, EVC / "rot_Q_I.pt")
    elif not args.Q:
        torch.save({"Q": Q.cpu(), "seed": args.seed}, EVC / "rot_Q.pt")
    rec = {"seed": args.seed, "hidden": H, "ce_bf16": ce_bf16, "ce_rot": ce_rot,
           "gate": args.gate, "out": str(out), "snapshot": str(SNAPSHOT),
           "Q_src": "identity" if args.identity else (args.Q or "random"), "note": "R1 全局旋轉;text-only;RMSNorm (1+γ) 已折歸零;lm_head 解綁",
           "timestamp": datetime.datetime.now().isoformat()}
    (EVC / args.rec).write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print("ROT_FOLD_DONE →", out, flush=True)


if __name__ == "__main__":
    main()

"""E72 外部基線量化器(prereg_e72_baselines.json):GPTQ / AWQ(llm-compressor)與 HQQ(hqq)→ 假量化 bf16 HF export + 我方帳面。

  .venv-base/bin/python -m p2_anchor.wringer.baseline_quant --method gptq --bits 2 --group 128 \
      --model Qwen/Qwen3-4B --revision <sha> --calib evidence/p1_grouping/calib_e70_traj.pt --lengths evidence/p1_grouping/calib_e70_len.pt \
      --out exports/e72_q_gptq2 --ledger evidence/p1_grouping/corkscrew/baseline_e72_q_gptq2.json

輸出:HF 目錄(config/tokenizer/model.safetensors,權重=量化後反量化的 bf16,vLLM 直接載入)+ ledger json:
  帳面由本檔統一計算:碼 bits×numel + 每群組 fp16 尺度(+ 非對稱 zero-point 以 bits 計)/ 群組;只計 decoder 層內 nn.Linear,embed/lm_head 不入帳。
只依賴 .venv-base(torch/transformers/llmcompressor/hqq),不 import 專案其他模組,避免環境耦合。
印 BASELINE_DONE {json} 或拋錯。
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYER_RE = re.compile(r"^model\.(language_model\.)?layers\.(\d+)\.")


# 2026-09-15:A1 的 linear_attn.in_proj_a / in_proj_b(各 1.97 M)Wringer 留 bf16,基線同樣排除(模組集合一致;E72 A1 gptq2 首跑曾量到 2-bit,已用 a1_wrap 還原)
EXCLUDE_RE = re.compile(r"linear_attn\.in_proj_[ab]$")
LLMC_IGNORE = ["lm_head", r"re:.*linear_attn\.in_proj_a$", r"re:.*linear_attn\.in_proj_b$"]


def body_linears(model):
    """decoder 層內的 nn.Linear(名稱、模組)= Wringer 本體同集合(排除 EXCLUDE_RE)。"""
    out = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and LAYER_RE.match(n) and not EXCLUDE_RE.search(n):
            out.append((n, m))
    return out


def ledger(model, bits, group, symmetric, method, zp_bits):
    tot_w = 0
    tot_bits = 0
    per_type = {}
    for n, m in body_linears(model):
        M, K = m.weight.shape
        numel = M * K
        ngroups = M * ((K + group - 1) // group)
        b = bits * numel + 16 * ngroups + (0 if symmetric else zp_bits * ngroups)
        tot_w += numel
        tot_bits += b
        t = n.split(".")[-1]
        per_type[t] = per_type.get(t, 0) + numel
    return {"method": method, "bits": bits, "group": group, "symmetric": symmetric, "zp_bits": 0 if symmetric else zp_bits,
            "weights_body": tot_w, "bpw_body": tot_bits / tot_w, "n_modules": len(body_linears(model)),
            "per_type_weights": per_type}


def load(args):
    kw = {"torch_dtype": torch.bfloat16, "revision": args.revision}
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=args.device, **kw)
    return model, tok


def calib_dataset(args):
    from datasets import Dataset
    ids = torch.load(args.calib, map_location="cpu", weights_only=False)
    lens = torch.load(args.lengths, map_location="cpu", weights_only=False) if args.lengths else torch.full((ids.shape[0],), ids.shape[1])
    n = min(args.n_calib, ids.shape[0])
    rows = []
    for i in range(n):
        L = min(int(lens[i]), args.max_len)
        rows.append({"input_ids": ids[i, :L].tolist(), "attention_mask": [1] * L})
    return Dataset.from_list(rows)


def run_llmc(args, model, tok):
    from llmcompressor import oneshot
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    wargs = QuantizationArgs(num_bits=args.bits, type="int", symmetric=args.symmetric, strategy="group", group_size=args.group)
    scheme = QuantizationScheme(targets=["Linear"], weights=wargs)
    if args.method == "gptq":
        from llmcompressor.modifiers.quantization import GPTQModifier
        mod = GPTQModifier(config_groups={"group_0": scheme}, ignore=LLMC_IGNORE, dampening_frac=args.damp, block_size=128)
    else:
        from llmcompressor.modifiers.awq import AWQModifier
        mod = AWQModifier(config_groups={"group_0": scheme}, ignore=LLMC_IGNORE)
    ds = calib_dataset(args)
    oneshot(model=model, tokenizer=tok, dataset=ds, recipe=mod, num_calibration_samples=len(ds),
            max_seq_length=args.max_len, shuffle_calibration_samples=False, save_compressed=False,
            pipeline=args.pipeline)
    # AWQ 只算尺度、前向時假量化(GPTQ 原地寫回);統一明確假量化寫回,之後以乾淨 bf16 落盤
    from compressed_tensors.quantization.lifecycle.forward import fake_quantize
    n_fq = 0
    with torch.no_grad():
        for n, m in body_linears(model):
            if hasattr(m, "weight_scale") and getattr(m, "quantization_scheme", None) is not None:
                W = fake_quantize(m.weight, m.weight_scale, m.weight_zero_point, m.quantization_scheme.weights,
                                  g_idx=getattr(m, "weight_g_idx", None))
                m.weight.data.copy_(W.to(m.weight.dtype))
                n_fq += 1
    print(f"fake_quantize written back: {n_fq} modules", flush=True)
    return model


def clean_save(model, tok, out: Path):
    """乾淨 bf16 HF checkpoint:剔除 compressed-tensors 的 scale/zp/g_idx 張量與 quantization_config;tied embedding 只存一份。"""
    from safetensors.torch import save_file
    drop_sfx = ("weight_scale", "weight_zero_point", "weight_g_idx", "input_scale", "output_scale", "input_zero_point", "output_zero_point")
    sd = {}
    for k, v in model.state_dict().items():
        if k.endswith(drop_sfx):
            continue
        sd[k] = v.detach().to(torch.bfloat16).contiguous().cpu()
    emb = [k for k in sd if k.endswith("embed_tokens.weight")]
    if "lm_head.weight" in sd and emb and getattr(model.config, "tie_word_embeddings", False):
        if torch.equal(sd["lm_head.weight"], sd[emb[0]]):
            del sd["lm_head.weight"]
    out.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    cfg = model.config.to_dict()
    cfg.pop("quantization_config", None)
    cfg.pop("compression_config", None)
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    gc = getattr(model, "generation_config", None)
    if gc is not None:
        gc.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    return len(sd)


@torch.no_grad()
def run_hqq(args, model):
    from hqq.core.quantize import Quantizer
    for n, m in body_linears(model):
        W = m.weight.data.float()
        dev = W.device
        Wq, meta = Quantizer.quantize(W, nbits=args.bits, channel_wise=True, group_size=args.group, optimize=True,
                                      round_zero=False, axis=1, bitpack=False, device=str(dev) if dev.type == "cuda" else "cpu")
        Wd = Quantizer.dequantize(Wq, meta)
        m.weight.data.copy_(Wd.reshape(W.shape).to(m.weight.dtype))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=("gptq", "awq", "hqq"), required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--symmetric", action="store_true", help="預設非對稱(GPTQ/HQQ 慣例);AWQ 4-bit 建議 --symmetric")
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--zp-bits", type=int, default=None, help="非對稱 zero-point 帳面位元;預設 gptq/awq=bits(packed int)、hqq=16(quant_zero=False 存 fp16)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--lengths", default=None)
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--pipeline", default="independent", help="llm-compressor 校準管線:independent / sequential / basic")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", required=True)
    args = ap.parse_args()
    t0 = time.time()
    model, tok = load(args)
    if args.method == "hqq":
        model = run_hqq(args, model)
    else:
        assert args.calib, "gptq/awq 需 --calib"
        model = run_llmc(args, model, tok)
    zp = args.zp_bits if args.zp_bits is not None else (16 if args.method == "hqq" else args.bits)
    led = ledger(model, args.bits, args.group, args.symmetric, args.method, zp)
    out = Path(args.out)
    led["n_tensors_saved"] = clean_save(model, tok, out)
    if str(getattr(model.config, "model_type", "")).endswith("_text"):
        # Qwen3.5 混合架構:AutoModelForCausalLM 只載文字子模型,包回 Wringer export 同一格式(vLLM 需 Qwen3_5ForConditionalGeneration)
        from p2_anchor.wringer.a1_wrap import wrap, snapshot_dir
        wrap(out, snapshot_dir(args.model, args.revision), EXCLUDE_RE.pattern, None)
        led["wrapped"] = "Qwen3_5ForConditionalGeneration(a1_wrap.py)"
    # 假量化落盤驗證:讀回本體權重,逐群組唯一值數 ≤ 2^bits(抽 8 張,每張首列首群);config 不得含 quantization_config
    from safetensors import safe_open
    with safe_open(str(out / "model.safetensors"), "pt") as f:
        keys = [k for k in f.keys() if LAYER_RE.match(k) and k.endswith("weight") and "norm" not in k
                and not EXCLUDE_RE.search(k.rsplit(".", 1)[0]) and len(f.get_slice(k).get_shape()) == 2]
        checks = []
        for key in keys[:: max(1, len(keys) // 8)][:8]:
            W = f.get_tensor(key).float()
            checks.append({"tensor": key, "unique_in_first_group": int(W[0, :args.group].unique().numel())})
        cfg_keys = list(f.keys())
    bad = [c for c in checks if c["unique_in_first_group"] > 2 ** args.bits]
    led["fakequant_check"] = {"max_allowed": 2 ** args.bits, "checked": checks, "n_bad": len(bad),
                              "leftover_quant_tensors": sum(k.endswith(("weight_scale", "weight_zero_point")) for k in cfg_keys),
                              "config_has_quantization_config": "quantization_config" in json.loads((out / "config.json").read_text())}
    assert not bad and led["fakequant_check"]["leftover_quant_tensors"] == 0 and not led["fakequant_check"]["config_has_quantization_config"], led["fakequant_check"]
    led.update({"model": args.model, "revision": args.revision, "out": str(out), "runtime_s": round(time.time() - t0, 1),
                "n_calib": args.n_calib if args.method != "hqq" else 0, "max_len": args.max_len})
    Path(args.ledger).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ledger).write_text(json.dumps(led, ensure_ascii=False, indent=1))
    print("BASELINE_DONE", json.dumps({k: led[k] for k in ("method", "bits", "group", "bpw_body", "fakequant_check", "runtime_s")}), flush=True)


if __name__ == "__main__":
    main()

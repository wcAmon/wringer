"""把 llm-compressor / hqq 落盤的 Qwen3.5 文字子模型 export(Qwen3_5TextModel:鍵 model.*、無 visual)包回
Wringer export 同一格式(Qwen3_5ForConditionalGeneration:鍵 model.language_model.* + model.visual.*、config 含 text_config)。
規則:HF 快照的每個鍵 → 若對應 export 的 model.* 張量存在且模組不在排除清單 → 用 export 的(含 AWQ 平滑後的 norm);否則用快照原始張量。
排除清單(預設 linear_attn.in_proj_a/in_proj_b):Wringer 留 bf16 的兩張小閘,基線也留原始,模組集合一致;帳面同步剔除。
用法:python -m p2_anchor.wringer.a1_wrap --out exports/e72_a_gptq2 --model InternScience/Agents-A1-4B --revision <rev> [--ledger <json>]
"""
import argparse, json, re, shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXCLUDE_DEFAULT = r"linear_attn\.in_proj_[ab]$"


def snapshot_dir(model, revision):
    return Path.home() / ".cache/huggingface/hub" / f"models--{model.replace('/', '--')}" / "snapshots" / revision


def wrap(out: Path, snap: Path, exclude_re: str, ledger: Path | None):
    idx = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    ex = re.compile(exclude_re) if exclude_re else None
    src = safe_open(str(out / "model.safetensors"), "pt")
    src_keys = set(src.keys())
    if any(k.startswith("model.language_model.") for k in src_keys):
        print("already wrapped"); return
    sd, n_from_export, n_excluded, n_snap = {}, 0, 0, 0
    files = {}
    for k, f in idx.items():
        files.setdefault(f, []).append(k)
    for f, keys in files.items():
        with safe_open(str(snap / f), "pt") as sf:
            for k in keys:
                ek = k.replace("model.language_model.", "model.", 1) if k.startswith("model.language_model.") else None
                mod = ek.rsplit(".", 1)[0] if ek else ""
                if ek in src_keys and not (ex and ex.search(mod)):
                    t = src.get_tensor(ek); n_from_export += 1
                else:
                    t = sf.get_tensor(k); n_snap += 1
                    if ek in src_keys:
                        n_excluded += 1
                sd[k] = t.to(torch.bfloat16).contiguous()
    tmp = out / "model.safetensors.wrapped"
    save_file(sd, str(tmp), metadata={"format": "pt"})
    del src
    tmp.replace(out / "model.safetensors")
    for f in ("config.json", "preprocessor_config.json", "processor_config.json", "chat_template.jinja", "generation_config.json"):
        if (snap / f).exists():
            shutil.copy(snap / f, out / f)
    print(f"wrapped: {len(sd)} tensors(export {n_from_export}、快照 {n_snap},其中排除模組還原 {n_excluded})")
    if ledger and ledger.exists():
        led = json.loads(ledger.read_text())
        drop = [t for t in led["per_type_weights"] if ex and ex.search("linear_attn." + t)]
        removed = sum(led["per_type_weights"][t] for t in drop)
        n_mod_removed = n_excluded // 1   # 每模組一張 weight
        led["weights_body"] -= removed
        led["n_modules"] -= n_mod_removed
        for t in drop:
            led["per_type_weights"].pop(t)
        led["excluded_modules"] = {"regex": exclude_re, "restored_to_base": n_excluded, "weights_removed_from_ledger": removed,
                                   "note": "Wringer 留 bf16 的 in_proj_a/b;bpw_body 為均勻配置不變"}
        led["wrapped"] = "Qwen3_5ForConditionalGeneration(a1_wrap.py)"
        ledger.write_text(json.dumps(led, ensure_ascii=False, indent=1))
        print(f"ledger: weights_body {led['weights_body']} n_modules {led['n_modules']} bpw {led['bpw_body']:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--model", required=True); ap.add_argument("--revision", required=True)
    ap.add_argument("--exclude", default=EXCLUDE_DEFAULT); ap.add_argument("--ledger", default=None)
    a = ap.parse_args()
    wrap(Path(a.out), snapshot_dir(a.model, a.revision), a.exclude, Path(a.ledger) if a.ledger else None)


if __name__ == "__main__":
    main()

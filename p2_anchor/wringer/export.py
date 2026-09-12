"""烘焙匯出雙形態:HF safetensors(fake-quant 材化)+ 緊湊位面包(hoops 合約)。

  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.export \
      --state-tag st3 [--alphas alphas_st3_pol50.npz] --out exports/t3 [--verify]

位面包格式(與 guava bitplanes_k2_c0.6.npz 同構,含 bit-exact 解碼驗證):
每 tensor 三件 —— idx((T1+1)*3+(T2+1)∈[0,8],兩值一 byte)、
alpha(fp32 逐 block)、inv(int32 欄置換);meta.c / meta.grid。
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from p2_anchor.wringer.modelio import enumerate_targets, load_model
from p2_anchor.wringer.state import apply_k2, load_state

EVC = Path("evidence/p1_grouping/corkscrew")
SNAPSHOT = (Path.home() / ".cache/huggingface/hub/"
            "models--InternScience--Agents-A1-4B/snapshots/"
            "945c40a4aa6f534d434a353207b8d42ecf7a5293")


def build_planes(st, alphas, out, c=0.6, grid=9):
    planes = {"meta.c": np.float32(c), "meta.grid": np.int32(grid)}
    for n, s in st.items():
        planes[f"{n}.grid"] = np.int32(s.get("grid", grid))
        M, nB, B = s["T1"].shape
        idx = ((s["T1"].numpy().astype(np.int16) + 1) * 3
               + (s["T2"].numpy().astype(np.int16) + 1)).astype(np.uint8)
        flat = idx.reshape(M, nB * B)          # 置換域,不解置換
        packed = (flat[:, 0::2] << 4) | flat[:, 1::2]
        planes[f"{n}.idx"] = packed
        a = (s["a0"] if alphas is None else torch.as_tensor(alphas[n]))
        planes[f"{n}.alpha"] = a.reshape(M, nB).float().numpy()
        planes[f"{n}.inv"] = s["inv"].numpy().astype(np.int32)
        if s.get("mix") is not None:           # E61:在線塊混合 (k,b,b) fp32
            planes[f"{n}.mix"] = s["mix"].float().numpy()
        if s.get("lmix") is not None:          # E62b:輸出側列塊 L (M/b,b,b) fp32
            planes[f"{n}.lmix"] = s["lmix"].float().numpy()
        if s.get("t2beta") is not None:        # E63 T2:殘差位面尺度 β 逐 block fp32(w = α t1 + β t2)
            planes[f"{n}.t2beta"] = s["t2beta"].float().numpy().reshape(M, nB)
        if s.get("rneg") is not None:          # E63:負尺度 r 逐列 (M,1) 或逐 block (M,nB)
            r = s["rneg"].float().numpy().reshape(M, -1)
            if s.get("rneg_cb") is not None:   # 量化版:存碼書 + 索引(位元帳 log2(k)/B b/w)
                cb = s["rneg_cb"].float().numpy().reshape(-1)
                idx = np.abs(r[:, :, None] - cb[None, None, :]).argmin(-1).astype(np.uint8)
                assert np.array_equal(cb[idx], r), "rneg 不在碼書上"
                planes[f"{n}.rneg_cb"] = cb.astype(np.float32)
                planes[f"{n}.rneg_idx"] = idx
            else:
                planes[f"{n}.rneg"] = r.astype(np.float32)
    path = out / f"bitplanes_k2_c{c}.npz"
    np.savez_compressed(path, **planes)
    print("planes →", path)
    return path


def decode_tensor(z, name):
    packed = z[f"{name}.idx"]
    alpha = z[f"{name}.alpha"]
    inv = z[f"{name}.inv"]
    M, nB = alpha.shape
    N = packed.shape[1] * 2           # block 由位面寬度推導(g32/g16 通用)
    idx = np.empty((M, N), np.uint8)
    idx[:, 0::2] = packed >> 4
    idx[:, 1::2] = packed & 0x0F
    t1 = (idx.astype(np.float32) // 3) - 1.0
    t2 = (idx % 3).astype(np.float32) - 1.0
    has_beta = f"{name}.t2beta" in z.files
    if has_beta:                      # E63 T2:w = α t1 (負側×r) + β t2(與 state.dense_weight 同路)
        w = (alpha[:, :, None] * t1.reshape(M, nB, N // nB)).reshape(M, N)
    else:
        v = (t1 + float(z["meta.c"]) * t2).reshape(M, nB, N // nB)
        w = (alpha[:, :, None] * v).reshape(M, N)
    if f"{name}.rneg" in z.files or f"{name}.rneg_idx" in z.files:   # E63:負碼尺度(與 state.dense_weight 同路)
        r = (z[f"{name}.rneg"] if f"{name}.rneg" in z.files
             else z[f"{name}.rneg_cb"][z[f"{name}.rneg_idx"]]).reshape(M, -1)
        r = np.broadcast_to(r[:, :, None], (M, r.shape[1], N // r.shape[1])).reshape(M, N)
        w = np.where(t1 < 0, w * r, w).astype(np.float32)
    if has_beta:
        beta = z[f"{name}.t2beta"].astype(np.float32)
        w = (w.reshape(M, nB, N // nB) + beta[:, :, None] * t2.reshape(M, nB, N // nB)
             ).reshape(M, N).astype(np.float32)
    return w[:, inv]


def verify_planes(planes_path, export_dir):
    """位面包 → bf16,逐 tensor 對 safetensors bit-exact。"""
    import glob

    from safetensors import safe_open
    z = np.load(planes_path)
    names = sorted({k.rsplit(".", 1)[0] for k in z.files if k.endswith(".idx")})
    ref = {}
    for f in glob.glob(str(export_dir / "*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                if k.endswith(".weight"):
                    ref[k] = f
    bad = 0
    for i, n in enumerate(names):
        key = n + ".weight"
        with safe_open(ref[key], framework="pt") as sf:
            r = sf.get_tensor(key)
        w = torch.from_numpy(decode_tensor(z, n))
        if f"{n}.mix" in z.files or f"{n}.lmix" in z.files:   # E61/E62b:與 apply_k2 同路
            from p2_anchor.wringer.state import mix_dense
            w = w.cuda()
            if f"{n}.mix" in z.files:
                w = w @ mix_dense(torch.from_numpy(z[f"{n}.mix"]).cuda())
            if f"{n}.lmix" in z.files:
                w = mix_dense(torch.from_numpy(z[f"{n}.lmix"]).cuda()) @ w
            w = w.cpu()
        w = w.to(torch.bfloat16)
        ok = torch.equal(w, r)
        bad += not ok
        if not ok or i % 50 == 0:
            print(f"[{i + 1}/{len(names)}] {n} bit-exact={ok}", flush=True)
    print(f"verified {len(names)} tensors, mismatches={bad}")
    assert bad == 0, "位面包 bit-exact 驗證 FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-tag", required=True)
    ap.add_argument("--alphas", default=None,
                    help="EVC 下 npz 檔名(打磨後 α);缺省用 a0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--strip-mix", action="store_true",
                    help="純碼診斷:丟棄輸入側 M(M=I),只材化 碼+α")
    ap.add_argument("--strip-lmix", action="store_true",
                    help="E62b 診斷:丟棄輸出側 L(L=I),只材化 碼+α+M")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, tok = load_model()
    targets = enumerate_targets(model)
    st = load_state(args.state_tag, targets)
    if args.strip_lmix:
        n_strip = sum(1 for v in st.values() if v.get("lmix") is not None)
        for v in st.values():
            v.pop("lmix", None)
        print(f"strip-lmix: dropped L on {n_strip} modules")
    if args.strip_mix:
        n_strip = sum(1 for v in st.values() if v.get("mix") is not None)
        for v in st.values():
            v.pop("mix", None)
        print(f"strip-mix: dropped M on {n_strip} modules")
    anpz = (dict(np.load(EVC / args.alphas)) if args.alphas else None)
    apply_k2(model, st, anpz)

    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    for f in ("chat_template.jinja", "preprocessor_config.json",
              "processor_config.json", "generation_config.json"):
        src = SNAPSHOT / f
        if src.exists() and not (out / f).exists():
            shutil.copy(src, out / f)

    gset = sorted({int(s["grid"]) for s in st.values()})
    grid = gset[0] if len(gset) == 1 else 0     # 0 = 混合;逐 tensor 見 .grid
    c = st[targets[0]]["c"]
    if any(g not in (3, 5, 7, 9) for g in gset):    # E68 整數格:位面包(k2 4bit idx)不適用,只出 HF 材化
        print(f"int-grid state {gset}: planes/verify skipped (HF safetensors only)")
        print("done →", out)
        return
    planes = build_planes(st, anpz, out, c=c, grid=grid)
    if args.verify:
        del model
        torch.cuda.empty_cache()
        verify_planes(planes, out)
    print("done →", out)


if __name__ == "__main__":
    main()

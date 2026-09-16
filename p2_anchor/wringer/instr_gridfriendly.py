"""儀器(不裁決):蓄水是否把目標權重搬到「格點友善」位置。
對每個量化模組:
  residual(state, target) = ||W_target − Ŵ(state)||_F / ||W_target||_F,Ŵ = 碼 × 尺度(pack_verify.materialize 同式)
  - zero:  state=rtnj0(bf16 → 最近點 + 閉式尺度), target=bf16 原權重
  - water: state=rtnj (含水 → 最近點 + 閉式尺度), target=含水權重(resA export;A1 為第二輪 e69_p3b_w_resA,故 gptq 對照用同目標的 p3b_w2、起始碼用 p3b_w)
  另:碼差異比例 codes(rtnj0) vs codes(rtnj)(蓄水改了多少最近點碼)、codes(rtnj) vs codes(GPTQ 排水態)(求解器改了多少碼)
用法:python -m p2_anchor.wringer.instr_gridfriendly --model qwen|a1 --out <json>
"""
import argparse, json, glob, re
from pathlib import Path
import torch
from safetensors import safe_open

CFG = {
 "qwen": dict(zero="e74_q_rtnj0", water="e73_q_rtnj", gptq="e70_p3a_w", start="e70_p3a", resA="exports/e70_p3a_resA/model.safetensors",
              snap=str(Path.home()/".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c")),
 "a1": dict(zero="e74_a_rtnj0", water="e73_a_rtnj", gptq="p3b_w2", start="p3b_w", resA="exports/e69_p3b_w_resA/model.safetensors",
            snap=sorted(glob.glob(str(Path.home()/".cache/huggingface/hub/models--InternScience--Agents-A1-4B/snapshots/*")))[0]),
}

class Snap:
    def __init__(self, d):
        d = Path(d)
        if (d/"model.safetensors.index.json").exists():
            wm = json.loads((d/"model.safetensors.index.json").read_text())["weight_map"]
            self.files = {k: d/f for k, f in wm.items()}
        else:
            self.files = {}
            for f in d.glob("*.safetensors"):
                with safe_open(str(f), "pt") as sf:
                    for k in sf.keys(): self.files[k] = f
        self.h = {}
    def get(self, k):
        f = self.files[k]
        if f not in self.h: self.h[f] = safe_open(str(f), "pt")
        return self.h[f].get_tensor(k)

def load_state(tag):
    st = {}
    for f in sorted(glob.glob(f"data/qs_{tag}/layer*.pt")):
        st.update(torch.load(f, weights_only=False, map_location="cpu"))
    return st

def what(s):  # (Ŵ in original column order, codes in original column order)
    T1 = s["T1"]; a = s["a0"].float(); inv = s["inv"].long()
    M = T1.shape[0]
    w = (T1.float() * a).reshape(M, -1)[:, inv]
    codes = T1.reshape(M, -1)[:, inv]
    return w, codes

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); c = CFG[a.model]
    zero, water, gptq, start = load_state(c["zero"]), load_state(c["water"]), load_state(c["gptq"]), load_state(c["start"])
    snap = Snap(c["snap"]); resA = safe_open(c["resA"], "pt")
    keys = sorted(zero)
    tot = {"zero": [0.0, 0.0], "water": [0.0, 0.0]}; by = {}
    flip_fill = [0, 0]; flip_solver = [0, 0]; flip_start = [0, 0]; flip_zero_solver = [0, 0]; per_mod = []
    for k in keys:
        wk = k + ".weight"
        Wb = snap.get(wk).float()
        rk = wk if wk in resA.keys() else wk.replace("model.language_model.", "model.", 1)
        Wr = resA.get_tensor(rk).float()
        wz, cz = what(zero[k]); ww, cw = what(water[k]); _, cg = what(gptq[k]); _, cs = what(start[k])
        ez, nz = (Wb - wz).pow(2).sum().item(), Wb.pow(2).sum().item()
        ew, nw = (Wr - ww).pow(2).sum().item(), Wr.pow(2).sum().item()
        g = f"g{zero[k]['grid']}"
        for name, e, n in (("zero", ez, nz), ("water", ew, nw)):
            tot[name][0] += e; tot[name][1] += n
            by.setdefault(g, {"zero": [0.0, 0.0], "water": [0.0, 0.0]})[name][0] += e; by[g][name][1] += n
        ff = (cz != cw).sum().item(); fs = (cw != cg).sum().item(); n = cz.numel()
        flip_fill[0] += ff; flip_fill[1] += n; flip_solver[0] += fs; flip_solver[1] += n
        flip_start[0] += (cw != cs).sum().item(); flip_start[1] += n; flip_zero_solver[0] += (cz != cs).sum().item(); flip_zero_solver[1] += n
        per_mod.append({"module": re.sub(r"^model\.(language_model\.)?layers\.\d+\.", "", k), "grid": g,
                        "res_zero": (ez/nz)**0.5, "res_water": (ew/nw)**0.5, "flip_fill": ff/n, "flip_solver": fs/n})
    r = lambda t: (t[0]/t[1])**0.5
    out = {"model": a.model, "states": {k: c[k] for k in ("zero","water","gptq","start")},
           "residual_rel_fro": {"zero_training(bf16 target)": r(tot["zero"]), "after_fill(watered target)": r(tot["water"])},
           "by_grid": {g: {"zero": r(v["zero"]), "water": r(v["water"])} for g, v in by.items()},
           "code_change_frac": {"nearest_codes_bf16_vs_watered": flip_fill[0]/flip_fill[1],
                                "nearest_vs_gptq_on_watered": flip_solver[0]/flip_solver[1],
                                "nearest_on_watered_vs_frozen_start_codes(P3 gptq)": flip_start[0]/flip_start[1],
                                "nearest_vs_gptq_on_bf16(zero training)": flip_zero_solver[0]/flip_zero_solver[1]},
           "n_modules": len(keys), "n_codes": flip_fill[1]}
    # 逐模組型別彙總
    import collections, statistics as st
    agg = collections.defaultdict(list)
    for m in per_mod: agg[m["module"]].append(m)
    out["by_module_type"] = {t: {"res_zero": st.mean(x["res_zero"] for x in v), "res_water": st.mean(x["res_water"] for x in v),
                                 "flip_fill": st.mean(x["flip_fill"] for x in v), "flip_solver": st.mean(x["flip_solver"] for x in v)} for t, v in agg.items()}
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("INSTR_DONE", json.dumps({k: out[k] for k in ("residual_rel_fro", "code_change_frac")}))

if __name__ == "__main__":
    main()

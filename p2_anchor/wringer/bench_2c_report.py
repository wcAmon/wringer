"""2c 結果彙整:讀 evidence/p1_grouping/corkscrew/bench_2c/{smoke_*,lat_*,tp_*}.json + 煙測日誌的載入記憶體,
寫 evidence/p1_grouping/corkscrew/bench_2c.md。印 REPORT_DONE。
"""
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "evidence/p1_grouping/corkscrew/bench_2c"
S = Path("$SCRATCH")
TAGS = ["bf16", "gptq4", "gptq23"]
CFGS = ["bs1_in512_out128", "bs1_in2048_out128", "bs8_in512_out128", "bs32_in512_out128"]


def load_mem(tag):
    for name in (f"smoke_{tag}.log", f"lat_{tag}_{CFGS[0]}.log"):
        p = S / name
        if p.exists():
            m = re.search(r"Model loading took ([\d.]+) GiB", p.read_text())
            if m:
                return float(m.group(1))
    return None


def main():
    lat = {(t, c): json.load(open(D / f"lat_{t}_{c}.json")) for t in TAGS for c in CFGS if (D / f"lat_{t}_{c}.json").exists()}
    tp = {t: json.load(open(D / f"tp_{t}.json")) for t in TAGS if (D / f"tp_{t}.json").exists()}
    smoke = {p.stem.replace("smoke_", ""): json.load(open(p)) for p in D.glob("smoke_*.json")}
    mem = {t: load_mem(t) for t in TAGS}
    L = ["# 2c:p3b_w2 GPTQ 版面在 vLLM 0.26 的記憶體與速度(對 bf16 材化 export)", "",
         "同一組權重數值(升位保值,見 pack_to_gptq.py):gptq4 = 2/3→4-bit + q/k/v 8-bit(原生 Marlin);gptq23 = 真 2/3-bit(exllama,需補丁,fp16 激活)。", "",
         "## 載入記憶體(vLLM「Model loading took」)", "", "| export | 權重 GiB | 檔案 GB |", "|---|---|---|"]
    for t in TAGS:
        p = {"bf16": "exports/e69_p3b_w2_pack", "gptq4": "exports/e69_p3b_w2_gptq4", "gptq23": "exports/e69_p3b_w2_gptq23"}[t]
        f = ROOT / p / "model.safetensors"
        L.append(f"| {t} | {mem[t] if mem[t] is not None else '—'} | {f.stat().st_size/1e9:.2f} |" if f.exists() else f"| {t} | — | — |")
    L += ["", "## 單批延遲(vllm bench latency,avg s;括號 = 每輸出 token ms)", "", "| 設定 | " + " | ".join(TAGS) + " |", "|---|" + "---|" * len(TAGS)]
    for c in CFGS:
        cells = []
        for t in TAGS:
            d = lat.get((t, c))
            cells.append(f"{d['avg_latency']:.3f} ({d['avg_latency']/128*1000:.1f})" if d else "—")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "## 離線吞吐(vllm bench throughput,random in1024/out256,256 prompts,max-num-seqs 64)", "", "| export | requests/s | total tok/s | output tok/s |", "|---|---|---|---|"]
    for t in TAGS:
        d = tp.get(t)
        if d:
            L.append(f"| {t} | {d['requests_per_second']:.2f} | {d['tokens_per_second']:.0f} | {d['num_requests'] * 256 / d['elapsed_time']:.0f} |")
    L += ["", "## 煙測(16 提示 greedy 96 token,對 bf16 逐 token 比)", "", "| export | 全程一致 | 平均一致前綴 | 首分歧位置 |", "|---|---|---|---|"]
    for k, v in sorted(smoke.items()):
        c = v.get("compare")
        L.append(f"| {k} ({v.get('dtype','auto')}) | {c['identical']}/{c['n']} | {c['mean_prefix_agree']} | {c['first_divergence']} |" if c else f"| {k} ({v.get('dtype','auto')}) | 參考 | — | — |")
    L += ["", "## 讀法", "",
          "- **VRAM**:權重 8.61 → 3.84(gptq4)→ 3.27 GiB(gptq23);餘下大頭是 bf16 embedding 1.27 GiB(與 lm_head 綁定)與視覺塔(vLLM 一律載入),本體 2/3-bit 已到位。",
          "- **decode(bs1/bs8)**:兩種版面都比 bf16 快 1.7–2.1×,gptq23 略快於 gptq4(位元少、fp16 激活);這是記憶體頻寬綁定段,低位元直接換速度。",
          "- **大批次/吞吐**:gptq4(Marlin)仍領先 bf16(bs32 1.3×、吞吐 +7%);gptq23(exllama)bs32 反慢於 gptq4、吞吐 −10% vs bf16——exllama 對大 M 走 dequant 路徑,核心層微基準(bench_gptq_kernel.md)已預告。",
          "- **正確性**:三個版面解回權重 95.9% 與 bf16 export 逐位相同、餘 1 bf16 ulp(int8-α fp16 化);greedy 一致率 gptq4 10/16、fp16 對照 9/16、gptq23 9/16(分歧位置與 fp16 對照幾乎重合 = 2/3-bit 核心無額外偏差)。",
          "- **部署建議**:今日可用 = gptq4(原生 vLLM、Marlin、bf16 激活、吞吐最佳);真 2.6 bpw 的 VRAM 與 bs1 延遲 = gptq23(需 vllm_patch_23bit,fp16 激活,大批次吞吐讓 10%);要兩者兼得需 Marlin 級 2/3-bit 核心(新工程)。",
          "- 條件差異注意:gptq23 用 --dtype float16(exllama 限制),bf16/gptq4 用 auto(bf16);fp16 對照煙測顯示 fp16 本身不影響正確性。"]
    (ROOT / "evidence/p1_grouping/corkscrew/bench_2c.md").write_text("\n".join(L) + "\n")
    print("REPORT_DONE", {t: mem[t] for t in TAGS}, flush=True)


if __name__ == "__main__":
    main()

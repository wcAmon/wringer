"""E67-U:llama-quantize --dry-run 輸出 → 主幹線性層 bpw(不含 embedding/output/norm)與整檔 bpw。
  用法:python gguf_dryrun_bpw.py <preset> [--tensor-type pat=type ...]"""
import os, re, subprocess, sys
Q = "~/llama.cpp/build/bin/llama-quantize"
preset = sys.argv[1]; extra = sys.argv[2:]
cmd = [Q, "--dry-run", "--token-embedding-type", "q4_k", "--output-tensor-type", "q6_k", *extra,
       os.environ.get("GGUF_BF16", "data/gguf/a1-4b-bf16.gguf"), "/tmp/dry.gguf", preset, "8"]   # E70:GGUF_BF16 覆寫
out = subprocess.run(cmd, capture_output=True, text=True).stdout + subprocess.run(cmd, capture_output=True, text=True).stderr
pat = re.compile(r"\]\s+(\S+)\s+-\s+\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\], type =\s+\S+, size =\s+([\d.]+) MiB(?: ->\s+([\d.]+) MiB \((\w+)\))?")
tot_b = tot_n = body_b = body_n = 0.0; types = {}
for m in pat.finditer(out):
    name = m.group(1); n = int(m.group(2)) * int(m.group(3)) * int(m.group(4)) * int(m.group(5))
    mib = float(m.group(7) or m.group(6)); ty = m.group(8) or "f32"
    bits = mib * 1048576 * 8
    tot_b += bits; tot_n += n
    if name.startswith("blk.") and "norm" not in name and n >= 1 << 16:
        body_b += bits; body_n += n; types[ty] = types.get(ty, 0) + 1
print(f"{preset} {' '.join(extra)}\n  bpw_body={body_b/max(body_n,1):.3f} bpw_whole={tot_b/max(tot_n,1):.3f} body_n={body_n/1e9:.3f}G types={types}")

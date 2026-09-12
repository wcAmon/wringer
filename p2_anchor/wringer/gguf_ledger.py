"""GGUF 位元帳(E67-U):逐張量 bpw,整檔 / 不含 embedding+output(=我們的帳口徑)兩欄 + 量化型態直方圖。

  PYTHONPATH=~/llama.cpp/gguf-py .venv/bin/python -m p2_anchor.wringer.gguf_ledger data/gguf/x.gguf
"""
import json
import sys
from collections import Counter

from gguf import GGUFReader, GGML_QUANT_SIZES


def main(path):
    r = GGUFReader(path)
    rows, hist = [], Counter()
    tot_bits = tot_n = 0
    body_bits = body_n = 0
    for t in r.tensors:
        n = 1
        for d in t.shape:
            n *= int(d)
        bs, ts = GGML_QUANT_SIZES[t.tensor_type]
        bits = n / bs * ts * 8
        name = t.name
        rows.append((name, t.tensor_type.name, n, bits / n))
        hist[t.tensor_type.name] += 1
        tot_bits += bits
        tot_n += n
        if name.startswith("blk.") and not name.endswith("norm.weight") and n >= 1 << 16:
            body_bits += bits
            body_n += n
    out = {"file": path, "bpw_whole": tot_bits / tot_n, "bpw_body_linear": body_bits / body_n,
           "weights_whole": tot_n, "weights_body": body_n, "types": dict(hist)}
    print(json.dumps(out, indent=1))
    big = sorted([r_ for r_ in rows if r_[2] >= 1 << 20], key=lambda x: x[0])
    for name, ty, n, bpw in big[:12]:
        print(f"  {name:40s} {ty:8s} {n/1e6:7.1f}M {bpw:.2f}")
    return out


if __name__ == "__main__":
    main(sys.argv[1])

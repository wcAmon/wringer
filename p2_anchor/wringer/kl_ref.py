"""E60 S0′:r_KL 參考點——held-out KL-to-bf16(calib_val)四態:
  g3ctl 直投(apply_k2 原基底)/ g3rot 隨機旋轉@Q0 / g3rotL1@Q_L1 / 載體 g3rotL1@Q_L2(CE 漂移態)。
  r_KL = 1 − KL/KL_ctl;bf16 自身 0。
"""
import json
import torch
from p2_anchor.wringer.modelio import enumerate_targets, load_model
from p2_anchor.wringer.state import apply_k2, load_state
from p2_anchor.wringer.rot_learn import val_kl, val_ce, EV, EVC
from transformers import AutoModelForImageTextToText


@torch.no_grad()
def main():
    val = torch.load(EV / "calib_val.pt", weights_only=False)
    teacher, tok = load_model()
    teacher.eval()
    out = {}
    model, _ = load_model()
    st = load_state("g3ctl", enumerate_targets(model))
    apply_k2(model, st)
    model.eval()
    out["g3ctl_direct"] = {"kl": val_kl(model, teacher, val), "ce": val_ce(model, val)}
    print("g3ctl_direct", out["g3ctl_direct"], flush=True)
    del model, st
    torch.cuda.empty_cache()
    for name, d in (("g3rot_random", "exports/g3rot_floor"),
                    ("g3rotL1_QL1", "exports/a1_rotL1_bf16__q"),   # 佔位:由 rot_learn --kd step0 量
                    ("carrier_g3rotL1_QL2", "exports/g3rotL1_QL2_carrier")):
        if name == "g3rotL1_QL1":
            continue
        m = AutoModelForImageTextToText.from_pretrained(d, dtype=torch.bfloat16).cuda().eval()
        out[name] = {"kl": val_kl(m, teacher, val), "ce": val_ce(m, val)}
        print(name, out[name], flush=True)
        del m
        torch.cuda.empty_cache()
    for k in out:
        out[k]["r_kl"] = 1 - out[k]["kl"] / out["g3ctl_direct"]["kl"]
    (EVC / "kl_ref_e60.json").write_text(json.dumps(out, indent=1))
    print(f"KLREF_DONE ctl {out['g3ctl_direct']['kl']:.4f} random {out['g3rot_random']['kl']:.4f}"
          f"(r {out['g3rot_random']['r_kl']:.3f}) carrierCE {out['carrier_g3rotL1_QL2']['kl']:.4f}"
          f"(r {out['carrier_g3rotL1_QL2']['r_kl']:.3f})", flush=True)


if __name__ == "__main__":
    main()

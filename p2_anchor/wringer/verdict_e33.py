"""E33 判準 Z1–Z6 裁決器(prereg_e33_7lane_v2.json @ceff4ff)。

**本檔在看到任何 E33 結果之前寫成**,以免操作化選擇被數據牽著走。
每個判準的實作方式、以及 prereg 文字留下的模糊處如何解讀,都在下方
明列;裁決時只執行、不改。

用法:
  PYTHONPATH=. .venv/bin/python -m p2_anchor.wringer.verdict_e33 \
    --out evidence/p1_grouping/corkscrew/verdict_e33.json
"""
import argparse
import json
from pathlib import Path

EV = Path("evidence/p1_grouping")
EVC = EV / "corkscrew"
BF16_IFEVAL = 0.9094269870609981          # 官方 IFEval strict prompt-level
E28_QAT_IFEVAL = 0.8853974121996303       # Z1 天花板基準

# prereg 常數
Z1_FLOOR = 0.8853974121996303             # e33pre_9e 須 >
Z2_FLOOR = 0.90 * BF16_IFEVAL             # = 0.81848… e33_7e 須 ≥
Z5_BAND = (0.55, 0.95)
Z6_MARGIN = 0.01                          # +1.0 個百分點


def strict(tag):
    p = EV / "a1eval" / tag / "summary.json"
    if not p.exists():
        return None
    return json.load(open(p))["strict"]["prompt_level"]


def windows(tag):
    p = EVC / f"train_{tag}.json"
    if not p.exists():
        return None
    return json.load(open(p))


def z3_sweep2_value(w1, w2):
    """Z3:sweep2 各窗末 loss < sweep1 同窗末 loss(中位數比較)。

    操作化:逐窗取 loss[-1] 的比值 r_w = end2/end1,判準 = median(r) < 1。
    同時報 r<1 的窗數佔比作為佐證。窗以 window 欄位配對,不以序號。

    ⚠ 已知混淆(裁決時必須連同結果揭露,但不改判準——prereg 已凍結):
    sweep1 用 cal1、sweep2 用 cal2,兩迴的 loss 是在**不同語料**上量的,
    重建目標 Y 也隨語料改變,故 end2 與 end1 並非同尺度。Z3 因此無法乾淨
    分離「重退火的淨值」與「兩份語料的難度差」。若 Z3 PASS,只能宣告
    「sweep2 在其自身語料上的末 loss 低於 sweep1 在其語料上的末 loss」,
    不能直接宣告重退火有增益。E34 若要乾淨測,需在同一語料上跑第二迴,
    或把兩迴末權重都拿到同一份 held-out 上比。
    """
    if not w1 or not w2:
        return {"verdict": "N/A", "reason": "缺兩迴之一"}
    m1 = {tuple(x["window"]): x["loss"][-1] for x in w1["windows"]}
    m2 = {tuple(x["window"]): x["loss"][-1] for x in w2["windows"]}
    keys = sorted(set(m1) & set(m2))
    if not keys:
        return {"verdict": "N/A", "reason": "無共同窗"}
    r = sorted(m2[k] / m1[k] for k in keys if m1[k] > 0)
    med = r[len(r) // 2]
    return {"verdict": "PASS" if med < 1 else "FAIL",
            "median_ratio_end2_over_end1": med,
            "frac_windows_improved": sum(1 for x in r if x < 1) / len(r),
            "n_windows": len(r),
            "confound": "sweep1=cal1 / sweep2=cal2,不同語料不同尺度;"
                        "PASS 僅能宣告各自語料上的末 loss 高低"}


def z4_stability(runs, e2es):
    """Z4:兩迴窗 loss 全程有限、無帶化連發;e2e STE 段 kd < soft 末段 ×2。

    v2 追加:探索平台段(前 plateau_frac 比例的 epoch)允許高原,不判帶化。
    操作化:帶化比值改用 loss[-1] / loss[i0],i0 = ceil(plateau_frac*epochs)
    的索引(平台結束後第一個 epoch),而非 loss[0]。單窗比值 > 1.5 記一次
    帶化;**連發 = 連續 ≥2 個窗帶化**才判 FAIL(prereg 原文「無帶化連發」)。
    """
    out = {}
    bad = []
    for tag, w in runs.items():
        if not w:
            out[tag] = {"verdict": "N/A"}
            continue
        hp = w.get("hparams", {})
        ep = int(hp.get("epochs", 30))
        pf = float(hp.get("plateau_frac", 0.0) or 0.0)
        i0 = min(int(-(-pf * ep // 1)), ep - 1) if pf > 0 else 0
        band, nonfinite = [], 0
        for x in w["windows"]:
            L = x["loss"]
            if any((not isinstance(v, (int, float))) or v != v or
                   v in (float("inf"), float("-inf")) for v in L):
                nonfinite += 1
            j = min(i0, len(L) - 1)
            band.append(L[-1] > L[j] * 1.5)
        run = 0
        mx = 0
        for b in band:
            run = run + 1 if b else 0
            mx = max(mx, run)
        ok = nonfinite == 0 and mx < 2
        out[tag] = {"verdict": "PASS" if ok else "FAIL",
                    "plateau_exempt_epochs": i0, "n_banded": sum(band),
                    "max_consecutive_banded": mx, "n_nonfinite_windows": nonfinite}
        if not ok:
            bad.append(tag)
    for tag, e in e2es.items():
        if not e:
            out[tag] = {"verdict": "N/A"}
            continue
        kd = [x["kd"] for x in e.get("kd_log", [])]
        if not kd:
            out[tag] = {"verdict": "N/A", "reason": "無 kd_log"}
            continue
        # STE 段 = kd_log 後半;soft 末段 = 前半的末尾一成
        h = len(kd) // 2
        soft_tail = kd[max(0, h - max(1, h // 10)):h] or kd[:1]
        ste = kd[h:]
        ok = (sum(ste) / len(ste)) < 2 * (sum(soft_tail) / len(soft_tail))
        out[tag] = {"verdict": "PASS" if ok else "FAIL",
                    "ste_mean": sum(ste) / len(ste),
                    "soft_tail_mean": sum(soft_tail) / len(soft_tail)}
        if not ok:
            bad.append(tag)
    out["verdict"] = "FAIL" if bad else (
        "PASS" if any(v.get("verdict") == "PASS" for k, v in out.items()
                      if isinstance(v, dict)) else "N/A")
    out["failed"] = bad
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(EVC / "verdict_e33.json"))
    args = ap.parse_args()

    pre = strict("ifeval_e33pre_9e")
    pre2 = strict("ifeval_e33pre2_9e")
    main7 = strict("ifeval_e33_7e")

    Z = {}
    Z["Z1_pre_ceiling"] = {
        "verdict": "N/A" if pre is None else ("PASS" if pre > Z1_FLOOR else "FAIL"),
        "e33pre_9e_strict": pre, "floor_e28_qat": Z1_FLOOR,
        "delta_pp": None if pre is None else (pre - Z1_FLOOR) * 100}
    Z["Z2_main"] = {
        "verdict": "N/A" if main7 is None else
                   ("PASS" if main7 >= Z2_FLOOR else "FAIL"),
        "e33_7e_strict": main7, "floor_90pct": Z2_FLOOR,
        "retention_vs_bf16": None if main7 is None else main7 / BF16_IFEVAL,
        "delta_pp": None if main7 is None else (main7 - Z2_FLOOR) * 100,
        "linear_law_prediction_pct": 84.0}
    Z["Z3_sweep2_value"] = z3_sweep2_value(windows("st9"), windows("st9s2"))
    Z["Z3_sweep2_value_main7"] = z3_sweep2_value(windows("st7"), windows("st7s2"))
    Z["Z4_stability"] = z4_stability(
        {"st9": windows("st9"), "st9s2": windows("st9s2"),
         "st7": windows("st7"), "st7s2": windows("st7s2")},
        {"e2e_st9s2": windows("st9s2_e2e"), "e2e_st7s2": windows("st7s2_e2e"),
         "e2e_st9s2_sg": windows("st9s2_e2e_sg")})
    ret = None if main7 is None else main7 / BF16_IFEVAL
    Z["Z5_honest_band"] = {
        "verdict": "N/A" if ret is None else
                   ("PASS" if Z5_BAND[0] <= ret <= Z5_BAND[1] else "FAIL"),
        "retention": ret, "band": list(Z5_BAND),
        "note": "誠實帶:落帶外代表量測或管線有問題,不是好消息"}
    Z["Z6_selfgen"] = {
        "verdict": "N/A" if (pre is None or pre2 is None) else
                   ("PASS" if pre2 > pre + Z6_MARGIN else "FAIL"),
        "e33pre2_9e_strict": pre2, "e33pre_9e_strict": pre,
        "margin_pp": 1.0,
        "delta_pp": None if (pre is None or pre2 is None) else (pre2 - pre) * 100}

    # 四科錨定協定:humaneval 為第二判準(懸崖偵測器)
    hv = None
    p = EV / "a1eval" / "humaneval_e33_7e" / "summary.json"
    if p.exists():
        hv = json.load(open(p))["score"]
    Z["second_criterion_humaneval"] = {
        "e33_7e_humaneval": hv,
        "bf16": 0.957317,
        "retention": None if hv is None else hv / 0.957317,
        "verdict": "N/A" if hv is None else
                   ("PASS" if hv / 0.957317 >= 0.85 else "CLIFF"),
        "note": "非 prereg 判準(prereg 已凍結);四科錨定 @ab0adc5 立的"
                "讀出協定:<85% 判『表面過關實質已裂』,不得當再降一級的起點"}

    res = {"experiment": "E33 v2", "prereg": "prereg_e33_7lane_v2.json @ceff4ff",
           "bf16_ifeval_strict": BF16_IFEVAL, "criteria": Z}
    out = Path(args.out)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for k, v in Z.items():
        print(f"{k:<28} {v.get('verdict', '?')}")
    print("wrote", out)


if __name__ == "__main__":
    main()

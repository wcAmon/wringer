"""技術報告圖 1:comp vs body b/w,兩模型並排;零訓練各法連線、Wringer 與訓練式對照為點。數字取自 prereg_e70/e72/e73/e74 與 E69。"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
Q = {  # Qwen3-4B
 "GPTQ g128": [(2.141, 0.0), (3.148, 0.945)],
 "HQQ g64": [(2.500, 0.0), (3.500, 0.823)],
 "GGUF imatrix": [(2.519, 0.652), (2.727, 0.887), (3.066, 0.943)],
 "same grid, GPTQ + joint scales (P3)": [(2.638, 0.814)],
 "same grid, nearest + joint scales": [(2.638, 0.335)],
 "AWQ 4-bit (asym)": [(4.156, 0.989)],
}
A = {  # A1
 "GPTQ g128": [(2.141, 0.0), (3.148, 0.985)],
 "HQQ g64": [(2.500, 0.0), (3.500, 0.842)],
 "GGUF imatrix": [(2.558, 0.322), (2.763, 0.755), (2.972, 0.896), (3.295, 0.986)],
 "same grid, GPTQ + joint scales (P3)": [(2.655, 0.814)],
 "same grid, nearest + joint scales": [(2.655, 0.389)],
 "Wringer 8-level, zero-training + α polish": [(3.206, 0.980)],
}
TQ = {"Wringer (fill + wring)": (2.638, 0.922), "STE-QAT, same budget": (2.638, 0.858)}
TA = {"Wringer (fill + wring)": (2.655, 0.947), "STE-QAT, same budget": (2.655, 0.926)}
sty = {"GPTQ g128": ("tab:blue", "o-"), "HQQ g64": ("tab:purple", "s-"), "GGUF imatrix": ("tab:green", "^-"),
       "same grid, GPTQ + joint scales (P3)": ("tab:gray", "D"), "same grid, nearest + joint scales": ("tab:gray", "x"),
       "AWQ 4-bit (asym)": ("tab:cyan", "P"), "Wringer 8-level, zero-training + α polish": ("tab:olive", "v")}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, D, T, title in ((axes[0], Q, TQ, "Qwen3-4B (dense, 36 layers)"), (axes[1], A, TA, "Agents-A1-4B (hybrid GatedDeltaNet, 32 layers)")):
    for name, pts in D.items():
        c, m = sty[name]; xs, ys = zip(*pts)
        ax.plot(xs, ys, m, color=c, label=name, ms=7, lw=1.5, mfc="none" if m[0] in "Dx" else c)
    for name, (x, y) in T.items():
        if "Wringer" in name:
            ax.plot(x, y, "*", color="tab:red", ms=16, label=name, zorder=5)
        else:
            ax.plot(x, y, "h", color="tab:orange", ms=9, label=name, zorder=4)
    ax.axvspan(2.5, 2.75, color="tab:red", alpha=0.06)
    ax.set_xlim(2.0, 4.3); ax.set_ylim(-0.03, 1.03); ax.grid(alpha=0.3)
    ax.set_xlabel("bits per quantized body weight (codes + scales)"); ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7.2, loc="lower right")
axes[0].set_ylabel("comp = mean retention of IFEval / HumanEval / GSM8K vs bf16")
fig.suptitle("Zero-training methods fall off a cliff between 2.5 and 2.75 b/w; fill-and-wring places a point on the cliff at the level of the 3.1 b/w plateau", fontsize=10)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"evidence/p1_grouping/corkscrew/fig_collapse_curve.{ext}", dpi=160)
print("FIG_DONE")

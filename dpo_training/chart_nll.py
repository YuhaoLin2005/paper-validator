"""Generate per-token NLL comparison chart for DEV.to article.

Standalone CLI — zero project imports. matplotlib Agg backend only.
Writes results/per_token_nll_comparison.png for article embedding.
Data hardcoded from verified analyze_logprobs.py output.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 10,
    "figure.dpi": 200, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

DOMAINS = ["Overclaim\nResistance", "Privacy\nAwareness", "Bias\nDetection",
           "Architecture\nDecision", "Knowledge\nInsight"]

BASE_NLL = [2.7044, 2.8777, 3.4149, 3.6414, 3.5278]
DPO_NLL  = [2.4244, 2.7450, 3.3945, 3.3226, 3.3865]
REDUCTIONS = [(b - d) / b * 100 for b, d in zip(BASE_NLL, DPO_NLL)]

CB = "#E67E22"; CD = "#2ECC71"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                gridspec_kw={'width_ratios': [2, 1]})

x = np.arange(len(DOMAINS)); w = 0.35
ax1.bar(x - w/2, BASE_NLL, w, label="Base (Qwen2.5-1.5B)",
        color=CB, alpha=0.85, edgecolor="white", linewidth=0.5)
ax1.bar(x + w/2, DPO_NLL, w, label="DPO (Causal-trained)",
        color=CD, alpha=0.85, edgecolor="white", linewidth=0.5)

for i, (bv, dv, red) in enumerate(zip(BASE_NLL, DPO_NLL, REDUCTIONS)):
    ax1.text(i - w/2, bv + 0.03, f"{bv:.2f}", ha="center", fontsize=8, c="#888")
    ax1.text(i + w/2, dv - 0.12, f"{dv:.2f}", ha="center", fontsize=8,
             c="#1a7a3a", fontweight="bold")
    ax1.text(i, max(bv, dv) + 0.08, f"-{red:.1f}%", ha="center", fontsize=8,
             c="#E74C3C", fontweight="bold")

ax1.set_xticks(x); ax1.set_xticklabels(DOMAINS, fontsize=9)
ax1.set_ylabel("Per-Token NLL (lower = more probable)", fontsize=10)
ax1.set_title("DPO Reduces Per-Token NLL on Causal Responses",
              fontweight="bold", fontsize=11)
ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)
ax1.grid(axis="y", alpha=0.2); ax1.set_ylim(2.2, 4.0)

mean_base = np.mean(BASE_NLL); mean_dpo = np.mean(DPO_NLL)
mean_red = (mean_base - mean_dpo) / mean_base * 100; mw = 0.5
ax2.bar([0], [mean_base], mw, color=CB, alpha=0.85,
        edgecolor="white", linewidth=0.5)
ax2.bar([1], [mean_dpo], mw, color=CD, alpha=0.85,
        edgecolor="white", linewidth=0.5)
ax2.text(0, mean_base + 0.03, f"{mean_base:.2f}", ha="center", fontsize=11, c="#888")
ax2.text(1, mean_dpo - 0.15, f"{mean_dpo:.2f}", ha="center", fontsize=11,
         c="#1a7a3a", fontweight="bold")
ax2.annotate(f"-{mean_red:.1f}%", xy=(1, mean_dpo), xytext=(0.5, 3.55),
            ha="center", fontsize=12, c="#E74C3C", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.5))
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Base", "DPO"], fontsize=10)
ax2.set_ylabel("Mean Per-Token NLL", fontsize=10)
ax2.set_title("Mean Across 5 Domains", fontweight="bold", fontsize=11)
ax2.grid(axis="y", alpha=0.2); ax2.set_ylim(2.8, 3.7)

fig.suptitle("Per-Token NLL: Base vs DPO Model", fontweight="bold",
             fontsize=14, y=0.98)

os.makedirs("results", exist_ok=True)
outpath = os.path.join("results", "per_token_nll_comparison.png")
fig.savefig(outpath, facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved: {outpath}")

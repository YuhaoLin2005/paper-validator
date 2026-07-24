"""Generate charts for DEV.to article from comparison JSON.

Usage:
    python dpo_training/visualize.py [comparison_json_path]
    If no path given, uses latest results/comparison_*.json
"""

import json, os, sys
from pathlib import Path
from glob import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = {"base": "#E74C3C", "dpo": "#2ECC71"}
DOMAIN_LABELS = {
    "ethics_overclaim": "Overclaim\nResistance",
    "ethics_privacy": "Privacy\nAwareness",
    "ethics_bias": "Bias\nDetection",
    "architecture_decision": "Architecture\nDecision",
    "knowledge_insight": "Knowledge\nInsight",
}


def load_data(path=None):
    if path:
        p = Path(path)
    else:
        candidates = sorted(glob("results/comparison_*.json"))
        if not candidates:
            print("ERROR: No comparison JSON found in results/")
            sys.exit(1)
        p = Path(candidates[-1])
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f), p.stem


def chart1_radar(data, outdir):
    """CIS 3-axis comparison: Behavioral / Discrimination / OOD Causal."""
    fig, ax = plt.subplots(figsize=(8, 6))

    metrics = ["Behavioral\nCompliance", "Discrimination\nAccuracy", "OOD Causal\nRate"]
    base_vals = [
        data["base"]["behavioral_compliance"],
        data["base"]["discrimination_accuracy"],
        data["base"]["ood_avg_causal"] / 4.0,
    ]
    dpo_vals = [
        data["dpo"]["behavioral_compliance"],
        data["dpo"]["discrimination_accuracy"],
        data["dpo"]["ood_avg_causal"] / 4.0,
    ]

    x = np.arange(len(metrics))
    w = 0.35
    bars1 = ax.bar(x - w/2, base_vals, w, label="Base (Qwen2.5-1.5B)", color=COLORS["base"], alpha=0.85)
    bars2 = ax.bar(x + w/2, dpo_vals, w, label="DPO (Causal-trained)", color=COLORS["dpo"], alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score")
    ax.set_title("Base vs DPO: 3-Axis Performance Comparison", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(outdir, "chart1_cis_axes.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def chart2_behavioral_detail(data, outdir):
    """Per-behavioral-test hit rate comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    tests = ["B1_read_after_write", "B2_overclaim", "B3_insight_deposit", "B4_decision_log"]
    labels = ["Read-after-Write", "Overclaim\nResistance", "Insight\nDeposit", "Decision\nRationale"]

    x = np.arange(len(tests))
    w = 0.35

    base_hits = [1 if data["base"]["behavioral"][t]["hit"] else 0 for t in tests]
    dpo_hits = [1 if data["dpo"]["behavioral"][t]["hit"] else 0 for t in tests]

    ax.bar(x - w/2, base_hits, w, label="Base Model", color=COLORS["base"], alpha=0.85)
    ax.bar(x + w/2, dpo_hits, w, label="DPO Model", color=COLORS["dpo"], alpha=0.85)

    for i, (bh, dh) in enumerate(zip(base_hits, dpo_hits)):
        if bh:
            ax.text(i - w/2, bh + 0.02, "✓", ha="center", fontsize=14, color=COLORS["base"])
        else:
            ax.text(i - w/2, 0.1, "✗", ha="center", fontsize=14, color=COLORS["base"])
        if dh:
            ax.text(i + w/2, dh + 0.02, "✓", ha="center", fontsize=14, color=COLORS["dpo"], fontweight="bold")
        else:
            ax.text(i + w/2, 0.1, "✗", ha="center", fontsize=14, color=COLORS["dpo"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.4)
    ax.set_ylabel("Behavioral Signal Detected")
    ax.set_title("Behavioral Test Results: Base vs DPO", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    base_cis = data["base"]["cis"]["score"]
    dpo_cis = data["dpo"]["cis"]["score"]
    ax.text(0.98, 0.92, f"CIS: {base_cis:.0f} -> {dpo_cis:.0f} (+{dpo_cis - base_cis:.0f})",
            transform=ax.transAxes, ha="right", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    path = os.path.join(outdir, "chart2_behavioral_detail.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def chart3_logprob_gate(data, outdir):
    """Logprob ratio: neural preference for causal over direct."""
    fig, ax = plt.subplots(figsize=(10, 5))

    base_lp = data["base"]["logprob"]["details"]
    dpo_lp = data["dpo"]["logprob"]["details"]

    domains = [d["domain"] for d in base_lp]
    domain_labels = [DOMAIN_LABELS.get(d, d) for d in domains]
    base_ratios = [d["ratio"] for d in base_lp]
    dpo_ratios = [d["ratio"] for d in dpo_lp]

    x = np.arange(len(domains))
    w = 0.35

    ax.bar(x - w/2, base_ratios, w, label="Base Model", color=COLORS["base"], alpha=0.85)
    ax.bar(x + w/2, dpo_ratios, w, label="DPO Model", color=COLORS["dpo"], alpha=0.85)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(domain_labels, fontsize=9)
    ax.set_ylabel("Logprob Ratio\n(positive = prefers causal)")
    ax.set_title("Neural Preference Gate: Logprob(chosen) - Logprob(rejected)", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    n_base = data["base"]["logprob"]["n_prefer_chosen"]
    n_dpo = data["dpo"]["logprob"]["n_prefer_chosen"]
    ax.text(0.98, 0.95, f"Prefers causal: Base={n_base}/5  DPO={n_dpo}/5",
            transform=ax.transAxes, ha="right", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    path = os.path.join(outdir, "chart3_logprob_gate.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def chart4_cis_scorecard(data, outdir):
    """CIS breakdown card — both models side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, key, title in [
        (axes[0], "base", "Base Model\n(Qwen2.5-1.5B-Instruct)"),
        (axes[1], "dpo", "DPO Model\n(+ Causal Preference Training)"),
    ]:
        r = data[key]
        cis = r["cis"]["score"]
        ood = r["ood_avg_causal"]
        disc = r["discrimination_accuracy"]
        behav = r["behavioral_compliance"]

        sizes = [ood * 0.4 * 100, disc * 0.3 * 100, behav * 0.3 * 100]
        colors_slice = ["#3498DB", "#F39C12", "#2ECC71"]

        wedges, texts = ax.pie(sizes, labels=None, colors=colors_slice,
                               startangle=90, wedgeprops=dict(width=0.4, edgecolor="white"))
        ax.text(0, 0, f"CIS\n{cis:.0f}", ha="center", va="center", fontsize=18, fontweight="bold")
        ax.set_title(title, fontweight="bold", fontsize=12)

    labels_legend = ["OOD Causal (x0.4)", "Discrimination (x0.3)", "Behavioral (x0.3)"]
    fig.legend(wedges, labels_legend, loc="lower center", ncol=3, framealpha=0.9)
    fig.suptitle("CIS (Causal Internalization Score) Breakdown", fontweight="bold", fontsize=14)

    path = os.path.join(outdir, "chart4_cis_scorecard.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def chart5_causal_density(data, outdir):
    """Causal pattern density: avg causal patterns per response across conditions."""
    fig, ax = plt.subplots(figsize=(9, 5))

    conditions = ["Behavioral", "Discrimination", "OOD (Ethics)"]
    base_causal = [
        np.mean([v["causal"] for v in data["base"]["behavioral"].values()]),
        np.mean([v["causal"] for v in data["base"]["discrimination"].values()]),
        np.mean([v["causal"] for v in data["base"]["ood"].values()]),
    ]
    dpo_causal = [
        np.mean([v["causal"] for v in data["dpo"]["behavioral"].values()]),
        np.mean([v["causal"] for v in data["dpo"]["discrimination"].values()]),
        np.mean([v["causal"] for v in data["dpo"]["ood"].values()]),
    ]

    x = np.arange(len(conditions))
    w = 0.35

    ax.bar(x - w/2, base_causal, w, label="Base Model", color=COLORS["base"], alpha=0.85)
    ax.bar(x + w/2, dpo_causal, w, label="DPO Model", color=COLORS["dpo"], alpha=0.85)

    for i, (bv, dv) in enumerate(zip(base_causal, dpo_causal)):
        ax.text(i - w/2, bv + 0.05, f"{bv:.1f}", ha="center", fontsize=10)
        ax.text(i + w/2, dv + 0.05, f"{dv:.1f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Avg Causal Patterns per Response")
    ax.set_title("Causal Reasoning Density by Condition", fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(outdir, "chart5_causal_density.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    data, stem = load_data(sys.argv[1] if len(sys.argv) > 1 else None)
    outdir = f"results/charts_{stem}"
    os.makedirs(outdir, exist_ok=True)

    print(f"Generating charts -> {outdir}/")
    chart1_radar(data, outdir)
    chart2_behavioral_detail(data, outdir)
    chart3_logprob_gate(data, outdir)
    chart4_cis_scorecard(data, outdir)
    chart5_causal_density(data, outdir)

    print(f"\nDone! {len(os.listdir(outdir))} charts saved to {outdir}/")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    main()

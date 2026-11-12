"""Website figures for the uncertainty-wire articles, from the saved metric tables.

    python scripts/website_figures.py --out figures_web
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, YELLOW, RED, GREEN = "#1F4E9C", "#E3A800", "#C8102E", "#5FA65A"
INK, MUTED, GRID = "#0f1830", "#55617c", "#e4e6ec"


def rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def style(ax):
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def team_ece(out):
    r = {(x["regime"], x["policy"]): float(x["ece"]) for x in rows(ROOT / "outputs" / "metrics_summary.csv")}
    policies = [("label_only_majority", "Majority\nvote"), ("confidence_weighted_raw", "Confidence\naverage"),
                ("confidence_weighted_calibrated", "Confidence average\n(calibrated)"),
                ("independent_log_odds_raw", "Log-odds"), ("independent_log_odds_calibrated", "Log-odds\n(calibrated)")]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    w = 0.36
    for i, (regime, label, colour) in enumerate((("calibrated", "Honest confidence", BLUE),
                                                ("overconfident", "Overconfident (+0.18)", YELLOW))):
        vals = [r[(regime, p)] for p, _ in policies]
        xs = [k + (i - 0.5) * w for k in range(len(policies))]
        bars = ax.bar(xs, vals, w, color=colour, label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3, color=INK)
    ax.set_xticks(range(len(policies)), [n for _, n in policies], fontsize=8)
    ax.set_ylabel("Calibration error (ECE), lower is better", color=INK)
    ax.set_title("Team calibration error for five ways to combine three agents", loc="left", fontsize=11, color=INK)
    ax.set_ylim(0, 0.19)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "agent-uncertainty-wire.png", dpi=150)
    plt.close(fig)


def inflation(out):
    by = defaultdict(dict)
    for x in rows(ROOT / "experiments" / "confidence_inflation" / "outputs" / "metrics_summary.csv"):
        by[(x["inflation"], x["seed"])][x["method"]] = float(x["brier"])
    levels = sorted({k[0] for k in by}, key=float)
    gain = [mean(v["raw_confidence"] - v["calibration_corrected"] for k, v in by.items() if k[0] == lv) for lv in levels]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar([str(float(lv)) if float(lv) else "0.0" for lv in levels], gain, 0.6, color=BLUE)
    ax.bar_label(bars, labels=["%.5f" % g for g in gain],
                 fontsize=8, padding=3, color=INK)
    ax.set_xlabel("Confidence added to every agent (inflation)", color=INK)
    ax.set_ylabel("Brier score reduction after the fix\n(higher is better, mean of 5 seeds)", color=INK)
    ax.set_title("The one-number fix helped at every level, but by less than 0.001", loc="left", fontsize=11, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "confidence-inflation-calibration.png", dpi=150)
    plt.close(fig)
    return levels, gain


def correlated(out):
    acc = defaultdict(list)
    for x in rows(ROOT / "experiments" / "correlated_errors" / "outputs" / "metrics_summary.csv"):
        if x["policy"] == "majority":
            acc[float(x["q"])].append(float(x["accuracy"]))
    qs = sorted(acc)
    ys = [100 * mean(acc[q]) for q in qs]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(qs, ys, color=BLUE, marker="o", lw=2, label="Majority vote of three agents")
    for q, y in zip(qs, ys):
        ax.annotate("%.1f%%" % y, (q, y), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.axhline(80, color=RED, ls="--", lw=1.4, label="Best single agent's set skill (80%)")
    ax.set_xticks(qs)
    ax.set_xlabel("Chance that all agents share the same failure (q)", color=INK)
    ax.set_ylabel("Accuracy (%), mean of 5 seeds", color=INK)
    ax.set_ylim(65, 83)
    ax.set_title("Majority vote got worse as the agents failed together", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, loc="lower left")
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "correlated-errors-multi-agent-decisions.png", dpi=150)
    plt.close(fig)
    return qs, ys


def left_right(out):
    r = {(x["agent"], x["lang"]): float(x["accuracy"]) for x in
         rows(ROOT / "experiments" / "real_llm_confidence" / "outputs" / "agent_metrics.csv")}
    agents = [("qwen_plain", "qwen2.5:3b"), ("qwen_checker", "qwen2.5:3b\n(checker prompt)"), ("llama_plain", "llama3.1:8b")]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    w = 0.36
    for i, (lang, label, colour) in enumerate((("en", "English", BLUE), ("ru", "Roman Urdu", YELLOW))):
        vals = [100 * r[(a, lang)] for a, _ in agents]
        bars = ax.bar([k + (i - 0.5) * w for k in range(len(agents))], vals, w, color=colour, label=label)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=3, color=INK)
    ax.set_xticks(range(len(agents)), [n for _, n in agents], fontsize=8)
    ax.set_ylim(70, 103)
    ax.set_ylabel("Questions answered correctly (%), 48 per language", color=INK)
    ax.set_title("All three models did worse on the same notes in Roman Urdu", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "roman-urdu-left-right-confidence.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures_web")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    team_ece(out)
    print("inflation", inflation(out))
    print("correlated", correlated(out))
    left_right(out)


if __name__ == "__main__":
    main()

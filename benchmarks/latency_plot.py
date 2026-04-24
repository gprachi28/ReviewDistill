"""
Latency benchmark visualisation — EXP-010 through EXP-015.

Per-stage stacked breakdown across the 6 structured latency experiments.
EXP-001–009 were quality/prompt experiments; latency was not the focus there.

Run: PYTHONPATH=. python benchmarks/latency_plot.py
Output: benchmarks/latency_benchmarks.png
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────

# Per-stage breakdown — steady-state latency (avg of 2nd & 3rd queries), ms.
# SQL + meta always <10ms — merged into one segment for clarity.
# Columns: [planner, retrieval, synthesizer, sql+meta]
STAGES = ["Query Planner", "Retrieval", "Synthesizer", "SQL + Meta"]
STAGE_COLORS = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#95E1D3"]

server_exps = [
    "EXP-010\nFull precision\nmodel",
    "EXP-011\n4-bit\nquantization",
    "EXP-012\nmax_tokens\n+ stop seq",
    "EXP-013\nCache\ncollection.count()",
    "EXP-014\nStartup\nwarmup",
    "EXP-015\nHNSW index\npre-load",
]
server_stages = np.array([
    #  planner  retrieval  synthesizer  sql+meta
    [  2_808,    2_136,     13_870,        7  ],   # EXP-010
    [    929,      996,      5_109,        3  ],   # EXP-011
    [    887,    1_255,      4_543,        3  ],   # EXP-012
    [    878,      933,      2_161,        1  ],   # EXP-013
    [    857,    1_108,      2_071,        2  ],   # EXP-014
    [    860,    1_046,      2_107,        2  ],   # EXP-015
], dtype=float)

# ── Figure setup ──────────────────────────────────────────────────────────────

BG    = "#ffffff"
PANEL = "#f6f8fa"
GRID  = "#d0d7de"
TEXT  = "#1f2328"
MUTED = "#57606a"

plt.rcParams.update({
    "font.family":     "monospace",
    "text.color":      TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color":     TEXT,
    "ytick.color":     TEXT,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
fig.suptitle(
    "Yelp Conversational Assistant — Latency Optimisation",
    fontsize=14, fontweight="bold", color=TEXT, y=1.01,
)

# ── Stacked bars ──────────────────────────────────────────────────────────────

ax.set_facecolor(PANEL)
for sp in ax.spines.values():
    sp.set_color(GRID)

x = np.arange(len(server_exps))
bar_w = 0.52
bottoms = np.zeros(len(server_exps))

for stage, color in zip(STAGES, STAGE_COLORS):
    i = STAGES.index(stage)
    vals_s = server_stages[:, i] / 1000
    ax.bar(x, vals_s, bar_w, bottom=bottoms, label=stage,
           color=color, alpha=0.88, edgecolor=BG, linewidth=0.7, zorder=3)
    for j, (val, bot) in enumerate(zip(vals_s, bottoms)):
        if val > 0.35:
            ax.text(j, bot + val / 2, f"{val:.1f}s",
                    ha="center", va="center", fontsize=9, color=TEXT, fontweight="bold")
    bottoms += vals_s

# Total labels above bars
totals_s = server_stages.sum(axis=1) / 1000
for j, tot in enumerate(totals_s):
    ax.text(j, tot + 0.25, f"{tot:.1f}s", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=TEXT)

# Speedup badges below x-axis
speedup_base = totals_s[0]
ax.text(0, -1.6, "baseline", ha="center", va="top", fontsize=9, color=MUTED)
for j, tot in enumerate(totals_s[1:], start=1):
    ax.text(j, -1.6, f"{speedup_base / tot:.1f}×", ha="center", va="top",
            fontsize=10, color="#b45309", fontweight="bold")
ax.text(len(server_exps) / 2 - 0.5, -2.9,
        "↑ cumulative speedup vs EXP-010 baseline",
        ha="center", fontsize=9, color=MUTED, style="italic")

ax.set_xticks(x)
ax.set_xticklabels(server_exps, fontsize=10)
ax.set_ylabel("Latency (seconds)", fontsize=11)
ax.set_ylim(-3.8, totals_s[0] + 2.5)
ax.yaxis.grid(True, color=GRID, linewidth=0.6, linestyle="--", zorder=0)
ax.set_title(
    "Per-stage breakdown  |  mlx_lm.server (Apple M4 Pro, MPS)  |  "
    "Steady-state: avg of queries 2 & 3\n"
    "(Query 1 excluded — absorbs mlx_lm.server KV-cache + embedding model cold-start)",
    fontsize=10, color=TEXT, pad=10,
)

ax.legend(loc="upper right", facecolor=PANEL, edgecolor=GRID,
          labelcolor=TEXT, fontsize=10, framealpha=0.9)

# ── Save ─────────────────────────────────────────────────────────────────────

plt.tight_layout()
out = "benchmarks/latency_benchmarks.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
print(f"Saved → {out}")

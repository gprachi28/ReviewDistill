"""
Latency benchmark visualisation — EXP-001 through EXP-015.

Two panels:
  1. End-to-end warm latency journey across all timed experiments (log scale)
  2. Per-stage stacked breakdown for the mlx_lm.server era (EXP-010 → EXP-015)

Run: PYTHONPATH=. python benchmarks/latency_plot.py
Output: benchmarks/latency_benchmarks.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────

# End-to-end warm latency in ms across all timed experiments.
# EXP-002/006/007 had no timing; EXP-008/009 were eval-only — skipped.
# EXP-003–005 used mlx_lm locally; values from experiments.md comments.
# EXP-010–015 used mlx_lm.server; warm = avg of Q2/Q3 (Q1 absorbs first-call overhead).
journey = [
    ("EXP-001\nBaseline\nmlx_lm",          172_000),
    ("EXP-003\nTimeout fix\n+ shorter q",    95_000),
    ("EXP-004\nCtx cap\n(max 5 biz)",       102_000),
    ("EXP-005\nSemantic\ndecoupled",         118_000),
    ("EXP-010\n→ mlx_lm.server\nfull model",   19_820),
    ("EXP-011\n4-bit\nquant",                 7_036),
    ("EXP-012\nmax_tokens\n+ stop seq",       6_687),
    ("EXP-013\nCache\ncollection.count()",    3_976),
    ("EXP-014\nStartup\nwarmup",              4_038),
    ("EXP-015\nHNSW\npre-load",              4_015),
]
exp_labels, latencies = zip(*journey)
latencies = list(latencies)

# Per-stage breakdown — warm avg (Q2+Q3), ms.  SQL+meta merged (always <10ms).
# Columns: [planner, retrieval, synthesizer, sql+meta]
STAGES = ["Query Planner", "Retrieval", "Synthesizer", "SQL + Meta"]
STAGE_COLORS = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#95E1D3"]

server_exps = [
    "EXP-010\nFull model",
    "EXP-011\n4-bit quant",
    "EXP-012\nmax_tokens",
    "EXP-013\nCache count()",
    "EXP-014\nStartup warmup",
    "EXP-015\nHNSW pre-load",
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

BG      = "#ffffff"
PANEL   = "#f6f8fa"
GRID    = "#d0d7de"
TEXT    = "#1f2328"
MUTED   = "#57606a"

plt.rcParams.update({
    "font.family": "monospace",
    "text.color":  TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

fig = plt.figure(figsize=(16, 11), facecolor=BG)
fig.suptitle(
    "Yelp Conversational Assistant — Latency Optimisation (15 Experiments)",
    fontsize=14, fontweight="bold", color=TEXT, y=0.98,
)

ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

# ── Panel 1: Journey (log scale) ─────────────────────────────────────────────

ax1.set_facecolor(PANEL)
for sp in ax1.spines.values():
    sp.set_color(GRID)

x = np.arange(len(exp_labels))
lat_s = [v / 1000 for v in latencies]

# Colour gradient along the line
cmap = plt.cm.plasma
norm_x = np.linspace(0.15, 0.95, len(x))
seg_colors = [cmap(v) for v in norm_x]

for i in range(len(x) - 1):
    ax1.semilogy(
        x[i : i + 2], lat_s[i : i + 2],
        color=seg_colors[i], linewidth=2.8, solid_capstyle="round", zorder=2,
    )

for i, (xi, yi) in enumerate(zip(x, lat_s)):
    ax1.scatter(xi, yi, color=seg_colors[i], s=110, zorder=4,
                edgecolors="white", linewidth=0.9)
    offset = 6 if i % 2 == 0 else -14
    ax1.annotate(
        f"{yi:.1f}s",
        (xi, yi),
        textcoords="offset points",
        xytext=(0, offset + 5),
        ha="center", fontsize=8.5, fontweight="bold", color=TEXT,
    )

# mlx_lm.server switch boundary
ax1.axvline(x=4.5, color="#FFD700", linestyle="--", linewidth=1.6, alpha=0.8)
ax1.text(4.55, 80, "  → switch to mlx_lm.server", color="#b45309", fontsize=8.5, va="top",
         style="italic")

# 15-second target
ax1.axhline(y=15, color="#15803d", linestyle=":", linewidth=1.5, alpha=0.85)
ax1.text(x[-1] + 0.08, 15, "  15s target", color="#15803d", fontsize=8,
         va="center")

# 97% reduction annotation
ax1.annotate(
    "97% reduction\n172s → 4.0s",
    xy=(x[-1], lat_s[-1]),
    xytext=(x[-1] - 1.5, lat_s[-1] * 6),
    arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.4),
    color="#dc2626", fontsize=8.5, fontweight="bold", ha="center",
)

ax1.set_xticks(x)
ax1.set_xticklabels(exp_labels, fontsize=7.5)
ax1.set_ylabel("End-to-End Latency (seconds, log scale)", fontsize=10)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}s"))
ax1.yaxis.grid(True, color=GRID, linewidth=0.6, linestyle="--", which="both")
ax1.set_xlim(-0.6, len(x) - 0.4)
ax1.set_title("End-to-End Warm Latency — All 10 Timed Experiments", fontsize=11,
              fontweight="bold", color=TEXT, pad=8)

# ── Panel 2: Per-stage stacked bars ──────────────────────────────────────────

ax2.set_facecolor(PANEL)
for sp in ax2.spines.values():
    sp.set_color(GRID)

x2 = np.arange(len(server_exps))
bar_w = 0.52
bottoms = np.zeros(len(server_exps))

for i, (stage, color) in enumerate(zip(STAGES, STAGE_COLORS)):
    vals_s = server_stages[:, i] / 1000
    bars = ax2.bar(x2, vals_s, bar_w, bottom=bottoms, label=stage,
                   color=color, alpha=0.88, edgecolor=BG, linewidth=0.7, zorder=3)
    # Label inside bar if tall enough
    for j, (val, bot) in enumerate(zip(vals_s, bottoms)):
        if val > 0.3:
            ax2.text(
                j, bot + val / 2,
                f"{val:.1f}s",
                ha="center", va="center", fontsize=8, color="#1f2328", fontweight="bold",
            )
    bottoms += vals_s

# Total labels above bars
totals_s = server_stages.sum(axis=1) / 1000
for j, tot in enumerate(totals_s):
    ax2.text(j, tot + 0.25, f"{tot:.1f}s", ha="center", va="bottom",
             fontsize=9.5, fontweight="bold", color=TEXT)

# Speedup badges
speedup_base = totals_s[0]
for j, tot in enumerate(totals_s[1:], start=1):
    sx = speedup_base / tot
    ax2.text(j, -1.8, f"{sx:.1f}×", ha="center", va="top",
             fontsize=9, color="#b45309", fontweight="bold")
ax2.text(0, -1.8, "baseline", ha="center", va="top", fontsize=9, color=MUTED)
ax2.text(len(server_exps) / 2 - 0.5, -3.2, "↑ cumulative speedup vs EXP-010 full model",
         ha="center", fontsize=8.5, color=MUTED, style="italic")

ax2.set_xticks(x2)
ax2.set_xticklabels(server_exps, fontsize=9)
ax2.set_ylabel("Latency (seconds)", fontsize=10)
ax2.set_ylim(-4, totals_s[0] + 2.5)
ax2.yaxis.grid(True, color=GRID, linewidth=0.6, linestyle="--", zorder=0)
ax2.set_title(
    "Per-Stage Breakdown — mlx_lm.server Era  |  Steady-state latency (avg of 2nd & 3rd queries;\n"
    "1st query excluded — absorbs mlx_lm.server KV-cache + embedding model cold-start overhead)",
    fontsize=10, fontweight="bold", color=TEXT, pad=8,
)

legend = ax2.legend(
    loc="upper right", facecolor=PANEL, edgecolor=GRID,
    labelcolor=TEXT, fontsize=10, framealpha=0.9,
)

# ── Final layout ─────────────────────────────────────────────────────────────

plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3.5)
out = "benchmarks/latency_benchmarks.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
print(f"Saved → {out}")

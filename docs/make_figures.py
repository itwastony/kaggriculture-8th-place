#!/usr/bin/env python3
"""Rebuild the data figures and three diagrams of docs/ from the logs in logs/ (no games, no network).

    python docs/make_figures.py        # needs matplotlib

Writes docs/images/{pipeline,appo,d7b_training,league_mix,bc_dip,lineage,selection,live_ratings}.png:
- pipeline, appo, lineage: diagrams (text and numbers written in this script, from docs/method.md);
- d7b_training, league_mix, bc_dip: from the logs in logs/;
- selection: the numbers of the final evaluation (docs/evaluation.md, section 3), written in this script;
- live_ratings: logs/live_ratings.csv (hourly Kaggle API snapshots of the two final submissions after the deadline).
The other images in docs/images (hindsight_label, model) are kept as images: two diagrams.
All curves come from TRAINING games of the final run d7b: the learning side samples its actions and PFSP shifts the
opponent mix, so they are not an evaluation. The logs are aggregated (blocks of iterations, rounded values), see
logs/README.md. The output is deterministic for a given matplotlib version (no time stamp in the PNG files).
"""

import csv
import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
LOGS, OUT = ROOT / "logs", ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8c8b86", "#e1e0d9", "#fcfcfb"
BOXF, BOXE = "#eef3fb", "#9db8dd"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
        "savefig.facecolor": SURF,
        "lines.linewidth": 2,
    }
)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def box(ax, x, y, w, h, text, fc=BOXF, ec=BOXE, fs=8.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    head, *rest = text.split("\n")
    ax.text(x + w / 2, y + h - 0.22, head, ha="center", va="top", fontsize=fs + 0.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.55, "\n".join(rest), ha="center", va="top", fontsize=fs, color=INK2, linespacing=1.35)


def arrow(ax, a, b, color=INK2, lw=1.4, rad=0.0, ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=13,
            color=color,
            lw=lw,
            connectionstyle=f"arc3,rad={rad}",
            linestyle=ls,
        )
    )


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set(xlim=(0, w), ylim=(0, h))
    ax.axis("off")
    return fig, ax


# ---------- pipeline (diagram of README.md and docs/method.md) ----------
fig, ax = canvas(15, 3.9)
ax.text(0.1, 3.85, "Pipeline", fontsize=13, fontweight="bold", va="top")
steps = [
    "Top-team games\nKaggle daily episode\ndatasets + replays\n(read-only)",
    "Tape worlds\nre-simulate seed, shops\nand both players' actions;\nkeep only if both final\nbanks match to the $",
    "BC\nstrong teams' seats;\nunit label = (target tile,\noperation) in hindsight;\n"
    "later lines start from a\nrecent RL model of ours",
    "Short APPO league\nself / past / best\n+ 101 code bots\n+ 9,954 tape seats; PFSP;\n"
    "reward = outcome; KL to a\nratchet anchor",
    "Select & submit\nEMA snapshots →\ngates written before the\nevaluation (LB field,\n"
    "head-to-head) → NumPy\nagent, fp16 weights",
]
for k, text in enumerate(steps):
    box(ax, 0.1 + k * 2.95, 1.35, 2.75, 2.0, text, fs=9)
    if k < len(steps) - 1:
        arrow(ax, (0.1 + k * 2.95 + 2.77, 2.35), (0.1 + (k + 1) * 2.95 - 0.02, 2.35))
arrow(ax, (13.27, 1.3), (7.4, 1.3), color=ORANGE, rad=-0.3, lw=1.8)
ax.text(
    9.6,
    0.1,
    "the top meta changes → BC on the newest top games, starting from a recent RL model → short APPO (almost daily)",
    ha="center",
    fontsize=9.5,
    color=INK2,
)
save(fig, "pipeline")

# ---------- APPO loop (diagram of rl/appo.py with the flags of configs/appo_d7b.json) ----------
fig, ax = canvas(15, 6.6)
ax.text(
    0.1,
    6.5,
    "APPO league: who plays, what is optimised, what keeps it stable",
    fontsize=13,
    fontweight="bold",
    va="top",
)
box(
    ax,
    0.1,
    3.6,
    4.2,
    2.2,
    "GPU actors (every rank)\n128 games per rank on a batched CUDA engine\n"
    "opponent: self 0.4 / past 0.5 / best 0.1\npast (argmax): own snapshots every 10 iters\n"
    "+ earlier lines' snapshots + 5 fixed checkpoints,\nPFSP sampling (1−P)²",
)
box(
    ax,
    0.1,
    0.6,
    4.2,
    2.6,
    "CPU actors (26 processes per rank)\n208 games per rank on kagsim (C++, bit-exact)\n"
    "101 code bots (ours + public notebooks)\n6 tape-world pools, 9,954 seats: top games\n"
    "of Sep 27–30, fast opening, weak field\nPFSP hard: w·max(0.1, (1−P)²)",
)
box(
    ax,
    5.3,
    1.45,
    4.3,
    3.95,
    "Learner (DDP over all ranks)\nbehaviour policy is 1 update old\n"
    "R = sign(d) + 0.15·clip(d/30k, ±1), d = Δbank\nA = GAE(λ 0.95, γ 1) + UPGO\n"
    "per-component PPO: clip 0.2, dual-clip 3\n+ KL(π‖anchor): 0.02 → 0.005 over 200 iters\n"
    "entropy 0; 2 epochs × 104 minibatches per rank\nearly epoch stop: k3-KL(behaviour‖current)\n"
    "of any head > 1 or joint > 5× median\nvalue: detached + privileged critic, lr 1e-4",
    fc="#e6eef9",
    fs=8.6,
)
box(
    ax,
    10.6,
    3.9,
    4.3,
    2.0,
    "EMA 0.999\nweight average after every step;\nema_XXXXX.pt every 10 iterations —\n"
    "these are what we evaluate and submit",
)
box(
    ax,
    10.6,
    0.6,
    4.3,
    2.9,
    "Anchor gate (ratchet)\nanchor := current policy only if it beats\nthe anchor (P ≥ 0.55 over a short window\n"
    "of iterations) and its win rate against the\nfield is at most 0.03 below the anchor's.\n"
    "Anchor = KL target and the 'best' opponent.",
)
arrow(ax, (4.32, 4.75), (5.28, 4.2))
arrow(ax, (4.32, 1.9), (5.28, 2.6))
ax.text(4.8, 3.4, "trajectories", ha="center", fontsize=8.5, color=INK2, rotation=90)
arrow(ax, (9.62, 4.4), (10.58, 4.9))
arrow(ax, (9.62, 2.6), (10.58, 2.0))
# The anchor is the KL target in the learner and is published to the GPU actor thread as the 'best' opponent.
arrow(ax, (10.58, 1.15), (9.62, 1.75), color=ORANGE, ls="--")
ax.text(
    9.5,
    0.95,
    "new anchor → KL target in the learner, and the 'best'\nopponent of the GPU actors (loaded with the actor weights)",
    ha="right",
    va="center",
    fontsize=8.5,
    color=INK2,
)
arrow(ax, (5.28, 5.25), (4.32, 5.6), color=MUTED, rad=-0.3, ls="--")
ax.text(4.8, 6.0, "actor weights refreshed every iteration", ha="center", fontsize=8.5, color=INK2)
save(fig, "appo")

# ---------- training curves of d7b (blocks of 10 iterations) ----------
rows = list(csv.DictReader(open(LOGS / "appo_d7b_curves.csv")))
lo = [int(r["iter_from"]) for r in rows]
hi = [int(r["iter_to"]) for r in rows]
mid = [(a + b) / 2 for a, b in zip(lo, hi)]


def col(name):
    return [float(r[name]) for r in rows]


fig, axs = plt.subplots(2, 2, figsize=(12, 7.4), sharex=True)
panels = [
    (axs[0, 0], "P(win) vs field: 101 code bots + 6 tape pools", [("field", col("P_field"), BLUE)], (0.6, 1.0)),
    (
        axs[0, 1],
        "P(win) vs top-team tapes and past versions",
        [("top-team tapes, Sep 30", col("P_tapes_sep30"), ORANGE), ("past versions (argmax)", col("P_past"), BLUE)],
        (0.0, 1.0),
    ),
    (axs[1, 0], "Policy entropy (per component)", [("entropy", col("ent"), BLUE)], (0.08, 0.15)),
    (axs[1, 1], "Value explained variance", [("EV", col("ev"), BLUE)], (0, 0.6)),
]
for ax, title, lines, yl in panels:
    for lab, ys, c in lines:
        ax.plot(mid, ys, color=c, lw=1.8, marker="o", ms=4, label=lab)
    ax.axvspan(109.5, max(hi) + 0.5, color="#ecebe6", zorder=0)
    ax.axvline(109.5, color=INK2, lw=1, ls="--")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_ylim(*yl)
    if len(lines) > 1:
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
axs[0, 0].text(110.5, 0.62, "after\nEMA 110\n(unused)", fontsize=8, color=INK2)
for ax in axs[1]:
    ax.set_xlabel("APPO iteration (block means over 10 iterations, plotted at block centres)")
    ax.set_xlim(-2, 120)
fig.suptitle(
    "Final run d7b: training games (sampled actions; PFSP shifts the opponent mix)",
    x=0.01,
    ha="left",
    fontsize=12,
    fontweight="bold",
)
fig.tight_layout()
save(fig, "d7b_training")

# ---------- opponent mix up to EMA 110 (shares only) ----------
mix = json.load(open(LOGS / "appo_d7b_league_mix.json"))["share_pct"]
NAMES = {
    "tapes: sep30": "top-team tapes, Sep 30 (3,931 seats)",
    "tapes: sep29": "top-team tapes, Sep 29 (874)",
    "tapes: sep28": "top-team tapes, Sep 28 (952)",
    "tapes: fast": "'fast opening' tapes (1,763)",
    "tapes: sep27": "top-team tapes, Sep 27 (1,190)",
    "tapes: weak": "weak-field tapes (1,244)",
}
items = [(k, v, BLUE) for k, v in mix.items() if k.startswith(("self", "past", "best"))]
items += sorted(((NAMES[k], v, ORANGE) for k, v in mix.items() if k.startswith("tapes")), key=lambda t: -t[1])
items += [(k, v, AQUA) for k, v in mix.items() if k.startswith("101")]
fig, ax = plt.subplots(figsize=(10.5, 5.0))
ys = list(range(len(items)))[::-1]
ax.barh(ys, [v for _, v, _ in items], color=[c for *_, c in items], height=0.7, edgecolor=SURF, lw=2)
for y, (_, v, _) in zip(ys, items):
    ax.text(v + 0.4, y, f"{v:.1f}%", va="center", fontsize=8.8, color=INK2)
ax.set_yticks(ys, [n for n, _, _ in items], fontsize=9)
ax.set_xlabel("share of learner trajectories, % (a game seen from the learning seat; a self game counts twice)")
ax.set_xlim(0, 32)
ax.grid(axis="y", visible=False)
ax.legend(
    handles=[
        Patch(color=BLUE, label="network opponents (GPU)"),
        Patch(color=ORANGE, label="tape worlds (CPU)"),
        Patch(color=AQUA, label="code bots (CPU)"),
    ],
    loc="lower right",
    frameon=False,
    fontsize=8.8,
)
ax.set_title("Who d7b actually trained against, iterations 0–109", loc="left", fontsize=11)
save(fig, "league_mix")

# ---------- dip after BC and recovery (blocks of 20 iterations) ----------
S = {}
for r in csv.DictReader(open(LOGS / "appo_d7b_vs_fixed_past.csv")):
    S.setdefault(r["opponent"], []).append(((int(r["iter_from"]) + int(r["iter_to"])) / 2, float(r["learner_P"])))
fig, ax = plt.subplots(figsize=(11, 4.4))
label_y = {}  # end labels, pushed apart where the curves end at the same value
for lab in sorted(S, key=lambda k: S[k][-1][1]):
    y = S[lab][-1][1]
    label_y[lab] = max([y] + [v + 0.06 for v in label_y.values()])
for (lab, pts), c in zip(S.items(), (BLUE, ORANGE, AQUA)):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=c, marker="o", ms=4, lw=1.8)
    ax.text(xs[-1] + 1.5, label_y[lab], lab, va="center", fontsize=9, color=c)
ax.axhline(0.5, color=INK2, lw=1)
ax.set(
    xlim=(-2, 140),
    ylim=(0, 1),
    xlabel="APPO iteration (block means over 20 iterations, plotted at block centres)",
    ylabel="learner win rate",
)
ax.set_title(
    "Learner (sampling) vs fixed past checkpoints (argmax): the parent d6 EMA 140 is overtaken around iteration 60",
    loc="left",
    fontsize=10.5,
)
save(fig, "bc_dip")

# ---------- lineage of the weights (docs/method.md, section 9) ----------
# No dates or run times per line: only what each BC learned from and where its weights came from.
fig, ax = canvas(15, 6.0)
ax.text(
    0.1,
    5.9,
    "Where the d7b_110 weights come from: from d2 on, every new BC starts from a recent RL model of ours (warm start)",
    fontsize=12.5,
    fontweight="bold",
    va="top",
)
row1 = [
    ("BC-3", "BC of c3bc: our modified\npublic V54-family notebook"),
    ("BC top-50", "5,833 trajectories\nof top-50 seats"),
    ("BC DSM + M&M", "1,024 trajectories\nof two top teams"),
    ("APPO d1 → soup", "s100_180 = mean of\n9 EMA snapshots (100..180)"),
    ("BC top-5 ('fast opening')\n→ APPO d2a", "777 trajectories\nEMA 200"),
]
row2 = [
    ("BC top-5 (newest games)\n→ APPO d3a", "533 trajectories\nEMA 350"),
    ("BC top-5 (newest games)\n→ APPO d4a", "605 trajectories\nEMA 370"),
    ("BC '4 quadrants'\n→ APPO d6 (land rule)", "543 trajectories\nEMA 250 = 2nd submission"),
    (
        "BC top-10 (Sep 30 games)",
        "809 trajectories (4 quadrants)\nfrom d6 EMA 140, an\nintermediate snapshot\n3,000 steps",
    ),
    ("APPO d7b → EMA 110", "110 iterations, land rule\n= final submission"),
]
W, H = 2.65, 1.65
for k, (t, s) in enumerate(row1):
    x = 0.15 + k * 2.97
    box(ax, x, 3.35, W, H, t + "\n" + s, fs=8.4)
    if k < 4:
        arrow(ax, (x + W + 0.02, 4.17), (x + 2.95, 4.17))
for k, (t, s) in enumerate(row2):
    x = 0.15 + k * 2.97
    hl = k >= 3
    box(ax, x, 0.75, W, H, t + "\n" + s, fs=8.4, fc=("#fde9df" if hl else BOXF), ec=(ORANGE if hl else BOXE))
    if k < 4:
        arrow(ax, (x + W + 0.02, 1.57), (x + 2.95, 1.57))
arrow(ax, (12.03 + W / 2, 3.33), (0.15 + W / 2, 2.42), rad=-0.08)
ax.text(9.4, 0.2, "orange = the final line (BC d7 + APPO d7b up to EMA 110)", fontsize=9, color=INK2)
save(fig, "lineage")

# ---------- choice of the final snapshot (docs/evaluation.md, section 3) ----------
# The submitted snapshot and the one the preregistered rule picked; both passed the gates. The evaluation games are not
# in this repository, so the values are written here. SE: per seed (head-to-head, about 0.02, not drawn), clustered by
# episode (top-4 panel, drawn as +-2 SE).
cands = ["d7b_110\n(submitted)", "rule's pick"]
h2h_d6 = [0.621, 0.504]
h2h_d3 = [0.901, 0.869]
top4, top4se = [0.073, 0.083], [0.032, 0.032]
cols = [BLUE, MUTED]
fig, axs = plt.subplots(1, 3, figsize=(11, 4.2))
for ax, vals, title in (
    (axs[0], h2h_d6, "Head-to-head vs d6_250\n(our other final submission)"),
    (axs[1], h2h_d3, "Head-to-head vs d3a_350\n(our previous best submission)"),
):
    ax.bar(range(2), vals, color=cols, width=0.55, edgecolor=SURF, lw=2)
    ax.axhline(0.5, color=INK2, lw=1)
    for x, v in enumerate(vals):
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(2), cands, fontsize=8.8)
    ax.set(xlim=(-0.6, 1.6), ylim=(0, 1))
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(axis="x", visible=False)
ax = axs[2]
ax.errorbar(range(2), top4, yerr=[2 * s for s in top4se], fmt="none", ecolor=MUTED, elinewidth=1.5, capsize=4)
ax.scatter(range(2), top4, color=cols, s=60, zorder=3)
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks(range(2), cands, fontsize=8.8)
ax.set_xlim(-0.6, 1.6)
ax.set_title("ΔP vs d6_250 on top-4 seats\n(tapes), ±2 SE", loc="left", fontsize=10)
ax.grid(axis="x", visible=False)
axs[0].set_ylabel("score share (W + 0.5D)/N")
fig.tight_layout()
save(fig, "selection")

# ---- live ratings of the final pair after the deadline (logs/live_ratings.csv) ----
# public_score of each submission from hourly snapshots of the Kaggle API (UTC); gaps = no snapshot, no interpolation.
series = {}
with open(LOGS / "live_ratings.csv") as f:
    for r in csv.DictReader(f):
        t = dt.datetime.strptime(r["utc"], "%Y-%m-%dT%H:%MZ")
        series.setdefault(r["submission"], []).append((t, float(r["public_score"])))
fig, ax = plt.subplots(figsize=(11, 3.8))
for name, color, dy in (("d7b_110", BLUE, 7), ("d6_250", MUTED, -7)):
    pts = series[name]
    ax.plot([t for t, _ in pts], [v for _, v in pts], color=color, lw=1.6, marker="o", ms=2.5)
    ax.annotate(
        f"{name} {pts[-1][1]:.1f}",
        (pts[-1][0], pts[-1][1]),
        xytext=(6, dy),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=INK,
    )
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
ax.set_ylabel("submission rating (API)")
ax.set_title(
    "Live rating of the final pair after the deadline, hourly snapshots (gaps = no snapshot)", loc="left", fontsize=10
)
fig.tight_layout()
save(fig, "live_ratings")

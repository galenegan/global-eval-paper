"""
Figure: air temperature evaluation (manuscript Fig. 2 tair_evaluation).

Scatter plots for the 2022 daily air temperature triple collocation analysis:
all pairwise combinations of (Spotter, ERA5, OAFlux) for both the E25 linear
regression and E25 neural network estimates.

Run from the repo root:
    PYTHONPATH=. python scripts/plot_tair_etc_scatter.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.methods import mae, bias, rmse

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
FIGPATH = REPO_ROOT / "figures"

params = {
    "axes.labelsize": 16,
    "font.size": 16,
    "legend.fontsize": 10,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern",
    "axes.grid": False,
}
plt.rcParams.update(params)

NAVY = "#012749"
MAGENTA = "#9f1853"

df = pd.read_csv(DATA / "spotter_daily_2022_full.csv")
assert df["air_temperature_era5"].mean() < 100, "ERA5 air temp expected in deg C"

LIMS = (-15, 40)
LABELS = {
    "lr": r"$T_{\textrm{E25lr}}$",
    "nn": r"$T_{\textrm{E25nn}}$",
    "era5": r"$T_{\textrm{ERA5}}$",
    "oaflux": r"$T_{\textrm{OAFlux}}$",
}


def scatter_panel(ax, x, y, xlab, ylab, title):
    one = np.linspace(*LIMS, 100)
    ax.plot(x, y, "o", markersize=2, color=NAVY, alpha=0.5)
    ax.plot(
        one,
        one,
        "-",
        color=MAGENTA,
        linewidth=3,
        label=(
            f"MAE = {mae(x, y):.2f}, RMSE = {rmse(x, y):.2f}, "
            f"Bias = {bias(x, y):.2f} " + r"$^\circ$C"
        ),
    )
    ax.legend(loc="upper left")
    ax.set_xlabel(xlab + r" ($^\circ$C)")
    ax.set_ylabel(ylab + r" ($^\circ$C)")
    ax.set_xlim(*LIMS)
    ax.set_ylim(*LIMS)
    ax.set_title(title)


fig = plt.figure(figsize=(15, 9.5))
# 6-column grid so each panel spans 2 columns. Bottom panels are offset by one
# column (cols 1-3 and 3-5) to center them under the 3 top panels.
gs = fig.add_gridspec(2, 6)
ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:4])
ax_c = fig.add_subplot(gs[0, 4:6])
ax_d = fig.add_subplot(gs[1, 1:3])
ax_e = fig.add_subplot(gs[1, 3:5])

# Row 1: E25 linear regression triplet
sub = df.dropna(
    subset=["air_temperature_era5", "air_temperature_oaflux", "estimated_air_temperature"]
)
print(f"E25lr triplet N = {len(sub)}")
scatter_panel(
    ax_a, sub["estimated_air_temperature"], sub["air_temperature_era5"],
    LABELS["lr"], LABELS["era5"], "(a)",
)
scatter_panel(
    ax_b, sub["estimated_air_temperature"], sub["air_temperature_oaflux"],
    LABELS["lr"], LABELS["oaflux"], "(b)",
)
scatter_panel(
    ax_c, sub["air_temperature_oaflux"], sub["air_temperature_era5"],
    LABELS["oaflux"], LABELS["era5"], "(c)",
)

# Row 2: E25 neural network triplet
sub = df.dropna(
    subset=["air_temperature_era5", "air_temperature_oaflux", "estimated_air_temperature_nn"]
)
print(f"E25nn triplet N = {len(sub)}")
scatter_panel(
    ax_d, sub["estimated_air_temperature_nn"], sub["air_temperature_era5"],
    LABELS["nn"], LABELS["era5"], "(d)",
)
scatter_panel(
    ax_e, sub["estimated_air_temperature_nn"], sub["air_temperature_oaflux"],
    LABELS["nn"], LABELS["oaflux"], "(e)",
)

fig.tight_layout(pad=0.5)
FIGPATH.mkdir(exist_ok=True)
plt.savefig(FIGPATH / "tair_evaluation.png", dpi=300)
print(f"Saved {FIGPATH / 'tair_evaluation.png'}")

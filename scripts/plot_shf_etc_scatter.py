"""
Figure: sensible heat flux evaluation (manuscript Fig. 4 shf_evaluation).

Scatter plots for the 2022 daily sensible heat flux triple collocation
analysis on the held-out test Spotter units: all pairwise combinations of
(Spotter COARE, ERA5, OAFlux), with the Spotter flux computed from the
uncorrected (top row, Q_SH,E25lr), bias-corrected (middle row, Q_SH,c), and
neural-network (bottom row, Q_SH,nn) air temperature estimates.

All three rows are evaluated on the SAME held-out test triplets, so the
uncorrected -> corrected -> NN progression is a fair, same-population, fully
out-of-sample comparison. Uses the test-split rows of the daily full csv.


Run from the repo root:
    PYTHONPATH=. python scripts/plot_shf_etc_scatter.py
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

DAILY = DATA / "spotter_daily_2022_full.csv"

LIMS = (-100, 250)
LABELS = {
    "sp_raw": r"$Q_{\textrm{SH,E25lr}}$",
    "sp_bc": r"$\hat{Q}_{\textrm{SH,c}}$",
    "sp_nn": r"$\hat{Q}_{\textrm{SH,nn}}$",
    "era5": r"$Q_{\textrm{SH,ERA5}}$",
    "oaflux": r"$Q_{\textrm{SH,OAFlux}}$",
}
UNITS = r" (W/m$^2$)"


def scatter_panel(ax, x, y, xlab, ylab, title):
    one = np.linspace(*LIMS, 100)
    ax.plot(x, y, "o", markersize=2, color=NAVY, alpha=0.5)
    ax.plot(
        one,
        one,
        "-",
        color=MAGENTA,
        linewidth=2,
        label=(
            f"MAE = {mae(x, y):.1f}, RMSE = {rmse(x, y):.1f}, "
            f"Bias = {bias(x, y):.1f} " + r"W/m$^2$"
        ),
    )
    ax.legend(loc="upper left")
    ax.set_xlabel(xlab + UNITS)
    ax.set_ylabel(ylab + UNITS)
    ax.set_xlim(*LIMS)
    ax.set_ylim(*LIMS)
    ax.set_title(title)


fig = plt.figure(figsize=(15, 14.25))
# 6-column grid so each panel spans 2 columns. Rows 2 and 3 hold only 2 panels
# each; offsetting them by one column (cols 1-3 and 3-5) centers them under the
# 3 full-width panels of row 1.
gs = fig.add_gridspec(3, 6)
ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:4])
ax_c = fig.add_subplot(gs[0, 4:6])
ax_d = fig.add_subplot(gs[1, 1:3])
ax_e = fig.add_subplot(gs[1, 3:5])
ax_f = fig.add_subplot(gs[2, 1:3])
ax_g = fig.add_subplot(gs[2, 3:5])

# All rows share the same held-out test triplets.
df = pd.read_csv(DAILY)
df = df[df["split"] == "test"].dropna(subset=[
    "sensible_heat_flux_spotter_uncorrected", "sensible_heat_flux_spotter_corrected",
    "sensible_heat_flux_spotter_nn", "sensible_heat_flux_era5", "sensible_heat_flux_oaflux",
])
era5 = df["sensible_heat_flux_era5"]
oaflux = df["sensible_heat_flux_oaflux"]
unc, corr, nn = (df["sensible_heat_flux_spotter_uncorrected"],
                 df["sensible_heat_flux_spotter_corrected"],
                 df["sensible_heat_flux_spotter_nn"])
print(f"Test triplet N = {len(df)}, {df['spotter_id'].nunique()} spotters")

# Row 1: flux from uncorrected E25lr air temperature
scatter_panel(ax_a, unc, era5, LABELS["sp_raw"], LABELS["era5"], "(a)")
scatter_panel(ax_b, unc, oaflux, LABELS["sp_raw"], LABELS["oaflux"], "(b)")
scatter_panel(ax_c, oaflux, era5, LABELS["oaflux"], LABELS["era5"], "(c)")

# Row 2: flux from bias-corrected air temperature
scatter_panel(ax_d, corr, era5, LABELS["sp_bc"], LABELS["era5"], "(d)")
scatter_panel(ax_e, corr, oaflux, LABELS["sp_bc"], LABELS["oaflux"], "(e)")

# Row 3: flux from the neural-network air temperature
scatter_panel(ax_f, nn, era5, LABELS["sp_nn"], LABELS["era5"], "(f)")
scatter_panel(ax_g, nn, oaflux, LABELS["sp_nn"], LABELS["oaflux"], "(g)")

fig.tight_layout(pad=0.5)
FIGPATH.mkdir(exist_ok=True)
plt.savefig(FIGPATH / "shf_evaluation.png", dpi=300)
print(f"Saved {FIGPATH / 'shf_evaluation.png'}")

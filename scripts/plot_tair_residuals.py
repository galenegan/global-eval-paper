"""
Figure: air temperature residuals (manuscript Fig. tair_residual).

Air temperature residual analysis on the hourly 2022 + 2023 dataset:
(a) residuals binned by ERA5 air temperature for the E25 linear regression
    estimate and its bias-corrected version,
(b) the relationship between the internal temperature increase above the
    nighttime value and the internal/external temperature difference in the
    subzero regime,
(c) time series from a single Southern Ocean Spotter (SPOT-010286) showing
    close tracking in open water (Jan 2023) and episodic divergence while
    surrounded by sea ice (Aug-Sep 2023).

Run from the repo root:
    PYTHONPATH=. python scripts/plot_tair_residuals.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
FIGPATH = REPO_ROOT / "figures"

params = {
    "axes.labelsize": 16,
    "font.size": 16,
    "legend.fontsize": 12,
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
PURPLE = "#6929c4"
TEAL = "#009d9a"
RED = "#da1e28"

USECOLS = [
    "time",
    "spotter_id",
    "air_temperature",
    "delta_night_air_temp",
    "estimated_air_temperature",
    "bias_corrected_air_temperature",
    "air_temperature_era5",
]

df = pd.read_csv(DATA / "spotter_hourly_full.csv", usecols=USECOLS)
assert df["air_temperature_era5"].mean() < 100, "ERA5 air temp expected in deg C"
df["time"] = pd.to_datetime(df["time"], format="mixed")

# %% Figure layout
fig = plt.figure(figsize=(13, 9.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.85])
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
# Broken-axis time series: width ratio approximates window durations
gs_c = gs[1, :].subgridspec(1, 2, width_ratios=[1, 2], wspace=0.05)
ax_c1 = fig.add_subplot(gs_c[0, 0])
ax_c2 = fig.add_subplot(gs_c[0, 1], sharey=ax_c1)

# %% (a) Binned residuals
df["residual_linear"] = df["estimated_air_temperature"] - df["air_temperature_era5"]
df["residual_bc"] = df["bias_corrected_air_temperature"] - df["air_temperature_era5"]
bins = np.arange(-15, 31, 5)
df["tair_bins"] = pd.cut(df["air_temperature_era5"], bins)

x_mids = [b.mid for b in df["tair_bins"].cat.categories]
for col, color, label in [
    ("residual_linear", NAVY, r"$T_{\textrm{E25lr}}$"),
    ("residual_bc", MAGENTA, r"$\hat{T}_{\textrm{E25lr,c}}$"),
]:
    grp = df.groupby("tair_bins", observed=False)[col]
    ax_a.errorbar(
        x_mids,
        grp.mean(),
        yerr=grp.std(),
        fmt="o-",
        color=color,
        ecolor=color,
        capsize=3,
        linewidth=2.5,
        alpha=0.7,
        label=label,
    )
ax_a.grid()
ax_a.set_xlabel(r"$T_{\textrm{ERA5}}$ ($^\circ$C)")
ax_a.set_ylabel(r"Residual $T - T_{\textrm{ERA5}}$ ($^\circ$C)")
ax_a.set_title("(a)")
ax_a.legend()

# %% (b) Cold-regime scatter of nighttime warming vs internal/external difference
cold = df[(df["air_temperature_era5"] < 0) & (df["delta_night_air_temp"] < 5)].dropna(
    subset=["delta_night_air_temp", "air_temperature", "air_temperature_era5"]
)
sc = ax_b.scatter(
    cold["delta_night_air_temp"],
    cold["air_temperature"] - cold["air_temperature_era5"],
    c=cold["air_temperature_era5"],
    s=12,
    cmap="Blues_r",
)
cb = fig.colorbar(sc, ax=ax_b)
cb.set_label(r"$T_{\textrm{ERA5}}$ ($^\circ$C)")
ax_b.set_xlabel(r"$\Delta T_{\textrm{night}}$ ($^\circ$C)")
ax_b.set_ylabel(r"$T_{\textrm{int}} - T_{\textrm{ERA5}}$ ($^\circ$C)")
ax_b.set_title("(b)")

# %% (c) Time series for a single Southern Ocean unit
SPOT = "SPOT-010286"
WINDOWS = [("2023-01-01", "2023-01-11"), ("2023-08-29", "2023-09-19")]

s = df[df["spotter_id"] == SPOT].set_index("time").sort_index()
for ax, (t0, t1) in zip([ax_c1, ax_c2], WINDOWS):
    w = s.loc[t0:t1]
    print(f"{SPOT} window {t0} to {t1}: n = {len(w)}")
    ax.plot(w.index, w["air_temperature_era5"], "-", color=PURPLE, linewidth=1.5, label="ERA5")
    ax.plot(w.index, w["air_temperature"], "-", color=TEAL, linewidth=1.5, label="Spotter Internal")
    ax.plot(w.index, w["estimated_air_temperature"], "--", color=RED, linewidth=1.5, label="E25lr")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.grid(alpha=0.3)

ax_c1.set_ylabel(r"Air Temperature ($^\circ$C)")

# Broken-axis styling
ax_c1.spines["right"].set_visible(False)
ax_c2.spines["left"].set_visible(False)
ax_c2.tick_params(labelleft=False, left=False)
ax_c2.yaxis.tick_right()

d = 0.012
kwargs = dict(transform=ax_c1.transAxes, color="k", clip_on=False)
ax_c1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
ax_c1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
kwargs = dict(transform=ax_c2.transAxes, color="k", clip_on=False)
ax_c2.plot((-d / 2, +d / 2), (-d, +d), **kwargs)
ax_c2.plot((-d / 2, +d / 2), (1 - d, 1 + d), **kwargs)
ax_c1.set_title("(c)", x=0.78)
ax_c2.legend(loc="upper right")

fig.tight_layout(pad=1)
FIGPATH.mkdir(exist_ok=True)
plt.savefig(FIGPATH / "tair_residual.png", dpi=300)
print(f"Saved {FIGPATH / 'tair_residual.png'}")

# Stats quoted in the manuscript discussion
w = s.loc[WINDOWS[1][0]:WINDOWS[1][1]]
dev = w["estimated_air_temperature"] - w["air_temperature_era5"]
print(f"Window 2 max deviation: {dev.max():.1f} C; mean ERA5: {w['air_temperature_era5'].mean():.1f} C")

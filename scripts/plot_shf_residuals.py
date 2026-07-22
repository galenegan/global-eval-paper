"""
Figure: sensible heat flux residuals (manuscript Fig. 5 heat_flux_residual).

Sensible heat flux residual analysis on the hourly 2022-2023 dataset:
(a) SHF residual binned by ERA5 air temperature, for the flux computed from
    the uncorrected and bias-corrected E25 linear regression air temperature,
(b) SHF residual as a function of air temperature residual (bias-corrected
    estimate), with a least-squares fit.

Run from the repo root:
    PYTHONPATH=. python scripts/plot_shf_residuals.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
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

BINS = np.arange(-15, 31, 5)

df_full = pd.read_csv(
    DATA / "spotter_hourly_full.csv",
    usecols=[
        "air_temperature_era5", "bias_corrected_air_temperature",
        "sensible_heat_flux_era5",
        "sensible_heat_flux_spotter_uncorrected",
        "sensible_heat_flux_spotter_corrected",
    ],
)
df_full = df_full.rename(columns={"air_temperature_era5": "tair_era5"})

# Residual = Spotter flux - ERA5 flux (both already W/m^2, positive upward).
df_lr = df_full.assign(
    shf_residual=df_full["sensible_heat_flux_spotter_uncorrected"] - df_full["sensible_heat_flux_era5"]
)
df_bc = df_full.assign(
    shf_residual=df_full["sensible_heat_flux_spotter_corrected"] - df_full["sensible_heat_flux_era5"]
)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 6))

# %% (a) Binned residuals
for df, color, label in [
    (df_lr, NAVY, r"$Q_{\textrm{SH,E25lr}}$"),
    (df_bc, MAGENTA, r"$\hat{Q}_{\textrm{SH,c}}$"),
]:
    df = df.dropna(subset=["shf_residual", "tair_era5"]).copy()
    df["tair_bins"] = pd.cut(df["tair_era5"], BINS)
    grp = df.groupby("tair_bins", observed=False)["shf_residual"]
    x_mids = [b.mid for b in df["tair_bins"].cat.categories]
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
ax_a.set_ylabel(r"Residual $Q - Q_{\textrm{SH,ERA5}}$ (W/m$^2$)")
ax_a.set_title("(a)")
ax_a.legend()

# %% (b) SHF residual vs air temperature residual (bias-corrected estimate)
sub = df_bc.copy()
sub["tair_residual"] = sub["bias_corrected_air_temperature"] - sub["tair_era5"]
sub = sub.dropna(subset=["tair_residual", "shf_residual"])

coeffs = np.polyfit(sub["tair_residual"], sub["shf_residual"], 1)
fit = np.poly1d(coeffs)
r2 = np.corrcoef(sub["tair_residual"], sub["shf_residual"])[0, 1] ** 2
print(f"(b) fit: slope = {coeffs[0]:.2f} W/m2/C, intercept = {coeffs[1]:.2f} W/m2, r2 = {r2:.2f}, N = {len(sub)}")

ax_b.plot(sub["tair_residual"], sub["shf_residual"], "o", markersize=1, color=NAVY, alpha=0.1)
xfit = np.linspace(sub["tair_residual"].min(), sub["tair_residual"].max(), 100)
sign = "+" if coeffs[1] >= 0 else "-"
ax_b.plot(
    xfit,
    fit(xfit),
    "-",
    color=MAGENTA,
    linewidth=2,
    label=rf"$y = {coeffs[0]:.2f}x {sign} {abs(coeffs[1]):.2f}$, $r^2 = {r2:.2f}$",
)
ax_b.legend(loc="upper right")
ax_b.set_xlabel(r"$\hat{T}_{\textrm{E25lr,c}} - T_{\textrm{ERA5}}$ ($^\circ$C)")
ax_b.set_ylabel(r"$\hat{Q}_{\textrm{SH,c}} - Q_{\textrm{SH,ERA5}}$ (W/m$^2$)")
ax_b.set_title("(b)")

fig.tight_layout(pad=0.5)
FIGPATH.mkdir(exist_ok=True)
plt.savefig(FIGPATH / "heat_flux_residual.png", dpi=300)
print(f"Saved {FIGPATH / 'heat_flux_residual.png'}")

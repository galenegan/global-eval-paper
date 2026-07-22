"""
Figure: neural-network air temperature evaluation (manuscript Fig. tair_neural_net).

Scatter plots of the final air-temperature neural network (model_sweep_8,
trial 15) against ERA5 for the training, validation, and hold-out test splits
(hourly 2022-2023 data, group split by instrument). Reads the NN prediction
(nn_air_temperature) and instrument split from the hourly full csv.

Run from the repo root:
    PYTHONPATH=. python scripts/plot_tair_nn_scatter.py
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

data_path = DATA / "spotter_hourly_full.csv"

LIMS = (-20, 35)

df = pd.read_csv(data_path, usecols=["split", "nn_air_temperature", "air_temperature_era5"])
df = df.dropna(subset=["split", "nn_air_temperature", "air_temperature_era5"])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
one = np.linspace(*LIMS, 100)

for ax, split, name, panel in zip(
    axes, ["train", "val", "test"], ["Training", "Validation", "Test"], "abc"
):
    sub = df.loc[df["split"] == split]
    x = sub["nn_air_temperature"].values
    y = sub["air_temperature_era5"].values
    print(f"{split}: N = {len(sub)}, MAE = {mae(x, y):.3f}, Bias = {bias(x, y):.3f}")
    ax.plot(x, y, "o", markersize=1, color=NAVY, alpha=0.3)
    ax.plot(
        one,
        one,
        "-",
        color=MAGENTA,
        linewidth=2,
        label=(
            f"MAE = {mae(x, y):.2f}, RMSE = {rmse(x, y):.2f}, "
            f"Bias = {bias(x, y):.2f} " + r"$^\circ$C"
        ),
    )
    ax.legend(loc="upper left")
    ax.set_xlabel(r"$\hat{T}_{\textrm{nn}}$ ($^\circ$C)")
    ax.set_ylabel(r"$T_{\textrm{ERA5}}$ ($^\circ$C)")
    ax.set_xlim(*LIMS)
    ax.set_ylim(*LIMS)
    ax.set_title(f"({panel}) {name}")

fig.tight_layout(pad=0.5)
FIGPATH.mkdir(exist_ok=True)
plt.savefig(FIGPATH / "tair_neural_net.png", dpi=300)
print(f"Saved {FIGPATH / 'tair_neural_net.png'}")

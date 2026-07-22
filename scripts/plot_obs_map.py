"""
Figure: Spotter drift tracks and latitude histogram (manuscript Fig. 1 map).

Produces the two component panels of the manuscript map figure:
- figures/spotter_map.png : global scatter of all Spotter drift positions,
- figures/lat_hist.png    : histogram of the latitudes sampled.

The published figure (figures/map.png in the manuscript) is these two panels
composited side-by-side and labelled (a)/(b) in a slide editor; this script
regenerates the underlying panels. Requires plotly + kaleido for static image
export.

Run from the repo root:
    PYTHONPATH=. python scripts/plot_obs_map.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
FIGPATH = REPO_ROOT / "figures"
FIGPATH.mkdir(exist_ok=True)

df = pd.read_csv(
    DATA / "spotter_hourly_full.csv",
    usecols=["latitude", "longitude", "spotter_id", "epoch"],
)
df["time"] = pd.to_datetime(df["epoch"], unit="s")

# %% Drift-track scatter
fig = px.scatter_geo(df, lat="latitude", lon="longitude", opacity=0.5)
fig.update_traces(marker=dict(size=2))
fig.update_layout(height=600, width=1200, font={"family": "Computer Modern", "size": 16})
fig.write_image(FIGPATH / "spotter_map.png", scale=5)
print(f"Saved {FIGPATH / 'spotter_map.png'}")

# %% Latitude histogram
fig = px.histogram(df, x="latitude", labels={"latitude": "Latitude"})
fig.update_layout(
    height=600, width=800, yaxis_title="Counts", font={"family": "Computer Modern", "size": 16}
)
fig.write_image(FIGPATH / "lat_hist.png", scale=5)
print(f"Saved {FIGPATH / 'lat_hist.png'}")

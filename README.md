# Global evaluation of Spotter air-sea observations — figure/table reproduction

This repository contains the minimal set of scripts and data needed to
reproduce the figures and tables in the manuscript *"Global meteorological
observations at the air-sea interface: evaluating and improving a proxy
sensing approach"* (Galen Egan, Pieter Smit), submitted to the *Journal of
Atmospheric and Oceanic Technology*.

The manuscript evaluates Spotter-buoy proxy estimates of near-surface air
temperature and sensible heat flux against ERA5 and OAFlux using Extended
Triple Collocation (ETC), then proposes a data-driven (neural-network)
improvement.

This repository and documentation is largely AI-generated from a much 
messier source code repository. 

## Layout

```
scripts/   plotting and table scripts (one per manuscript figure/table)
utils/     shared error metrics and the ETC estimator
data/       two datasets (not tracked in git; see "Data")
figures/   output figures written by the scripts
```

## Data

All analysis reads from just two files, both with a single consistent
convention (air temperature in °C; every sensible heat flux in W/m², positive
upward; one QC'd instrument population):

| File                               | Rows | Units | Description |
|------------------------------------|---|---|---|
| `data/spotter_hourly_full.csv`     | 1,015,447 | 255 | Hourly 2022 + 2023, Spotter + ERA5, with a `year` column. Drives the residual figures, the NN air-temperature figure, and the map. |
| `data/spotter_daily_2022_full.csv` | 11,261 | 166 | Daily-averaged 2022, OAFlux-collocated. Drives the ETC table and the air-temperature / heat-flux evaluation scatters. |

The `data/` folder is not tracked in git because of
its size. You can access the source data files from Dropbox
at [this link](https://www.dropbox.com/scl/fo/4sx64z60e4lgmr8uslfhk/AAyU0zznILtM5AJ5VDPxlBU?rlkey=nfx6k1d1kjqc7a98splqjz9tw&st=kahwqgwa&dl=0)



## Setup

The scripts require Python ≥ 3.11 and a working LaTeX installation (the plots
use `text.usetex=True`). Install dependencies with either tool:

```bash
uv sync            # or:  pip install -e .
```

## Reproducing the figures and tables

Run each script from the repository root with the root on `PYTHONPATH`:

```bash
PYTHONPATH=. python scripts/plot_tair_etc_scatter.py   # Fig. tair_evaluation
PYTHONPATH=. python scripts/plot_tair_nn_scatter.py    # Fig. tair_neural_net
PYTHONPATH=. python scripts/plot_shf_etc_scatter.py    # Fig. shf_evaluation
PYTHONPATH=. python scripts/plot_tair_residuals.py     # Fig. tair_residual
PYTHONPATH=. python scripts/plot_shf_residuals.py      # Fig. heat_flux_residual
PYTHONPATH=. python scripts/plot_obs_map.py            # Fig. map (two panels)
PYTHONPATH=. python scripts/print_etc_table.py         # Table 2 (ETC error std)
```

Figures are written to `figures/`. Numbers for Table 2 are printed to stdout.

### Notes

- `plot_obs_map.py` writes the two component panels (`spotter_map.png`,
  `lat_hist.png`); the published map figure composites these side-by-side and
  adds the (a)/(b) labels in a slide editor. It requires `plotly` + `kaleido`.
- The manuscript's Table 1 is a hand-authored list of variable definitions and
  is not produced by any script.
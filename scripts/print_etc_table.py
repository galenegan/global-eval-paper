"""
Table: triple collocation error standard deviations (manuscript Table 2).

Reproduces every row of the ETC error-standard-deviation table from the single
daily dataset (data/spotter_daily_2022_full.csv):
- "All"  rows: the full QC'd daily-averaged 2022 dataset (166 units).
- "Test" rows: the held-out test Spotter units.

For each triplet, column 0 is the ERA5 reference, column 1 the Spotter estimate
named in the row, and column 2 OAFlux; the printed "Spotter/ERA5/OAFlux" error
standard deviations correspond directly to the table columns. All fluxes are in
W/m^2 (positive upward) and all temperatures in deg C.

Run from the repo root:
    PYTHONPATH=. python scripts/print_etc_table.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from utils.methods import ETC

REPO_ROOT = Path(__file__).resolve().parent.parent
d = pd.read_csv(REPO_ROOT / "data" / "spotter_daily_2022_full.csv")
test = d[d["split"] == "test"]


def etc_row(estimate, subset, df, cols):
    """cols = [ERA5 reference, Spotter estimate, OAFlux]."""
    sub = df.dropna(subset=cols).reset_index(drop=True)
    X = np.stack([sub[c].values for c in cols], axis=1)
    err_var, _, _, _ = ETC(X)
    era5, spotter, oaflux = np.sqrt(err_var)
    return (estimate, subset, len(sub), spotter, era5, oaflux)


ERA5_T, OA_T = "air_temperature_era5", "air_temperature_oaflux"
ERA5_Q, OA_Q = "sensible_heat_flux_era5", "sensible_heat_flux_oaflux"

air_rows = [
    etc_row("T_E25lr", "All", d, [ERA5_T, "estimated_air_temperature", OA_T]),
    etc_row("T_E25nn", "All", d, [ERA5_T, "estimated_air_temperature_nn", OA_T]),
    etc_row("T_E25lr,c", "All", d, [ERA5_T, "bias_corrected_air_temperature", OA_T]),
    etc_row("T_E25lr,c", "Test", test, [ERA5_T, "bias_corrected_air_temperature", OA_T]),
    etc_row("T_nn", "Test", test, [ERA5_T, "nn_air_temperature", OA_T]),
]
flux_rows = [
    etc_row("Q_SH,E25lr", "All", d, [ERA5_Q, "sensible_heat_flux_spotter_uncorrected", OA_Q]),
    etc_row("Q_SH,c", "All", d, [ERA5_Q, "sensible_heat_flux_spotter_corrected", OA_Q]),
    etc_row("Q_SH,E25lr", "Test", test, [ERA5_Q, "sensible_heat_flux_spotter_uncorrected", OA_Q]),
    etc_row("Q_SH,c", "Test", test, [ERA5_Q, "sensible_heat_flux_spotter_corrected", OA_Q]),
    etc_row("Q_SH,nn", "Test", test, [ERA5_Q, "sensible_heat_flux_spotter_nn", OA_Q]),
]


def _print_block(title, block, prec):
    print(title)
    print(f"{'Estimate':<14}{'Subset':<7}{'N':>7}{'Spotter':>10}{'ERA5':>8}{'OAFlux':>8}")
    for est, subset, n, sp, er, oa in block:
        print(f"{est:<14}{subset:<7}{n:>7}{sp:>10.{prec}f}{er:>8.{prec}f}{oa:>8.{prec}f}")


print("=" * 55)
print("Manuscript Table 2 -- ETC error standard deviations")
print("=" * 55)
_print_block("\nAir temperature (deg C)", air_rows, 2)
_print_block("\nSensible heat flux (W/m^2)", flux_rows, 1)

"""
This script draws the heat map of a saved scan in results/, in the same style as 2D_scan.py,
without recomputing anything. By default it plots the photon-fusion (gamma gamma -> chi chibar)
scan from boson_merge; point SCAN_FILE at an s-channel file to plot that instead.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ===== Parameters you can adjust =====

RESULTS = Path(__file__).parent / "results"
SCAN_FILE = RESULTS / "s_channel_13.6TeV_eps0.01_gD0.3_21x21.csv"
TITLE = r"$\gamma\gamma \to \chi\bar\chi$,  $\sqrt{s}$ = 13.6 TeV, $\epsilon$ = 0.01, $g_D$ = 0.3"
COLOR_MAP = "Blues"  # any matplotlib colormap name
COLOR_RANGE = (
    None  # (lowest, highest) cross section in pb for the color scale; None = automatic
)

# ===== End of adjustable parameters =====


def log_edges(values):
    """Cell edges halfway between log-spaced scan points, for pcolormesh."""
    step = np.sqrt(values[1] / values[0])
    return np.geomspace(values[0] / step, values[-1] * step, len(values) + 1)


columns = np.loadtxt(SCAN_FILE, delimiter=",", unpack=True)
m_d_values, m_chi_values = np.unique(columns[0]), np.unique(columns[1])
# rows are written with m_D as the outer loop, so a reshape restores the grid
totals = columns[2].reshape(len(m_d_values), len(m_chi_values))

fig, ax = plt.subplots(figsize=(7.5, 5.5))
mesh = ax.pcolormesh(
    log_edges(m_chi_values),
    log_edges(m_d_values),
    totals,
    norm="log",
    cmap=COLOR_MAP,
    shading="flat",
)
if COLOR_RANGE is not None:
    mesh.set_clim(*COLOR_RANGE)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Dark monopole mass $M_\chi$ (GeV)")
ax.set_ylabel(r"Dark photon mass $m_D$ (GeV)")
ax.set_title(TITLE)
fig.colorbar(mesh, label=r"$\sigma(pp \to \chi\bar{\chi})$ (pb)")
fig.tight_layout()
output = SCAN_FILE.with_suffix(".png")
fig.savefig(output, dpi=160)
print(f"Saved the figure to {output}")
plt.show()

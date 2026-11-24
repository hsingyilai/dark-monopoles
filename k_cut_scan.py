"""
This script scans the s-channel cross section of fermionic dark magnetic monopole pairs in
the dark monopole mass and the dark photon mass, like 2D_scan.py, but keeps only collisions
with sqrt(s_hat) >= K_CUT * m_D.

Why the cut: a massive dark photon confines the monopoles, with a flux-tube length of about
1 / m_D. When the pair is made at sqrt(s_hat) >> m_D, it forms at much shorter distances and
the free-monopole calculation in fermion.py gives the right total; confinement only changes
the final state. Near or below sqrt(s_hat) ~ m_D that separation fails, so those collisions
are left out. With K_CUT > 1 the dark photon resonance at sqrt(s_hat) = m_D is cut away.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

import fermion

# ===== Parameters you can adjust =====

# Keep only sqrt(s_hat) >= K_CUT * m_D. K_CUT = 0 gives back 2D_scan.py.
K_CUT = 3.0

# Scan range of the dark monopole mass, in GeV
M_CHI_MIN = 1.0
M_CHI_MAX = 100.0
# 14 log-spaced points keep about the spacing of 2D_scan.py's 21 points over 1-1000 GeV
N_M_CHI = 14

# Scan range of the dark photon mass, in GeV
M_D_MIN = 1.0
M_D_MAX = 10000.0
# 17 log-spaced points are exactly the points of 2D_scan.py's 21-point scan above 1 GeV
N_M_D = 17

# These override the values in fermion.py for this scan
fermion.EPS = 0.01  # kinetic mixing
fermion.G_D = 0.3  # dark magnetic coupling; the dark photon width follows from these
fermion.S = 13600.0**2  # squared collision energy of the two protons, in GeV^2

EPSREL = 1e-4  # relative accuracy of each integral; smaller is slower
COLOR_MAP = "Blues"  # any matplotlib colormap name

# The PDF set and the list of quarks are set in fermion.py

# ===== End of adjustable parameters =====


def sigma_from_quark_cut(pid, q, m_q, s_hat_min, epsrel=1e-4):
    """Like fermion.sigma_from_quark, but integrating only over s_hat >= s_hat_min.
    Returns the cross section from one quark flavour in GeV^-2 and its error estimate.
    Uses the current module values of M_CHI, M_D, EPS, G_D and S in fermion.
    """
    # Start at the pair threshold, the quark threshold or the cut, whichever is highest
    v_min = np.log(max(4 * fermion.M_CHI**2, 4 * m_q**2, s_hat_min) / fermion.S)
    if v_min >= 0:
        # the cut is above the proton-proton collision energy: nothing left
        return 0.0, 0.0

    # Same treatment of the resonance as fermion.sigma_from_quark. It only matters when
    # the pole is above the cut, i.e. for K_CUT < 1.
    v_pole = np.log(fermion.M_D**2 / fermion.S)
    half_width = 30 * fermion.dark_photon_width(fermion.M_D, fermion.M_CHI) / fermion.M_D
    if v_min < v_pole < 0:
        v_limits = sorted(
            {
                v_min,
                max(v_min, v_pole - half_width),
                min(0.0, v_pole + half_width),
                0.0,
            }
        )
    else:
        v_limits = [v_min, 0.0]

    sigma_q = 0.0
    error = 0.0
    for v_low, v_high in zip(v_limits, v_limits[1:]):
        piece, piece_error = quad(
            fermion.integrand,
            v_low,
            v_high,
            args=(pid, q, m_q),
            epsabs=0,
            epsrel=epsrel,
            limit=200,
        )
        sigma_q += piece
        error += piece_error
    return sigma_q, error


def log_edges(values):
    """Cell edges halfway between log-spaced scan points, for pcolormesh."""
    step = np.sqrt(values[1] / values[0])
    return np.geomspace(values[0] / step, values[-1] * step, len(values) + 1)


m_chi_values = np.geomspace(M_CHI_MIN, M_CHI_MAX, N_M_CHI)
m_d_values = np.geomspace(M_D_MIN, M_D_MAX, N_M_D)
totals = np.zeros((N_M_D, N_M_CHI))  # totals[i, j]: m_d_values[i], m_chi_values[j]

for i, m_d in enumerate(m_d_values):
    for j, m_chi in enumerate(m_chi_values):
        # fermion.integrand reads these two values from the fermion module
        fermion.M_D = m_d
        fermion.M_CHI = m_chi

        s_hat_min = (K_CUT * m_d) ** 2
        for pid, q, m_q in fermion.QUARKS:
            sigma_q, error = sigma_from_quark_cut(pid, q, m_q, s_hat_min, epsrel=EPSREL)
            totals[i, j] += sigma_q * fermion.GEV2_TO_PB
        print(
            f"M_D = {m_d:.2f} GeV, M_CHI = {m_chi:.2f} GeV: total {totals[i, j]:.3e} pb"
        )

# Save every point to results/ before plotting, so closing the plot window can't lose the scan.
# The file name records the settings. An existing file is never overwritten: _2, _3, ... is added.
results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)
name = (
    f"s_channel_kcut{K_CUT:g}_{np.sqrt(fermion.S) / 1000:g}TeV"
    f"_eps{fermion.EPS:g}_gD{fermion.G_D:g}_{N_M_D}x{N_M_CHI}"
)
path = results_dir / f"{name}.csv"
copy = 2
while path.exists():
    path = results_dir / f"{name}_{copy}.csv"
    copy += 1
header = (
    "sigma(pp -> chi chibar), s-channel q qbar -> gamma* -> A' -> chi chibar, "
    "from k_cut_scan.py\n"
    f"only sqrt(s_hat) >= K_CUT * m_D with K_CUT = {K_CUT:g}; zero where that is above "
    "sqrt(S)\n"
    f"sqrt(S) = {np.sqrt(fermion.S) / 1000:g} TeV, EPS = {fermion.EPS:g}, "
    f"G_D = {fermion.G_D:g}, EPSREL = {EPSREL:g}, PDF {fermion.pdf.set().name}\n"
    f"m_D: {N_M_D} log points {M_D_MIN:g} to {M_D_MAX:g} GeV, "
    f"m_chi: {N_M_CHI} log points {M_CHI_MIN:g} to {M_CHI_MAX:g} GeV\n"
    'load: m_D, m_chi, sigma = np.loadtxt(path, delimiter=",", unpack=True)\n'
    "m_D_GeV,m_chi_GeV,sigma_pb"
)
# one row per point, in the same order as the loop: m_D outer, m_chi inner
m_d_grid, m_chi_grid = np.meshgrid(m_d_values, m_chi_values, indexing="ij")
rows = np.column_stack([m_d_grid.ravel(), m_chi_grid.ravel(), totals.ravel()])
np.savetxt(path, rows, delimiter=",", fmt="%.10g", header=header)
print(f"Saved the scan to {path}")

fig, ax = plt.subplots(figsize=(7.5, 5.5))
# Points with nothing left after the cut can't go on a log color scale: hide them, and
# let the gray background show through there
mesh = ax.pcolormesh(
    log_edges(m_chi_values),
    log_edges(m_d_values),
    np.ma.masked_where(totals <= 0, totals),
    norm="log",
    cmap=COLOR_MAP,
    shading="flat",
)
ax.set_facecolor("#d9d9d9")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Dark monopole mass $M_\chi$ (GeV)")
ax.set_ylabel(r"Dark photon mass $m_D$ (GeV)")
ax.set_title(
    rf"$\sqrt{{S}}$ = {np.sqrt(fermion.S) / 1000:.1f} TeV, "
    rf"$\epsilon$ = {fermion.EPS:g}, $g_D$ = {fermion.G_D:g}, "
    rf"$\sqrt{{\hat s}} \geq {K_CUT:g}\, m_D$ (gray: nothing left)"
)
fig.colorbar(mesh, label=r"$\sigma(pp \to \chi\bar{\chi})$ (pb)")
fig.tight_layout()
plt.show()

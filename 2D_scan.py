"""
This script scans the total cross section of fermionic dark magnetic monopole production
in the dark monopole mass and the dark photon mass, and plots the result.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

import fermion

# ===== Parameters you can adjust =====

# Scan range of the dark monopole mass, in GeV. Below about 0.83 GeV the collision energy
# 2 * M_CHI drops under the PDF grid's lowest Q (1.65 GeV) and LHAPDF extrapolates silently.
M_CHI_MIN = 1.0
M_CHI_MAX = 10.0
N_M_CHI = 21  # number of log-spaced points; the run time grows as N_M_CHI * N_M_D

# Scan range of the dark photon mass, in GeV
M_D_MIN = 0.1
M_D_MAX = 100.0
N_M_D = 21  # number of log-spaced points

# These override the values in fermion.py for this scan
fermion.L_DECAY = 0.01  # dark photon decay rate, GeV
fermion.EPS_G_D_CUBE = 1  # kinetic mixing times dark magnetic coupling cubed
fermion.S = 13600.0**2  # squared collision energy of the two protons, in GeV^2

EPSREL = 1e-4  # relative accuracy of each integral; smaller is slower
COLOR_MAP = "Blues"  # any matplotlib colormap name

# The PDF set and the list of quarks are set in fermion.py

# ===== End of adjustable parameters =====


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

        for pid, q, m_q in fermion.QUARKS:
            # Start at the threshold: quad has trouble with the step in sigma_hat there
            v_min = np.log(max(4 * m_chi**2, 4 * m_q**2) / fermion.S)
            sigma_q, error = quad(
                fermion.integrand,
                v_min,
                0.0,
                args=(pid, q, m_q),
                epsabs=0,
                epsrel=EPSREL,
                limit=200,
            )
            totals[i, j] += sigma_q * fermion.GEV2_TO_PB
        print(
            f"M_D = {m_d:.2f} GeV, M_CHI = {m_chi:.2f} GeV: total {totals[i, j]:.3e} pb"
        )

fig, ax = plt.subplots(figsize=(7.5, 5.5))
mesh = ax.pcolormesh(
    log_edges(m_chi_values),
    log_edges(m_d_values),
    totals,
    norm="log",
    cmap=COLOR_MAP,
    shading="flat",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Dark monopole mass $M_\chi$ (GeV)")
ax.set_ylabel(r"Dark photon mass $m_D$ (GeV)")
ax.set_title(
    rf"$\sqrt{{S}}$ = {np.sqrt(fermion.S) / 1000:.1f} TeV, "
    rf"$\Gamma_D$ = {fermion.L_DECAY} GeV, "
    rf"$\epsilon g_D^3$ = {fermion.EPS_G_D_CUBE}"
)
fig.colorbar(mesh, label=r"$\sigma(pp \to \chi\bar{\chi})$ (pb)")
fig.tight_layout()
plt.show()

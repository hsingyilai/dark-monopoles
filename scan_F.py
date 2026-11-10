"""
This script scans |F_2|^2, the magnitude square of the Pauli form factor returned by
f2_square in fermion.py, in beta and l = lambda^2 / M^2, and plots the result.
"""

import matplotlib.pyplot as plt
import numpy as np

import fermion

# ===== Parameters you can adjust =====

# Scan range of beta = sqrt(1 - 4 M^2 / q^2), on a linear scale.
# Keep it strictly between 0 and 1: the formula divides by beta and by 1 - beta.
BETA_MIN = 0.01
BETA_MAX = 0.99
N_BETA = 99  # number of evenly spaced points

# Scan range of l = lambda^2 / M^2, on a log scale
L_MIN = 0.01
L_MAX = 1000.0
N_L = 101  # number of log-spaced points; the run time grows as N_BETA * N_L

COLOR_MAP = "Blues"  # any matplotlib colormap name

# ===== End of adjustable parameters =====


def linear_edges(values):
    """Cell edges halfway between evenly spaced scan points, for pcolormesh."""
    half_step = (values[1] - values[0]) / 2
    return np.linspace(values[0] - half_step, values[-1] + half_step, len(values) + 1)


def log_edges(values):
    """Cell edges halfway between log-spaced scan points, for pcolormesh."""
    step = np.sqrt(values[1] / values[0])
    return np.geomspace(values[0] / step, values[-1] * step, len(values) + 1)


beta_values = np.linspace(BETA_MIN, BETA_MAX, N_BETA)
l_values = np.geomspace(L_MIN, L_MAX, N_L)
f2_values = np.zeros((N_L, N_BETA))  # f2_values[i, j]: l_values[i], beta_values[j]

for i, l in enumerate(l_values):
    for j, beta in enumerate(beta_values):
        # f2_square returns an 80-digit mpmath number; float() makes it an ordinary one
        f2_values[i, j] = float(fermion.f2_square(beta, l))

print(f"|F_2|^2 ranges from {f2_values.min():.3e} to {f2_values.max():.3e}")

fig, ax = plt.subplots(figsize=(7.5, 5.5))
mesh = ax.pcolormesh(
    linear_edges(beta_values),
    log_edges(l_values),
    f2_values,
    norm="log",
    cmap=COLOR_MAP,
    shading="flat",
)
ax.set_yscale("log")
ax.set_xlabel(r"$\beta = \sqrt{1 - 4M^2/q^2}$")
ax.set_ylabel(r"$\ell = \lambda^2 / M^2$")
ax.set_title(r"Pauli form factor, $|F_2^{(1)}|^2$")
fig.colorbar(mesh, label=r"$|F_2^{(1)}|^2$")
fig.tight_layout()
plt.show()

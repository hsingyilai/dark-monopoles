"""
This script plots |F_2|^2, returned by f2_square in fermion.py, against the monopole mass M
at a fixed collision energy sqrt(q^2), for a few vector boson masses lambda, and compares
it with a line proportional to M^4.
"""

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

import fermion

# ===== Parameters you can adjust =====

SQRT_Q2 = 13600.0  # fixed collision energy sqrt(q^2), in GeV

LAMBDA_VALUES = [0.1, 1.0, 10.0]  # vector boson masses lambda in GeV, one curve each

# Range of the monopole mass M in GeV, on a log scale. M must stay below SQRT_Q2 / 2.
M_MIN = 0.01
M_MAX = 0.99 * SQRT_Q2 / 2
N_M = 80  # number of log-spaced points per curve

# One color per curve, in the same order as LAMBDA_VALUES; add one if you add a lambda
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

# The M^4 reference line starts this many times above the first curve at M_MIN
REF_FACTOR = 10.0

# ===== End of adjustable parameters =====

m_values = np.geomspace(M_MIN, M_MAX, N_M)
# beta is computed with mpmath: for M << sqrt(q^2) it is so close to 1 that a plain float
# would keep only a few digits of 1 - beta^2, which F_2 is proportional to
beta_values = [mp.sqrt(1 - 4 * mp.mpf(m) ** 2 / mp.mpf(SQRT_Q2) ** 2) for m in m_values]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
f2_at_m_min = []  # each curve's value at M_MIN, used to place the reference line
for lam, color in zip(LAMBDA_VALUES, COLORS):
    # f2_square takes l = lambda^2 / M^2, which changes with M at fixed lambda.
    # It returns an 80-digit mpmath number; float() makes it an ordinary one.
    f2_values = [
        float(fermion.f2_square(beta, lam**2 / m**2))
        for beta, m in zip(beta_values, m_values)
    ]
    ax.loglog(
        m_values, f2_values, color=color, linewidth=2, label=rf"$\lambda$ = {lam:g} GeV"
    )
    f2_at_m_min.append(f2_values[0])

# Only the slope of the reference line matters, so scale it to sit near the curves
ref_scale = REF_FACTOR * f2_at_m_min[0] / M_MIN**4
ax.loglog(
    m_values,
    ref_scale * m_values**4,
    "--",
    color="gray",
    linewidth=1.5,
    label=r"$|F_2|^2 \propto M^4$",
)
ax.set_xlabel(r"Monopole mass $M$ (GeV)")
ax.set_ylabel(r"$|F_2^{(1)}|^2$")
ax.set_title(rf"Pauli form factor against $M$, at $\sqrt{{q^2}}$ = {SQRT_Q2:g} GeV")
ax.grid(which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()

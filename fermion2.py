"""
This script is for production of fermionic dark magnetic monopole, with the loop on the monopole vertex.

sigma_hat depends only on s = tau * S, so the cross section is split as
    sigma = integral d(ln tau) tau * sigma_hat(tau * S) * [integral dy luminosity]
and the slow, high-precision sigma_hat is computed once per tau instead of once per (x1, x2).
"""

import lhapdf
import mpmath as mp
import numpy as np
from scipy.integrate import quad

# Eq. 151 cancels terms of size l^2 / beta^5 down to F2 ~ 1/(3 l), so it needs far more digits
# than numpy's 16. 80 digits is enough for l up to ~1e8; for larger l, compare with 120.
mp.mp.dps = 80


def psi(p: float, xi: float) -> float:
    """Psi function defined in eq.149 of hep-ph/9707496v2
    Args:
        p: real number between 0 and 1.
        xi: real numbe > 0.
    """
    if xi < 4:
        term_1 = -2 * mp.atan(mp.sqrt((4 - xi) / xi)) ** 2
        term_2 = 2 * mp.re(
            mp.polylog(2, (1 + 0.5 * p * (-2 + xi + 1j * mp.sqrt(4 * xi - xi**2))))
        )
        term_3 = -1.5 * mp.pi**2 + 4 * mp.pi * mp.atan(mp.sqrt(4 * xi - xi**2) / xi)
        term_4 = 2 * mp.pi * mp.atan((2 * p + xi - 2) / mp.sqrt(4 * xi - xi**2))
        return term_1 + term_2 + term_3 + term_4
    else:
        term_1 = 0.5 * mp.log(0.5 * (xi - 2 + mp.sqrt(xi**2 - 4 * xi))) ** 2
        term_2 = mp.re(
            mp.polylog(2, (1 + 0.5 * p * (-2 + xi + mp.sqrt(xi**2 - 4 * xi)) + 0j))
        )
        term_3 = mp.re(
            mp.polylog(2, (1 + 0.5 * p * (-2 + xi - mp.sqrt(xi**2 - 4 * xi)) + 0j))
        )
        term_4 = -0.5 * mp.pi**2
        return term_1 + term_2 + term_3 + term_4


def phi(xi: float) -> float:
    """Phi function defined in eq.148 of hep-ph/9707496v2
    xi: real number bigger than 0.
    """
    if xi > 4:
        u = mp.sqrt(xi**2 - 4 * xi)
        return 0.5 * u * mp.log((xi - u) / (xi + u))
    elif xi == 0:
        return 0
    else:
        v = mp.sqrt(4 * xi - xi**2)
        return v * mp.atan(v / xi)


def re_f2(beta: float, l: float) -> float:
    """The real part of F_2 Pauli form factor
    beta: sqrt(1-4M^2/q^2)
    l: lambda^2/M^2, where lambda is the vector boson mass.
    """
    # Make sure every operation below is done with mp.mp.dps digits, not in plain floats
    beta = mp.mpf(beta)
    l = mp.mpf(l)
    p = (1 - beta) / (1 + beta)
    term_1 = (
        (1 - beta**2) ** 2
        / (4 * beta**3)
        * (1 + 3 * (1 - beta**2) / (8 * beta**2) * l)
        * l
        * psi(p, l)
    )
    term_2 = (
        (1 - beta**2)
        / (2 * beta**2)
        * (1 + (3 - 5 * beta**2) / (4 * beta**2) * l)
        * phi(l)
    )
    term_3 = (
        (1 + beta) ** 2
        * (1 - beta**2)
        / (16 * beta**4)
        * (
            (3 - 5 * beta**2) / (1 + beta) ** 2
            + 3 * (1 - beta) ** 2 / (2 * beta) * mp.log(p)
        )
        * l**2
        * mp.log(l)
    )
    term_4 = (
        (1 - beta**2)
        / (2 * beta**2)
        * (1 + (1 - beta**2) / (2 * beta) * mp.log(p))
        * l
        * mp.log(l)
    )
    term_5 = 3 * (1 - beta**2) ** 3 / (64 * beta**5) * mp.log(p) ** 2 * l**2
    term_6 = (
        (1 - beta**2)
        / (4 * beta**2)
        * (1 + (1 - beta**2) / (2 * beta) * mp.log(p) * (3 + mp.log(p)))
        * l
    )
    term_7 = (1 - beta**2) / (4 * beta) * mp.log(p)
    return term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7


def im_f2(beta: float, l: float) -> float:
    """The imaginary part of F_2 Pauli form factor
    beta: sqrt(1-4M^2/q^2)
    l: lambda^2/M^2, where lambda is the vector boson mass.
    """
    beta = mp.mpf(beta)
    l = mp.mpf(l)
    p = (1 - beta) / (1 + beta)
    term_1 = (1 - beta**2) / (4 * beta)
    term_2 = 3 * (1 - beta**2) ** 2 / (8 * beta**3) * l
    term_3 = (
        (1 - beta**2) ** 2
        / (4 * beta**3)
        * (1 + 3 * (1 - beta**2) / (8 * beta**2) * l)
        * l
        * mp.log(1 + (1 - p) ** 2 / (l * p))
    )
    return mp.pi * (term_1 + term_2 - term_3)


def f2_square(beta: float, l: float) -> float:
    """Magnitude square of F_2
    beta: sqrt(1-4M^2/q^2)
    l: lambda^2/M^2, where lambda is the vector boson mass.
    """
    return re_f2(beta, l) ** 2 + im_f2(beta, l) ** 2


def sigma_hat(
    s: float,
    m_chi: float,
    m_D: float,
    epsilon_g_D: float,
    alpha_D: float,
    q: float,
    l_decay: float,
) -> float:
    """Spin-averaged cross section for quark + antiquark (a specific color) -> fermionic dark monopole + antimonopole, in GeV^-2.
    Args:
        s: squared collision energy of the quark and antiquark, in GeV^2.
        m_chi: dark monopole mass, in GeV.
        m_D: dark photon mass, in GeV.
        epsilon_g_D: kinetic mixing times the dark magnetic coupling (dimensionless).
        alpha_D: g_D^2 / (4 pi), the dark magnetic coupling inside the monopole's loop.
        q: quark charge, 1/3 or 2/3.
        l_decay: decay rate of dark photon in GeV.
    """
    # Below threshold, s < (2 m_chi)^2, there isn't enough energy to make the pair
    if s <= 4 * m_chi**2:
        return 0.0

    # Velocity of the monopole in the quark-antiquark rest frame
    beta = mp.sqrt(1 - 4 * m_chi**2 / s)

    alpha = 1 / 137  # May need to update due to running coupling
    # Couplings: alpha q^2 from the tree-level quark vertex, and (alpha_D / pi)^2 from
    # F2 = (alpha_D / pi) F2^(1) on the monopole. (scalar.py had the loop on the quark: (alpha q^2 / pi)^2.)
    c_1 = alpha * q**2 * alpha_D**2 / 3 / np.pi**2 * epsilon_g_D**2
    # Split the formula into parts for programming
    c_2 = 4 * m_D**4 / m_chi**2
    c_3 = float(f2_square(beta, m_D**2 / m_chi**2))
    # Same loop-vertex spin sum as scalar.py, but the plain quark vertex replaces the scalar
    # current: (beta_q/2) (beta_chi/2)^3 becomes 2 (beta_chi/2)^3, for a massless quark
    c_4 = 2 * (0.25 - m_chi**2 / s) ** 1.5 / ((s - m_D**2) ** 2 + m_D**2 * l_decay**2)

    return c_1 * c_2 * c_3 * c_4


def luminosity(x1, x2, pid):
    """How often proton 1 gives a quark with momentum fraction x1 and proton 2 the
    matching antiquark with x2, plus the swapped case. pid: 1 = d, 2 = u, 3 = s, 4 = c."""
    Q = np.sqrt(
        x1 * x2 * S
    )  # collision energy of the quark pair, used as the PDF scale

    # LHAPDF returns x*f(x), so divide by x to get f(x)
    quark_1 = pdf.xfxQ(pid, x1, Q) / x1
    antiquark_2 = pdf.xfxQ(-pid, x2, Q) / x2
    antiquark_1 = pdf.xfxQ(-pid, x1, Q) / x1
    quark_2 = pdf.xfxQ(pid, x2, Q) / x2

    return quark_1 * antiquark_2 + antiquark_1 * quark_2


def parton_luminosity(tau, pid):
    """Luminosity integrated over the rapidity y at fixed tau = x1*x2,
    with x1 = sqrt(tau) e^y and x2 = sqrt(tau) e^-y."""
    y_max = -0.5 * np.log(tau)

    def f(y):
        x1 = np.sqrt(tau) * np.exp(y)
        x2 = np.sqrt(tau) * np.exp(-y)
        return luminosity(x1, x2, pid)

    # PDFs are interpolated from a grid, so they aren't smooth enough to ask for much better than 1e-4
    result, _ = quad(f, -y_max, y_max, epsabs=0, epsrel=1e-4, limit=200)
    return result


def integrand(ln_tau, pid, q):
    tau = np.exp(ln_tau)
    s_hat = sigma_hat(tau * S, M_CHI, M_D, EPS_G_D, ALPHA_D, q, L_DECAY)
    # dx1 dx2 = dtau dy, and dtau = tau d(ln tau): hence the extra factor of tau
    return tau * parton_luminosity(tau, pid) * s_hat


lhapdf.setVerbosity(0)
pdf = lhapdf.mkPDF("NNPDF31_lo_as_0118", 0)

S = 13600.0**2  # squared collision energy of the two protons (LHC Run 3), in GeV^2

M_CHI = 1000.0  # dark monopole mass, GeV
M_D = 1000.0  # dark photon mass, GeV
L_DECAY = 10  # dark photon decay rate, GeV
EPS_G_D = 0.001  # kinetic mixing times dark magnetic coupling
ALPHA_D = 0.1  # g_D^2 / (4 pi) in the monopole's loop; placeholder, set from your model
GEV2_TO_PB = 3.894e8  # 1 GeV^-2 = 3.894e8 picobarns

# Each quark type: (PDG code, electric charge)
QUARKS = [
    (1, 1 / 3),  # down
    (2, 2 / 3),  # up
    (3, 1 / 3),  # strange
    (4, 2 / 3),  # charm
    (5, 1 / 3),  # bottom
]


tau0 = 4 * M_CHI**2 / S  # smallest x1*x2 with enough energy to make the pair

total = 0.0
for pid, q in QUARKS:
    sigma_q, error = quad(
        integrand,
        np.log(tau0),
        0.0,  # ln(tau) runs from ln(tau0) to 0
        args=(pid, q),
        epsabs=0,
        epsrel=1e-4,
    )
    print(
        f"quark {pid}: {sigma_q * GEV2_TO_PB:.3e} pb   (error {error * GEV2_TO_PB:.1e} pb)"
    )
    total += sigma_q

print(f"total: {total * GEV2_TO_PB:.3e} pb")

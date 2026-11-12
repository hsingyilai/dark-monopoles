"""
This script is for production of fermionic dark magnetic monopole
"""

import lhapdf
from scipy.integrate import quad
import mpmath as mp
import numpy as np


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
    beta = mp.mpf(beta)
    l = mp.mpf(l)
    return re_f2(beta, l) ** 2 + im_f2(beta, l) ** 2


def sigma_hat(
    s: float,
    m_chi: float,
    m_D: float,
    m_q: float,
    epsilon_g_D_cube: float,
    q: float,
    l_decay: float,
) -> float:
    """Spin-averaged cross section for quark + antiquark (a specific color) -> fermionic dark monopole + antimonopole, in GeV^-2.
    Args:
        s: squared collision energy of the quark and antiquark, in GeV^2.
        m_chi: dark monopole mass, in GeV.
        m_D: dark photon mass, in GeV.
        m_q: mass of quark in GeV.
        epsilon_g_D_cube: kinetic mixing times the dark magnetic coupling^3 (dimensionless).
        q: quark charge, 1/3 or 2/3.
        l_decay: decay rate of dark photon in GeV.
    """
    # Below threshold there isn't enough energy: s must exceed (2 m_chi)^2 for the pair,
    # and (2 m_q)^2 for the quark velocity in c_4 to be real
    if s <= max(4 * m_chi**2, 4 * m_q**2):
        return 0.0

    e = 0.3  # electric coupling

    c_1 = 1 / np.pi * (epsilon_g_D_cube * q * e * m_D / 8 / np.pi**2 / m_chi) ** 2
    # Split the formula into parts for programming
    c_2 = (
        m_D**2
        * (m_D**2 + l_decay**2)
        / s**3
        / ((s - m_D**2) ** 2 + m_D**2 * l_decay**2)
    )
    c_3 = f2_square(np.sqrt(1 - 4 * m_chi**2 / s), m_D**2 / m_chi**2)
    c_4 = (s / 2 / np.sqrt(s - 4 * m_q**2) - np.sqrt(s / 4 - m_q**2) / 3) * (
        s / 4 - m_chi**2
    ) ** 1.5

    return c_1 * c_2 * c_3 * c_4


def parton_luminosity(v, pid):
    """Lum_q(tau) with tau = e^v: integral over u = ln(x1) from v to 0 of L_q(x1, tau/x1)."""

    def f(u):
        # x1 = e^u, x2 = tau / x1 = e^(v - u), and dx1/x1 = du
        return luminosity(np.exp(u), np.exp(v - u), pid)

    result, _ = quad(f, v, 0, epsabs=0, epsrel=1e-4, limit=200)
    return result


def integrand(v, pid, q, m_q):
    tau = np.exp(v)
    s_hat = float(sigma_hat(tau * S, M_CHI, M_D, m_q, EPS_G_D_CUBE, q, L_DECAY))
    # dtau = tau dv: hence the extra factor of tau
    return tau * s_hat * parton_luminosity(v, pid)


def sigma_from_quark(pid, q, m_q, epsrel=1e-4):
    """Total pp cross section from one quark flavour, in GeV^-2, and its error estimate.
    Uses the current module values of M_CHI, M_D, L_DECAY, EPS_G_D_CUBE and S.
    pid, q, m_q: PDG code, electric charge and mass of the quark, as in QUARKS.
    """
    # Start at the threshold: quad has trouble with the step in sigma_hat there
    v_min = np.log(max(4 * M_CHI**2, 4 * m_q**2) / S)

    # The resonance at s = m_D^2 is a peak about L_DECAY / m_D wide in v. Give it its own
    # short range so the peak can't fall between quad's sample points.
    v_pole = np.log(M_D**2 / S)
    half_width = 30 * L_DECAY / M_D
    if v_min < v_pole < 0:
        # a set drops duplicate limits when the window touches an end of the range
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
            integrand,
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


lhapdf.setVerbosity(0)
pdf = lhapdf.mkPDF("NNPDF31_lo_as_0118", 0)

S = 13600.0**2  # squared collision energy of the two protons (LHC Run 3), in GeV^2

M_CHI = 1.0  # dark monopole mass, GeV
M_D = 2.0  # dark photon mass, GeV
L_DECAY = 0.01  # dark photon decay rate, GeV
EPS_G_D_CUBE = 1  # kinetic mixing times dark magnetic coupling cubed
GEV2_TO_PB = 3.894e8  # 1 GeV^-2 = 3.894e8 picobarns

# Each quark type: (PDG code, electric charge, mass in GeV)
QUARKS = [
    (1, 1 / 3, 4.7e-3),  # down
    (2, 2 / 3, 2.2e-3),  # up
    (3, 1 / 3, 0.094),  # strange
    (4, 2 / 3, 1.27),  # charm
    (5, 1 / 3, 4.18),
]  # bottom


# Only runs for "python fermion.py", not when another script does "import fermion"
if __name__ == "__main__":
    total = 0.0
    for pid, q, m_q in QUARKS:
        sigma_q, error = sigma_from_quark(pid, q, m_q)
        print(
            f"quark {pid}: {sigma_q * GEV2_TO_PB:.3e} pb   (error {error * GEV2_TO_PB:.1e} pb)"
        )
        total += sigma_q

    print(f"total: {total * GEV2_TO_PB:.3e} pb")

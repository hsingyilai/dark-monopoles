import lhapdf
import numpy as np
from scipy.integrate import dblquad


def sigma_hat(s: float, m_chi: float, m_D: float, m_q: float, epsilon_g_D: float, q: float) -> float:
    """Spin-averaged cross section for quark + antiquark (a specific color) -> scalar dark monopole + antimonopole, in GeV^-2.
    Args:
        s: squared collision energy of the quark and antiquark, in GeV^2.
        m_chi: dark monopole mass, in GeV.
        m_D: dark photon mass, in GeV.
        m_q: quark mass, in GeV (up quark: 2.2e-3).
        epsilon_g_D: kinetic mixing times the dark magnetic coupling (dimensionless).
        q: quark charge, 1/3 or 2/3.
    """
    # Below threshold, s < (2 m_chi)^2, there isn't enough energy to make the pair
    if s <= 4 * m_chi**2:
        return 0.0               

    # 1 - beta^2 = 4 m_q^2 / s is computed directly: for light quarks it is
    # ~1e-12 or smaller, so computing 1 - beta would round to exactly 0.
    beta = np.sqrt(1 - 4 * m_q**2 / s)
    one_minus_beta2 = 4 * m_q**2 / s

    alpha = 1 / 137  # May need to update due to running coupling
    # Couplings
    c_1 = alpha**3/ 3 / np.pi**2 * epsilon_g_D**2 * q**6
    # Split the formula into parts for programming
    c_2 = m_D**4 / m_q**2 * (one_minus_beta2/ 2 / beta)**2
    c_3 = (np.log((1 + beta)**2/one_minus_beta2)**2 + np.pi**2)
    c_4 = np.sqrt(0.25 - m_q**2 / s) * (0.25 - m_chi**2 / s)**1.5 / (s - m_D**2)**2

    return c_1 * c_2 * c_3 * c_4


def luminosity(x1, x2, pid):
    """How often proton 1 gives a quark with momentum fraction x1 and proton 2 the
    matching antiquark with x2, plus the swapped case. pid: 1 = d, 2 = u, 3 = s, 4 = c."""
    Q = np.sqrt(x1 * x2 * S)   # collision energy of the quark pair, used as the PDF scale

    # LHAPDF returns x*f(x), so divide by x to get f(x)
    quark_1     = pdf.xfxQ(pid,  x1, Q) / x1
    antiquark_2 = pdf.xfxQ(-pid, x2, Q) / x2
    antiquark_1 = pdf.xfxQ(-pid, x1, Q) / x1
    quark_2     = pdf.xfxQ(pid,  x2, Q) / x2

    return quark_1 * antiquark_2 + antiquark_1 * quark_2


def integrand(x2, x1, pid, q, m_q):   # dblquad wants the INNER variable (x2) first
    s = x1 * x2 * S
    return luminosity(x1, x2, pid) * sigma_hat(s, M_CHI, M_D, m_q, EPS_G_D, q)

lhapdf.setVerbosity(0)
pdf = lhapdf.mkPDF("NNPDF31_lo_as_0118", 0)

S = 13600.0**2   # squared collision energy of the two protons (LHC Run 3), in GeV^2

M_CHI = 1000.0         # dark monopole mass, GeV
M_D = 500.0            # dark photon mass, GeV
EPS_G_D = 1.0          # kinetic mixing times dark magnetic coupling
GEV2_TO_PB = 3.894e8   # 1 GeV^-2 = 3.894e8 picobarns

# Each quark type: (PDG code, electric charge, mass in GeV)
QUARKS = [(1, 1/3, 4.7e-3),   # down
          (2, 2/3, 2.2e-3),   # up
          (3, 1/3, 0.094),    # strange
          (4, 2/3, 1.27),     # charm
          (5, 1/3, 4.18)]     # bottom


tau0 = 4 * M_CHI**2 / S   # smallest x1*x2 with enough energy to make the pair

total = 0.0
for pid, q, m_q in QUARKS:
    sigma_q, error = dblquad(integrand,
                             tau0, 1,                 # x1 runs from tau0 to 1
                             lambda x1: tau0 / x1, 1, # x2 runs from tau0/x1 to 1
                             args=(pid, q, m_q),
                             epsabs=0, epsrel=1e-4)
    print(f"quark {pid}: {sigma_q * GEV2_TO_PB:.3e} pb   (error {error * GEV2_TO_PB:.1e} pb)")
    total += sigma_q

print(f"total: {total * GEV2_TO_PB:.3e} pb")
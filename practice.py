import lhapdf
import numpy as np
import matplotlib.pyplot as plt


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
    c_2 = m_D**4 / m_q**2 *(one_minus_beta2/ 2 / beta)**2
    c_3 = (np.log((1 + beta)**2/one_minus_beta2)**2 + np.pi**2)
    c_4 = np.sqrt(0.25 - m_q**2 / s) * (0.25 - m_chi**2 / s)**1.5 / (s - m_D**2)**2

    return c_1 * c_2 * c_3 * c_4

for E in [1500.0, 2001.0, 2500.0, 4000.0, 10000.0]:
    print(E, "GeV:", "sigma", sigma_hat(E**2, 1000, 500, 2.2*10**(-3), 1, 2 / 3), "GeV^-2")



lhapdf.setVerbosity(0)
pdf = lhapdf.mkPDF("NNPDF31_lo_as_0118", 0)

Q = 200   # Collision scale in GeV

xs = np.logspace(-3, 0, 200)
ups = [pdf.xfxQ(2, x, Q) for x in xs]

plt.plot(xs, ups)
plt.xscale("log")
plt.xlabel("x (fraction of the proton's momentum)")
plt.ylabel("x f(x)")
plt.title(f"Up quark in the proton, Q = {Q} GeV")
plt.grid(alpha=0.3)
plt.show()
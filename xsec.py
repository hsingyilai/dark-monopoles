"""Leading-order cross sections for p p -> X Xbar through q qbar annihilation.

    sigma        = sum_q ∫dx1 dx2 [f_q(x1) f_qbar(x2) + f_qbar(x1) f_q(x2)] sigma_hat(x1 x2 S)
    sigma_hat(s) = beta/(32 pi s) ∫ dcos(theta) <|M|^2>,   beta = sqrt(1 - 4 M^2/s)

theta is the angle between the quark and X in the q qbar rest frame, in both beam
orderings, so the two orderings share one sigma_hat. (Lab-frame distributions will
need to tell them apart.) The squared amplitude is any me2(s, t, M, pid); see
amplitudes.py.

Two integrators for the same formula:
    hadronic_xsec_quad    deterministic Gauss-Legendre: fast and precise total cross section
    hadronic_xsec_vegas   Monte Carlo: slower, but extends to cuts and histograms
"""
from functools import lru_cache

import lhapdf
import numpy as np
import vegas

GEV2_TO_PB = 0.3893793721e9   # (hbar c)^2: 1 GeV^-2 = 3.894e8 pb
QUARKS_4F = (1, 2, 3, 4)      # d u s c: MadGraph's default proton
QUARKS_5F = (1, 2, 3, 4, 5)
DEFAULT_PDF = "NNPDF31_lo_as_0118"
DEFAULT_SQRT_S = 13600.0      # GeV, LHC Run 3

lhapdf.setVerbosity(0)


@lru_cache(maxsize=None)
def get_pdf(name=DEFAULT_PDF, member=0):
    return lhapdf.mkPDF(name, member)


def velocity(s, M):
    """beta of X in the q qbar rest frame (0 below threshold)."""
    return np.sqrt(np.clip(1 - 4*M**2/s, 0, None))


def mandelstam_t(s, M, cos_theta):
    return M**2 - 0.5*s*(1 - velocity(s, M)*cos_theta)


def gauss_legendre(n, a, b):
    """Gauss-Legendre nodes and weights on [a, b]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5*(b - a)*x + 0.5*(b + a), 0.5*(b - a)*w


def xfx(pdf, pid, x, mu):
    """x*f(x, mu) for arrays of any (broadcastable) shape."""
    x, mu = np.broadcast_arrays(x, mu)
    return np.asarray(pdf.xfxQ(pid, x.ravel(), mu.ravel())).reshape(x.shape)


def qqbar_luminosity(pdf, pid, x1, x2, mu):
    """f_q(x1) f_qbar(x2) + f_qbar(x1) f_q(x2). LHAPDF returns x*f(x), hence the division."""
    return (xfx(pdf, pid, x1, mu)*xfx(pdf, -pid, x2, mu)
            + xfx(pdf, -pid, x1, mu)*xfx(pdf, pid, x2, mu)) / (x1*x2)


def partonic_xsec(me2, s, M, pid, n_cos=48):
    """sigma_hat(s) in GeV^-2 for q qbar -> X Xbar; s may be an array."""
    s_arr = np.atleast_1d(np.asarray(s, dtype=float))
    sigma = np.zeros_like(s_arr)
    above = s_arr > 4*M**2
    s_above = s_arr[above]
    c, w = np.polynomial.legendre.leggauss(n_cos)
    me2_grid = me2(s_above[:, None], mandelstam_t(s_above[:, None], M, c), M, pid)
    sigma[above] = velocity(s_above, M)/(32*np.pi*s_above) * (w*me2_grid).sum(axis=1)
    return sigma if np.ndim(s) else float(sigma[0])


def hadronic_xsec_quad(me2, M, sqrt_s=DEFAULT_SQRT_S, pdf_name=DEFAULT_PDF, member=0,
                       quarks=QUARKS_4F, mu_factor=1.0, n_tau=64, n_y=96, n_cos=48):
    """Total sigma(p p -> X Xbar) in pb, by Gauss-Legendre quadrature.

    Variables: tau = x1 x2 = tau0^(1 - v^2), a log scale for the steep fall-off
    in which v^2 also smooths the beta^n threshold; y = ln(x1/x2)/2, the rapidity
    of the q qbar pair. The factorisation scale is mu_F = mu_factor*sqrt(s_hat);
    mu_factor = 1 matches MadGraph's dynamical_scale_choice = 4.

    Precision is about 1e-5 relative. It is limited by the y integral, whose ends
    have one x -> 1, where PDFs are not smooth.
    """
    S = sqrt_s**2
    tau0 = 4*M**2/S
    if tau0 >= 1:
        return 0.0
    pdf = get_pdf(pdf_name, member)

    v, w_v = gauss_legendre(n_tau, 0.0, 1.0)
    tau = tau0**(1 - v**2)
    jac_tau = w_v * tau*np.log(1/tau0)*2*v         # dtau = tau ln(1/tau0) 2v dv
    s = tau*S

    u, w_u = np.polynomial.legendre.leggauss(n_y)
    ymax = -0.5*np.log(tau)[:, None]
    y = ymax*u                                     # shape (n_tau, n_y)
    x1 = np.sqrt(tau)[:, None]*np.exp(y)
    x2 = np.sqrt(tau)[:, None]*np.exp(-y)
    mu = mu_factor*np.sqrt(s)[:, None]

    total = 0.0
    for pid in quarks:
        lum = (qqbar_luminosity(pdf, pid, x1, x2, mu) * ymax*w_u).sum(axis=1)   # ∫dy
        total += np.sum(jac_tau*lum*partonic_xsec(me2, s, M, pid, n_cos))
    return total*GEV2_TO_PB


def hadronic_xsec_vegas(me2, M, sqrt_s=DEFAULT_SQRT_S, pdf_name=DEFAULT_PDF, member=0,
                        quarks=QUARKS_4F, mu_factor=1.0, neval=20000, nitn=10):
    """Total sigma(p p -> X Xbar) in pb, by vegas Monte Carlo over (tau, y, cos theta).

    Returns a gvar: .mean and .sdev in pb. A .Q above ~0.1 means the
    iterations agree with each other.
    """
    S = sqrt_s**2
    tau0 = 4*M**2/S
    if tau0 >= 1:
        return 0.0
    pdf = get_pdf(pdf_name, member)

    @vegas.lbatchintegrand
    def integrand(r):              # r has shape (n_points, 3), each column in [0, 1]
        tau = tau0**(1 - r[:, 0])
        jac = tau*np.log(1/tau0)
        ymax = -0.5*np.log(tau)
        y = ymax*(2*r[:, 1] - 1)
        jac *= 2*ymax
        cos_theta = 2*r[:, 2] - 1
        jac *= 2
        x1, x2 = np.sqrt(tau)*np.exp(y), np.sqrt(tau)*np.exp(-y)
        s = tau*S
        t = mandelstam_t(s, M, cos_theta)
        mu = mu_factor*np.sqrt(s)
        weight = sum(qqbar_luminosity(pdf, pid, x1, x2, mu)*me2(s, t, M, pid) for pid in quarks)
        return jac*velocity(s, M)/(32*np.pi*s)*weight*GEV2_TO_PB

    integ = vegas.Integrator(3*[[0, 1]])
    integ(integrand, nitn=nitn, neval=neval)       # first pass only trains the sampling grid
    return integ(integrand, nitn=nitn, neval=neval)

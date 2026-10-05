"""Spin- and colour-averaged squared amplitudes <|M|^2> for q qbar -> X Xbar.

Every function here has the signature  me2(s, t, M, pid):
    s, t   Mandelstam invariants in GeV^2, with t = (p_q - p_X)^2  (numpy arrays allowed)
    M      mass of X in GeV
    pid    PDG id of the incoming quark (1 = d, 2 = u, 3 = s, 4 = c, 5 = b)
and returns <|M|^2> = (1/4)(1/9) * sum over spins and colours of |M|^2 (dimensionless).

Extra model parameters are best bound with functools.partial, e.g.
    me2 = functools.partial(me2_monopole, g_D=..., eps=...)
"""
import numpy as np

ALPHA_EM = 1/127.9   # MadGraph 'sm' default, so the test process can be compared with MadGraph
QUARK_CHARGE = {1: -1/3, 2: 2/3, 3: -1/3, 4: 2/3, 5: -1/3}


def me2_test_fermion(s, t, M, pid):
    """q qbar -> gamma* -> F Fbar, with F a Dirac fermion of unit charge and mass M.

    A process with a known answer, used to validate the integration code. In MadGraph
    it is `generate p p > a > ta+ ta-` with the tau mass set to M.
    """
    e2 = 4*np.pi*ALPHA_EM
    u = 2*M**2 - s - t
    colour_factor = 3/9   # sum over colours / average over the 9 initial colour states
    spin_avg = 2*e2**2*QUARK_CHARGE[pid]**2 * ((M**2 - t)**2 + (M**2 - u)**2 + 2*M**2*s) / s**2
    return colour_factor*spin_avg


def sigma_hat_test_fermion(s, M, pid):
    """Analytic partonic cross section of me2_test_fermion, in GeV^-2."""
    beta = np.sqrt(np.clip(1 - 4*M**2/s, 0, None))
    return QUARK_CHARGE[pid]**2/3 * 4*np.pi*ALPHA_EM**2/(3*s) * beta*(3 - beta**2)/2


def me2_monopole(s, t, M, pid):
    """q qbar -> dark monopole + dark antimonopole.  TODO: insert your amplitude."""
    raise NotImplementedError("Insert the dark-monopole <|M|^2> in amplitudes.me2_monopole")

"""Validate xsec.py on a process with a known answer: q qbar -> gamma* -> F Fbar.

1. Partonic: numerical cos(theta) integral of the amplitude vs the analytic sigma_hat.
2. Hadronic: quadrature (and its convergence) vs vegas Monte Carlo.
These test the code against itself and against analytic results. The independent
check of the PDF convolution is MadGraph; see README.md.
"""
import time

import numpy as np

import amplitudes as amp
import xsec


def check_partonic():
    print("1. Partonic sigma_hat: numerical cos(theta) integral vs analytic formula")
    worst = 0.0
    for M in (300.0, 1000.0, 3000.0):
        s = (2*M)**2*np.array([1.0001, 1.01, 1.5, 4.0, 100.0])
        for pid in (1, 2):
            num = xsec.partonic_xsec(amp.me2_test_fermion, s, M, pid)
            ana = amp.sigma_hat_test_fermion(s, M, pid)
            worst = max(worst, np.max(np.abs(num/ana - 1)))
    ok = worst < 1e-10
    print(f"   max relative difference {worst:.1e}  ->  {'OK' if ok else 'FAIL'}\n")
    return ok


def check_hadronic():
    print(f"2. sigma(pp -> F Fbar) at sqrt(S) = {xsec.DEFAULT_SQRT_S/1000:g} TeV, "
          f"{xsec.DEFAULT_PDF}, mu = sqrt(s_hat)")
    print(f"   {'M [GeV]':>8} {'quadrature [pb]':>16} {'quad conv.':>11} {'vegas [pb]':>22} {'pull':>6}")
    ok = True
    for M in (250.0, 500.0, 1000.0, 2000.0, 3000.0):
        quad = xsec.hadronic_xsec_quad(amp.me2_test_fermion, M)
        quad_fine = xsec.hadronic_xsec_quad(amp.me2_test_fermion, M, n_tau=128, n_y=192, n_cos=96)
        mc = xsec.hadronic_xsec_vegas(amp.me2_test_fermion, M)
        conv = abs(quad/quad_fine - 1)
        pull = (mc.mean - quad)/mc.sdev
        ok &= conv < 1e-4 and abs(pull) < 3   # 1e-4: far below the percent-level PDF uncertainty
        print(f"   {M:8.0f} {quad:16.6e} {conv:11.1e} {mc.mean:13.6e} ± {mc.sdev:.1e} {pull:+6.2f}")
    print("   quad conv. = relative change when doubling the quadrature points")
    print("   pull = (vegas - quadrature)/vegas error; should be within about ±2")
    print(f"   ->  {'OK' if ok else 'FAIL'}\n")
    return ok


if __name__ == "__main__":
    t0 = time.perf_counter()
    results = [check_partonic(), check_hadronic()]
    print(f"{'ALL CHECKS PASSED' if all(results) else 'SOME CHECKS FAILED'} "
          f"({time.perf_counter() - t0:.1f} s)")

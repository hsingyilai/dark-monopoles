"""sigma(p p -> X Xbar) versus mass, with scale and PDF uncertainties, saved to CSV.

Switch ME2 and LABEL to the monopole once amplitudes.me2_monopole is filled in.
"""
import csv
import time
from pathlib import Path

import lhapdf
import numpy as np

import amplitudes as amp
import xsec

ME2 = amp.me2_test_fermion
LABEL = "test_fermion"
SQRT_S = 13600.0                          # GeV
MASSES = np.arange(250.0, 4001.0, 250.0)  # GeV
PDF_NAME = xsec.DEFAULT_PDF


def sigma_with_uncertainties(M, pdfset):
    """Central sigma (PDF member 0) and the edges of its scale and PDF bands, all in pb."""
    central = xsec.hadronic_xsec_quad(ME2, M, SQRT_S, PDF_NAME)

    # Scale: at LO with no alpha_s, only mu_F moves (through the PDFs). Vary it by 1/2 and 2.
    varied = [central] + [xsec.hadronic_xsec_quad(ME2, M, SQRT_S, PDF_NAME, mu_factor=k)
                          for k in (0.5, 2.0)]

    # PDF: one cross section per set member. alternative=True gives the 68% interval of
    # the replicas, not a symmetric standard deviation: at multi-TeV masses the replica
    # spread (large-x antiquarks) is wide and lopsided.
    per_member = [xsec.hadronic_xsec_quad(ME2, M, SQRT_S, PDF_NAME, member=i)
                  for i in range(pdfset.size)]
    unc = pdfset.uncertainty(per_member, alternative=True)
    return central, min(varied), max(varied), unc.central - unc.errminus, unc.central + unc.errplus


def main():
    pdfset = lhapdf.getPDFSet(PDF_NAME)
    outfile = Path("results")/f"sigma_vs_mass_{LABEL}_{SQRT_S/1000:g}TeV.csv"
    outfile.parent.mkdir(exist_ok=True)

    print(f"sigma(pp -> X Xbar), X = {LABEL}, sqrt(S) = {SQRT_S/1000:g} TeV, {PDF_NAME}")
    print(f"{'M [GeV]':>8} {'sigma [pb]':>12} {'scale band':>16} {'PDF band (68%)':>18}")
    t0 = time.perf_counter()
    rows = []
    for M in MASSES:
        sigma, s_lo, s_hi, p_lo, p_hi = sigma_with_uncertainties(M, pdfset)
        rows.append([M, sigma, s_lo, s_hi, p_lo, p_hi])
        pct = lambda edge: f"{100*(edge/sigma - 1):+.0f}%"
        print(f"{M:8.0f} {sigma:12.4e} {pct(s_lo):>8} {pct(s_hi):>7} {pct(p_lo):>10} {pct(p_hi):>7}")

    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M_GeV", "sigma_pb", "scale_low_pb", "scale_high_pb", "pdf_low_pb", "pdf_high_pb"])
        writer.writerows(rows)
    print(f"\nWrote {outfile} ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()

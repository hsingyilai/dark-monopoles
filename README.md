# dark-monopoles

LHC production of dark monopole pairs, q q̄ → M M̄, from an analytic amplitude
convolved with proton PDFs.

## Setup

Python 3.12 venv in `.venv`, with LHAPDF compiled into it (LHAPDF is not on PyPI).

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
./scripts/install_lhapdf.sh      # builds LHAPDF 6.5.6 into .venv, downloads NNPDF31_lo_as_0118
```

More PDF sets: `.venv/bin/lhapdf install <set name>` (grids live in `.venv/share/LHAPDF`).
For MadGraph later, point it at `.venv/bin/lhapdf-config` so both use the same PDFs.

## Code

| File | What it does |
|---|---|
| `amplitudes.py` | Squared amplitudes `me2(s, t, M, pid)`: a test process with a known answer, and `me2_monopole` (to fill in) |
| `xsec.py` | Partonic and hadronic cross sections: `hadronic_xsec_quad` (deterministic, ~20 ms) and `hadronic_xsec_vegas` (Monte Carlo, ~1 s; for cuts and histograms later) |
| `validate.py` | Checks the integration code against analytic results and itself (~5 s) |
| `sigma_vs_mass.py` | σ(M) with scale and PDF (68% replica interval) bands, written to `results/*.csv` (~35 s) |

```bash
.venv/bin/python validate.py
.venv/bin/python sigma_vs_mass.py
```

### Adding the monopole amplitude

1. Fill in `amplitudes.me2_monopole`: return the spin- and colour-averaged |M|²,
   with t = (p_quark − p_monopole)², vectorised over numpy arrays `s` and `t`.
   Bind extra couplings with `functools.partial`.
2. In `sigma_vs_mass.py`, set `ME2 = amp.me2_monopole` and `LABEL = "monopole"`.

### Cross-check against MadGraph (later)

The only check of the PDF convolution independent of this code. The test process
in MadGraph is `import model sm`, `generate p p > a > ta+ ta-`, with
`MTA = M` in the param_card. In the run_card set: `ebeam1 = ebeam2 = 6800`,
`pdlabel = lhapdf`, `lhaid = 315000`, `dynamical_scale_choice = 4`, and no
generation cuts on the taus. Compare with the quadrature column of `validate.py`
(e.g. 1.0021e-04 pb at M = 1 TeV).

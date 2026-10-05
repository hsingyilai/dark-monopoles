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

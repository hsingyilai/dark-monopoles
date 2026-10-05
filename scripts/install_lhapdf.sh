#!/usr/bin/env bash
# LHAPDF is not on PyPI, so build it from source and install it into the
# project venv: afterwards `import lhapdf` and the `lhapdf` command work there.
set -euo pipefail

LHAPDF_VERSION="${LHAPDF_VERSION:-6.5.6}"
PDF_SETS=(NNPDF31_lo_as_0118)

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_DIR/.venv"
export PATH="$VENV/bin:$PATH"   # so configure finds the venv's python and cython
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

# Needed to build LHAPDF's Python bindings
"$VENV/bin/pip" install --upgrade setuptools cython

cd "$BUILD_DIR"
curl -fsSL "https://lhapdf.hepforge.org/downloads/?f=LHAPDF-${LHAPDF_VERSION}.tar.gz" | tar xz
cd "LHAPDF-${LHAPDF_VERSION}"
./configure --prefix="$VENV" PYTHON="$VENV/bin/python"
make -j"$(sysctl -n hw.ncpu)"
make install

# PDF grids go to .venv/share/LHAPDF
"$VENV/bin/lhapdf" install "${PDF_SETS[@]}"

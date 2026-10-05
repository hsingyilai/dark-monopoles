# CLAUDE.md

LHC phenomenology of dark magnetic monopoles: σ(pp → M M̄) through q q̄ annihilation at
leading order, from an analytic amplitude convolved with LHAPDF PDFs. Magnetic charges have
no standard local Lagrangian, so FeynRules → UFO → MadGraph cannot be used directly.

Keep this file current: when a change affects anything documented here (status, interfaces,
conventions, environment), update it in the same commit.

## Working with the owner

Fluent in Python only, new to LHAPDF, MadGraph, Pythia, Fortran/C++ and git. Explain HEP
tools and git steps in plain terms. Ask before rewriting git history or force-pushing.

## Environment

- `.venv`: Python 3.12 (Homebrew `python3.12` on macOS). Run everything with `.venv/bin/python`.
- LHAPDF 6.5.6 is not on PyPI. `scripts/install_lhapdf.sh` compiles it into `.venv`
  (Python module, `lhapdf`, `lhapdf-config`). PDF grids live in `.venv/share/LHAPDF`.
- Default PDF: `NNPDF31_lo_as_0118` (LHAID 315000), 100 replicas plus member 0.
- The install script and README are macOS-specific (`sysctl -n hw.ncpu`, Homebrew Python
  path). On Linux use `nproc`. On Windows use WSL2: LHAPDF and MadGraph don't run natively.

## Code

- `amplitudes.py`: every amplitude is `me2(s, t, M, pid)`:
  - Returns the spin- and colour-averaged ⟨|M|²⟩, dimensionless.
  - t = (p_quark − p_X)²; `pid` is the quark PDG id (1–5).
  - Must be numpy-vectorised: s and t may be 2-D arrays.
  - Bind extra couplings with `functools.partial`.

  `me2_test_fermion` (q q̄ → γ* → F F̄, with analytic `sigma_hat_test_fermion`) is the
  validation process. `me2_monopole` is **not implemented** (raises), so every number
  produced so far is for the test fermion, not the monopole.
- `xsec.py`:
  - `partonic_xsec`: Gauss–Legendre over cos θ.
  - `hadronic_xsec_quad`: deterministic, ~20 ms, the default for totals. Variables are
    τ = τ0^(1−v²) and the q q̄ rapidity y.
  - `hadronic_xsec_vegas`: Monte Carlo, ~1 s, for cuts and histograms.
- `validate.py`: run after any change to `xsec.py`; all checks must pass (~5 s).
- `sigma_vs_mass.py`: σ(M) with bands, written to `results/` (generated output, not
  committed). Switch `ME2`/`LABEL` there to the monopole once it is implemented.

## Conventions

- Natural units, GeV. σ̂ in GeV⁻², σ reported in pb (`GEV2_TO_PB`).
- Heaviside–Lorentz charges: α = e²/4π, Dirac quantization e·g = 2πn. Many monopole papers
  use Gaussian units (α = e², e·g = n/2): convert with e_HL = √(4π) e_G, g_HL = √(4π) g_G.
- MadGraph-compatible defaults, so results can be compared with MadGraph:
  - α = 1/127.9;
  - a 4-flavour proton (d u s c), the same as MadGraph's `p`;
  - μ_F = `mu_factor`·√ŝ, matching `dynamical_scale_choice = 4`;
  - √S = 13.6 TeV.
- θ is measured from the quark in both beam orderings, so both orderings share σ̂.
  Lab-frame distributions must distinguish the orderings.

## Numerics: known, intended behaviour

- LHAPDF's Python `xfxQ(pid, x_array, q_array)` works element by element and is fast;
  `xsec.xfx` flattens N-d arrays for it.
- The quadrature converges to about 1e-5 relative. The limit comes from the y integral:
  at its ends one x → 1, where PDFs are not smooth. This is not a bug, and the
  convergence threshold in `validate.py` is 1e-4.
- Uncertainties:
  - Scale: μ_F ×½ and ×2. At LO with no α_s, only the PDFs move.
  - PDF: the 68% replica interval (`pdfset.uncertainty(..., alternative=True)`), with
    member 0 as the central value. The replica spread is skewed at multi-TeV masses
    (large-x antiquarks): −48%/+170% at 3 TeV, so a symmetric standard deviation misleads.

## Status and plan

- Done: Step 1, the total cross section with validation (PR #1, merged).
- Next: the owner supplies the monopole ⟨|M|²⟩. Check units and the averaging factors
  (1/4 spin, 1/3 colour), and add checks for its symmetries and threshold behaviour.
- Pending: an independent MadGraph check of the PDF convolution.
  - Process: `import model sm`, `generate p p > a > ta+ ta-` with `MTA = M`.
  - Run card: `lhaid = 315000`, `dynamical_scale_choice = 4`, and no cuts on the taus.
  - Target: 1.0021e-04 pb at M = 1 TeV.
- Step 2, only if events are needed: replace the matrix element in a MadGraph process.
  - Placeholder UFO: PDG code 4110000, the monopole's spin, one s-channel diagram.
  - Commands: `set group_subprocesses False`, then `output madevent ... --hel_recycling=False`.
  - Overwrite `ANS` at the end of `SMATRIX` in each `SubProcesses/P*/matrix.f`.
  - Validate with the test formula before inserting the monopole.
- Physics caveats:
  - Dark monopoles are invisible, so observability needs a recoiling jet (monojet).
  - In Delphes, add `add EnergyFraction {4110000} {0.0 0.0}`.
  - The PDF set choice (LO NNPDF3.1 vs PDF4LHC21 or NNPDF4.0) is the owner's decision.

"""
This script estimates the heaviest dark monopole that gives at least one produced pair at the
LHC, with the kinetic mixing epsilon at the largest value the dark photon limits in
dark_photon.py allow, for:
  the s-channel  q qbar -> gamma* -> A' -> chi chibar  (results/ file written by 2D_scan.py)
  photon fusion  gamma gamma -> chi chibar             (results/ file from boson_merge)
  and their sum (different initial states, so they do not interfere).

Both scans were run at one epsilon (EPS_SCAN) and are rescaled point by point:
  s-channel    sigma ~ eps^2, times the change of BR(A' -> chi chibar) when m_D > 2 M_CHI
  photon fusion  sigma ~ eps^4 (exact up to the A' width, which is negligible there)

Which limits apply depends on how the dark photon decays:
  m_D > 2 M_CHI  it decays into chi chibar: the invisible limits (dark_photon.LIMITS), and,
                 with WITH_DILEPTON, also dilepton searches through its small visible branching
  m_D < 2 M_CHI  it decays visibly: dilepton bound without dark decays plus the electroweak
                 precision bounds, which hold whatever the decays. Only where the dilepton
                 curve exists (3-68 GeV): above it, LHC dilepton searches limit a visible dark
                 photon to eps ~ 1e-3, but that curve is not digitized.
Where no digitized curve covers a mass, epsilon is unknown and the point is skipped.

Photon fusion grows as eps^4 and barely depends on m_D above ~30 GeV, so its reach is set by
the heaviest dark photon allowed in. Above M_D_MAX the compilation has only the CMS monojet
bound (eps up to ~1 at 2 TeV), so those masses are left out of the main numbers and shown
separately.
"""

from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import dark_photon as dp
import fermion

# ===== Parameters you can adjust =====

RESULTS = Path(__file__).parent / "results"
S_CHANNEL_FILE = RESULTS / "s_channel_13.6TeV_eps0.01_gD0.3_21x21.csv"
GAMMA_GAMMA_FILE = RESULTS / "gamma_gamma_13.6TeV_eps0.01_gD0.3_21x21.csv"
EPS_SCAN = 0.01  # epsilon used in both scans
G_D = 0.3  # dark coupling used in both scans

LUMINOSITIES = {"LHC Run 3": 300.0, "HL-LHC": 3000.0}  # integrated luminosity in fb^-1

# Heaviest dark photon used in the main numbers: the electroweak precision curve
# (EWPT_CURTIN) ends here, and above it only the CMS monojet bound remains
M_D_MAX = 283.9
# Other values of M_D_MAX to show how the photon-fusion reach depends on it
M_D_MAX_SCAN = [68.0, 150.0, 283.9, 562.0, 1000.0, 2114.0]

# Photon fusion is smooth in both masses, so it is also interpolated onto a fine grid
N_FINE = 800  # points per axis of the fine grid

OUTPUT_PNG = RESULTS / "reach_13.6TeV_gD0.3.png"

# ===== End of adjustable parameters =====

ALPHA_D = G_D**2 / (4 * np.pi)


def load_grid(path):
    """Read a results file into (m_D values, m_chi values, sigma[i_D, i_chi])."""
    columns = np.loadtxt(path, delimiter=",", unpack=True)
    m_d_values, m_chi_values = np.unique(columns[0]), np.unique(columns[1])
    return m_d_values, m_chi_values, columns[2].reshape(len(m_d_values), len(m_chi_values))


def curve_at(mass, curve):
    """A limit curve's epsilon at the given mass(es), log-log interpolated. NaN outside its
    mass range, and where the curve is at epsilon = 1 (a vetoed window, no limit)."""
    mass = np.asarray(mass, dtype=float)
    out = np.full(mass.shape, np.nan)
    inside = (mass >= curve[0, 0]) & (mass <= curve[-1, 0])
    out[inside] = 10 ** np.interp(
        np.log10(mass[inside]), np.log10(curve[:, 0]), np.log10(curve[:, 1])
    )
    out[out >= 1] = np.nan
    return out


def strongest(*limits):
    """The smallest epsilon among limits; NaN only where none applies."""
    return reduce(np.fmin, limits)


def width_shape(x):
    """(1 + 2x^2) sqrt(1 - 4x^2): the mass dependence of Gamma(A' -> chi chibar), x = m_chi/m_D."""
    return (1 + 2 * x**2) * np.sqrt(np.clip(1 - 4 * x**2, 0, None))


def z_invisible_curve():
    """Harigaya et al. Z invisible width bound for a 0.5% width, with the vertical end
    segments reduced to the points that join the rest of the curve."""
    curve = dp.Z_INVISIBLE_BY_WIDTH[0.005]
    masses = np.unique(curve[:, 0])
    first, last = curve[curve[:, 0] == masses[0]], curve[curve[:, 0] == masses[-1]]
    middle = curve[(curve[:, 0] > masses[0]) & (curve[:, 0] < masses[-1])]
    return np.vstack([first[-1:], middle, last[:1]])


Z_INVISIBLE = z_invisible_curve()


def eps_invisible(m_d, m_chi, with_dilepton):
    """Largest allowed epsilon when the dark photon decays into chi chibar."""
    eps = dp.epsilon_max(m_d)
    if not with_dilepton:
        return eps
    x = m_chi / m_d
    # The Alonso et al. dilepton curves assume m_chi = m_D / 3. A dark photon with our x has
    # the invisible width of one with alpha_eff at x = 1/3, so interpolate the curves in
    # log(alpha_D) at alpha_eff.
    alpha_eff = ALPHA_D * width_shape(x) / width_shape(1 / 3)
    keys = np.array(sorted(dp.DILEPTON_BY_ALPHA_D))
    log_eps = np.log10([curve_at(m_d, dp.DILEPTON_BY_ALPHA_D[k]) for k in keys])
    position = np.clip(np.interp(np.log10(alpha_eff), np.log10(keys), np.arange(len(keys))),
                       0, len(keys) - 1 - 1e-12)
    low = np.floor(position).astype(int)
    fraction = position - low
    rows = np.indices(np.shape(m_d))
    below = log_eps[(low, *rows)]
    above = log_eps[(low + 1, *rows)]
    dilepton = 10 ** ((1 - fraction) * below + fraction * above)
    # Harigaya et al. LHC dilepton bound above the Z: scaled from their 0.5% width curve
    # as (invisible width / mass)^(1/4)
    width_fraction = ALPHA_D / 3 * width_shape(x)
    lhc = curve_at(m_d, dp.LHC_DILEPTON_BY_WIDTH[0.005]) * (width_fraction / 0.005) ** 0.25
    # Z invisible width bound for a 0.5% width, not rescaled
    z_inv = curve_at(m_d, Z_INVISIBLE)
    return strongest(eps, dilepton, lhc, z_inv)


def eps_visible(m_d):
    """Largest allowed epsilon when the dark photon cannot decay into chi chibar. NaN outside
    the dilepton curve's mass range, where the strongest bound (dilepton) is missing."""
    eps = strongest(
        curve_at(m_d, dp.DILEPTON_NO_DARK_DECAYS),
        curve_at(m_d, dp.EWPT_CURTIN),
        curve_at(m_d, dp.EWPO_ALONSO),
        curve_at(m_d, dp.PEW_HARIGAYA),
    )
    dilepton = dp.DILEPTON_NO_DARK_DECAYS
    outside = (m_d < dilepton[0, 0]) | (m_d > dilepton[-1, 0])
    return np.where(outside, np.nan, eps)


def eps_allowed(m_d, m_chi, with_dilepton):
    m_d, m_chi = np.broadcast_arrays(np.asarray(m_d, float), np.asarray(m_chi, float))
    return np.where(
        m_d > 2 * m_chi, eps_invisible(m_d, m_chi, with_dilepton), eps_visible(m_d)
    )


def branching_to_chi(m_d, m_chi, eps):
    """BR(A' -> chi chibar), with the same widths as fermion.dark_photon_width."""
    pair = np.vectorize(fermion.width_to_pair)
    gamma_chi = pair(ALPHA_D, m_d, m_chi)
    alpha_mix = eps**2 * fermion.ALPHA
    gamma_sm = sum(pair(alpha_mix, m_d, m) for m in fermion.LEPTON_MASSES)
    gamma_sm = gamma_sm + fermion.R_HADRONS * pair(alpha_mix, m_d, fermion.LEPTON_MASSES[1])
    return gamma_chi / (gamma_chi + gamma_sm)


def rescaled(m_d, m_chi, sigma_s, sigma_gg, with_dilepton):
    """Both cross sections at the largest allowed epsilon, and that epsilon."""
    eps = eps_allowed(m_d, m_chi, with_dilepton)
    br_factor = np.where(
        m_d > 2 * m_chi,
        branching_to_chi(m_d, m_chi, eps) / branching_to_chi(m_d, m_chi, EPS_SCAN),
        1.0,
    )
    s = None if sigma_s is None else sigma_s * (eps / EPS_SCAN) ** 2 * br_factor
    gg = sigma_gg * (eps / EPS_SCAN) ** 4
    return eps, s, gg


def heaviest(m_d, m_chi, events, eps, m_d_max=M_D_MAX):
    """The heaviest m_chi with at least one event and m_D <= m_d_max, and the m_D, epsilon
    and events there."""
    ok = (events >= 1) & np.isfinite(events) & (m_d <= m_d_max)
    if not ok.any():
        return None
    best = np.argmax(np.where(ok, m_chi, -np.inf))
    index = np.unravel_index(best, m_chi.shape)
    return m_chi[index], m_d[index], eps[index], events[index]


m_d_values, m_chi_values, sigma_s = load_grid(S_CHANNEL_FILE)
_, _, sigma_gg = load_grid(GAMMA_GAMMA_FILE)
M_D, M_CHI = np.meshgrid(m_d_values, m_chi_values, indexing="ij")

# fine grid for photon fusion: interpolate log(sigma) in (log m_D, log m_chi)
interpolate = RegularGridInterpolator(
    (np.log(m_d_values), np.log(m_chi_values)), np.log(sigma_gg)
)
fine_d = np.geomspace(m_d_values[0], m_d_values[-1], N_FINE)
fine_chi = np.geomspace(m_chi_values[0], m_chi_values[-1], N_FINE)
FINE_D, FINE_CHI = np.meshgrid(fine_d, fine_chi, indexing="ij")
sigma_gg_fine = np.exp(interpolate(np.stack([np.log(FINE_D), np.log(FINE_CHI)], axis=-1)))

results = {}
for with_dilepton in (False, True):
    option = "invisible + dilepton limits" if with_dilepton else "invisible limits only"
    eps, s, gg = rescaled(M_D, M_CHI, sigma_s, sigma_gg, with_dilepton)
    eps_fine, _, gg_fine = rescaled(FINE_D, FINE_CHI, None, sigma_gg_fine, with_dilepton)
    print(f"\n===== {option} (g_D = {G_D:g}, 13.6 TeV, produced events, no cuts, "
          f"m_D <= {M_D_MAX:g} GeV) =====")
    for name, lumi in LUMINOSITIES.items():
        to_events = 1000 * lumi  # pb -> events
        print(f"--- {name}, {lumi:g} fb^-1 ---")
        rows = [
            ("s-channel only (grid)", M_D, M_CHI, s * to_events, eps),
            ("photon fusion only (grid)", M_D, M_CHI, gg * to_events, eps),
            ("photon fusion only (fine)", FINE_D, FINE_CHI, gg_fine * to_events, eps_fine),
            ("both channels (grid)", M_D, M_CHI, (s + gg) * to_events, eps),
        ]
        for label, md, mc, events, e in rows:
            found = heaviest(md, mc, events, e)
            if found is None:
                print(f"  {label:28s} no point reaches one event")
                continue
            m_chi_best, m_d_best, eps_best, n = found
            print(f"  {label:28s} M_CHI = {m_chi_best:7.1f} GeV  at m_D = {m_d_best:8.1f} GeV, "
                  f"eps = {eps_best:.3g}, {n:.2g} events")
        results[(with_dilepton, name)] = (eps_fine, gg_fine * to_events, eps, s * to_events,
                                          (s + gg) * to_events)

# ----- how the photon-fusion reach depends on the heaviest dark photon allowed in -----
print("\n===== photon fusion (fine grid): heaviest M_CHI against the largest m_D allowed in =====")
for name in LUMINOSITIES:
    for with_dilepton in (False, True):
        eps_fine, gg_events = results[(with_dilepton, name)][:2]
        option = "invisible + dilepton" if with_dilepton else "invisible only"
        cells = []
        for m_d_max in M_D_MAX_SCAN:
            found = heaviest(FINE_D, FINE_CHI, gg_events, eps_fine, m_d_max)
            cells.append("   none" if found is None
                         else f"{found[0]:5.0f} ({found[1]:.0f}, {found[2]:.2g})")
        print(f"  {name:9s} {option:21s} " + " | ".join(cells))
print("  columns: m_D <= " + ", ".join(f"{m:g}" for m in M_D_MAX_SCAN)
      + " GeV; each cell: M_CHI (at m_D, eps)")

# ----- figure: heaviest reachable M_CHI against m_D, HL-LHC -----
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK_2, MUTED, GRID = "#52514e", "#a3a29c", "#e6e5e1"
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, with_dilepton in zip(axes, (False, True)):
    eps_fine, gg_events, eps, s_events, sum_events = results[(with_dilepton, "HL-LHC")]

    def row_reach(md_values, mc_values, events):
        out = np.full(len(md_values), np.nan)
        for i in range(len(md_values)):
            ok = np.where(events[i] >= 1)[0]
            if len(ok):
                out[i] = mc_values[ok.max()]
        return out

    ax.plot(fine_d, row_reach(fine_d, fine_chi, gg_events), color=ORANGE, lw=2,
            label=r"photon fusion $\gamma\gamma \to \chi\bar\chi$")
    ax.plot(m_d_values, row_reach(m_d_values, m_chi_values, s_events), "o", color=BLUE, ms=7,
            mec="white", mew=1.5, label=r"s-channel $q\bar q \to A' \to \chi\bar\chi$ (grid)")
    ax.plot(m_d_values, row_reach(m_d_values, m_chi_values, sum_events), "s", color=INK_2,
            ms=4, label="sum (grid)")
    ax.plot(fine_d, fine_d / 2, color=MUTED, lw=1, ls=(0, (2, 2)), label=r"$M_\chi = m_{A'}/2$")
    ax.axvspan(M_D_MAX, fine_d[-1], color="#f1f0ed", zorder=0)
    ax.text(M_D_MAX * 1.15, 1.0, "only the CMS monojet\nbound on $\\epsilon$ here", color=INK_2,
            fontsize=8.5, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(fine_d[0], fine_d[-1])
    ax.set_ylim(0.8, 1200)
    ax.set_xlabel(r"Dark photon mass $m_{A'}$ (GeV)")
    ax.set_title("invisible + dilepton limits" if with_dilepton else "invisible limits only",
                 fontsize=11)
    ax.grid(which="major", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel(r"Heaviest $M_\chi$ with $\geq 1$ event (GeV)")
axes[0].legend(fontsize=9, loc="upper left")
fig.suptitle(rf"HL-LHC, 3000 fb$^{{-1}}$, 13.6 TeV, $g_D$ = {G_D:g}, $\epsilon$ at its largest "
             "allowed value; produced events, no cuts", fontsize=10.5, color=INK_2)
fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=160)
print(f"\nSaved the figure to {OUTPUT_PNG}")
plt.show()

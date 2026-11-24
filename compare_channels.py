"""
This script compares the two production channels of dark monopole pairs, read from results/:
  s-channel  q qbar -> gamma* -> A' -> chi chibar, loop-level F_2 vertex (written by 2D_scan.py)
  t-channel  gamma gamma -> chi chibar, photon fusion at tree level (SFGen_dark in
             ~/PythonProjects/boson_merge, collected by repro/collect_scan21_13p6.py)
Both files must be on the same grid of dark photon and monopole masses.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

# ===== Parameters you can adjust =====

RESULTS = Path(__file__).parent / "results"
S_CHANNEL_FILE = RESULTS / "s_channel_13.6TeV_eps0.01_gD0.3_21x21.csv"
GAMMA_GAMMA_FILE = RESULTS / "gamma_gamma_13.6TeV_eps0.01_gD0.3_21x21.csv"
TITLE = r"$\sqrt{s}$ = 13.6 TeV, $\epsilon$ = 0.01, $g_D$ = 0.3, no cuts"

# Rows of the m_D grid (0 = lightest) that get their own panel of sigma against M_CHI
SLICE_ROWS = [0, 5, 10, 15, 20]

LUMINOSITY = 3000.0  # integrated luminosity in fb^-1 for the one-event line (3000 = HL-LHC)

OUTPUT_PNG = RESULTS / "compare_channels_13.6TeV_eps0.01_gD0.3.png"

# ===== End of adjustable parameters =====

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e1"


def load_grid(path):
    """Read a results file into (m_D values, m_chi values, sigma[i_D, i_chi])."""
    columns = np.loadtxt(path, delimiter=",", unpack=True)
    m_d, m_chi, sigma = columns[0], columns[1], columns[2]
    m_d_values, m_chi_values = np.unique(m_d), np.unique(m_chi)
    if len(sigma) != len(m_d_values) * len(m_chi_values):
        raise SystemExit(f"{path.name}: {len(sigma)} points, not a full grid")
    # rows are written with m_D as the outer loop, so a reshape restores the grid
    return m_d_values, m_chi_values, sigma.reshape(len(m_d_values), len(m_chi_values))


def log_edges(values):
    """Cell edges halfway between log-spaced points, for pcolormesh."""
    step = np.sqrt(values[1] / values[0])
    return np.geomspace(values[0] / step, values[-1] * step, len(values) + 1)


m_d_values, m_chi_values, sigma_s = load_grid(S_CHANNEL_FILE)
m_d_gg, m_chi_gg, sigma_gg = load_grid(GAMMA_GAMMA_FILE)
if not (np.allclose(m_d_values, m_d_gg) and np.allclose(m_chi_values, m_chi_gg)):
    raise SystemExit("the two files are on different mass grids")
ratio = sigma_gg / sigma_s
log_ratio = np.log10(ratio)
one_event = 1 / (1000 * LUMINOSITY)  # sigma in pb that gives one event

# ----- printed summary -----
s_wins = np.argwhere(ratio < 1)
print(f"gamma gamma is larger at {ratio.size - len(s_wins)} of {ratio.size} points, "
      f"by up to 10^{log_ratio.max():.1f}")
print(f"the s-channel is larger at {len(s_wins)} points:")
for i, j in s_wins:
    print(f"  m_D = {m_d_values[i]:8.2f} GeV, M_CHI = {m_chi_values[j]:7.2f} GeV: "
          f"s-channel {sigma_s[i, j]:.3e} pb, gamma gamma {sigma_gg[i, j]:.3e} pb, "
          f"ratio s/gg = {1 / ratio[i, j]:.2f}")

# ----- figure: one panel per slice, and the ratio map -----
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2,
                     "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlecolor": INK})
fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
panels = axes.flat

shown = np.concatenate([sigma_s[SLICE_ROWS], sigma_gg[SLICE_ROWS]])
y_low = 10.0 ** np.floor(np.log10(shown.min()))
y_high = 10.0 ** np.ceil(np.log10(shown.max()) + 1)
for k, row in enumerate(SLICE_ROWS):
    ax = panels[k]
    ax.loglog(m_chi_values, sigma_s[row], color=BLUE, lw=2,
              label=r"s-channel  $q\bar q \to \gamma^* \to A' \to \chi\bar\chi$")
    ax.loglog(m_chi_values, sigma_gg[row], color=ORANGE, lw=2, ls=(0, (5, 2)),
              label=r"t-channel  $\gamma\gamma \to \chi\bar\chi$")
    ax.axhline(one_event, color=MUTED, lw=1, ls=":")
    if k == 0:
        ax.text(m_chi_values[-1], one_event * 3, f"1 event at {LUMINOSITY:g} fb$^{{-1}}$",
                ha="right", color=INK_2, fontsize=8.5)
    resonance = m_d_values[row] / 2  # the s-channel peaks just below M_CHI = m_D / 2
    if m_chi_values[0] < resonance < m_chi_values[-1]:
        ax.axvline(resonance, color=MUTED, lw=1, ls=(0, (1, 3)))
        ax.text(resonance * 1.12, y_low * 100, r"$M_\chi = m_{A'}/2$", color=INK_2,
                fontsize=8.5, rotation=90)
    m_d_label = f"{m_d_values[row]:.3g}" if m_d_values[row] < 1000 else f"{m_d_values[row]:.0f}"
    ax.set_title(rf"$m_{{A'}}$ = {m_d_label} GeV", fontsize=11)
    ax.set_ylim(y_low, y_high)
    ax.grid(which="major", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if k % 3 == 0:
        ax.set_ylabel(r"$\sigma(pp \to \chi\bar\chi)$ (pb)")
    if k >= 3:
        ax.set_xlabel(r"Dark monopole mass $M_\chi$ (GeV)")

ax = panels[5]
cmap = LinearSegmentedColormap.from_list("blue_gray_orange", [BLUE, "#ecebe8", ORANGE])
norm = TwoSlopeNorm(vmin=min(-1.0, np.floor(log_ratio.min())), vcenter=0.0,
                    vmax=np.ceil(log_ratio.max()))
mesh = ax.pcolormesh(log_edges(m_chi_values), log_edges(m_d_values), log_ratio, cmap=cmap,
                     norm=norm, shading="flat")
ax.contour(m_chi_values, m_d_values, log_ratio, levels=[0.0], colors=INK, linewidths=1.2)
ax.plot(m_chi_values, 2 * m_chi_values, color=INK_2, lw=1, ls=(0, (2, 2)))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$M_\chi$ (GeV)")
ax.set_ylabel(r"$m_{A'}$ (GeV)")
ax.set_title(r"$\log_{10}\,(\sigma_{\gamma\gamma} / \sigma_{q\bar q})$", fontsize=11)
ax.legend(handles=[Line2D([], [], color=INK, lw=1.2, label="equal"),
                   Line2D([], [], color=INK_2, lw=1, ls=(0, (2, 2)), label=r"$m_{A'} = 2M_\chi$")],
          loc="upper left", fontsize=8.5, frameon=True, framealpha=0.9)
colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
# ticks on both sides of 0: the blue side (s-channel larger) is short and gets none by default
colorbar.set_ticks(sorted({norm.vmin, 0.0, *np.arange(5.0, norm.vmax + 1, 5.0)}))

handles, labels = panels[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10.5,
           bbox_to_anchor=(0.5, 0.995))
fig.suptitle(TITLE, y=0.945, fontsize=10, color=INK_2)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUTPUT_PNG, dpi=160)
print(f"Saved the figure to {OUTPUT_PNG}")
plt.show()

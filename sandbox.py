import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spence


def li2(z):
    """Dilogarithm Li2(z). SciPy's spence uses an older convention: Li2(z) = spence(1 - z)."""
    return spence(1 - z)


# Check the convention against known values
print("Li2(1)   =", li2(1.0),  "  expected pi^2/6 =", np.pi**2 / 6)
print("Li2(-1)  =", li2(-1.0), "  expected -pi^2/12 =", -np.pi**2 / 12)
print("Li2(0.5) =", li2(0.5),  "  expected pi^2/12 - ln(2)^2/2 =", np.pi**2/12 - np.log(2)**2/2)

# Gotcha: a REAL input beyond 1 gives nan; make it complex
print("Li2(3.0)  =", li2(3.0))
print("Li2(3+0j) =", li2(3.0 + 0j))

# Li2 along the real axis, approached from just above and just below
x = np.linspace(-5, 5, 1001)
eps = 1e-12
above = li2(x + 1j * eps)
below = li2(x - 1j * eps)

fig, (ax_re, ax_im) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))

ax_re.plot(x, above.real, label="just above the axis (x + iε)")
ax_re.plot(x, below.real, "--", label="just below the axis (x − iε)")
ax_re.set_ylabel("Re Li₂(x)")
ax_re.set_title("Dilogarithm along the real axis")
ax_re.legend()
ax_re.grid(alpha=0.3)

ax_im.plot(x, above.imag, label="just above the axis (x + iε)")
ax_im.plot(x, below.imag, "--", label="just below the axis (x − iε)")
ax_im.axvline(1, color="gray", linewidth=0.8)   # the branch cut starts at x = 1
ax_im.set_xlabel("x")
ax_im.set_ylabel("Im Li₂(x)")
ax_im.legend()
ax_im.grid(alpha=0.3)

fig.tight_layout()
plt.show()
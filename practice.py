import lhapdf
import numpy as np
import matplotlib.pyplot as plt

lhapdf.setVerbosity(0)
pdf = lhapdf.mkPDF("NNPDF31_lo_as_0118", 0)

Q = 200   # Collision scale in GeV

xs = np.logspace(-3, 0, 200)
ups = [pdf.xfxQ(2, x, Q) for x in xs]

plt.plot(xs, ups)
plt.xscale("log")
plt.xlabel("x (fraction of the proton's momentum)")
plt.ylabel("x f(x)")
plt.title(f"Up quark in the proton, Q = {Q} GeV")
plt.grid(alpha=0.3)
plt.show()
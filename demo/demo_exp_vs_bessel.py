import os
import numpy as np
import matplotlib.pyplot as plt

from ThePlottingModule import TheMatplotlibStyler
from scipy.special import k0

os.system("clear")

def main():
  fig, ax = plt.subplots()
  ## generate data
  x        = np.logspace(0, 4, 1000)
  x_scale  = 100
  y_exp    = np.exp(-x/x_scale)
  y_bessel = k0(x/x_scale)
  ## plot data
  ax.plot(x, y_exp,    c="r", ls="--", lw=2, label=r"$\exp\big(-x/100\big)$")
  ax.plot(x, y_bessel, c="b", ls="-",  lw=2, label=r"$K_0\big(x/100\big)$")
  ## tune figure axis
  ax.set_xlabel(r"$x$")
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_ylim([ 10**(-10), 10 ])
  ax.legend(loc="lower left", fontsize=20)
  ## save figure
  fig.savefig("compare_exp_vs_bessel.png")
  plt.close(fig)
  print("Saved figure.")

## PROGRAM ENTRY POINT
if __name__ == "__main__":
  main()

## END OF PROGRAM
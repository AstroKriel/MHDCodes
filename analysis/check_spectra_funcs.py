#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from TheFittingModule import FitMHDScales

def main():
  ## generate data
  func_full = FitMHDScales.SpectraModels.kinetic_linear
  func_simple = FitMHDScales.SpectraModels.magnetic_simple_linear
  data_x = np.linspace(1, 500, 100)
  params = [ 10**(-6), -5, 1/5 ]
  data_y_full   = func_full(data_x, *params)
  data_y_simple = func_simple(data_x, *params)
  ## plot data
  fig, ax = plt.subplots()
  ax.plot(data_x, data_y_full, "b.")
  ax.plot(data_x, data_y_simple, "r.")
  ## save figure
  ax.set_xscale("log")
  ax.set_yscale("log")
  fig.savefig("bessel_vs_exp.png")

if __name__ == "__main__":
  main()

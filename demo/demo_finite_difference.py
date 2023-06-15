#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np
import collections.abc

## 'tmpfile' needs to be loaded before any 'matplotlib' libraries,
## so matplotlib stores its cache in a temporary directory.
## (necessary when plotting in parallel)
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
import matplotlib.pyplot as plt
import matplotlib.colors as colors

## load user defined modules
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def gradient_1ofd(field, cell_width):
  F = -1 # shift forwards
  return (
    np.roll(field, F) - field
  ) / cell_width

def gradient_2ocd(field, cell_width):
  F = -1 # shift forwards
  B = +1 # shift backwards
  return (
    np.roll(field, F) - np.roll(field, B)
  ) / (2*cell_width)

def gradient_4ocd(field, cell_width):
  F = -1 # shift forwards
  B = +1 # shift backwards
  return (
    - np.roll(field, 2*F) + 8*np.roll(field, F)
    + np.roll(field, 2*B) - 8*np.roll(field, B)
  ) / (12*cell_width)


## ###############################################################
## OPPERATOR HANDLING PLOT CALLS
## ###############################################################
def plotTNB():
  ncols = 1
  nrows = 3
  fig, axs = plt.subplots(
    ncols   = ncols,
    nrows   = nrows,
    figsize = (ncols*10, nrows*5)
  )
  list_dict_methods = [
    {"func" : gradient_1ofd, "color" : "orange"},
    {"func" : gradient_2ocd, "color" : "red"},
    {"func" : gradient_4ocd, "color" : "forestgreen"},
  ]
  x = np.linspace(0, 2*np.pi, 100)[:-1]
  y = np.sin(x)
  dydx = np.cos(x)
  axs[0].plot(x, y,    marker="o", ls="-", lw=2, color="black")
  axs[1].plot(x, dydx, marker="o", ls="-", lw=2, color="black")
  for method_index in range(len(list_dict_methods)):
    for num_points in [ 5, 10, 20, 40, 100, 200 ]:
      x = np.linspace(0, 2*np.pi, num_points+1)[:-1]
      y = np.sin(x)
      dydx_analytic = np.cos(x)
      dydx_numeric = list_dict_methods[method_index]["func"](y, 2*np.pi/num_points)
      err = np.abs(dydx_analytic - dydx_numeric)
      if num_points == 40:
        axs[1].plot(x, dydx_numeric, marker="o", ls="-", lw=2, color=list_dict_methods[method_index]["color"])
      axs[2].plot(num_points, err[0], marker="o", ls="-", lw=2, color=list_dict_methods[method_index]["color"])
      axs[2].set_xscale("log")
      axs[2].set_yscale("log")
  p = np.linspace(5, 200, 10)
  axs[2].plot(p, p**(-1), ls="-", lw=2, color="orange")
  axs[2].plot(p, 5*p**(-2), ls="-", lw=2, color="red")
  axs[2].plot(p, 50*p**(-4), ls="-", lw=2, color="forestgreen")
  ## save figure
  print("Saving figure...")
  PlotFuncs.saveFigure(fig, f"field_.png", bool_draft=False)


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  plotTNB()
  sys.exit()


## END OF PROGRAM
#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import sys
import cmasher as cmr
import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib.colors as colors

plt.rcParams["axes.axisbelow"] = False


## load user defined modules
from TheFlashModule import LoadData, SimParams
from TheUsefulModule import WWFnF
from TheFittingModule import FitMHDScales
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def getSimLabel(mach_regime, Re):
  return "$" + mach_regime.replace("Mach", "\mathcal{M}") + "\\text{Re}" + f"{Re:.0f}" + "\\text{Pm}5$"

def getSimPath(scratch_path, sim_suite, mach_regime, sim_res):
  return f"{scratch_path}/{sim_suite}/{mach_regime}/Pm5/{int(sim_res):d}/"

def initFigure(ncols=1, nrows=1):
  return plt.subplots(
    ncols   = ncols,
    nrows   = nrows,
    figsize = (7*ncols, 7*nrows)
  )

def addText(
    ax, pos, text,
    rotation = 0,
    fontsize = 20,
    va       = "bottom",
    ha       = "left",
  ):
  ax.text(
    pos[0], pos[1],
    text,
    va            = va,
    ha            = ha,
    transform     = ax.transAxes,
    rotation      = rotation,
    fontsize      = fontsize,
    rotation_mode = "anchor",
    color         = "black",
    zorder        = 10
  )

def yFunc(phi, chi):
  k = np.logspace(0, 6, 10**3)
  keta = 100
  ## first method
  arg2 = (k / keta)**chi
  gamma = lambda phi_, k_: scipy.special.gammainc(phi_, k_)
  func = lambda phi_: gamma(2*phi_ - chi*phi_, arg2) - 2*k**(phi_*(1-chi)) * gamma(phi_, arg2)
  coef = keta**(chi+1) * (chi / phi)**(1/chi)
  numer = np.sum(func(phi+1))
  denom = np.sum(func(phi))
  return 1 / (coef * numer / denom)
  # ## second method
  # exp_vals = np.exp( -(k/keta)**(chi) )
  # model = k**(phi) * exp_vals
  # kp = (phi * keta**chi / chi)**(1/chi)
  # numer = np.sum(model)
  # denom = np.sum(k**(phi-1) * exp_vals)
  # kcor = numer / denom
  # return kcor / kp


## ###############################################################
## PLOT B-FIELD SCALES
## ###############################################################
def plotScaleComparison():
    ## initialise figure
  print("Initialising figure...")
  fig, ax = initFigure()
  ## compute data
  print("Computing solution...")
  result     = np.zeros((50, 50))
  phi_values = np.linspace(1, 2, len(result[:,0]))
  chi_values   = np.linspace(0.5, 1, len(result[0,:]))
  for row, phi in enumerate(phi_values):
    for col, chi in enumerate(chi_values):
      result[row, col] = yFunc(phi, chi)
  ## plot data
  print("Plotting solution...")
  plt_obj = ax.imshow(
    result,
    norm   = colors.LogNorm(),
    origin = "lower",
    extent = [
      np.min(chi_values), np.max(chi_values),
      np.min(phi_values), np.max(phi_values),
    ]
  )
  ## label axis
  ax.axhline(y=3/2, ls="--", lw=2, color="red")
  ax.set_xlabel(r"$\chi$")
  ax.set_ylabel(r"$\phi$")
  ## add colorbar
  ax_cbar = fig.add_axes([ 0.125, 0.9, 0.775, 0.06 ])
  fig.colorbar(mappable=plt_obj, cax=ax_cbar, orientation="horizontal")
  ax_cbar.set_title(r"$k_{\rm cor} / k_{\rm p}$", fontsize=20, pad=10)
  ax_cbar.xaxis.set_ticks_position("top")
  ## save figure
  print("Saving figure...")
  filepath_fig = f"{PATH_PLOT}/k_cor.png"
  fig.savefig(filepath_fig, dpi=100)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  WWFnF.createFolder(PATH_PLOT, bool_verbose=False)
  plotScaleComparison()


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
COLOR_SUBSONIC   = "#B80EF6"
COLOR_SUPERSONIC = "#F4A123"
PATH_PLOT = "/home/586/nk7952/MHDCodes/kriel2023/spectra/"


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
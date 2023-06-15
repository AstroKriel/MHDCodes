#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys, copy
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

## load user defined modules
from TheFlashModule import SimParams, FileNames
from TheUsefulModule import WWFnF, WWObjs
from TheFittingModule import UserModels
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def getVal(val):
  return val if (val is None) else np.nan

def addText(ax, pos, text):
  ax.text(
    pos[0], pos[1],
    text,
    va        = "center",
    ha        = "left",
    transform = ax.transAxes,
    color     = "black",
    fontsize  = 18,
    zorder    = 10
  )

def plotScale(ax, x, y_median, y_1sig, color):
  ax.errorbar(
    x, y_median,
    yerr   = y_1sig,
    mfc    = "whitesmoke" if color is None else color,
    ecolor = "black" if color is None else color,
    fmt="o", mec="black", elinewidth=1, markersize=7, capsize=7.5, linestyle="None", zorder=5
  )

def plotLogisticModel(ax, fit_params, color):
  func = UserModels.ListOfModels.logistic_growth_increasing
  data_x = np.logspace(np.log10(1), np.log10(10**4), 100)
  data_y = func(data_x, *fit_params)
  ax.plot(data_x, data_y, color=color, ls="-", lw=2)

def fitLogisticModel(ax, list_nres, val_group_nres, std_group_nres, color="black"):
  func = UserModels.ListOfModels.logistic_growth_increasing
  fit_params, fit_cov = curve_fit(
    f      = func,
    xdata  = list_nres,
    ydata  = val_group_nres,
    sigma  = std_group_nres,
    bounds = (
      ## amplitude, turnover scale, turnover rate
      (1.0, 0.1, 0.0),
      (1e3, 1e3, 5.0)
    ),
    absolute_sigma=True, maxfev=10**5,
  )
  plotLogisticModel(ax, fit_params, color)

def fitScales(ax, list_nres, scales_group_nres, color="black"):
  list_nres = copy.deepcopy(list_nres)
  list_nres.append(2*list_nres[-1])
  scales_group_nres = copy.deepcopy(scales_group_nres)
  scales_group_nres.append(scales_group_nres[-1])
  list_nres = [
    list_nres[nres_index]
    for nres_index in range(len(list_nres))
    if scales_group_nres[nres_index]["val"] is not None
  ]
  val_group_nres = [
    scales["val"]
    for scales in scales_group_nres
    if scales["val"] is not None
  ]
  std_group_nres = [
    scales["std"]
    for scales in scales_group_nres
    if scales["val"] is not None
  ]
  fitLogisticModel(ax, list_nres, val_group_nres, std_group_nres, color)


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class PlotConvergence():
  def __init__(self, axs, sim_name, color):
    self.ax_knu, self.ax_keta, self.ax_kp = axs
    self.sim_name = sim_name
    self.color = color

  def performRoutine(self):
    self._readScales()
    self._plotScales()
    self._fitScales()

  def _readScales(self):
    self.list_nres        = []
    self.k_nu_group_nres  = []
    self.k_eta_group_nres = []
    self.k_p_group_nres   = []
    dict_scales_group_sim = WWObjs.readJsonFile2Dict(
      filepath     = PATH_PAPER,
      filename     = "dataset.json",
      bool_verbose = False
    )
    dict_sim = dict_scales_group_sim[self.sim_name]
    for sim_res in [ "18", "36", "72", "144", "288", "576" ]:
      self.list_nres.append(int(sim_res))
      self.k_nu_group_nres.append(dict_sim["k_nu_vel"][sim_res])
      self.k_eta_group_nres.append(dict_sim["k_eta_cur"][sim_res])
      self.k_p_group_nres.append(dict_sim["k_p"][sim_res])

  def _plotScales(self):
    for sim_res_index, nres in enumerate(self.list_nres):
      plotScale(
        ax       = self.ax_knu,
        x        = nres,
        y_median = self.k_nu_group_nres[sim_res_index]["val"],
        y_1sig   = self.k_nu_group_nres[sim_res_index]["std"],
        color    = self.color
      )
      plotScale(
        ax       = self.ax_keta,
        x        = nres,
        y_median = self.k_eta_group_nres[sim_res_index]["val"],
        y_1sig   = self.k_eta_group_nres[sim_res_index]["std"],
        color    = self.color
      )
      plotScale(
        ax       = self.ax_kp,
        x        = nres,
        y_median = self.k_p_group_nres[sim_res_index]["val"],
        y_1sig   = self.k_p_group_nres[sim_res_index]["std"],
        color    = self.color
      )

  def _fitScales(self):
    fitScales(self.ax_knu,  self.list_nres, self.k_nu_group_nres,  self.color)
    fitScales(self.ax_keta, self.list_nres, self.k_eta_group_nres, self.color)
    fitScales(self.ax_kp,   self.list_nres, self.k_p_group_nres,   self.color)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## initialise figure
  print("Initialising figure...")
  figscale = 1
  fig, axs = plt.subplots(
    nrows   = 3,
    figsize = (6*figscale, 3*4*figscale),
    sharex  = True
  )
  fig.subplots_adjust(hspace=0.05)
  ## plot subsonic scales
  obj_subsonic = PlotConvergence(axs, "Mach0.3_Rm3000_Pm5", COLOR_SUBSONIC)
  obj_subsonic.performRoutine()
  ## plot supersonic scales
  obj_supersonic = PlotConvergence(axs, "Mach5_Rm3000_Pm5", COLOR_SUPERSONIC)
  obj_supersonic.performRoutine()
  ## label figure
  axs[-1].set_xlabel(r"$N_{\rm res}$", fontsize=22)
  axs[-1].set_xscale("log")
  axs[-1].set_xlim([ 10, 10**(4) ])
  axs[0].set_ylabel(r"$k_\nu$", fontsize=22)
  axs[1].set_ylabel(r"$k_\eta$", fontsize=22)
  axs[2].set_ylabel(r"$k_\mathrm{p}$", fontsize=22)
  axs[0].set_yscale("log")
  axs[1].set_yscale("log")
  axs[2].set_yscale("log")
  axs[0].set_ylim([ 0.8*10**(1), 1.2*10**(2) ])
  axs[1].set_ylim([ 0.8*10**(0), 1.2*10**(2) ])
  axs[2].set_ylim([ 0.8*10**(0), 1.2*10**(1) ])
  ## add legends
  PlotFuncs.addLegend_fromArtists(
    axs[0],
    list_legend_labels = [
      r"$\mathcal{M}0.3{\rm Re}600{\rm Pm}5$",
      r"$\mathcal{M}5{\rm Re}600{\rm Pm}5$",
    ],
    list_artists       = [ "-" ],
    list_marker_colors = [ COLOR_SUBSONIC, COLOR_SUPERSONIC ],
    label_color        = "white",
    loc                = "lower right",
    bbox               = (1.0, 0.0),
    fontsize           = 17
  )
  addText(axs[0], (0.6, 0.23), r"$\mathcal{M}0.3{\rm Re}600{\rm Pm}5$")
  addText(axs[0], (0.6, 0.115), r"$\mathcal{M}5{\rm Re}600{\rm Pm}5$")
  ## save figure
  print("Saving figure...")
  fig_name     = f"nres_scales.pdf"
  filepath_fig = f"{PATH_PAPER}/{fig_name}"
  fig.savefig(filepath_fig, dpi=100)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
COLOR_SUBSONIC   = "#C85DEF"
COLOR_SUPERSONIC = "#FFAB1A"
PATH_PAPER = "/home/586/nk7952/MHDCodes/kriel2023/"


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np

## 'tmpfile' needs to be loaded before any 'matplotlib' libraries,
## so matplotlib stores its cache in a temporary directory.
## (necessary when plotting in parallel)
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

## load user defined modules
from TheFlashModule import LoadData, SimParams
from TheUsefulModule import WWLists
from TheFittingModule import FitFuncs
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def addText(ax, pos, text, rotation=0):
  ax.text(
    pos[0], pos[1],
    text,
    transform = ax.transAxes,
    va        = "center",
    ha        = "left",
    rotation  = rotation,
    rotation_mode = "anchor",
    color     = "black",
    fontsize  = 17,
    zorder    = 10
  )

## ###############################################################
## OPERATOR CLASS
## ###############################################################
class PlotTurbData():
  def __init__(
      self,
      fig, axs, ax_inset, filepath_sim_res, color,
      time_start_exp, time_end_exp, time_start_sat
    ):
    print("Looking at:", filepath_sim_res)
    ## save input arguments
    self.fig              = fig
    self.axs              = axs
    self.ax_inset         = ax_inset
    self.filepath_sim_res = filepath_sim_res
    self.color            = color
    self.time_start_exp   = time_start_exp
    self.time_end_exp     = time_end_exp
    self.time_start_sat   = time_start_sat
    ## read simulation input parameters
    self.dict_sim_inputs  = SimParams.readSimInputs(self.filepath_sim_res)

  def performRoutines(self):
    print("Loading volume integrated quantities...")
    self._loadData()
    print("Plotting volume integrated quantities...")
    self._plotMach()
    self._plotEnergyRatio()
    self._fitData()

  def _loadData(self):
    ## load Mach data
    _, data_Mach = LoadData.loadVIData(
      directory  = self.filepath_sim_res,
      field_name = "mach",
      t_turb     = self.dict_sim_inputs["t_turb"],
      time_start = 1e-1,
      time_end   = np.inf
    )
    ## load kinetic energy
    _, data_Ekin = LoadData.loadVIData(
      directory  = self.filepath_sim_res,
      field_name = "kin",
      t_turb     = self.dict_sim_inputs["t_turb"],
      time_start = 1e-1,
      time_end   = np.inf
    )
    ## load magnetic energy
    data_time, data_Emag = LoadData.loadVIData(
      directory  = self.filepath_sim_res,
      field_name = "mag",
      t_turb     = self.dict_sim_inputs["t_turb"],
      time_start = 1e-1,
      time_end   = np.inf
    )
    ## Only relevant when loading data while the simulation is running.
    ## Only grab portion of data that has been sefely written, for all quantities.
    max_len = min([
      len(data_time),
      len(data_Mach),
      len(data_Ekin),
      len(data_Emag)
    ])
    ## save data
    self.data_time = data_time[:max_len]
    self.data_Mach = data_Mach[:max_len]
    self.data_Emag = data_Emag[:max_len]
    self.data_Ekin = data_Ekin[:max_len]
    ## compute and save energy ratio: 'mag_energy / kin_energy'
    self.data_E_ratio = [
      mag_energy / kin_energy
      for mag_energy, kin_energy in zip(
        self.data_Emag,
        self.data_Ekin
      )
    ]

  def _plotMach(self):
    self.axs[0].plot(
      self.data_time,
      self.data_Mach,
      color=self.color, ls="-", lw=2, zorder=3
    )

  def _plotEnergyRatio(self):
    for ax in [ self.axs[1], self.ax_inset ]:
      ax.plot(
        self.data_time,
        self.data_E_ratio,
        color=self.color, ls="-", lw=2, zorder=3
      )

  def _fitData(self):
    linestyle_exp  = "-"
    linestyle_lin  = "--"
    linestyle_sat  = ":"
    ## SATURATED REGIME
    ## ----------------
    ## get index and time associated with saturated regime
    index_start_sat = WWLists.getIndexClosestValue(self.data_time, self.time_start_sat)
    index_end_sat   = len(self.data_time)-1
    FitFuncs.fitConstFunc(
      ax              = self.axs[1],
      data_x          = self.data_time,
      data_y          = self.data_E_ratio,
      str_label       = r"$\left(E_{\rm kin} / E_{\rm mag}\right)_{\rm sat} =$ ",
      index_start_fit = index_start_sat,
      index_end_fit   = index_end_sat,
      linestyle       = linestyle_sat
    )
    ## EXPONENTIAL GROWTH REGIME
    ## -------------------------
    index_start_exp = WWLists.getIndexClosestValue(self.data_time, self.time_start_exp)
    index_end_exp   = WWLists.getIndexClosestValue(self.data_time, self.time_end_exp)
    FitFuncs.fitExpFunc(
      ax              = self.axs[1],
      data_x          = self.data_time,
      data_y          = self.data_E_ratio,
      index_start_fit = index_start_exp,
      index_end_fit   = index_end_exp,
      linestyle       = linestyle_exp
    )
    ## MACH NUMBER
    ## -----------
    FitFuncs.fitConstFunc(
      ax              = self.axs[0],
      data_x          = self.data_time,
      data_y          = self.data_Mach,
      str_label       = r"$\mathcal{M} =$ ",
      index_start_fit = index_start_exp,
      index_end_fit   = index_end_exp,
      linestyle       = linestyle_exp
    )
    ## LINEAR GROWTH REGIME
    ## --------------------
    FitFuncs.fitLinearFunc(
      ax              = self.ax_inset,
      data_x          = self.data_time,
      data_y          = self.data_E_ratio,
      index_start_fit = index_end_exp,
      index_end_fit   = index_start_sat,
      linestyle       = linestyle_lin
    )


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## initialise figure
  print("Initialising figure...")
  figscale = 1.1
  fig, axs = plt.subplots(
    nrows   = 2,
    figsize = (6*figscale, 2*4*figscale),
    sharex  = True
  )
  fig.subplots_adjust(hspace=0.075)
  ## add inset axis
  ax_inset = PlotFuncs.addInsetAxis(
    axs[1],
    ax_inset_bounds = [
      0.37, 0.05,
      0.6, 0.33
    ],
    label_x         = None,
    label_y         = None,
    fontsize        = 20
  )
  ## plot subsonic data
  obj_plot_turb = PlotTurbData(
    fig              = fig,
    axs              = axs,
    ax_inset         = ax_inset,
    filepath_sim_res = f"{PATH_SCRATCH}/Rm3000/Mach0.3/Pm10/288/",
    color            = COLOR_SUBSONIC,
    time_start_exp   = 5.5,
    time_end_exp     = 16,
    time_start_sat   = 26,
  )
  obj_plot_turb.performRoutines()
  ## plot supersonic data
  obj_plot_turb = PlotTurbData(
    fig              = fig,
    axs              = axs,
    ax_inset         = ax_inset,
    filepath_sim_res = f"{PATH_SCRATCH}/Rm3000/Mach5/Pm10/288/",
    color            = COLOR_SUPERSONIC,
    time_start_exp   = 5.5,
    time_end_exp     = 38,
    time_start_sat   = 53,
  )
  obj_plot_turb.performRoutines()
  ## label figure
  axs[1].set_xlabel(r"$t / t_\mathrm{turb}$")
  axs[0].set_ylabel(r"$\mathcal{M}$")
  axs[1].set_ylabel(r"$E_\mathrm{mag} / E_\mathrm{kin}$")
  axs[0].set_yscale("log")
  axs[1].set_yscale("log")
  axs[0].set_xlim([ -2, 102 ])
  axs[0].set_ylim([ 10**(-2), 10**(1) ])
  axs[1].set_ylim([ 10**(-11), 10**(1) ])
  ## add log axis-ticks
  PlotFuncs.addAxisTicks_log10(
    axs[1],
    bool_major_ticks = True,
    num_major_ticks  = 7
  )
  ## label inset axis
  ax_inset.tick_params(axis="x", bottom=True, top=True, labelbottom=False, labeltop=True)
  ax_inset.set_xlim([ 7, 73 ])
  ax_inset.set_xticks([ 10, 20, 30, 40, 50, 60, 70 ])
  ax_inset.set_xticklabels([ 10, "", 30, "", 50, "", 70 ])
  ax_inset.xaxis.set_minor_locator(NullLocator())
  ax_inset.set_ylim([ -0.09, 0.8 ])
  ## annotate figure
  axs[0].axvspan(0, 5, alpha=0.25, color="grey", ls=None, lw=0)
  axs[1].axvspan(0, 5, alpha=0.25, color="grey", ls=None, lw=0)
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
  addText(axs[0], (0.675, 0.195), r"$\mathcal{M}0.3{\rm Re}600{\rm Pm}5$")
  addText(axs[0], (0.675, 0.095), r"$\mathcal{M}5{\rm Re}600{\rm Pm}5$")
  PlotFuncs.addLegend_fromArtists(
    axs[1],
    list_artists       = [
      "-",
      "--",
      ":"
    ],
    list_legend_labels = [
      "exponential growth",
      "linear growth",
      "saturated",
    ],
    list_marker_colors = [ "k" ],
    label_color        = "white",
    loc                = "upper right",
    bbox               = (1.0, 0.835),
    fontsize           = 17
  )
  addText(axs[1], (0.047, 0.5), r"transient phase", rotation=90)
  addText(axs[1], (0.62, 0.74), r"exponential growth")
  addText(axs[1], (0.62, 0.63), r"linear growth")
  addText(axs[1], (0.62, 0.53), r"saturated")
  ## save figure
  print("Saving figure...")
  fig_name     = f"time_evolution.pdf"
  filepath_fig = f"{PATH_PLOT}/{fig_name}"
  fig.savefig(filepath_fig, dpi=100)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
COLOR_SUBSONIC   = "#C85DEF"
COLOR_SUPERSONIC = "#FFAB1A"
PATH_SCRATCH     = "/scratch/ek9/nk7952/"
PATH_PLOT        = "/home/586/nk7952/MHDCodes/kriel2023/"


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
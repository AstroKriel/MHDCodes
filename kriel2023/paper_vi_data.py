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
## OPERATOR CLASS
## ###############################################################
class PlotTurbData():
  def __init__(
      self,
      fig, axs, filepath_sim_res, color,
      time_start_growth, time_end_growth, time_start_sat
    ):
    print("Looking at:", filepath_sim_res)
    ## save input arguments
    self.fig               = fig
    self.axs               = axs
    self.filepath_sim_res  = filepath_sim_res
    self.color             = color
    self.time_start_growth = time_start_growth
    self.time_end_growth   = time_end_growth
    self.time_start_sat    = time_start_sat
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
      color=self.color, ls="-", lw=1.5, zorder=3
    )

  def _plotEnergyRatio(self):
    self.axs[1].plot(
      self.data_time,
      self.data_E_ratio,
      color=self.color, ls="-", lw=1.5, zorder=3
    )

  def _fitData(self):
    linestyle_kin  = "--"
    linestyle_sat  = ":"
    ## SATURATED REGIME
    ## ----------------
    ## get index and time associated with saturated regime
    index_start_sat = WWLists.getIndexClosestValue(self.data_time, self.time_start_sat)
    index_end_sat   = len(self.data_time)-1
    ## find saturated energy ratio
    self.E_ratio_sat, _ = FitFuncs.fitConstFunc(
      ax              = self.axs[1],
      data_x          = self.data_time,
      data_y          = self.data_E_ratio,
      str_label       = r"$\left(E_{\rm kin} / E_{\rm mag}\right)_{\rm sat} =$ ",
      index_start_fit = index_start_sat,
      index_end_fit   = index_end_sat,
      linestyle       = linestyle_sat
    )
    ## GROWTH REGIME
    ## -------------
    index_start_growth = WWLists.getIndexClosestValue(self.data_time, self.time_start_growth)
    index_end_growth   = WWLists.getIndexClosestValue(self.data_time, self.time_end_growth)
    ## find growth rate of exponential
    self.E_growth_rate = FitFuncs.fitExpFunc(
      ax              = self.axs[1],
      data_x          = self.data_time,
      data_y          = self.data_E_ratio,
      index_start_fit = index_start_growth,
      index_end_fit   = index_end_growth,
      linestyle       = linestyle_kin
    )
    ## MACH NUMBER
    ## -----------
    self.rms_Mach_growth, self.std_Mach_growth = FitFuncs.fitConstFunc(
      ax              = self.axs[0],
      data_x          = self.data_time,
      data_y          = self.data_Mach,
      str_label       = r"$\mathcal{M} =$ ",
      index_start_fit = index_start_growth,
      index_end_fit   = index_end_growth,
      linestyle       = linestyle_kin
    )


## ###############################################################
## OPPERATOR HANDLING PLOT CALLS
## ###############################################################
def plotSimData():
  ## initialise figure
  print("Initialising figure...")
  figscale = 1.2
  fig, axs = plt.subplots(nrows=2, figsize=(6*figscale, 2*4*figscale), sharex=True)
  fig.subplots_adjust(hspace=0.075)
  ## plot subsonic data
  obj_plot_turb = PlotTurbData(
    fig               = fig,
    axs               = axs,
    filepath_sim_res  = f"{PATH_SCRATCH}/Rm3000/Mach0.3/Pm5/288/",
    color             = "#66c2a5",
    time_start_growth = 5,
    time_end_growth   = 15,
    time_start_sat    = 33,
  )
  obj_plot_turb.performRoutines()
  ## plot supersonic data
  obj_plot_turb = PlotTurbData(
    fig               = fig,
    axs               = axs,
    filepath_sim_res  = f"{PATH_SCRATCH}/Rm3000/Mach5/Pm5/288/",
    color             = "#fc8d62",
    time_start_growth = 5,
    time_end_growth   = 35,
    time_start_sat    = 62,
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
  ## add legends
  PlotFuncs.addLegend_fromArtists(
    axs[1],
    list_artists       = [
      "--",
      ":"
    ],
    list_legend_labels = [
      r"kinematic phase",
      r"saturated phase",
    ],
    list_marker_colors = [ "k" ],
    label_color        = "black",
    loc                = "lower right",
    bbox               = (1.0, 0.0),
    fontsize           = 18
  )
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
PATH_SCRATCH = "/scratch/ek9/nk7952/"
# PATH_SCRATCH = "/scratch/jh2/nk7952/"
PATH_PLOT    = "/home/586/nk7952/MHDCodes/kriel2023/"


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  plotSimData()
  sys.exit()


## END OF PROGRAM
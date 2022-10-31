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
from matplotlib.gridspec import GridSpec

## load user defined modules
from ThePlottingModule import PlotFuncs
from TheUsefulModule import WWLists, WWFnF
from TheLoadingModule import LoadFlashData
from TheFittingModule import FitFuncs

## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## PLOT INTEGRATED QUANTITIES
## ###############################################################
class PlotTurbData():
  def __init__(
      self,
      axs, filepath_data
    ):
    ## input arguments
    self.axs            = axs
    self.filepath_data  = filepath_data
    ## quantities to measure
    self.time_exp_start = None
    self.time_exp_end   = None
    self.rms_Mach       = None
    self.Gamma          = None
    self.E_sat_ratio    = None
    self.bool_fitted    = False
    ## perform routines
    print("Loading volume integrated data...")
    self.__loadData()
    self.__plotMach()
    self.__plotEnergyRatio()
    self.__fitData()

  def getFittedParams(self):
    return {
      "time_start"  : self.time_exp_start,
      "time_end"    : self.time_exp_end,
      "rms_Mach"    : self.rms_Mach,
      "Gamma"       : self.Gamma,
      "E_sat_ratio" : self.E_sat_ratio
    }

  def __loadData(self):
    ## load kinetic energy
    _, data_E_K = LoadFlashData.loadTurbData(
      filepath_data = self.filepath_data,
      var_y         = 9, # 9 (new), 6 (old)
      t_turb        = T_TURB,
      time_start    = 0.1,
      time_end      = np.inf
    )
    ## load magnetic energy
    _, data_E_B = LoadFlashData.loadTurbData(
      filepath_data = self.filepath_data,
      var_y         = 11, # 11 (new), 29 (old)
      t_turb        = T_TURB,
      time_start    = 0.1,
      time_end      = np.inf
    )
    ## load Mach data
    data_time, data_Mach = LoadFlashData.loadTurbData(
      filepath_data = self.filepath_data,
      var_y         = 13, # 13 (new), 8 (old)
      t_turb        = T_TURB,
      time_start    = 0.1,
      time_end      = np.inf
    )
    ## Only relevant when loading data while a simulation is running
    ## (i.e., dealing w/ data synchronisation). So, only grab the portion
    ## (i.e., time-range) of data that has been sefely written for all quantities
    max_len = min([
      len(data_time),
      len(data_Mach),
      len(data_E_B),
      len(data_E_K)
    ])
    ## save data
    self.data_time = data_time[:max_len]
    self.data_Mach = data_Mach[:max_len]
    self.data_E_B  = data_E_B[:max_len]
    self.data_E_K  = data_E_K[:max_len]
    ## compute and save energy ratio: 'E_B / E_K'
    self.data_E_ratio = [
      E_B / E_K
      for E_B, E_K in zip(
        data_E_B[:max_len],
        data_E_K[:max_len]
      )
    ]
    ## define plot domain
    self.max_time = max([
      100,
      max(self.data_time[:max_len])
    ])

  def __plotMach(self):
    self.axs[0].plot(self.data_time, self.data_Mach, color="orange", ls="-", lw=1.5, zorder=3)
    self.axs[0].set_ylabel(r"$\mathcal{M}$")
    self.axs[0].set_xlim([ 0, self.max_time ])
  
  def __plotEnergyRatio(self):
    self.axs[1].plot(self.data_time, self.data_E_ratio, color="orange", ls="-", lw=1.5, zorder=3)
    ## define y-axis range for the energy ratio plot
    min_E_ratio         = min(self.data_E_ratio)
    log_min_E_ratio     = np.log10(min_E_ratio)
    new_log_min_E_ratio = np.floor(log_min_E_ratio)
    num_decades         = 1 + (-new_log_min_E_ratio)
    new_min_E_ratio     = 10**new_log_min_E_ratio
    num_y_major_ticks   = np.ceil(num_decades / 2)
    ## label axis
    self.axs[1].set_xlabel(r"$t / t_\mathrm{turb}$")
    self.axs[1].set_ylabel(r"$E_\mathrm{mag} / E_\mathrm{kin}$")
    self.axs[1].set_yscale("log")
    self.axs[1].set_xlim([ 0, self.max_time ])
    self.axs[1].set_ylim([ new_min_E_ratio, 10**(1) ])
    ## add log axis-ticks
    PlotFuncs.addLogAxisTicks(
      self.axs[1],
      bool_major_ticks = True,
      num_major_ticks  = num_y_major_ticks
    )

  def __fitData(self):
    linestyle_kin  = "--"
    linestyle_sat  = ":"
    label_Esat     = r"$\left(E_{\rm kin} / E_{\rm mag}\right)_{\rm sat} =$ "
    label_Mach     = r"$\mathcal{M} =$ "
    t_start_index  = WWLists.getIndexClosestValue(self.data_time, 5.0)
    growth_percent = self.data_E_ratio[-1] / self.data_E_ratio[t_start_index]
    ## IF DYNAMO GROWTH OCCURS
    ## -----------------------
    if growth_percent > 100:
      ## find saturated energy ratio
      index_sat_start = WWLists.getIndexClosestValue(
        self.data_time,
        0.75 * self.data_time[-1]
      )
      self.E_sat_ratio = FitFuncs.fitConstFunc(
        ax              = self.axs[1],
        data_x          = self.data_time,
        data_y          = self.data_E_ratio,
        str_label       = label_Esat,
        index_start_fit = index_sat_start,
        index_end_fit   = len(self.data_time)-1,
        linestyle       = linestyle_sat
      )
      ## get index range corresponding with kinematic phase of the dynamo
      index_E_lo = WWLists.getIndexClosestValue(self.data_E_ratio, 10**(-8))
      index_E_hi = WWLists.getIndexClosestValue(self.data_E_ratio, self.E_sat_ratio/100)
      index_start_fit = min([ index_E_lo, index_E_hi ])
      index_end_fit   = max([ index_E_lo, index_E_hi ])
      ## find growth rate of exponential
      self.Gamma = FitFuncs.fitExpFunc(
        ax              = self.axs[1],
        data_x          = self.data_time,
        data_y          = self.data_E_ratio,
        index_start_fit = index_start_fit,
        index_end_fit   = index_end_fit,
        linestyle       = linestyle_kin
      )
    ## IF NO GROWTH OCCURS
    ## -------------------
    else:
      ## get index range corresponding with end of the simulation
      index_start_fit = WWLists.getIndexClosestValue(self.data_time, (0.75 * self.data_time[-1]))
      index_end_fit   = len(self.data_time)-1
      ## find average energy ratio
      self.E_sat_ratio = FitFuncs.fitConstFunc(
        ax              = self.axs[1],
        data_x          = self.data_time,
        data_y          = self.data_E_ratio,
        str_label       = label_Esat,
        index_start_fit = index_start_fit,
        index_end_fit   = index_end_fit,
        linestyle       = linestyle_sat
      )
    ## find average Mach number
    self.rms_Mach = FitFuncs.fitConstFunc(
      ax              = self.axs[0],
      data_x          = self.data_time,
      data_y          = self.data_Mach,
      str_label       = label_Mach,
      index_start_fit = index_start_fit,
      index_end_fit   = index_end_fit,
      linestyle       = linestyle_kin
    )
    ## INIDCATE THAT FIT OCCURED SUCCESSFULLY
    ## --------------------------------------
    self.bool_fitted = True
    ## ANNOTATE FIGURE
    ## ---------------
    ## add legend
    legend_ax0 = self.axs[0].legend(frameon=False, loc="lower left", fontsize=18)
    legend_ax1 = self.axs[1].legend(frameon=False, loc="lower right", fontsize=18)
    self.axs[0].add_artist(legend_ax0)
    self.axs[1].add_artist(legend_ax1)
    ## store time range bounds corresponding with the exponential phase of the dynamo
    self.time_exp_start = self.data_time[index_start_fit]
    self.time_exp_end   = self.data_time[index_end_fit]


## ###############################################################
## FIGURE INITIALISATION AND SAVING
## ###############################################################
def plotSimData(filepath_data, filepath_vis, sim_name):
  ## CREATE FIGURE
  ## -------------
  print("Initialising figure...")
  fig, fig_grid = PlotFuncs.initFigureGrid(
    fig_scale        = 1.0,
    fig_aspect_ratio = (5.0, 8.0),
    num_rows         = 2,
    num_cols         = 2
  )
  ax_Mach   = fig.add_subplot(fig_grid[0, 0])
  ax_energy = fig.add_subplot(fig_grid[1, 0])
  ## PLOT INTEGRATED QUANTITIES (Turb.dat)
  ## -------------------------------------
  PlotTurbData(
    axs           = [ ax_Mach, ax_energy ],
    filepath_data = filepath_data
  )
  ## SAVE FIGURE
  ## -----------
  ## save the figure
  print("Saving figure...")
  fig_name = f"{sim_name}_time_evolution.png"
  fig_filepath = WWFnF.createFilepath([ filepath_vis, fig_name ])
  plt.savefig(fig_filepath)
  plt.close()
  print("Figure saved:", fig_name)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## LOOK AT EACH SIMULATION
  ## -----------------------
  ## loop over each simulation suite
  for suite_folder in LIST_SUITE_FOLDER:

    ## loop over each simulation folder
    for sim_folder in LIST_SIM_FOLDER:

      ## CHECK THE SIMULATION EXISTS
      ## ---------------------------
      filepath_sim = WWFnF.createFilepath([
        BASEPATH, suite_folder, SONIC_REGIME, sim_folder
      ])
      if not os.path.exists(filepath_sim): continue
      str_message = f"Looking at suite: {suite_folder}, sim: {sim_folder}"
      print(str_message)
      print("=" * len(str_message))
      print(" ")

      ## loop over each resolution
      for sim_res in LIST_SIM_RES:

        ## CHECK THE RESOLUTION RUN EXISTS
        ## -------------------------------
        filepath_sim_res = f"{filepath_sim}/{sim_res}/"
        ## check that the filepath exists
        if not os.path.exists(filepath_sim_res): continue

        ## MAKE SURE A VISUALISATION FOLDER EXISTS
        ## ---------------------------------------
        filepath_sim_res_plot = WWFnF.createFilepath([
          filepath_sim_res, "vis_folder"
        ])
        WWFnF.createFolder(filepath_sim_res_plot, bool_hide_updates=True)

        ## PLOT SIMULATION DATA
        ## --------------------
        sim_name = f"{suite_folder}_{sim_folder}"
        plotSimData(filepath_sim_res, filepath_sim_res_plot, sim_name)

        ## create empty space
        print(" ")
      print(" ")
    print(" ")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BASEPATH          = "/scratch/ek9/nk7952/"
SONIC_REGIME      = "super_sonic"
FILENAME_TURB     = "Turb.dat"
K_TURB            = 2.0
RMS_MACH          = 5.0
T_TURB            = 1 / (K_TURB * RMS_MACH) # ell_turb / (Mach * c_s)
LIST_SUITE_FOLDER = [ "Re10", "Re500", "Rm3000" ]
LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
LIST_SIM_RES      = [ "18", "36", "72" ]
# LIST_SIM_RES      = [ "144", "288", "576" ]


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
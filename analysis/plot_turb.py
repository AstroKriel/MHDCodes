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
    self.Mach           = None
    self.Gamma          = None
    self.E_sat_ratio    = None
    ## perform routine
    print("Loading volume integrated data...")
    self.__loadData()
    self.__plotMach()
    self.__plotEnergyRatio()
    self.__fitData()

  def getExpTimeBounds(self):
    return self.time_exp_start, self.time_exp_end

  def getMach(self):
    return self.Mach

  def getGamma(self):
    return self.Gamma

  def getEsatRatio(self):
    return self.E_sat_ratio

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
    ## load mach data
    data_time, data_Mach = LoadFlashData.loadTurbData(
      filepath_data = self.filepath_data,
      var_y         = 13, # 13 (new), 8 (old)
      t_turb        = T_TURB,
      time_start    = 0.1,
      time_end      = np.inf
    )
    ## only relevant when loading data while a simulation is running (i.e., dealing w/ data synchronisation)
    ## check the largest sample of data that has been safely written
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
      bool_major_ticks    = True,
      max_num_major_ticks = num_y_major_ticks
    )

  def __fitData(self):
    ls_kin         = "--"
    ls_sat         = ":"
    str_label_Esat = r"$\left(E_{\rm kin} / E_{\rm mag}\right)_{\rm sat} =$ "
    str_label_Mach = r"$\mathcal{M} =$ "
    growth_percent = self.data_E_ratio[-1] / self.data_E_ratio[
      WWLists.getIndexClosestValue(self.data_time, 5.0)
    ]
    ## IF DYNAMO GROWTH OCCURS
    ## -----------------------
    if growth_percent > 100:
      ## find saturated energy ratio
      self.E_sat_ratio = FitFuncs.fitConstFunc(
        ax              = self.axs[1],
        data_x          = self.data_time,
        data_y          = self.data_E_ratio,
        str_label       = str_label_Esat,
        index_start_fit = WWLists.getIndexClosestValue(
          self.data_time, (0.75 * self.data_time[-1])
        ),
        index_end_fit   = len(self.data_time)-1,
        linestyle       = ls_sat
      )
      ## get index range corresponding with kinematic phase of the dynamo
      index_exp_start = WWLists.getIndexClosestValue(self.data_E_ratio, 10**(-8))
      index_exp_end   = WWLists.getIndexClosestValue(self.data_E_ratio, self.E_sat_ratio/100)
      index_start_fit = min([ index_exp_start, index_exp_end ])
      index_end_fit   = max([ index_exp_start, index_exp_end ])
      ## find growth rate of exponential
      self.Gamma = FitFuncs.fitExpFunc(
        ax              = self.axs[1],
        data_x          = self.data_time,
        data_y          = self.data_E_ratio,
        index_start_fit = index_start_fit,
        index_end_fit   = index_end_fit,
        linestyle       = ls_kin
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
        str_label       = str_label_Esat,
        index_start_fit = index_start_fit,
        index_end_fit   = index_end_fit,
        linestyle       = ls_sat
      )
    ## find average mach number
    self.Mach = FitFuncs.fitConstFunc(
      ax              = self.axs[0],
      data_x          = self.data_time,
      data_y          = self.data_Mach,
      str_label       = str_label_Mach,
      index_start_fit = index_start_fit,
      index_end_fit   = index_end_fit,
      linestyle       = ls_kin
    )
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
def plotSimData(filepath_sim, filepath_plot, sim_name):
  ## CREATE FIGURE
  ## -------------
  print("Initialising figure...")
  fig_scale          = 1.0
  fig_aspect_ratio   = (5.0, 8.0)
  num_rows, num_cols = (2, 2)
  fig = plt.figure(
    constrained_layout = True,
    figsize            = (
      fig_scale * fig_aspect_ratio[1] * num_cols,
      fig_scale * fig_aspect_ratio[0] * num_rows
  ))
  gs  = GridSpec(num_rows, num_cols, figure=fig)
  ax_mach   = fig.add_subplot(gs[0,  0])
  ax_energy = fig.add_subplot(gs[1,  0])
  ## PLOT INTEGRATED QUANTITIES (Turb.dat)
  ## -------------------------------------
  PlotTurbData(
    axs           = [ ax_mach, ax_energy ],
    filepath_data = filepath_sim
  )
  ## SAVE FIGURE
  ## -----------
  ## save the figure
  print("Saving figure...")
  fig_name = f"{sim_name}_time_evolution.png"
  fig_filepath = WWFnF.createFilepath([ filepath_plot, fig_name ])
  plt.savefig(fig_filepath)
  plt.close()
  print("Figure saved:", fig_name)


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BOOL_FIT_MODEL = 0
BOOL_DEBUG     = 0
BASEPATH       = "/scratch/ek9/nk7952/"
SONIC_REGIME   = "super_sonic"
FILENAME_TURB  = "Turb.dat"
K_TURB         = 2.0
MACH           = 5.0
T_TURB         = 1 / (K_TURB * MACH) # ell_turb / (Mach * c_s)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## LOOK AT EACH SIMULATION
  ## -----------------------
  ## loop over each simulation suite
  for suite_folder in [
      "Re10",
      # "Re500",
      "Rm3000"
    ]:

    ## loop over each resolution
    for sim_res in [
        # "72",
        # "144",
        # "288",
        "576"
      ]:

      ## CHECK THAT THE VISUALISATION FOLDER EXISTS
      ## ------------------------------------------
      filepath_plot = WWFnF.createFilepath([
        BASEPATH, suite_folder, sim_res, SONIC_REGIME, "vis_folder"
      ])
      if not os.path.exists(filepath_plot):
        print("{} does not exist.".format(filepath_plot))
        continue

      ## print to the terminal which suite is being looked at
      str_message = f"Looking at suite: {suite_folder}, Nres = {sim_res}"
      print(str_message)
      print("=" * len(str_message))
      print("Saving figures in:", filepath_plot)
      print(" ")

      ## PLOT SIMULATION DATA
      ## --------------------
      ## loop over each simulation folder
      for sim_folder in [
          "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"
        ]: # "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"

        ## create filepath to the simulation folder
        filepath_sim = WWFnF.createFilepath([
          BASEPATH, suite_folder, sim_res, SONIC_REGIME, sim_folder
        ])
        ## check that the filepath exists
        if not os.path.exists(filepath_sim): continue
        ## plot simulation data
        sim_name = f"{suite_folder}_{sim_folder}"
        plotSimData(filepath_sim, filepath_plot, sim_name)

        ## create empty space
        print(" ")
      print(" ")
    print(" ")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
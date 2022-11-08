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
from TheUsefulModule import WWLists, WWFnF, WWObjs
from TheJobModule import SimInputParams
from TheLoadingModule import LoadFlashData
from ThePlottingModule import PlotFuncs
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
      fig, axs, filepath_data, dict_sim_params
    ):
    ## save input arguments
    self.fig            = fig
    self.axs            = axs
    self.filepath_data  = filepath_data
    self.t_turb         = dict_sim_params["t_turb"]
    self.N_res          = int(dict_sim_params["sim_res"])
    self.Re             = dict_sim_params["Re"]
    self.Rm             = dict_sim_params["Rm"]
    self.Pm             = dict_sim_params["Pm"]
    ## initialise quantities to measure
    self.time_exp_start = None
    self.time_exp_end   = None
    self.rms_Mach       = None
    self.Gamma          = None
    self.E_sat_ratio    = None
    ## flag to check that quantities have been measured
    self.bool_fitted    = False

  def performRoutines(self):
    self.__loadData()
    self.__plotMach()
    self.__plotEnergyRatio()
    self.__fitData()
    self.bool_fitted = True
    self.__labelPlots()

  def getFittedParams(self):
    if not self.bool_fitted: self.performRoutines()
    return {
      "time_growth_start" : self.time_exp_start,
      "time_growth_end"   : self.time_exp_end,
      "rms_Mach"          : self.rms_Mach,
      "Gamma"             : self.Gamma,
      "E_sat_ratio"       : self.E_sat_ratio
    }

  def saveFittedParams(self, filepath_sim):
    dict_params = self.getFittedParams()
    WWObjs.saveDict2JsonFile(f"{filepath_sim}/sim_outputs.json", dict_params)

  def __loadData(self):
    print("Loading volume integrated data...")
    ## check if the Turb.dat file is formatted
    with open(f"{self.filepath_data}/Turb.dat") as fp:
      file_first_line = fp.readline()
    bool_format_new = "#01_time" in file_first_line.split() # new if #01_time else #00_time
    ## load kinetic energy
    _, data_kin_energy = LoadFlashData.loadTurbData(
      filepath   = self.filepath_data,
      var_y      = 9 if bool_format_new else 6, # 9+1 (new), 6 (old)
      t_turb     = self.t_turb,
      time_start = 0.1,
      time_end   = np.inf
    )
    ## load magnetic energy
    _, data_mag_energy = LoadFlashData.loadTurbData(
      filepath   = self.filepath_data,
      var_y      = 11 if bool_format_new else 29, # 11+1 (new), 29 (old)
      t_turb     = self.t_turb,
      time_start = 0.1,
      time_end   = np.inf
    )
    ## load Mach data
    data_time, data_Mach = LoadFlashData.loadTurbData(
      filepath   = self.filepath_data,
      var_y      = 13 if bool_format_new else 8, # 13+1 (new), 8 (old)
      t_turb     = self.t_turb,
      time_start = 0.1,
      time_end   = np.inf
    )
    ## Only relevant when loading data while a simulation is running
    ## (i.e., dealing w/ data synchronisation). So, only grab the portion
    ## (i.e., time-range) of data that has been sefely written for all quantities
    max_len = min([
      len(data_time),
      len(data_Mach),
      len(data_mag_energy),
      len(data_kin_energy)
    ])
    ## save data
    self.data_time       = data_time[:max_len]
    self.data_Mach       = data_Mach[:max_len]
    self.data_mag_energy = data_mag_energy[:max_len]
    self.data_kin_energy = data_kin_energy[:max_len]
    ## compute and save energy ratio: 'mag_energy / kin_energy'
    self.data_E_ratio = [
      mag_energy / kin_energy
      for mag_energy, kin_energy in zip(
        self.data_mag_energy,
        self.data_kin_energy
      )
    ]
    ## define plot domain
    self.max_time = max([
      100,
      max(self.data_time)
    ])

  def __plotMach(self):
    self.axs[0].plot(
      self.data_time,
      self.data_Mach,
      color="orange", ls="-", lw=1.5, zorder=3
    )
    self.axs[0].set_ylabel(r"$\mathcal{M}$")
    self.axs[0].set_xlim([ 0, self.max_time ])
  
  def __plotEnergyRatio(self):
    self.axs[1].plot(
      self.data_time,
      self.data_E_ratio,
      color="orange", ls="-", lw=1.5, zorder=3
    )
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
      index_start_fit = max([ t_start_index, min([ index_E_lo, index_E_hi ]) ])
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
      index_start_fit = WWLists.getIndexClosestValue(
        self.data_time,
        0.75 * self.data_time[-1]
      )
      index_end_fit = len(self.data_time)-1
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
    ## store time range bounds corresponding with the exponential phase of the dynamo
    self.time_exp_start = self.data_time[index_start_fit]
    self.time_exp_end   = self.data_time[index_end_fit]

  def __labelPlots(self):
    ## annotate simulation parameters
    PlotFuncs.addBoxOfLabels(
      self.fig, self.axs[0],
      box_alignment = (1.0, 0.0),
      xpos          = 0.95,
      ypos          = 0.05,
      alpha         = 0.5,
      fontsize      = 18,
      list_labels   = [
        r"${\rm N}_{\rm res} = $ " + "{:d}".format(int(self.N_res)),
        r"${\rm Re} = $ "          + "{:d}".format(int(self.Re)),
        r"${\rm Rm} = $ "          + "{:d}".format(int(self.Rm)),
        r"${\rm Pm} = $ "          + "{:d}".format(int(self.Pm)),
      ]
    )
    ## annotate measured quantities
    legend_ax0 = self.axs[0].legend(frameon=False, loc="lower left", fontsize=18)
    legend_ax1 = self.axs[1].legend(frameon=False, loc="lower right", fontsize=18)
    self.axs[0].add_artist(legend_ax0)
    self.axs[1].add_artist(legend_ax1)


## ###############################################################
## HANDLING PLOT CALLS
## ###############################################################
def plotSimData(filepath_sim, filepath_vis, sim_name):
  ## GET SIMULATION PARAMETERS
  ## -------------------------
  obj_sim_params  = SimInputParams.readSimInputParams(filepath_sim)
  dict_sim_params = obj_sim_params.getSimParams()
  ## INITIALISE FIGURE
  ## -----------------
  print("Initialising figure...")
  fig, fig_grid = PlotFuncs.createFigure_grid(
    fig_scale        = 1.0,
    fig_aspect_ratio = (5.0, 8.0),
    num_rows         = 2,
    num_cols         = 2
  )
  ax_Mach    = fig.add_subplot(fig_grid[0, 0])
  ax_E_ratio = fig.add_subplot(fig_grid[1, 0])
  ## LOAD AND PLOT INTEGRATED QUANTITIES
  ## -----------------------------------
  obj_plot_turb = PlotTurbData(
    fig             = fig,
    axs             = [ ax_Mach, ax_E_ratio ],
    filepath_data   = filepath_sim,
    dict_sim_params = dict_sim_params
  )
  obj_plot_turb.saveFittedParams(filepath_sim)
  obj_plot_turb.performRoutines()
  ## SAVE FIGURE
  ## -----------
  print("Saving figure...")
  fig_name     = f"{sim_name}_time_evolution.png"
  fig_filepath = f"{filepath_vis}/{fig_name}"
  plt.savefig(fig_filepath)
  plt.close()
  print("Saved figure:", fig_filepath)


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
      str_message = f"Looking at suite: {suite_folder}, sim: {sim_folder}, regime: {SONIC_REGIME}"
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
        filepath_sim_res_plot = f"{filepath_sim_res}/vis_folder/"
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

LIST_SUITE_FOLDER = [ "Re10", "Re500", "Rm3000" ]
LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
LIST_SIM_RES      = [ "18", "36", "72", "144", "288", "576" ]

# LIST_SUITE_FOLDER = [ "Rm3000" ]
# LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm5" ]
# LIST_SIM_RES      = [ "288" ]


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
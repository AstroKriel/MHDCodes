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
from TheSimModule import SimParams
from TheLoadingModule import LoadFlashData, FileNames
from ThePlottingModule import PlotFuncs
from TheFittingModule import FitFuncs


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
      fig, axs, filepath_sim_res, dict_sim_inputs,
      bool_verbose = True
    ):
    ## save input arguments
    self.fig              = fig
    self.axs              = axs
    self.filepath_sim_res = filepath_sim_res
    self.t_turb           = dict_sim_inputs["t_turb"]
    self.N_res            = int(dict_sim_inputs["sim_res"])
    self.Re               = dict_sim_inputs["Re"]
    self.Rm               = dict_sim_inputs["Rm"]
    self.Pm               = dict_sim_inputs["Pm"]
    self.bool_verbose     = bool_verbose

  def performRoutines(self):
    self._initialiseQuantities()
    self._loadData()
    self._plotMach()
    self._plotEnergyRatio()
    if max(self.data_time) > 5:
      self._fitData()
      self.bool_fitted = True
    self._labelPlots()

  def getFittedParams(self):
    self._checkAnyQuantitiesNotMeasured()
    if not self.bool_fitted: self.performRoutines()
    return {
      "plots_per_eddy"    : self.plots_per_eddy,
      "time_growth_start" : self.time_exp_start,
      "time_growth_end"   : self.time_exp_end,
      "rms_Mach"          : self.rms_Mach,
      "std_Mach"          : self.std_Mach,
      "Gamma"             : self.Gamma,
      "E_sat_ratio"       : self.E_sat_ratio
    }

  def saveFittedParams(self, filepath_sim):
    dict_params = self.getFittedParams()
    WWObjs.saveDict2JsonFile(f"{filepath_sim}/{FileNames.FILENAME_SIM_OUTPUTS}", dict_params, self.bool_verbose)

  def _initialiseQuantities(self):
    ## flag to check that all required quantities have been measured
    self.bool_fitted    = False
    ## initialise quantities to measure
    self.plots_per_eddy = None
    self.time_exp_start = None
    self.time_exp_end   = None
    self.rms_Mach       = None
    self.std_Mach       = None
    self.Gamma          = None
    self.E_sat_ratio    = None

  def _checkAnyQuantitiesNotMeasured(self):
    ## no need to check growth rate (Gamma) and saturated ratio (E_sat_ratio)
    list_quantities_check = [
      self.plots_per_eddy,
      self.time_exp_start,
      self.time_exp_end,
      self.rms_Mach,
      self.std_Mach
    ]
    list_quantities_undefined = [ 
      index_quantity
      for index_quantity, quantity in enumerate(list_quantities_check)
      if quantity is None
    ]
    if len(list_quantities_undefined) > 0: raise Exception("Error: the following quantities were not measured:", list_quantities_undefined)

  def _loadData(self):
    ## extract the number of plt-files per eddy-turnover-time from 'Turb.log'
    self.plots_per_eddy = LoadFlashData.getPlotsPerEddy_fromFlashLog(
      filepath     = self.filepath_sim_res,
      bool_verbose = False
    )
    if self.bool_verbose: print("Loading volume integrated data...")
    ## check how the integrated quantities are ordered in file
    with open(f"{self.filepath_sim_res}/{FileNames.FILENAME_FLASH_VOL}") as fp: file_first_line = fp.readline()
    ## load kinetic energy
    _, data_kin_energy = LoadFlashData.loadTurbData(
      filepath   = self.filepath_sim_res,
      quantity   = "kin",
      t_turb     = self.t_turb,
      time_start = 0.1,
      time_end   = np.inf
    )
    ## load magnetic energy
    _, data_mag_energy = LoadFlashData.loadTurbData(
      filepath   = self.filepath_sim_res,
      quantity   = "mag",
      t_turb     = self.t_turb,
      time_start = 0.1,
      time_end   = np.inf
    )
    ## load Mach data
    data_time, data_Mach = LoadFlashData.loadTurbData(
      filepath   = self.filepath_sim_res,
      quantity   = "mach",
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

  def _plotMach(self):
    self.axs[0].plot(
      self.data_time,
      self.data_Mach,
      color="orange", ls="-", lw=1.5, zorder=3
    )
    self.axs[0].set_ylabel(r"$\mathcal{M}$")
    self.axs[0].set_xlim([ 0, self.max_time ])
  
  def _plotEnergyRatio(self):
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
    PlotFuncs.addAxisTicks_log10(
      self.axs[1],
      bool_major_ticks = True,
      num_major_ticks  = num_y_major_ticks
    )

  def _fitData(self):
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
      self.E_sat_ratio, _ = FitFuncs.fitConstFunc(
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
      self.E_sat_ratio, _ = FitFuncs.fitConstFunc(
        ax              = self.axs[1],
        data_x          = self.data_time,
        data_y          = self.data_E_ratio,
        str_label       = label_Esat,
        index_start_fit = index_start_fit,
        index_end_fit   = index_end_fit,
        linestyle       = linestyle_sat
      )
    ## find average Mach number
    self.rms_Mach, self.std_Mach = FitFuncs.fitConstFunc(
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

  def _labelPlots(self):
    ## annotate simulation parameters
    PlotFuncs.addBoxOfLabels(
      self.fig, self.axs[0],
      bbox        = (1.0, 0.0),
      xpos        = 0.95,
      ypos        = 0.05,
      alpha       = 0.5,
      fontsize    = 18,
      list_labels = [
        r"${\rm N}_{\rm res} = $ " + "{:d}".format(int(self.N_res)),
        r"${\rm Re} = $ "          + "{:d}".format(int(self.Re)),
        r"${\rm Rm} = $ "          + "{:d}".format(int(self.Rm)),
        r"${\rm Pm} = $ "          + "{:d}".format(int(self.Pm)),
      ]
    )
    ## annotate measured quantities
    if self.bool_fitted:
      legend_ax0 = self.axs[0].legend(frameon=False, loc="lower left", fontsize=18)
      legend_ax1 = self.axs[1].legend(frameon=False, loc="lower right", fontsize=18)
      self.axs[0].add_artist(legend_ax0)
      self.axs[1].add_artist(legend_ax1)


## ###############################################################
## HANDLING PLOT CALLS
## ###############################################################
def plotSimData(
    filepath_sim_res,
    lock            = None,
    bool_check_only = False,
    bool_verbose    = True
  ):
  print("Looking at:", filepath_sim_res)
  ## get simulation parameters
  dict_sim_inputs = SimParams.readSimInputs(filepath_sim_res, bool_verbose)
  ## make sure a visualisation folder exists
  filepath_vis = f"{filepath_sim_res}/vis_folder/"
  WWFnF.createFolder(filepath_vis, bool_verbose=False)
  ## INITIALISE FIGURE
  ## -----------------
  if bool_verbose: print("Initialising figure...")
  fig, fig_grid = PlotFuncs.createFigure_grid(
    fig_scale        = 1.0,
    fig_aspect_ratio = (5.0, 8.0),
    num_rows         = 2,
    num_cols         = 2
  )
  ax_Mach         = fig.add_subplot(fig_grid[0, 0])
  ax_energy_ratio = fig.add_subplot(fig_grid[1, 0])
  ## PLOT INTEGRATED QUANTITIES
  ## --------------------------
  obj_plot_turb = PlotTurbData(
    fig              = fig,
    axs              = [ ax_Mach, ax_energy_ratio ],
    filepath_sim_res = filepath_sim_res,
    dict_sim_inputs  = dict_sim_inputs,
    bool_verbose     = bool_verbose
  )
  obj_plot_turb.performRoutines()
  ## SAVE FIGURE + DATASET
  ## ---------------------
  if lock is not None: lock.acquire()
  if not(bool_check_only): obj_plot_turb.saveFittedParams(filepath_sim_res)
  sim_name = SimParams.getSimName(dict_sim_inputs)
  fig_name = f"{sim_name}_time_evolution.png"
  PlotFuncs.saveFigure(fig, f"{filepath_vis}/{fig_name}", bool_verbose=True)
  if lock is not None: lock.release()
  if bool_verbose: print(" ")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  SimParams.callFuncForAllSimulations(
    func               = plotSimData,
    bool_mproc         = BOOL_MPROC,
    bool_check_only    = BOOL_CHECK_ONLY,
    basepath           = BASEPATH,
    list_suite_folders = LIST_SUITE_FOLDERS,
    list_sonic_regimes = LIST_SONIC_REGIMES,
    list_sim_folders   = LIST_SIM_FOLDERS,
    list_sim_res       = LIST_SIM_RES
  )


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
BOOL_MPROC         = 1
BOOL_CHECK_ONLY    = 1
BASEPATH           = "/scratch/ek9/nk7952/"

# ## PLASMA PARAMETER SET
# LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm3000" ]
# LIST_SONIC_REGIMES = [ "Mach0.3", "Mach5" ]
# LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
# # LIST_SIM_RES       = [ "18", "36", "72", "144", "288", "576" ]
# LIST_SIM_RES       = [ "144", "288" ]

## MACH NUMBER SET
LIST_SUITE_FOLDERS = [ "Re300" ]
LIST_SONIC_REGIMES = [ "Mach0.3", "Mach1", "Mach10" ]
LIST_SIM_FOLDERS   = [ "Pm4" ]
LIST_SIM_RES       = [ "288" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
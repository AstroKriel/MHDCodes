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
from TheFlashModule import LoadData, SimParams, FileNames
from TheUsefulModule import WWLists, WWFnF, WWObjs
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
      fig, axs, filepath_sim_res, dict_sim_inputs,
      bool_verbose = True
    ):
    ## save input arguments
    self.fig              = fig
    self.axs              = axs
    self.filepath_sim_res = filepath_sim_res
    self.dict_sim_inputs  = dict_sim_inputs
    self.bool_verbose     = bool_verbose

  def performRoutines(self):
    if self.bool_verbose: print("Loading volume integrated quantities...")
    self._loadData()
    if self.bool_verbose: print("Plotting volume integrated quantities...")
    self._plotMach()
    self._plotEnergyRatio()
    self.bool_fitted = False
    if max(self.data_time) > 5:
      self._fitData()
      self.bool_fitted = True
    self._labelPlots()

  def getFittedParams(self):
    if not self.bool_fitted: self.performRoutines()
    return {
      "outputs_per_t_turb" : self.outputs_per_t_turb,
      "E_growth_rate"      : {
        "val" : self.E_growth_rate_stats[0],
        "std" : self.E_growth_rate_stats[1]
      },
      "E_ratio_sat"        : {
        "val" : self.E_ratio_sat_val_stats[0],
        "std" : self.E_ratio_sat_val_stats[1]
      },
      "Mach"               : {
        "val" : self.Mach_stats[0],
        "std" : self.Mach_stats[1]
      },
      "time_bounds_growth" : self.time_bounds_growth,
      "time_start_sat"     : self.time_start_sat,
    }

  def saveFittedParams(self, filepath_sim):
    dict_params = self.getFittedParams()
    WWObjs.saveDict2JsonFile(f"{filepath_sim}/{FileNames.FILENAME_SIM_OUTPUTS}", dict_params, self.bool_verbose)

  def _loadData(self):
    ## extract the number of plt-files per eddy-turnover-time from 'Turb.log'
    self.outputs_per_t_turb = LoadData.getPlotsPerEddy_fromFlashLog(
      directory    = self.filepath_sim_res,
      max_num_t_turb   = self.dict_sim_inputs["max_num_t_turb"],
      bool_verbose = False
    )
    ## load Mach data
    _, data_Mach = LoadData.loadVIData(
      directory  = self.filepath_sim_res,
      field_name = "mach",
      t_turb     = self.dict_sim_inputs["t_turb"],
      time_start = 2.0,
      time_end   = np.inf
    )
    ## load kinetic energy
    _, data_Ekin = LoadData.loadVIData(
      directory  = self.filepath_sim_res,
      field_name = "kin",
      t_turb     = self.dict_sim_inputs["t_turb"],
      time_start = 2.0,
      time_end   = np.inf
    )
    ## load magnetic energy
    data_time, data_Emag = LoadData.loadVIData(
      directory  = self.filepath_sim_res,
      field_name = "mag",
      t_turb     = self.dict_sim_inputs["t_turb"],
      time_start = 2.0,
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
    ## define plot domain
    self.max_time = max([ 100, max(self.data_time) ])

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
    ## SATURATED REGIME
    ## ----------------
    if  (np.max(self.data_E_ratio) > 10**(-2)) or (
        (np.max(self.data_E_ratio) > 10**(-4)) and (np.max(self.data_time[-1]) > 20)
      ):
      ## get index and time associated with saturated regime
      time_start_sat      = 0.75 * self.data_time[-1]
      index_start_sat     = WWLists.getIndexClosestValue(self.data_time, time_start_sat)
      self.time_start_sat = self.data_time[index_start_sat]
      index_end_sat       = len(self.data_time)-1
      ## find saturated energy ratio
      self.E_ratio_sat_val_stats = FitFuncs.fitConstFunc(
        ax              = self.axs[1],
        data_x          = self.data_time,
        data_y          = self.data_E_ratio,
        str_label       = r"$\left(E_{\rm kin} / E_{\rm mag}\right)_{\rm sat} =$ ",
        index_start_fit = index_start_sat,
        index_end_fit   = index_end_sat,
        linestyle       = linestyle_sat
      )
    else:
      self.time_start_sat  = None
      self.E_ratio_sat_val_stats = None, None
    ## GROWTH REGIME
    ## -------------
    index_start_growth = WWLists.getIndexClosestValue(self.data_time, 5.0)
    if self.time_start_sat is not None:
      index_end_growth = WWLists.getIndexClosestValue(self.data_E_ratio, self.E_ratio_sat_val_stats[0] / 100)
    else: index_end_growth = len(self.data_E_ratio)-1
    ## find growth rate of exponential
    self.E_growth_rate_stats = FitFuncs.fitExpFunc(
      ax              = self.axs[1],
      data_x          = self.data_time,
      data_y          = self.data_E_ratio,
      index_start_fit = index_start_growth,
      index_end_fit   = index_end_growth,
      linestyle       = linestyle_kin
    )
    ## store time range of growth
    self.time_bounds_growth = [
      self.data_time[index_start_growth],
      self.data_time[index_end_growth]
    ]
    ## MACH NUMBER
    ## -----------
    self.Mach_stats = FitFuncs.fitConstFunc(
      ax              = self.axs[0],
      data_x          = self.data_time,
      data_y          = self.data_Mach,
      str_label       = r"$\mathcal{M} =$ ",
      index_start_fit = index_start_growth,
      index_end_fit   = index_end_growth,
      linestyle       = linestyle_kin
    )

  def _labelPlots(self):
    ## annotate simulation parameters
    SimParams.addLabel_simInputs(
      fig             = self.fig,
      ax              = self.axs[0],
      dict_sim_inputs = self.dict_sim_inputs,
      bbox            = (1,0),
      vpos            = (0.95, 0.05),
      bool_show_res   = True
    )
    ## annotate measured quantities
    if self.bool_fitted:
      self.axs[0].legend(frameon=False, loc="lower left", fontsize=18)
      if self.time_start_sat is not None:
        self.axs[1].legend(frameon=False, loc="lower right", fontsize=18)
      else: self.axs[1].legend(frameon=False, loc="upper right", fontsize=18)


## ###############################################################
## OPPERATOR HANDLING PLOT CALLS
## ###############################################################
def plotSimData(
    filepath_sim_res,
    lock            = None,
    bool_check_only = False,
    bool_verbose    = True
  ):
  print("Looking at:", filepath_sim_res)
  ## read simulation input parameters
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
    list_base_paths    = LIST_BASE_PATHS,
    list_suite_folders = LIST_SUITE_FOLDERS,
    list_mach_regimes  = LIST_MACH_REGIMES,
    list_sim_folders   = LIST_SIM_FOLDERS,
    list_sim_res       = LIST_SIM_RES
  )


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
BOOL_MPROC      = 1
BOOL_CHECK_ONLY = 0
LIST_BASE_PATHS = [
  "/scratch/ek9/nk7952/",
  # "/scratch/jh2/nk7952/"
]
LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Re2000", "Rm500", "Rm3000" ]
LIST_MACH_REGIMES  = [ "Mach0.3", "Mach1", "Mach5", "Mach10" ]
LIST_SIM_FOLDERS   = [
  "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm30", "Pm50", "Pm125", "Pm250", "Pm300"
]
LIST_SIM_RES = [ "576" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
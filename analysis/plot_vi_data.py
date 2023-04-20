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
    if self.bool_verbose: print("Loading volume integrated data...")
    self._loadData()
    if self.bool_verbose: print("Plotting volume integrated data...")
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
      "E_growth_rate"      : self.E_growth_rate,
      "E_growth_percent"   : self.E_growth_percent,
      "E_ratio_sat"        : self.E_ratio_sat,
      "rms_Mach_growth"    : self.rms_Mach_growth,
      "std_Mach_growth"    : self.std_Mach_growth,
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
    _, data_kin_energy = LoadData.loadVIData(
      directory  = self.filepath_sim_res,
      field_name = "kin",
      t_turb     = self.dict_sim_inputs["t_turb"],
      time_start = 2.0,
      time_end   = np.inf
    )
    ## load magnetic energy
    data_time, data_mag_energy = LoadData.loadVIData(
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
      len(data_kin_energy),
      len(data_mag_energy)
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
    ## get index and time associated with saturated regime
    time_start_sat      = 0.75 * self.data_time[-1]
    index_start_sat     = WWLists.getIndexClosestValue(self.data_time, time_start_sat)
    self.time_start_sat = self.data_time[index_start_sat]
    index_end_sat       = len(self.data_time)-1
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
    t_start_index = WWLists.getIndexClosestValue(self.data_time, 5.0)
    self.E_growth_percent = self.E_ratio_sat / self.data_E_ratio[t_start_index]
    if self.E_growth_percent > 10**2:
      ## get index range corresponding with growth phase
      index_E_lo = WWLists.getIndexClosestValue(self.data_E_ratio, 10**(-8))
      index_E_hi = WWLists.getIndexClosestValue(self.data_E_ratio, self.E_ratio_sat/100)
      index_start_growth = max([ t_start_index, min([ index_E_lo, index_E_hi ]) ])
      index_end_growth   = max([ index_E_lo, index_E_hi ])
      ## find growth rate of exponential
      self.E_growth_rate = FitFuncs.fitExpFunc(
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
      ## when growth occurs, measure Mach number over growth regime
      index_start_Mach = index_start_growth
      index_end_Mach   = index_end_growth
    else:
      ## undefined growth rate
      self.E_growth_rate = None
      ## indicate that there is no time of growth
      self.time_bounds_growth = [ None, None ]
      ## when no growth occurs, measure Mach number over saturated regime
      index_start_Mach = index_start_sat
      index_end_Mach   = index_end_sat
    ## MACH NUMBER
    ## -----------
    self.rms_Mach_growth, self.std_Mach_growth = FitFuncs.fitConstFunc(
      ax              = self.axs[0],
      data_x          = self.data_time,
      data_y          = self.data_Mach,
      str_label       = r"$\mathcal{M} =$ ",
      index_start_fit = index_start_Mach,
      index_end_fit   = index_end_Mach,
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
      legend_ax0 = self.axs[0].legend(frameon=False, loc="lower left", fontsize=18)
      legend_ax1 = self.axs[1].legend(frameon=False, loc="lower right", fontsize=18)
      self.axs[0].add_artist(legend_ax0)
      self.axs[1].add_artist(legend_ax1)


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
    basepath           = BASEPATH,
    list_suite_folders = LIST_SUITE_FOLDERS,
    list_sonic_regimes = LIST_SONIC_REGIMES,
    list_sim_folders   = LIST_SIM_FOLDERS,
    list_sim_res       = LIST_SIM_RES
  )


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
BOOL_MPROC      = 1
BOOL_CHECK_ONLY = 0
BASEPATH        = "/scratch/ek9/nk7952/"

## PLASMA PARAMETER SET
LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm3000" ]
LIST_SONIC_REGIMES = [ "Mach0.3", "Mach5" ]
LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
LIST_SIM_RES       = [ "18", "36", "72", "144", "288", "576" ]

# ## MACH NUMBER SET
# LIST_SUITE_FOLDERS = [ "Re300" ]
# LIST_SONIC_REGIMES = [ "Mach0.3", "Mach1", "Mach5", "Mach10" ]
# LIST_SIM_FOLDERS   = [ "Pm4" ]
# LIST_SIM_RES       = [ "18", "36", "72", "144", "288" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
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
import matplotlib as mpl
import matplotlib.pyplot as plt

## load user defined modules
from TheFlashModule import SimParams
from TheUsefulModule import WWFnF
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
plt.switch_backend("agg") # use a non-interactive plotting backend


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
  ## get simulation parameters
  dict_sim_inputs = SimParams.readSimInputs(filepath_sim_res, bool_verbose=False)
  ## make sure a visualisation folder exists
  filepath_vis = f"{filepath_sim_res}/vis_folder/"
  WWFnF.createFolder(filepath_vis, bool_verbose=False)
  ## INITIALISE FIGURE
  ## -----------------
  if bool_verbose: print("Initialising figure...")
  fig, fig_grid = PlotFuncs.createFigure_grid(
    fig_scale        = 0.6,
    fig_aspect_ratio = (6.0, 10.0), # height, width
    num_rows         = 4,
    num_cols         = 3
  )
  ## volume integrated qunatities
  ax_Mach         = fig.add_subplot(fig_grid[0, 0])
  ax_energy_ratio = fig.add_subplot(fig_grid[1, 0])
  ## power spectra data
  ax_spectra_ratio    = fig.add_subplot(fig_grid[2:, 0])
  ax_spectra_mag      = fig.add_subplot(fig_grid[0, 1])
  ax_spectra_vel_lgt  = fig.add_subplot(fig_grid[1, 1])
  ax_spectra_vel_trv  = fig.add_subplot(fig_grid[2, 1])
  axs_spectra         = [
    ax_spectra_mag,
    ax_spectra_vel_lgt,
    ax_spectra_vel_trv
  ]
  ## reynolds spectra
  ax_reynolds_mag     = fig.add_subplot(fig_grid[0, 2])
  ax_reynolds_vel_lgt = fig.add_subplot(fig_grid[1, 2])
  ax_reynolds_vel_trv = fig.add_subplot(fig_grid[2, 2])
  axs_reynolds        = [
    ax_reynolds_mag,
    ax_reynolds_vel_lgt,
    ax_reynolds_vel_trv
  ]
  ## measured scales
  ax_scales = fig.add_subplot(fig_grid[3, 1:])
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
  if not(bool_check_only): obj_plot_turb.saveFittedParams(filepath_sim_res)
  dict_turb_params = obj_plot_turb.getFittedParams()
  ## PLOT SPECTRA + MEASURED SCALES
  ## ------------------------------
  obj_plot_spectra = PlotSpectra(
    fig             = fig,
    dict_axs        = {
      "axs_spectra"      : axs_spectra,
      "axs_reynolds"     : axs_reynolds,
      "ax_spectra_ratio" : ax_spectra_ratio,
      "ax_scales"        : ax_scales,
    },
    filepath_spect     = f"{filepath_sim_res}/spect/",
    dict_sim_inputs    = dict_sim_inputs,
    outputs_per_t_turb = dict_turb_params["outputs_per_t_turb"],
    time_bounds_growth = dict_turb_params["time_bounds_growth"],
    time_start_sat     = dict_turb_params["time_start_sat"],
    bool_verbose       = bool_verbose
  )
  obj_plot_spectra.performRoutines()
  ## SAVE FIGURE + DATASET
  ## ---------------------
  if lock is not None: lock.acquire()
  if not(bool_check_only): obj_plot_spectra.saveFittedParams(filepath_sim_res)
  sim_name = SimParams.getSimName(dict_sim_inputs)
  fig_name = f"{sim_name}_dataset.png"
  PlotFuncs.saveFigure(fig, f"{filepath_vis}/{fig_name}", bool_verbose=True)
  if lock is not None: lock.release()
  if bool_verbose: print(" ")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  list_filepath_sim_res = SimParams.getListOfSimFilepaths(
    basepath           = BASEPATH,
    list_suite_folders = LIST_SUITE_FOLDERS,
    list_sonic_regimes = [ "Mach5" ],
    list_sim_folders   = LIST_SIM_FOLDERS,
    list_sim_res       = [ "288" ]
  )
  dict_data = {
    "list_Re" : [],
    "list_Rm" : [],
    "list_Pm" : [],
    "list_E_growth_rate" : [],
    "list_E_ratio_sat" : []
  }
  for filepath_sim_res in list_filepath_sim_res:
    dict_sim_inputs  = SimParams.readSimInputs(filepath_sim_res, bool_verbose=False)
    dict_sim_outputs = SimParams.readSimOutputs(filepath_sim_res, bool_verbose=False)
    dict_data["list_Re"].append(dict_sim_inputs["Re"])
    dict_data["list_Rm"].append(dict_sim_inputs["Rm"])
    dict_data["list_Pm"].append(dict_sim_inputs["Pm"])
    dict_data["list_E_growth_rate"].append(dict_sim_outputs["E_growth_rate"])
    dict_data["list_E_ratio_sat"].append(dict_sim_outputs["E_ratio_sat"])
  print(dict_data)
  fig, fig_grid = PlotFuncs.createFigure_grid(
    fig_scale        = 1.0,
    fig_aspect_ratio = (6.0, 10.0), # height, width
    num_rows         = 1,
    num_cols         = 2
  )
  ax_E_growth_rate = fig.add_subplot(fig_grid[0])
  ax_E_ratio_sat   = fig.add_subplot(fig_grid[1])
  ax_E_growth_rate.scatter(
    x = dict_data["list_Re"],
    y = dict_data["list_Rm"],
    c = -np.array(dict_data["list_E_growth_rate"]),
    norm = mpl.colors.LogNorm()
  )
  ax_E_growth_rate.set_xlabel(r"Re")
  ax_E_growth_rate.set_ylabel(r"Rm")
  ax_E_growth_rate.set_xscale("log")
  ax_E_growth_rate.set_yscale("log")
  ax_E_ratio_sat.scatter(
    x    = dict_data["list_Re"],
    y    = dict_data["list_Rm"],
    c    = dict_data["list_E_ratio_sat"],
    norm = mpl.colors.LogNorm()
  )
  ax_E_ratio_sat.set_xlabel(r"Re")
  ax_E_ratio_sat.set_ylabel(r"Rm")
  ax_E_ratio_sat.set_xscale("log")
  ax_E_ratio_sat.set_yscale("log")
  fig_name = f"fig_Mach5_plasma_space.png"
  PlotFuncs.saveFigure(fig, f"{BASEPATH}/{fig_name}", bool_verbose=True)



## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
BASEPATH = "/scratch/ek9/nk7952/"

## PLASMA PARAMETER SET
LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm3000" ]
LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
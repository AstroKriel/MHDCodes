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
## MAIN PROGRAM
## ###############################################################
def main():
  print("Filtering simulations...")
  list_filepath_sim_res = SimParams.getListOfSimFilepaths(
    list_base_paths = [
      "/scratch/ek9/nk7952/",
      "/scratch/jh2/nk7952/"
    ],
    list_suite_folders = LIST_SUITE_FOLDERS,
    list_mach_regimes  = [ MACH_REGIME ],
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
  print("Loading simulation data...")
  for filepath_sim_res in list_filepath_sim_res:
    print("Looking at:", filepath_sim_res)
    dict_sim_inputs  = SimParams.readSimInputs(filepath_sim_res, bool_verbose=False)
    dict_sim_outputs = SimParams.readSimOutputs(filepath_sim_res, bool_verbose=False)
    dict_data["list_Re"].append(dict_sim_inputs["Re"])
    dict_data["list_Rm"].append(dict_sim_inputs["Rm"])
    dict_data["list_Pm"].append(dict_sim_inputs["Pm"])
    dict_data["list_E_growth_rate"].append(dict_sim_outputs["E_growth_rate"]["val"])
    dict_data["list_E_ratio_sat"].append(dict_sim_outputs["E_ratio_sat"]["val"])
  print("Plotting data...")
  fig, fig_grid = PlotFuncs.createFigure_grid(
    fig_scale        = 1.0,
    fig_aspect_ratio = (6.0, 10.0), # height, width
    num_rows         = 2,
    num_cols         = 1
  )
  ax_E_growth_rate = fig.add_subplot(fig_grid[0])
  ax_E_ratio_sat   = fig.add_subplot(fig_grid[1])
  ## growth rate
  ax_E_growth_rate.scatter(
    x = dict_data["list_Re"],
    y = dict_data["list_Rm"],
    c = dict_data["list_E_growth_rate"],
    norm = mpl.colors.LogNorm()
  )
  ax_E_growth_rate.set_xlabel(r"Re")
  ax_E_growth_rate.set_ylabel(r"Rm")
  ax_E_growth_rate.set_xscale("log")
  ax_E_growth_rate.set_yscale("log")
  x = np.logspace(0, 5, 100)
  PlotFuncs.plotData_noAutoAxisScale(
    ax = ax_E_growth_rate,
    x  = x,
    y  = x**(1/2),
    ls = ":"
  )
  PlotFuncs.plotData_noAutoAxisScale(
    ax = ax_E_growth_rate,
    x  = x,
    y  = 10*x**(1/2),
    ls = ":"
  )
  PlotFuncs.plotData_noAutoAxisScale(
    ax = ax_E_growth_rate,
    x  = x,
    y  = 100*x**(1/2),
    ls = ":"
  )
  ## energy ratio
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
  PlotFuncs.plotData_noAutoAxisScale(
    ax = ax_E_ratio_sat,
    x  = x,
    y  = x**(1/2),
    ls = ":"
  )
  PlotFuncs.plotData_noAutoAxisScale(
    ax = ax_E_ratio_sat,
    x  = x,
    y  = 10*x**(1/2),
    ls = ":"
  )
  PlotFuncs.plotData_noAutoAxisScale(
    ax = ax_E_ratio_sat,
    x  = x,
    y  = 100*x**(1/2),
    ls = ":"
  )
  print("Saving figure...")
  fig_name = f"fig_{MACH_REGIME}_plasma_space.png"
  PlotFuncs.saveFigure(fig, f"/scratch/ek9/nk7952/{fig_name}", bool_verbose=True)


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
MACH_REGIME        = "Mach0.3"
LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Re2000", "Rm500", "Rm3000" ]
LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
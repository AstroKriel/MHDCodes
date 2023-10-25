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
def plotData(ax, filepath_sim_res):
  dict_sim_inputs = SimParams.readSimInputs(filepath_sim_res, True)
  Mach = dict_sim_inputs["desired_Mach"]
  t_turb = dict_sim_inputs["t_turb"]
  Re = dict_sim_inputs["Re"]
  Rm = dict_sim_inputs["Rm"]
  Pm = dict_sim_inputs["Pm"]
  if not(Pm == 5): return
  ## load kinetic energy
  _, data_Ekin = LoadData.loadVIData(
    directory  = filepath_sim_res,
    field_name = "kin",
    t_turb     = dict_sim_inputs["t_turb"],
    time_start = 2.0,
    time_end   = np.inf
  )
  ## load magnetic energy
  data_time, data_Emag = LoadData.loadVIData(
    directory  = filepath_sim_res,
    field_name = "mag",
    t_turb     = dict_sim_inputs["t_turb"],
    time_start = 2.0,
    time_end   = np.inf
  )
  ## measure rate of change
  dydx = t_turb * np.diff(data_Emag) / np.diff(data_time)
  dydx = np.insert(dydx, 0, 0)
  ## plot data
  if Mach == 0.3:
    zorder = 7
    color = "forestgreen"
  elif Mach == 1:
    zorder = 5
    color = "orange"
  elif Mach == 5:
    zorder = 3
    color = "skyblue"
  elif Mach == 10:
    zorder = 1
    color = "royalblue"
  else:
    print("Weird Mach:", Mach)
    return
  data = np.array(data_Emag) / np.array(data_Ekin)
  ax.plot(
    data,
    dydx,
    color="black", ls="-", lw=2, zorder=zorder
  )
  ax.plot(
    data,
    dydx,
    color=color, ls="-", lw=0.75, zorder=zorder+1
  )


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  list_filepaths = SimParams.getListOfSimFilepaths(
    list_base_paths    = LIST_BASE_PATHS,
    list_suite_folders = LIST_SUITE_FOLDERS,
    list_mach_regimes  = LIST_MACH_REGIMES,
    list_sim_folders   = LIST_SIM_FOLDERS,
    list_sim_res       = LIST_SIM_RES
  )
  ## plot data
  fig, ax = plt.subplots()
  for filepath_sim_res in list_filepaths:
    plotData(ax, filepath_sim_res)
  ## label axis
  ax.set_xscale("log")
  # ax.set_yscale("log")
  ax.set_xlim([ 10**(-3), 1 ])
  # ax.set_ylim([ 10**(-11), 10 ])
  ax.set_xlabel(r"$E_{\rm mag} / E_{\rm kin}$")
  ax.set_ylabel(r"${\rm d} E_{\rm mag} / {\rm d} (t / t_{\rm turb})$")
  # plot_dict = { "color":"red", "ls":"--", "lw":2, "zorder":10 }
  # ax.axhline(y=1, **plot_dict)
  # x = np.logspace(-15, 3, 100)
  # for factor in [ 1/100, 1/10, 1, 10, 100 ]:
  #   PlotFuncs.plotData_noAutoAxisScale(ax, x, factor*x, **plot_dict)
  # PlotFuncs.addAxisTicks_log10(ax, bool_y_axis=False, bool_major_ticks=True, num_major_ticks=12)
  # PlotFuncs.addAxisTicks_log10(ax, bool_y_axis=True,  bool_major_ticks=True, num_major_ticks=12)
  ## save figure
  filepath_fig = f"{PLOT_PATH}/phase_plane.png"
  print("Saving figure...")
  fig.savefig(filepath_fig, dpi=200)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
PLOT_PATH          = "/home/586/nk7952/MHDCodes/kriel2023"
LIST_BASE_PATHS    = [ "/scratch/ek9/nk7952/", "/scratch/jh2/nk7952/" ]
LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Re2000", "Rm500", "Rm3000" ]
LIST_MACH_REGIMES  = [ "Mach0.3", "Mach1", "Mach5", "Mach10" ]
LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm30", "Pm50", "Pm125", "Pm250", "Pm300" ]
LIST_SIM_RES       = [ "288" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
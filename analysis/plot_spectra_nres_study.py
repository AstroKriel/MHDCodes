#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np
import matplotlib.pyplot as plt

## load user defined modules
from TheUsefulModule import WWFnF, WWObjs
from TheLoadingModule import LoadFlashData
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def getAveSpectra(filepath_sim, str_field):
  print(f"Reading in '{str_field}' data:", filepath_sim)
  ## load relevant data from simulation folder
  plots_per_eddy = LoadFlashData.getPlotsPerEddy_fromTurbLog(filepath_sim, bool_hide_updates=True)
  dict_sim_data = WWObjs.loadJsonFile2Dict(
    filepath = filepath_sim,
    filename = f"sim_outputs.json",
    bool_hide_updates = True
  )
  try:
    list_k = dict_sim_data["list_k"]
    if "v" in str_field:   list_power_ave = dict_sim_data[f"list_kin_power_ave"]
    elif "m" in str_field: list_power_ave = dict_sim_data[f"list_mag_power_ave"]
  except:
    ## load energy spectra
    list_k_group_t, list_power_group_t, _ = LoadFlashData.loadAllSpectraData(
      filepath          = f"{filepath_sim}/spect/",
      str_spectra_type  = str_field,
      file_start_time   = dict_sim_data["time_growth_start"],
      file_end_time     = dict_sim_data["time_growth_end"],
      plots_per_eddy    = plots_per_eddy,
      bool_hide_updates = True
    )
    list_k = list_k_group_t[0]
    ## normalise and time-average energy spectra
    list_power_norm_group_t = [
      np.array(list_power) / sum(list_power)
      for list_power in list_power_group_t
    ]
    list_power_ave = np.mean(list_power_norm_group_t, axis=0)
  return list_k, list_power_ave

def plotAveSpectra(ax, filepath_sim, sim_res, str_field):
  dict_plot_style = {
    "18"  : "r--",
    "36"  : "g--",
    "72"  : "b--",
    "144" : "r-",
    "288" : "g-",
    "576" : "b-"
  }
  list_k, list_spectra_ave = getAveSpectra(filepath_sim, str_field)
  ax.plot(list_k, list_spectra_ave, dict_plot_style[sim_res], label=sim_res)


## ###############################################################
## MAIN PROGRAM
## ###############################################################

class PlotSpectraConvergence():
  def __init__(
      self,
      filepath_sim, filepath_vis, sim_name
    ):
    self.filepath_sim = filepath_sim
    self.filepath_vis = filepath_vis
    self.sim_name     = sim_name

  def plotSpectra(self):
    fig, fig_grid = PlotFuncs.createFigure_grid(num_rows=2)
    ax_kin = fig.add_subplot(fig_grid[0])
    ax_mag = fig.add_subplot(fig_grid[1])
    for sim_res in LIST_SIM_RES:
      filepath_sim_res = f"{self.filepath_sim}/{sim_res}/"
      if not os.path.exists(f"{filepath_sim_res}/spect/"): continue
      plotAveSpectra(ax_kin, filepath_sim_res, sim_res, "vel")
      plotAveSpectra(ax_mag, filepath_sim_res, sim_res, "mag")
    ## label kinetic energy spectra
    ax_kin.legend(loc="upper right")
    ax_kin.set_xscale("log")
    ax_kin.set_yscale("log")
    ax_kin.set_ylabel(r"$\widehat{\mathcal{P}}_{\rm kin}(k)$")
    ## label magnetic energy spectra
    ax_mag.legend(loc="upper right")
    ax_mag.set_xscale("log")
    ax_mag.set_yscale("log")
    ax_mag.set_ylabel(r"$\widehat{\mathcal{P}}_{\rm mag}(k)$")
    ## save figure
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{self.sim_name}_nres_spectra.png")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## LOOK AT EACH SIMULATION FOLDER
  ## ------------------------------
  ## loop over the simulation suites
  for suite_folder in LIST_SUITE_FOLDER:

    ## COMMUNICATE PROGRESS
    ## --------------------
    str_message = f"Looking at suite: {suite_folder}, regime: {SONIC_REGIME}"
    print(str_message)
    print("=" * len(str_message))
    print(" ")

    ## loop over the simulation folders
    for sim_folder in LIST_SIM_FOLDER:

      ## define name of simulation dataset
      sim_name = f"{suite_folder}_{sim_folder}"
      ## define filepath to simulation
      filepath_sim = WWFnF.createFilepath([ 
        BASEPATH, suite_folder, SONIC_REGIME, sim_folder
      ])

      ## CHECK THE NRES=288 DATASET EXISTS
      ## ---------------------------------
      ## check that the simulation spect-subfolder exists at Nres=288
      if not os.path.exists(f"{filepath_sim}/288/spect/"): continue
      print(f"Looking at sim: {sim_folder}...")

      ## MAKE SURE A VISUALISATION FOLDER EXISTS
      ## ---------------------------------------
      ## where plots/dataset of converged data will be stored
      filepath_vis = f"{filepath_sim}/vis_folder/"
      WWFnF.createFolder(filepath_vis, bool_hide_updates=True)

      ## MEASURE HOW WELL SCALES ARE CONVERGED
      ## -------------------------------------
      obj = PlotSpectraConvergence(filepath_sim, filepath_vis, sim_name)
      obj.plotSpectra()

      if BOOL_DEBUG: return
      ## create empty space
      print(" ")
    print(" ")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BOOL_DEBUG        = 0
BASEPATH          = "/scratch/ek9/nk7952/"
SONIC_REGIME      = "super_sonic"
LIST_SUITE_FOLDER = [ "Re10", "Re500", "Rm3000" ]
LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
LIST_SIM_RES      = [ "18", "36", "72", "144", "288", "576" ]


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np
import matplotlib.pyplot as plt

## load user defined modules
from TheSimModule import SimParams
from TheUsefulModule import WWFnF, WWObjs
from TheLoadingModule import LoadFlashData
from ThePlottingModule import PlotFuncs
from TheFittingModule import FitMHDScales


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def getAveSpectra(filepath_sim_res, str_field, bool_get_fit=False):
  print(f"Reading in '{str_field}' data:", filepath_sim_res)
  ## load relevant data from simulation folder
  dict_sim_outputs = SimParams.readSimOutputs(filepath_sim_res)
  ## read in averaged spectrum
  try:
    list_k = dict_sim_outputs["list_k"]
    if "v" in str_field:   list_power_ave = dict_sim_outputs["list_kin_power_ave"]
    elif "m" in str_field: list_power_ave = dict_sim_outputs["list_mag_power_ave"]
  except:
    ## number of plt/spect-files per eddy-turn-over-time
    plots_per_eddy = LoadFlashData.getPlotsPerEddy_fromTurbLog(filepath_sim_res, bool_hide_updates=True)
    ## load energy spectra
    dict_spect_data = LoadFlashData.loadAllSpectraData(
      filepath          = f"{filepath_sim_res}/spect/",
      spect_field       = str_field,
      file_start_time   = dict_sim_outputs["time_growth_start"],
      file_end_time     = dict_sim_outputs["time_growth_end"],
      plots_per_eddy    = plots_per_eddy,
      bool_hide_updates = True
    )
    ## extract data
    list_k             = dict_spect_data["list_k_group_t"][0]
    list_power_group_t = dict_spect_data["list_power_group_t"]
    ## normalise and time-average energy spectra
    list_power_norm_group_t = [
      np.array(list_power) / sum(list_power)
      for list_power in list_power_group_t
    ]
    list_power_ave = np.mean(list_power_norm_group_t, axis=0)
  ## generate fitted spectrum
  if bool_get_fit:
    data_k_fit = np.logspace(0, 3, 1000)
    list_fit_params = dict_sim_outputs["fit_params_kin_ave"]
    data_power_fit = FitMHDScales.SpectraModels.kinetic_linear(data_k_fit, *list_fit_params)
  else:
    data_k_fit = None
    data_power_fit = None
  return list_k, list_power_ave, data_k_fit, data_power_fit

def plotAveSpectra(axs, filepath_sim_res, sim_res, str_field, bool_plot_fit=False):
  dict_plot_style = {
    "18"  : "r--",
    "36"  : "g--",
    "72"  : "b--",
    "144" : "r-",
    "288" : "g-",
    "576" : "b-"
  }
  list_k, list_power_ave, data_k_fit, data_power_fit = getAveSpectra(filepath_sim_res, str_field, bool_plot_fit)
  if "v" in str_field:
    list_power_comp = np.array(list_power_ave) * np.array(list_k)**2
    data_power_fit_comp = np.array(data_power_fit) * np.array(data_k_fit)**2
  else: list_power_comp = np.array(list_power_ave) * np.array(list_k)**(-3/2)
  axs[0].plot(list_k, list_power_ave, dict_plot_style[sim_res], label=sim_res)
  axs[1].plot(list_k, list_power_comp, dict_plot_style[sim_res])
  if bool_plot_fit:
    PlotFuncs.plotData_noAutoAxisScale(axs[0], data_k_fit, data_power_fit)
    PlotFuncs.plotData_noAutoAxisScale(axs[1], data_k_fit, data_power_fit_comp)


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
    fig, fig_grid = PlotFuncs.createFigure_grid(num_rows=2, num_cols=2)
    ax_kin_data = fig.add_subplot(fig_grid[0,0])
    ax_kin_comp = fig.add_subplot(fig_grid[0,1])
    ax_mag_data = fig.add_subplot(fig_grid[1,0])
    ax_mag_comp = fig.add_subplot(fig_grid[1,1])
    for sim_res in LIST_SIM_RES:
      filepath_sim_res = f"{self.filepath_sim}/{sim_res}/"
      if not os.path.exists(f"{filepath_sim_res}/spect/"): continue
      plotAveSpectra([ ax_kin_data, ax_kin_comp ], filepath_sim_res, sim_res, "vel", bool_plot_fit=True)
      plotAveSpectra([ ax_mag_data, ax_mag_comp ], filepath_sim_res, sim_res, "mag")
    ## label kinetic energy spectra
    ax_kin_data.legend(loc="upper right")
    ax_kin_data.set_xscale("log")
    ax_kin_data.set_yscale("log")
    ax_kin_data.set_ylabel(r"$\widehat{\mathcal{P}}_{\rm kin}(k)$")
    ax_kin_comp.set_xscale("log")
    ax_kin_comp.set_yscale("log")
    ax_kin_comp.set_ylabel(r"$\widehat{\mathcal{P}}_{\rm kin}(k) k^2$")
    ## label magnetic energy spectra
    ax_mag_data.legend(loc="upper right")
    ax_mag_data.set_xscale("log")
    ax_mag_data.set_yscale("log")
    ax_mag_data.set_xlabel(r"$k$")
    ax_mag_data.set_ylabel(r"$\widehat{\mathcal{P}}_{\rm mag}(k)$")
    ax_mag_comp.set_xscale("log")
    ax_mag_comp.set_yscale("log")
    ax_mag_comp.set_xlabel(r"$k$")
    ax_mag_comp.set_ylabel(r"$\widehat{\mathcal{P}}_{\rm mag}(k) k^{-3/2}$")
    ## load simulation parameters
    dict_sim_inputs = SimParams.readSimInputs(f"{self.filepath_sim}/288/")
    ## annotate simulation parameters
    PlotFuncs.addBoxOfLabels(
      fig, ax_kin_data,
      box_alignment = (0.0, 0.0),
      xpos          = 0.05,
      ypos          = 0.05,
      alpha         = 0.5,
      fontsize      = 18,
      list_labels   = [
        r"${\rm N}_{\rm res} = $ " + "{:d}".format(int(dict_sim_inputs["sim_res"])),
        r"${\rm Re} = $ "          + "{:d}".format(int(dict_sim_inputs["Re"])),
        r"${\rm Rm} = $ "          + "{:d}".format(int(dict_sim_inputs["Rm"])),
        r"${\rm Pm} = $ "          + "{:d}".format(int(dict_sim_inputs["Pm"])),
      ]
    )
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
# LIST_SUITE_FOLDER = [ "Rm3000" ]
LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
LIST_SIM_RES      = [ "18", "36", "72", "144", "288", "576" ]


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
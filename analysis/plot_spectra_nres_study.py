#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np
import matplotlib.pyplot as plt

## load user defined modules
from TheSimModule import SimParams
from TheUsefulModule import WWFnF
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def createSubAxis(fig, fig_grid, row_index):
  return [
    fig.add_subplot(fig_grid[row_index, 0]),
    fig.add_subplot(fig_grid[row_index, 1])
  ]

def getSpectraComp(list_k, list_power, comp_factor):
  return np.array(list_power) * np.array(list_k)**(comp_factor)

def plotSpectra_res(axs, filepath_sim_res, sim_res):
  ## helper function
  def plotData(
      ax_row,
      list_k_data, list_power_data, comp_factor
    ):
    plot_style = dict_plot_style[sim_res]
    list_power_data_comp = getSpectraComp(list_k_data, list_power_data, comp_factor)
    axs[ax_row][0].plot(list_k_data, list_power_data,      plot_style, label=sim_res)
    axs[ax_row][1].plot(list_k_data, list_power_data_comp, plot_style, label=sim_res)
  ## look up table for plot styles
  dict_plot_style = {
    "18"  : "r--",
    "36"  : "g--",
    "72"  : "b--",
    "144" : "r-",
    "288" : "g-",
    "576" : "b-"
  }
  ## load relevant data
  print(f"Reading in data:", filepath_sim_res)
  dict_sim_outputs = SimParams.readSimOutputs(filepath_sim_res, bool_verbose=True)
  ## read in time-averaged spectra
  list_k_data            = dict_sim_outputs["list_k"]
  list_mag_power_tot     = dict_sim_outputs["list_mag_power_tot_ave"]
  list_kin_power_tot     = dict_sim_outputs["list_kin_power_tot_ave"]
  list_kin_power_lgt     = dict_sim_outputs["list_kin_power_lgt_ave"]
  list_kin_power_trv     = dict_sim_outputs["list_kin_power_trv_ave"]
  ## plot total magnetic energy spectra
  plotData(
    ax_row          = 0,
    list_k_data     = list_k_data,
    list_power_data = list_mag_power_tot,
    comp_factor     = -3/2
  )
  ## plot total kinetic energy spectra
  plotData(
    ax_row          = 1,
    list_k_data     = list_k_data,
    list_power_data = list_kin_power_tot,
    comp_factor     = 2.0
  )
  ## plot longitudinal kinetic energy spectra
  plotData(
    ax_row          = 2,
    list_k_data     = list_k_data,
    list_power_data = list_kin_power_lgt,
    comp_factor     = 2.0
  )
  ## plot transverse kinetic energy spectra
  plotData(
    ax_row          = 3,
    list_k_data     = list_k_data,
    list_power_data = list_kin_power_trv,
    comp_factor     = 2.0
  )


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
    fig, fig_grid = PlotFuncs.createFigure_grid(num_rows=4, num_cols=2)
    axs_mag_tot   = createSubAxis(fig, fig_grid, row_index=0)
    axs_kin_tot   = createSubAxis(fig, fig_grid, row_index=1)
    axs_kin_lgt   = createSubAxis(fig, fig_grid, row_index=2)
    axs_kin_trv   = createSubAxis(fig, fig_grid, row_index=3)
    axs = [
      axs_mag_tot,
      axs_kin_tot,
      axs_kin_lgt,
      axs_kin_trv
    ]
    for sim_res in LIST_SIM_RES:
      filepath_sim_res = f"{self.filepath_sim}/{sim_res}/"
      if not os.path.exists(f"{filepath_sim_res}/spect/"): continue
      plotSpectra_res(axs, filepath_sim_res, sim_res)
    ## adjust figure axis
    for row_index in range(len(axs)):
      axs[row_index][0].legend(loc="upper right")
      axs[row_index][1].legend(loc="upper right")
      axs[row_index][0].set_xscale("log")
      axs[row_index][0].set_yscale("log")
      axs[row_index][1].set_xscale("log")
      axs[row_index][1].set_yscale("log")
    ## label axis
    label_mag_tot = r"$\widehat{\mathcal{P}}_{\rm mag, tot}(k)$"
    label_kin_tot = r"$\widehat{\mathcal{P}}_{\rm kin, tot}(k)$"
    label_kin_lgt = r"$\widehat{\mathcal{P}}_{\rm kin, \parallel}(k)$"
    label_kin_trv = r"$\widehat{\mathcal{P}}_{\rm kin, \perp}(k)$"
    axs[0][0].set_ylabel(label_mag_tot)
    axs[1][0].set_ylabel(label_kin_tot)
    axs[2][0].set_ylabel(label_kin_lgt)
    axs[3][0].set_ylabel(label_kin_trv)
    axs[0][1].set_ylabel(r"$k^{-3/2} \,$" + label_mag_tot)
    axs[1][1].set_ylabel(r"$k^2 \,$"      + label_kin_tot)
    axs[2][1].set_ylabel(r"$k^2 \,$"      + label_kin_lgt)
    axs[3][1].set_ylabel(r"$k^2 \,$"      + label_kin_trv)
    axs[-1][0].set_xlabel(r"$k$")
    axs[-1][1].set_xlabel(r"$k$")
    SimParams.addLabel_simInputs(
      filepath_sim_res = f"{self.filepath_sim}/288/",
      fig              = fig,
      ax               = axs[0][0]
    )
    ## save figure
    fig_name = f"{self.sim_name}_nres_spectra.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")


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
      WWFnF.createFolder(filepath_vis, bool_verbose=True)

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

# LIST_SUITE_FOLDER = [ "Rm3000" ]
# LIST_SIM_FOLDER   = [ "Pm1" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
#!/usr/bin/env python3

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
from matplotlib.collections import LineCollection

## load user defined modules
from TheUsefulModule import WWObjs, WWFnF
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def addLegend_suites(ax):
  PlotFuncs.addLegend(
    ax,
    list_artists       = [ "s", "D", "o" ],
    list_legend_labels = [
      r"$\mathrm{Re} = 10$",
      r"$\mathrm{Re} = 500$",
      r"$\mathrm{Rm} = 3000$",
    ],
    list_marker_colors = [ "k" ],
    label_color        = "black",
    loc                = "upper left",
    bbox               = (-0.05, 1.05)
  )

def addLegend_Re(ax):
  args = { "va":"bottom", "ha":"right", "transform":ax.transAxes, "fontsize":15 }
  ax.text(0.925, 0.225, r"Re $< 100$", color="blue", **args)
  ax.text(0.925, 0.1,   r"Re $> 100$", color="red",  **args)

def plotDataNoAutoAxisScale(ax, x, y, c="k", ls=":"):
  col = LineCollection([ np.column_stack((x, y)) ], colors=c, linestyles=ls)
  ax.add_collection(col, autolim=False)

def plotErrorBar_1D(ax, x, array_y, color="k", marker="o"):
  y_median = np.percentile(array_y, 50)
  y_p16    = np.percentile(array_y, 16)
  y_p84    = np.percentile(array_y, 84)
  y_1sig   = np.vstack([
    y_median - y_p16,
    y_p84 - y_median
  ])
  ax.errorbar(
    x, y_median,
    yerr  = y_1sig,
    color = color,
    fmt   = marker,
    markersize=7, elinewidth=2, linestyle="None", markeredgecolor="black", capsize=7.5, zorder=10
  )


## ###############################################################
## LOAD + PLOT MHD SCALES
## ###############################################################
class PlotSimScales():
  def __init__(self, filepath_vis):
    self.filepath_vis = filepath_vis
    print("Saving figures in:", self.filepath_vis)
    print(" ")
    ## INITIALISE DATA CONTAINERS
    ## --------------------------
    ## simulation parameters
    self.Re_group        = []
    self.Rm_group        = []
    self.Pm_group        = []
    self.color_group     = []
    self.marker_group    = []
    ## measured quantities
    self.list_alpha_kin_group = []
    self.list_k_nu_group      = []
    self.list_k_p_group       = []
    self.list_k_eq_group      = []
    self.__loadAllSimulationData()

  def __loadAllSimulationData(self):
    ## loop over the simulation suites
    for suite_folder in LIST_SUITE_FOLDER:
      ## check the suite's figure folder exists
      filepath_suite_output = WWFnF.createFilepath([ BASEPATH, suite_folder, SONIC_REGIME ])
      if not os.path.exists(filepath_suite_output):
        print("{} does not exist.".format(filepath_suite_output))
        continue
      str_message = "Loading datasets from suite: {}".format(suite_folder)
      print(str_message)
      print("=" * len(str_message))
      ## loop over the simulation folders
      for sim_folder in LIST_SIM_FOLDER:
        ## check that the fitted spectra data exists for the Nres=288 simulation setup
        filepath_sim = WWFnF.createFilepath([
          BASEPATH, suite_folder, STR_SIM_RES, SONIC_REGIME, sim_folder
        ])
        dataset_name = f"{suite_folder}_{sim_folder}_dataset.json"
        filepath_dataset = f"{filepath_sim}/{dataset_name}"
        if not os.path.isfile(filepath_dataset): continue
        ## load scales
        print(f"\t> Loading '{sim_folder}' dataset.")
        if suite_folder == "Re10": self.marker_group.append("s")
        elif suite_folder == "Re500": self.marker_group.append("D")
        elif suite_folder == "Rm3000": self.marker_group.append("o")
        self.__getParams(filepath_sim, dataset_name)
      ## create empty space
      print(" ")

  def __getParams(self, filepath_data, dataset_name):
    ## load spectra-fit data as a dictionary
    dict = WWObjs.loadJson2Dict(
      filepath          = filepath_data,
      filename          = dataset_name,
      bool_hide_updates = True
    )
    ## extract plasma Reynolds numbers
    Re = int(dict["Re"])
    Rm = int(dict["Rm"])
    Pm = int(dict["Pm"])
    self.Re_group.append(Re)
    self.Rm_group.append(Rm)
    self.Pm_group.append(Pm)
    self.color_group.append( "cornflowerblue" if Re < 100 else "orangered" )
    ## extract kinetic energy scales
    self.list_alpha_kin_group.append(dict["list_alpha_kin"])
    self.list_k_nu_group.append(dict["list_k_nu"])
    ## extract magnetic energy scales
    self.list_k_p_group.append(dict["list_k_p"])
    self.list_k_eq_group.append(dict["list_k_eq"])

  def plotDependance_knu(self):
    fig, ax = plt.subplots(1, 1, figsize=(7/1.1, 4/1.1))
    for sim_index in range(len(self.Pm_group)):
      plotErrorBar_1D(
        ax,
        self.Re_group[sim_index]**(2/3),
        self.list_k_nu_group[sim_index],
        color  = self.color_group[sim_index],
        marker = self.marker_group[sim_index]
      )
    ## plot reference lines
    x = np.linspace(10**(-1), 10**(3), 10**4)
    plotDataNoAutoAxisScale(ax, x, x / 2.5)
    ## label figure
    addLegend_suites(ax)
    addLegend_Re(ax)
    ax.set_xlabel(r"$\mathrm{Re}^{2/3}$", fontsize=20)
    ax.set_ylabel(r"$k_\nu$", fontsize=20)
    ## adjust axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    ## save plot
    fig_name = f"fig_dependance_knu_{STR_SIM_RES}.png"
    fig_filepath = WWFnF.createFilepath([ self.filepath_vis, fig_name ])
    plt.savefig(fig_filepath)
    plt.close(fig)
    print("Saved figure:", fig_name)

  def plotDependance_kp(self):
    fig, ax = plt.subplots(1, 1, figsize=(7/1.1, 4/1.1))
    for sim_index in range(len(self.Pm_group)):
      plotErrorBar_1D(
        ax,
        self.Re_group[sim_index]**(2/3) * self.Pm_group[sim_index],
        self.list_k_p_group[sim_index],
        color  = self.color_group[sim_index],
        marker = self.marker_group[sim_index]
      )
    ## plot reference lines
    x = np.linspace(10**(-2), 10**(4), 100)
    # ax.plot(x, x**(1/4), "k:")
    plotDataNoAutoAxisScale(ax, x, 1.15*x**(1/4))
    ## label figure
    addLegend_suites(ax)
    addLegend_Re(ax)
    ax.set_xlabel(r"$\mathrm{Re}^{2/3}\, \mathrm{Pm}$", fontsize=20)
    ax.set_ylabel(r"$k_{\rm p}$", fontsize=20)
    ## adjust axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    ## save plot
    fig_name = f"fig_dependance_kp_{STR_SIM_RES}.png"
    fig_filepath = WWFnF.createFilepath([ self.filepath_vis, fig_name ])
    plt.savefig(fig_filepath)
    plt.close(fig)
    print("Saved figure:", fig_name)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  plot_obj = PlotSimScales(BASEPATH)
  plot_obj.plotDependance_knu()
  plot_obj.plotDependance_kp()


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BASEPATH          = "/scratch/ek9/nk7952/"
SONIC_REGIME      = "super_sonic"
LIST_SUITE_FOLDER = [ "Re10", "Re500", "Rm3000" ]
LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
STR_SIM_RES       = "576"


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()

 
## END OF PROGRAM
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

## load user defined modules
from TheSimModule import SimParams
from TheUsefulModule import WWFnF, WWObjs
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def addLegend_Re(ax):
  args = { "va":"bottom", "ha":"right", "transform":ax.transAxes, "fontsize":15 }
  ax.text(0.925, 0.225, r"Re $< 100$", color="blue", **args)
  ax.text(0.925, 0.1,   r"Re $> 100$", color="red",  **args)

def plotScale(ax, x, y_median, y_1sig, color, marker):
  ax.errorbar(
    x, y_median,
    yerr  = y_1sig,
    color = color,
    fmt   = marker,
    markersize=7, markeredgecolor="black", capsize=7.5, elinewidth=2,
    linestyle="None", zorder=10
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
    self.Re_group     = []
    self.Rm_group     = []
    self.Pm_group     = []
    self.color_group  = []
    self.marker_group = []
    ## measured quantities
    self.k_nu_adj_trv_stats_group_sim       = []
    self.k_nu_adj_trv_stats_group_sim_fixed = []
    self.k_p_stats_group_sim                = []
    self.__loadAllSimulationData()

  def __loadAllSimulationData(self):
    ## loop over the simulation suites
    for suite_folder in LIST_SUITE_FOLDER:
      str_message = f"Loading datasets from suite: {suite_folder}"
      print(str_message)
      print("=" * len(str_message))
      ## loop over the simulation folders
      for sim_folder in LIST_SIM_FOLDER:
        ## check that the fitted spectra data exists for the Nres=288 simulation setup
        filepath_sim = WWFnF.createFilepath([
          BASEPATH, suite_folder, SONIC_REGIME, sim_folder
        ])
        if not os.path.isfile(f"{filepath_sim}/scales.json"): continue
        ## load scales
        print(f"\t> Loading '{sim_folder}' dataset.")
        bool_skip = self.__getParams(filepath_sim)
        if bool_skip: continue
        if suite_folder == "Re10":     self.marker_group.append("s")
        elif suite_folder == "Re500":  self.marker_group.append("D")
        elif suite_folder == "Rm3000": self.marker_group.append("o")
      ## create empty space
      print(" ")

  def __getParams(self, filepath_data):
    ## load spectra-fit data as a dictionary
    dict_sim_inputs = SimParams.readSimInputs(
      filepath     = f"{filepath_data}/288/",
      bool_verbose = False
    )
    dict_scales = WWObjs.readJsonFile2Dict(
      filepath     = filepath_data,
      filename     = "scales.json",
      bool_verbose = False
    )
    ## extract plasma Reynolds numbers
    Re = int(dict_sim_inputs["Re"])
    Rm = int(dict_sim_inputs["Rm"])
    Pm = int(dict_sim_inputs["Pm"])
    # if Re < 100: return True
    self.Re_group.append(Re)
    self.Rm_group.append(Rm)
    self.Pm_group.append(Pm)
    self.color_group.append( "cornflowerblue" if Re < 100 else "orangered" )
    ## extract measured scales
    self.k_nu_adj_trv_stats_group_sim.append(dict_scales["k_nu_adj_trv_stats"])
    self.k_nu_adj_trv_stats_group_sim_fixed.append(dict_scales["k_nu_adj_trv_stats_fixed"])
    self.k_p_stats_group_sim.append(dict_scales["k_p_stats"])
    return False

  def plotDependance_knu(self):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), sharex=True)
    for sim_index in range(len(self.Pm_group)):
      # plotScale(
      #   ax       = ax,
      #   x        = self.Re_group[sim_index],
      #   y_median = self.k_nu_adj_trv_stats_group_sim[sim_index][0],
      #   y_1sig   = self.k_nu_adj_trv_stats_group_sim[sim_index][1],
      #   color    = "orange", # self.color_group[sim_index],
      #   marker   = self.marker_group[sim_index]
      # )
      plotScale(
        ax       = ax,
        x        = self.Re_group[sim_index],
        y_median = self.k_nu_adj_trv_stats_group_sim_fixed[sim_index][0],
        y_1sig   = self.k_nu_adj_trv_stats_group_sim_fixed[sim_index][1],
        color    = "blue", # self.color_group[sim_index],
        marker   = self.marker_group[sim_index]
      )
    ## plot reference lines
    x = np.linspace(10**(-1), 10**(5), 10**4)
    PlotFuncs.plotData_noAutoAxisScale(ax, x, 1.5*x**(1/3),  ls=":")
    PlotFuncs.plotData_noAutoAxisScale(ax, x, 3*x**(1/3),    ls=":")
    PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.25*x**(2/3), ls="--")
    ## label figure
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
      loc                = "lower right",
      bbox               = (1.0, 0.0)
    )
    # addLegend_Re(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim([ 10**(-3), 10**(2) ])
    ax.set_ylabel(r"$k_{\nu, \perp}^{1 / \alpha}$", fontsize=20)
    ax.set_xlabel(r"$\mathrm{Re}$", fontsize=20)
    ## adjust axis
    ## save plot
    fig_name = f"fig_dependance_knu.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")
    print(" ")

  def plotDependance_kp(self):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for sim_index in range(len(self.Pm_group)):
      plotScale(
        ax       = ax,
        x        = self.Re_group[sim_index]**(2/3) * self.Pm_group[sim_index],
        y_median = self.k_p_stats_group_sim[sim_index][0],
        y_1sig   = self.k_p_stats_group_sim[sim_index][1],
        color    = self.color_group[sim_index],
        marker   = self.marker_group[sim_index]
      )
    ## plot reference lines
    x = np.linspace(10**(-2), 10**(4), 100)
    PlotFuncs.plotData_noAutoAxisScale(ax, x, 1.15*x**(1/4))
    ## label figure
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
    # addLegend_Re(ax)
    ax.set_ylim([ 1, 30 ])
    ax.set_xlabel(r"$\mathrm{Re}^{2/3}\, \mathrm{Pm}$", fontsize=20)
    ax.set_ylabel(r"$k_{\rm p}$", fontsize=20)
    ## adjust axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    ## save plot
    fig_name = f"fig_dependance_kp.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")
    print(" ")


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
LIST_SIM_RES      = [ "18", "36", "72", "144", "288", "576" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
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
from TheFlashModule import SimParams, FileNames
from TheUsefulModule import WWFnF, WWObjs
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
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
    self.k_p_stats_group_sim      = []
    self.k_eta_stats_group_sim    = []
    self.k_nu_lgt_stats_group_sim = []
    self.k_nu_trv_stats_group_sim = []
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
        if not os.path.isfile(f"{filepath_sim}/{FileNames.FILENAME_SIM_SCALES}"): continue
        ## load scales
        print(f"\t> Loading '{sim_folder}' dataset")
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
      filename     = FileNames.FILENAME_SIM_SCALES,
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
    self.k_p_stats_group_sim.append(dict_scales["k_p_stats_converge"])
    self.k_eta_stats_group_sim.append(dict_scales["k_eta_stats_converge"])
    self.k_nu_lgt_stats_group_sim.append(dict_scales["k_nu_lgt_stats_converge"])
    self.k_nu_trv_stats_group_sim.append(dict_scales["k_nu_trv_stats_converge"])
    return False

  def plotDependance_knu(self):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), sharex=True)
    for sim_index in range(len(self.Pm_group)):
      if "super" in SONIC_REGIME:
        knu_theory = self.Re_group[sim_index]**(2/3)
      else: knu_theory = self.Re_group[sim_index]**(3/4)
      plotScale(
        ax       = ax,
        x        = knu_theory,
        y_median = self.k_nu_trv_stats_group_sim[sim_index][0],
        y_1sig   = self.k_nu_trv_stats_group_sim[sim_index][1],
        color    = self.color_group[sim_index],
        marker   = self.marker_group[sim_index]
      )
    ## plot reference lines
    x = np.linspace(10**(-3), 10**(5), 10**4)
    if "super" in SONIC_REGIME:
      PlotFuncs.plotData_noAutoAxisScale(ax, x, x, ls=":")
    else: PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.025*x, ls=":")
    ## label figure
    PlotFuncs.addLegend_fromArtists(
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
      bbox               = (0.0, 1.0)
    )
    # addLegend_Re(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_xlim(left=10**(0))
    # ax.set_ylim([ 10**(-3), 10**(3) ])
    ax.set_ylabel(r"$k_{\nu, \perp}$", fontsize=20)
    if "super" in SONIC_REGIME:
      ax.set_xlabel(r"$\mathrm{Re}^{2/3}$", fontsize=20)
    else: ax.set_xlabel(r"$\mathrm{Re}^{3/4}$", fontsize=20)
    ## adjust axis
    ## save plot
    fig_name = f"fig_dependance_{SONIC_REGIME}_knu.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")
    print(" ")

  def plotDependance_keta(self):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), sharex=True)
    for sim_index in range(len(self.Pm_group)):
      # if "super" in SONIC_REGIME:
      #   keta_theory = self.Re_group[sim_index]**(2/3) * self.Pm_group[sim_index]**(1/4)
      # else: keta_theory = self.Re_group[sim_index]**(3/4) * self.Pm_group[sim_index]**(1/4)
      plotScale(
        ax       = ax,
        x        = self.Pm_group[sim_index], # keta_theory,
        y_median = self.k_eta_stats_group_sim[sim_index][0] / self.Re_group[sim_index]**(2/3),
        y_1sig   = self.k_eta_stats_group_sim[sim_index][1] / self.Re_group[sim_index]**(2/3),
        color    = self.color_group[sim_index],
        marker   = self.marker_group[sim_index]
      )
    ## plot reference lines
    x = np.linspace(10**(-3), 10**(5), 10**4)
    if "super" in SONIC_REGIME:
      PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.5*x**(1/2), ls="-")
      PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.75*x**(1/4), ls=":")
    else: PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.15*x, ls=":")
    ## label figure
    PlotFuncs.addLegend_fromArtists(
      ax,
      list_artists       = [ "-", ":" ],
      list_legend_labels = [
        r"Pm$^{1/2}$",
        r"Pm$^{1/4}$",
      ],
      list_marker_colors = [ "k" ],
      label_color        = "black",
      loc                = "lower right",
      bbox               = (1.0, 0.0),
      lw                 = 1
    )
    PlotFuncs.addLegend_fromArtists(
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
      bbox               = (0.0, 1.0)
    )
    # addLegend_Re(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_xlim(left=10**(0))
    # ax.set_ylim([ 10**(-3), 10**(3) ])
    ax.set_ylabel(r"$k_\eta / {\rm Re}^{2/3}$", fontsize=20)
    ax.set_xlabel(r"${\rm Pm}$", fontsize=20)
    # if "super" in SONIC_REGIME:
    #   ax.set_xlabel(r"${\rm Re}^{2/3} {\rm Pm}^{1/4}$", fontsize=20)
    # else: ax.set_xlabel(r"${\rm Re}^{3/4} {\rm Pm}^{1/4}$", fontsize=20)
    ## save plot
    fig_name = f"fig_dependance_{SONIC_REGIME}_keta.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")
    print(" ")

  def plotDependance_kp(self):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for sim_index in range(len(self.Pm_group)):
      if "super" in SONIC_REGIME:
        keta_theory = self.Re_group[sim_index]**(2/3) * self.Pm_group[sim_index]**(1/2)
      else: keta_theory = self.Re_group[sim_index]**(3/4) * self.Pm_group[sim_index]**(1/2)
      plotScale(
        ax       = ax,
        x        = keta_theory,
        y_median = self.k_p_stats_group_sim[sim_index][0],
        y_1sig   = self.k_p_stats_group_sim[sim_index][1],
        color    = self.color_group[sim_index],
        marker   = self.marker_group[sim_index]
      )
    ## plot reference lines
    x = np.linspace(10**(-2), 10**(4), 100)
    if "super" in SONIC_REGIME:
      mean_kp = np.mean([
        self.k_p_stats_group_sim[sim_index][0]
        for sim_index in range(len(self.Pm_group))
      ])
      ax.axhline(y=mean_kp, ls=":", c="black")
    else: PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.025*x, ls=":")
    ## label figure
    PlotFuncs.addLegend_fromArtists(
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
    if "super" in SONIC_REGIME:
      ax.set_xlabel(r"${\rm Re}^{2/3} {\rm Pm}^{1/2}$", fontsize=20)
    else: ax.set_xlabel(r"${\rm Re}^{3/4} {\rm Pm}^{1/2}$", fontsize=20)
    ax.set_ylabel(r"$k_{\rm p}$", fontsize=20)
    ## adjust axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    ## save plot
    fig_name = f"fig_dependance_{SONIC_REGIME}_kp.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")
    print(" ")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  plot_obj = PlotSimScales(BASEPATH)
  # plot_obj.plotDependance_knu()
  plot_obj.plotDependance_keta()
  # plot_obj.plotDependance_kp()


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
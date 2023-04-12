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
from TheFlashModule import SimParams, FileNames, LoadData
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

def plotScale(ax, x_median, y_median, y_1sig, color, marker, x_1sig=None):
  ax.errorbar(
    x_median, y_median,
    xerr  = x_1sig,
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
    if len(LIST_SONIC_REGIME) == 1:
      self.plot_name = LIST_SONIC_REGIME[0]
      self.bool_supersonic = LoadData.getNumberFromString(LIST_SONIC_REGIME[0], "Mach") > 1
    elif len(LIST_SONIC_REGIME) > 1:
      self.plot_name = "Mach_varied"
      self.bool_supersonic = None
    else: raise Exception("Error: need to provide sonic-regimes to look at")
    ## simulation parameters
    self.Mach_group_sim   = []
    self.Re_group_sim     = []
    self.Rm_group_sim     = []
    self.Pm_group_sim     = []
    self.color_group_sim  = []
    self.marker_group_sim = []
    ## measured quantities
    self.k_p_stats_group_sim       = []
    self.k_eta_cur_stats_group_sim = []
    self.k_eta_mag_stats_group_sim = []
    self.k_nu_lgt_stats_group_sim  = []
    self.k_nu_trv_stats_group_sim  = []
    self.__loadAllSimulationData()

  def __loadAllSimulationData(self):
    ## loop over the simulation suites
    for suite_folder in LIST_SUITE_FOLDER:
      str_message = f"Loading datasets from suite: {suite_folder}"
      print(str_message)
      print("=" * len(str_message))
      ## loop over the simulation folders
      for sim_folder in LIST_SIM_FOLDER:
        for sonic_regime in LIST_SONIC_REGIME:
          ## check that the fitted spectra data exists for the Nres=288 simulation setup
          filepath_sim = WWFnF.createFilepath([
            BASEPATH, suite_folder, sonic_regime, sim_folder
          ])
          if not os.path.isfile(f"{filepath_sim}/{FileNames.FILENAME_SIM_SCALES}"): continue
          ## load scales
          print(f"\t> Loading {sim_folder}, {sonic_regime} dataset")
          self.__getParams(filepath_sim)
          if suite_folder == "Re10":     self.marker_group_sim.append("s")
          elif suite_folder == "Re500":  self.marker_group_sim.append("D")
          elif suite_folder == "Rm3000": self.marker_group_sim.append("o")
          else: self.marker_group_sim.append("o")
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
    Mach = dict_sim_inputs["desired_Mach"]
    self.Mach_group_sim.append(Mach)
    self.Re_group_sim.append(Re)
    self.Rm_group_sim.append(Rm)
    self.Pm_group_sim.append(Pm)
    self.color_group_sim.append( "cornflowerblue" if Re < 100 else "orangered" )
    ## extract measured scales
    self.k_p_stats_group_sim.append(dict_scales["k_p_stats_nres"])
    self.k_eta_cur_stats_group_sim.append(dict_scales["k_eta_cur_stats_nres"])
    self.k_eta_mag_stats_group_sim.append(dict_scales["k_eta_mag_stats_nres"])
    self.k_nu_lgt_stats_group_sim.append(dict_scales["k_nu_lgt_stats_nres"])
    self.k_nu_trv_stats_group_sim.append(dict_scales["k_nu_trv_stats_nres"])

  def plotDependance_knu(self):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), sharex=True)
    for sim_index in range(len(self.Pm_group_sim)):
      plotScale(
        ax       = ax,
        x_median = self.Re_group_sim[sim_index],
        y_median = self.k_nu_trv_stats_group_sim[sim_index][0],
        y_1sig   = self.k_nu_trv_stats_group_sim[sim_index][1],
        color    = self.color_group_sim[sim_index],
        marker   = self.marker_group_sim[sim_index]
      )
    ## plot reference lines
    x = np.linspace(10**(-3), 10**(5), 10**4)
    PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.6*x**(2/3), ls=":")
    # PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.025*x, ls=":")
    ## label figure
    PlotFuncs.addLegend_fromArtists(
      ax,
      list_artists       = [ ":" ],
      list_legend_labels = [ r"$\propto {\rm Re}^{2/3}$" ],
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
    ## adjust axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Re", fontsize=20)
    ax.set_ylabel(r"$k_{\nu, \perp}$", fontsize=20)
    ## save plot
    fig_name = f"fig_dependance_{self.plot_name}_knu.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")
    print(" ")

  def plotDependance_keta_mag(self):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), sharex=True)
    for sim_index in range(len(self.Pm_group_sim)):
      v1 = self.k_eta_mag_stats_group_sim[sim_index][0]
      d1 = self.k_eta_mag_stats_group_sim[sim_index][1]
      v2 = self.k_nu_trv_stats_group_sim[sim_index][0]
      d2 = self.k_nu_trv_stats_group_sim[sim_index][1]
      plotScale(
        ax       = ax,
        x_median = self.Pm_group_sim[sim_index],
        y_median = v1 / v2,
        y_1sig   = (v1 / v2) * np.sqrt((d1 / v1)**2 + (d2 / v2)**2),
        color    = self.color_group_sim[sim_index],
        marker   = self.marker_group_sim[sim_index]
      )
    ## plot reference lines
    x = np.linspace(10**(-3), 10**(5), 10**4)
    PlotFuncs.plotData_noAutoAxisScale(ax, x, x**(1/3), ls=":")
    # PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.15*x, ls=":")
    ## label figure
    PlotFuncs.addLegend_fromArtists(
      ax,
      list_artists       = [ ":" ],
      list_legend_labels = [ r"$= {\rm Pm}^{1/3}$" ],
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
    ## adjust axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"${\rm Pm}$", fontsize=20)
    ax.set_ylabel(r"$k_{\eta, \mathbf{B}} / k_{\nu, \perp}$", fontsize=20)
    ## save plot
    fig_name = f"fig_dependance_{self.plot_name}_keta_mag.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")
    print(" ")
  
  def plotDependance_keta_cur(self):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), sharex=True)
    for sim_index in range(len(self.Pm_group_sim)):
      v1 = self.k_eta_cur_stats_group_sim[sim_index][0]
      d1 = self.k_eta_cur_stats_group_sim[sim_index][1]
      v2 = self.k_nu_trv_stats_group_sim[sim_index][0]
      d2 = self.k_nu_trv_stats_group_sim[sim_index][1]
      plotScale(
        ax       = ax,
        x_median = self.Pm_group_sim[sim_index],
        y_median = v1 / v2,
        y_1sig   = (v1 / v2) * np.sqrt((d1 / v1)**2 + (d2 / v2)**2),
        color    = self.color_group_sim[sim_index],
        marker   = self.marker_group_sim[sim_index]
      )
    ## plot reference lines
    x = np.linspace(10**(-3), 10**(5), 10**4)
    PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.35*x**(1/2), ls=":")
    # PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.15*x, ls=":")
    ## label figure
    PlotFuncs.addLegend_fromArtists(
      ax,
      list_artists       = [ ":" ],
      list_legend_labels = [ r"$\propto {\rm Pm}^{1/2}$" ],
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
    ## adjust axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"${\rm Pm}$", fontsize=20)
    ax.set_ylabel(r"$k_{\eta, \nabla\times\mathbf{B}} / k_{\nu, \perp}$", fontsize=20)
    ## save plot
    fig_name = f"fig_dependance_{self.plot_name}_keta_cur.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")
    print(" ")

  def plotDependance_kp_mag(self):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for sim_index in range(len(self.Pm_group_sim)):
      plotScale(
        ax       = ax,
        x_median = self.k_eta_mag_stats_group_sim[sim_index][0],
        x_1sig   = self.k_eta_mag_stats_group_sim[sim_index][1],
        y_median = self.k_p_stats_group_sim[sim_index][0],
        y_1sig   = self.k_p_stats_group_sim[sim_index][1],
        color    = self.color_group_sim[sim_index],
        marker   = self.marker_group_sim[sim_index]
      )
    ## plot reference lines
    x = np.linspace(10**(-2), 10**(4), 100)
    ax.axhline(y=5.0, ls=":", c="black")
    # PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.025*x, ls=":")
    # label figure
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
    ## adjust axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim([ 1, 30 ])
    ax.set_xlabel(r"$k_{\eta, \mathbf{B}}$", fontsize=20)
    ax.set_ylabel(r"$k_{\rm p}$", fontsize=20)
    ## save plot
    fig_name = f"fig_dependance_{self.plot_name}_kp_mag.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")
    print(" ")

  def plotDependance_kp_cur(self):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for sim_index in range(len(self.Pm_group_sim)):
      plotScale(
        ax       = ax,
        x_median = self.k_eta_cur_stats_group_sim[sim_index][0],
        x_1sig   = self.k_eta_cur_stats_group_sim[sim_index][1],
        y_median = self.k_p_stats_group_sim[sim_index][0],
        y_1sig   = self.k_p_stats_group_sim[sim_index][1],
        color    = self.color_group_sim[sim_index],
        marker   = self.marker_group_sim[sim_index]
      )
    ## plot reference lines
    x = np.linspace(10**(-2), 10**(4), 100)
    ax.axhline(y=5.0, ls=":", c="black")
    # PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.025*x, ls=":")
    # label figure
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
    ## adjust axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim([ 1, 30 ])
    ax.set_xlabel(r"$k_{\eta, \nabla\times\mathbf{B}}$", fontsize=20)
    ax.set_ylabel(r"$k_{\rm p}$", fontsize=20)
    ## save plot
    fig_name = f"fig_dependance_{self.plot_name}_kp_cur.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")
    print(" ")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  plot_obj = PlotSimScales(BASEPATH)
  plot_obj.plotDependance_knu()
  plot_obj.plotDependance_keta_mag()
  plot_obj.plotDependance_keta_cur()
  plot_obj.plotDependance_kp_mag()
  plot_obj.plotDependance_kp_cur()


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BASEPATH          = "/scratch/ek9/nk7952/"

## PLASMA PARAMETER SET
LIST_SONIC_REGIME = [ "Mach5" ]
LIST_SUITE_FOLDER = [ "Re10", "Re500", "Rm3000" ]
LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]

# ## MACH NUMBER SET
# LIST_SONIC_REGIME = [ "Mach0.3", "Mach1", "Mach5", "Mach10" ]
# LIST_SUITE_FOLDER = [ "Re300" ]
# LIST_SIM_FOLDER   = [ "Pm4" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
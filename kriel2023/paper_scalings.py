#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np
import matplotlib.pyplot as plt

## load user defined modules
from TheFlashModule import SimParams, FileNames
from TheUsefulModule import WWLists, WWFnF, WWObjs
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
    markersize=8, markeredgecolor="black", capsize=7.5, elinewidth=2,
    linestyle="None", zorder=10
  )


## ###############################################################
## LOAD + PLOT MHD SCALES
## ###############################################################
class PlotSimScales():
  def __init__(self):
    if len(LIST_MACH_REGIMES) > 1:
      self.plot_name = "Mach_varied"
      self.bool_supersonic = None
    else:
      self.plot_name = LIST_MACH_REGIMES[0]
      self.bool_supersonic = float(LIST_MACH_REGIMES[0].split("Mach")[1]) > 1
    self.filepath_vis = f"{PATH_PLOT}/{self.plot_name}/"
    WWFnF.createFolder(self.filepath_vis, bool_verbose=False)
    print("Saving figures in:", self.filepath_vis)
    print(" ")
    ## simulation parameters
    self.Mach_group_sim   = []
    self.Re_group_sim     = []
    self.Rm_group_sim     = []
    self.Pm_group_sim     = []
    self.color_group_sim  = []
    self.marker_group_sim = []
    ## measured quantities
    self.k_p_mag_stats_group_sim   = []
    self.k_eta_cur_stats_group_sim = []
    self.k_eta_mag_stats_group_sim = []
    self.k_nu_kin_stats_group_sim  = []
    self.__loadAllSimulationData()

  def __loadAllSimulationData(self):
    ## loop over mach regimes
    for scratch_path in LIST_SCRATCH_PATHS:
      ## loop over the simulation suites
      for suite_folder in LIST_SUITE_FOLDERS:
        str_message = f"Loading datasets from suite: {suite_folder}"
        print(str_message)
        print("=" * len(str_message))
        ## loop over the simulation folders
        for sim_folder in LIST_SIM_FOLDERS:
          for mach_regime in LIST_MACH_REGIMES:
            ## check that the fitted spectra data exists for the Nres=288 simulation setup
            filepath_sim = WWFnF.createFilepath([
              scratch_path, suite_folder, mach_regime, sim_folder
            ])
            if not os.path.isfile(f"{filepath_sim}/{FileNames.FILENAME_SIM_SCALES}"): continue
            ## load scales
            print(f"\t> Loading {sim_folder}, {mach_regime} dataset")
            self.__getParams(filepath_sim)
            if self.bool_supersonic is not None:
              if   suite_folder == "Re10":   self.marker_group_sim.append("s")
              elif suite_folder == "Re500":  self.marker_group_sim.append("D")
              elif suite_folder == "Rm3000": self.marker_group_sim.append("o")
            else:
              if   mach_regime == "Mach0.3": self.marker_group_sim.append("s")
              elif mach_regime == "Mach1":   self.marker_group_sim.append("D")
              elif mach_regime == "Mach5":   self.marker_group_sim.append("o")
              elif mach_regime == "Mach10":  self.marker_group_sim.append("^")
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
    self.Mach_group_sim.append(float(dict_sim_inputs["desired_Mach"]))
    Re = int(dict_sim_inputs["Re"])
    self.Re_group_sim.append(Re)
    self.Rm_group_sim.append(int(dict_sim_inputs["Rm"]))
    self.Pm_group_sim.append(int(dict_sim_inputs["Pm"]))
    if self.bool_supersonic is not None:
      self.color_group_sim.append("darkorange")
    else:
      cmap, norm = PlotFuncs.createCmap(
        cmap_name = "Blues",
        cmin      = 0.1,
        vmin      = 0.0,
        vmax      = 4
      )
      list_Re = [ 24, 300, 600, 300 ]
      mach_index = WWLists.getIndexClosestValue(list_Re, Re)
      self.color_group_sim.append(cmap(norm(mach_index)))
    ## extract measured scales
    self.k_p_mag_stats_group_sim.append(dict_scales["k_p_mag_stats_nres"])
    self.k_eta_cur_stats_group_sim.append(dict_scales["k_eta_cur_stats_nres"])
    self.k_eta_mag_stats_group_sim.append(dict_scales["k_eta_mag_stats_nres"])
    self.k_nu_kin_stats_group_sim.append(dict_scales["k_nu_kin_stats_nres"])

  def plotRoutines(self):
    self.plotDependance_knu_Re()
    self.plotDependance_keta_Pm(self.k_eta_mag_stats_group_sim,  "keta_mag")
    self.plotDependance_keta_Pm(self.k_eta_cur_stats_group_sim,  "keta_cur")
    self.plotDependance_kp_keta()

  def plotDependance_knu_Re(self):
    ## plot data
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for sim_index in range(len(self.Pm_group_sim)):
      plotScale(
        ax       = ax,
        x_median = self.Re_group_sim[sim_index],
        y_median = self.k_nu_kin_stats_group_sim[sim_index][0],
        y_1sig   = self.k_nu_kin_stats_group_sim[sim_index][1],
        color    = self.color_group_sim[sim_index],
        marker   = self.marker_group_sim[sim_index]
      )
    ## plot reference lines
    x = np.linspace(10**(-1), 10**(5), 10**4)
    PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.6*x**(2/3), ls=":")
    PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.25*x**(0.7), ls="--")
    PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.2*x**(3/4), ls="-")
    ## label figure
    PlotFuncs.addLegend_fromArtists(
      ax,
      list_artists       = [ ":", "--", "-" ],
      list_legend_labels = [
        r"$\propto {\rm Re}^{2/3}$",
        r"$\propto {\rm Re}^{0.7}$",
        r"$\propto {\rm Re}^{3/4}$"
      ],
      list_marker_colors = [ "k" ],
      label_color        = "black",
      loc                = "lower right",
      bbox               = (1.0, 0.0),
      lw                 = 1
    )
    if self.bool_supersonic is not None:
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
        bbox               = (-0.05, 1.025)
      )
    else:
      PlotFuncs.addLegend_fromArtists(
        ax,
        list_artists       = [ "s", "D", "o", "^" ],
        list_legend_labels = [
          r"$\mathcal{M} = 0.3$",
          r"$\mathcal{M} = 1$",
          r"$\mathcal{M} = 5$",
          r"$\mathcal{M} = 10$",
        ],
        list_marker_colors = [ "k" ],
        label_color        = "black",
        loc                = "upper left",
        bbox               = (-0.05, 1.025)
      )
    ## adjust axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim([ 0.9, 200 ])
    ax.set_xlabel(r"Re", fontsize=20)
    ax.set_ylabel(r"$k_{\nu, {\rm kin}}$", fontsize=20)
    ## save plot
    fig_name = f"fig_dependance_{self.plot_name}_knu_kin_Re.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")
    print(" ")

  def plotDependance_keta_Pm(self, k_eta_stats_group_sim, domain_name):
    ## plot data
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for sim_index in range(len(self.Pm_group_sim)):
      v1 = k_eta_stats_group_sim[sim_index][0]
      d1 = k_eta_stats_group_sim[sim_index][1]
      v2 = self.k_nu_kin_stats_group_sim[sim_index][0]
      d2 = self.k_nu_kin_stats_group_sim[sim_index][1]
      plotScale(
        ax       = ax,
        x_median = self.Pm_group_sim[sim_index],
        y_median = v1 / v2,
        y_1sig   = (v1 / v2) * np.sqrt((d1 / v1)**2 + (d2 / v2)**2),
        color    = self.color_group_sim[sim_index],
        marker   = self.marker_group_sim[sim_index]
      )
    ## plot reference lines
    x = np.linspace(10**(-1), 10**(5), 10**4)
    if   "mag" in domain_name.lower():
      PlotFuncs.plotData_noAutoAxisScale(ax, x, x**(1/3), ls=":")
      PlotFuncs.plotData_noAutoAxisScale(ax, x, x**(1/4), ls="--")
      list_artists = [ ":", "--" ]
      list_labels = [
        r"$= {\rm Pm}^{1/3}$",
        r"$= {\rm Pm}^{1/4}$",
      ]
    elif "cur" in domain_name.lower():
      PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.35*x**(1/2), ls=":")
      PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.45*x**(1/2), ls=":")
      list_artists = [ ":" ]
      list_labels = [ r"$\propto {\rm Pm}^{1/2}$" ]
    ## label figure
    PlotFuncs.addLegend_fromArtists(
      ax,
      list_artists       = list_artists,
      list_legend_labels = list_labels,
      list_marker_colors = [ "k" ],
      label_color        = "black",
      loc                = "lower right",
      bbox               = (1.0, 0.0),
      lw                 = 1
    )
    if self.bool_supersonic is not None:
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
        bbox               = (-0.05, 1.025)
      )
    else:
      PlotFuncs.addLegend_fromArtists(
        ax,
        list_artists       = [ "s", "D", "o", "^" ],
        list_legend_labels = [
          r"$\mathcal{M} = 0.3$",
          r"$\mathcal{M} = 1$",
          r"$\mathcal{M} = 5$",
          r"$\mathcal{M} = 10$",
        ],
        list_marker_colors = [ "k" ],
        label_color        = "black",
        loc                = "upper left",
        bbox               = (-0.05, 1.025)
      )
    ## adjust axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    if   "mag" in domain_name.lower(): ax.set_ylim([ 0.9, 11 ])
    elif "cur" in domain_name.lower(): ax.set_ylim([ 0.3, 11 ])
    ax.set_xlabel(r"${\rm Pm}$", fontsize=20)
    if   "mag" in domain_name.lower(): ax.set_ylabel(r"$k_{\eta, {\rm mag}} / k_{\nu, {\rm kin}}$", fontsize=20)
    elif "cur" in domain_name.lower(): ax.set_ylabel(r"$k_{\eta, {\rm cur}} / k_{\nu, {\rm kin}}$", fontsize=20)
    ## save plot
    fig_name = f"fig_dependance_{self.plot_name}_{domain_name}_Pm.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")
    print(" ")

  def plotDependance_kp_keta(self):
    ## plot data
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for sim_index in range(len(self.Pm_group_sim)):
      plotScale(
        ax       = ax,
        x_median = self.k_eta_cur_stats_group_sim[sim_index][0],
        x_1sig   = self.k_eta_cur_stats_group_sim[sim_index][1],
        y_median = self.k_p_mag_stats_group_sim[sim_index][0],
        y_1sig   = self.k_p_mag_stats_group_sim[sim_index][1],
        color    = self.color_group_sim[sim_index],
        marker   = self.marker_group_sim[sim_index]
      )
    ## plot reference lines
    ax.axhline(y=5.0, ls=":", c="black")
    x = np.linspace(10**(-1), 10**(5), 10**4)
    PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.4*x, ls="--")
    # label figure
    PlotFuncs.addLegend_fromArtists(
      ax,
      list_artists       = [ ":", "--" ],
      list_legend_labels = [
        r"$= k_{\rm p} = 5$",
        r"$\propto k_{\eta, {\rm cur}}$"
      ],
      list_marker_colors = [ "k" ],
      label_color        = "black",
      loc                = "lower right",
      bbox               = (1.0, 0.0),
      lw                 = 1
    )
    if self.bool_supersonic is not None:
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
        bbox               = (-0.05, 1.025)
      )
    else:
      PlotFuncs.addLegend_fromArtists(
        ax,
        list_artists       = [ "s", "D", "o", "^" ],
        list_legend_labels = [
          r"$\mathcal{M} = 0.3$",
          r"$\mathcal{M} = 1$",
          r"$\mathcal{M} = 5$",
          r"$\mathcal{M} = 10$",
        ],
        list_marker_colors = [ "k" ],
        label_color        = "black",
        loc                = "upper left",
        bbox               = (-0.05, 1.025)
      )
    ## adjust axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim([ 0.9, 30 ])
    ax.set_xlim([ 5, 70 ])
    ax.set_xlabel(r"$k_{\eta, {\rm cur}}$", fontsize=20)
    ax.set_ylabel(r"$k_{\rm p}$", fontsize=20)
    ## save plot
    fig_name = f"fig_dependance_{self.plot_name}_kp_keta_cur.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")
    print(" ")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  plot_obj_mach5 = PlotSimScales()
  plot_obj_mach5.plotRoutines()


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
PATH_PLOT = "/home/586/nk7952/MHDCodes/kriel2023/"
LIST_SCRATCH_PATHS = [
  "/scratch/ek9/nk7952/",
  # "/scratch/jh2/nk7952/"
]

# ## PLASMA PARAMETER SET
# LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm3000" ]
# LIST_MACH_REGIMES = [ "Mach5" ]
# LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]

## MACH NUMBER SET
LIST_SUITE_FOLDERS = [ "Rm3000" ]
LIST_MACH_REGIMES  = [ "Mach0.3", "Mach1", "Mach5", "Mach10" ]
LIST_SIM_FOLDERS   = [ "Pm1", "Pm5", "Pm10", "Pm125" ]

# ## BOTTLENECK RUN
# LIST_SUITE_FOLDERS = [ "Re2000" ]
# LIST_MACH_REGIMES = [ "Mach0.3", "Mach5" ]
# LIST_SIM_FOLDERS   = [ "Pm5" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
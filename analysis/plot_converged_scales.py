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
    markersize=8, markeredgecolor="black", capsize=7.5, elinewidth=2,
    linestyle="None", zorder=10
  )


## ###############################################################
## LOAD + PLOT MHD SCALES
## ###############################################################
class PlotSimScales():
  def __init__(self, filepath_vis):
    self.filepath_vis = filepath_vis
    WWFnF.createFolder(self.filepath_vis, bool_verbose=False)
    print("Saving figures in:", self.filepath_vis)
    print(" ")
    ## INITIALISE DATA CONTAINERS
    ## --------------------------
    if len(LIST_MACH_REGIMES) == 1:
      self.plot_name = LIST_MACH_REGIMES[0]
      self.bool_supersonic = LoadData.getNumberFromString(LIST_MACH_REGIMES[0], "Mach") > 1
    elif len(LIST_MACH_REGIMES) > 1:
      self.plot_name = "Mach_varied"
      self.bool_supersonic = None
    else: raise Exception("Error: you need to specify which sonic-regimes to look at")
    ## simulation parameters
    self.Mach_group_sim   = []
    self.Re_group_sim     = []
    self.Rm_group_sim     = []
    self.Pm_group_sim     = []
    self.color_group_sim  = []
    self.marker_group_sim = []
    ## measured quantities
    self.k_p_rho_stats_group_sim      = []
    self.k_p_mag_stats_group_sim      = []
    self.k_eta_cur_stats_group_sim    = []
    self.k_eta_mag_stats_group_sim    = []
    self.k_nu_kin_stats_group_sim     = []
    self.k_nu_vel_tot_stats_group_sim = []
    self.k_nu_vel_lgt_stats_group_sim = []
    self.k_nu_vel_trv_stats_group_sim = []
    self.__loadAllSimulationData()

  def __loadAllSimulationData(self):
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
            PATH_SCRATCH, suite_folder, mach_regime, sim_folder
          ])
          if not os.path.isfile(f"{filepath_sim}/{FileNames.FILENAME_SIM_SCALES}"):
            print(filepath_sim)
            continue
          ## load scales
          print(f"\t> Loading {sim_folder}, {mach_regime} dataset")
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
    self.Mach_group_sim.append(float(dict_sim_inputs["desired_Mach"]))
    self.Re_group_sim.append(int(dict_sim_inputs["Re"]))
    self.Rm_group_sim.append(int(dict_sim_inputs["Rm"]))
    self.Pm_group_sim.append(int(dict_sim_inputs["Pm"]))
    self.color_group_sim.append("darkorange")
    # self.color_group_sim.append( "cornflowerblue" if Re < 100 else "orangered" )
    ## extract measured scales
    self.k_p_rho_stats_group_sim.append(dict_scales["k_p_rho_stats_nres"])
    self.k_p_mag_stats_group_sim.append(dict_scales["k_p_mag_stats_nres"])
    self.k_eta_cur_stats_group_sim.append(dict_scales["k_eta_cur_stats_nres"])
    self.k_eta_mag_stats_group_sim.append(dict_scales["k_eta_mag_stats_nres"])
    self.k_nu_kin_stats_group_sim.append(dict_scales["k_nu_kin_stats_nres"])
    self.k_nu_vel_tot_stats_group_sim.append(dict_scales["k_nu_vel_tot_stats_nres"])
    self.k_nu_vel_lgt_stats_group_sim.append(dict_scales["k_nu_vel_lgt_stats_nres"])
    self.k_nu_vel_trv_stats_group_sim.append(dict_scales["k_nu_vel_trv_stats_nres"])

  def plotRoutines(self):
    self.plotDependance_knu_Re(self.k_nu_kin_stats_group_sim,     "knu_kin")
    self.plotDependance_knu_Re(self.k_nu_vel_tot_stats_group_sim, "knu_vel_tot")
    self.plotDependance_knu_Re(self.k_nu_vel_lgt_stats_group_sim, "knu_vel_lgt")
    self.plotDependance_knu_Re(self.k_nu_vel_trv_stats_group_sim, "knu_vel_trv")
    self.plotDependance_keta_Pm(self.k_eta_mag_stats_group_sim,   "keta_mag")
    self.plotDependance_keta_Pm(self.k_eta_cur_stats_group_sim,   "keta_cur")
    self.plotDependance_kp_scale(self.k_nu_kin_stats_group_sim,   "knu_kin")
    self.plotDependance_kp_scale(self.k_eta_mag_stats_group_sim,  "keta_mag")
    self.plotDependance_kp_scale(self.k_eta_cur_stats_group_sim,  "keta_cur")
    self.plotDependance_kp_scale(self.k_p_rho_stats_group_sim,    "krho")
    self.plotDependance_kp_number(self.Re_group_sim,              "Re")
    self.plotDependance_kp_number(self.Rm_group_sim,              "Rm")
    self.plotDependance_kp_number(self.Pm_group_sim,              "Pm")
    self.plotDependance_krho_knu()
    if self.bool_supersonic is None: self.plotDependance_kp_Mach()

  def plotDependance_knu_Re(self, k_nu_stats_group_sim, domain_name):
    ## check for valid input paramaters
    if not("vel" in domain_name.lower()) and not("kin" in domain_name.lower()):
      raise Exception(f"Error: '{domain_name}' is an invalid input")
    ## plot data
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for sim_index in range(len(self.Pm_group_sim)):
      plotScale(
        ax       = ax,
        x_median = self.Re_group_sim[sim_index],
        y_median = k_nu_stats_group_sim[sim_index][0],
        y_1sig   = k_nu_stats_group_sim[sim_index][1],
        color    = self.color_group_sim[sim_index],
        marker   = self.marker_group_sim[sim_index]
      )
    ## plot reference lines
    x = np.linspace(10**(-1), 10**(5), 10**4)
    PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.6*x**(2/3), ls=":")
    PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.5*x**(2/3), ls=":")
    PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.3*x**(3/4), ls="--")
    PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.75*x**(1/2), ls="-")
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
    # if self.bool_supersonic is not None: ax.set_ylim([ 0.9, 200 ])
    ax.set_xlabel(r"Re", fontsize=20)
    if   "kin" in domain_name.lower(): ax.set_ylabel(r"$k_{\nu, {\rm kin}}$", fontsize=20)
    elif "tot" in domain_name.lower(): ax.set_ylabel(r"$k_{\nu, {\rm vel}}$", fontsize=20)
    elif "lgt" in domain_name.lower(): ax.set_ylabel(r"$k_{\nu, {\rm vel}, \parallel}$", fontsize=20)
    elif "trv" in domain_name.lower(): ax.set_ylabel(r"$k_{\nu, {\rm vel}, \perp}$", fontsize=20)
    ## save plot
    fig_name = f"fig_dependance_{self.plot_name}_{domain_name}_Re.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")
    print(" ")

  def plotDependance_keta_Pm(self, k_eta_stats_group_sim, domain_name):
    ## check for valid input paramaters
    if not("mag" in domain_name.lower()) and not("cur" in domain_name.lower()):
      raise Exception(f"Error: '{domain_name}' is an invalid input")
    ## plot data
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for sim_index in range(len(self.Pm_group_sim)):
      v1 = k_eta_stats_group_sim[sim_index][0]
      d1 = k_eta_stats_group_sim[sim_index][1]
      v2 = self.k_nu_vel_lgt_stats_group_sim[sim_index][0]
      d2 = self.k_nu_vel_lgt_stats_group_sim[sim_index][1]
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
      label_ref_line = r"$= {\rm Pm}^{1/3}$"
      PlotFuncs.plotData_noAutoAxisScale(ax, x, x**(1/3), ls=":")
      PlotFuncs.plotData_noAutoAxisScale(ax, x, x**(1/4), ls="--")
      PlotFuncs.plotData_noAutoAxisScale(ax, x, x**(0.1), ls="-")
    elif "cur" in domain_name.lower():
      label_ref_line = r"$\propto {\rm Pm}^{1/2}$"
      PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.4*x**(1/2), ls=":")
      PlotFuncs.plotData_noAutoAxisScale(ax, x, 1.5*x**(1/2), ls=":")
    ## label figure
    PlotFuncs.addLegend_fromArtists(
      ax,
      list_artists       = [ ":" ],
      list_legend_labels = [ label_ref_line ],
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
    # if self.bool_supersonic is not None:
    #   if   "mag" in domain_name.lower(): ax.set_ylim([ 0.9, 11 ])
    #   elif "cur" in domain_name.lower(): ax.set_ylim([ 0.3, 11 ])
    ax.set_xlabel(r"${\rm Pm}$", fontsize=20)
    if   "mag" in domain_name.lower(): ax.set_ylabel(r"$k_{\eta, \mathbf{B}} / k_{\nu, {\rm vel}}$", fontsize=20)
    elif "cur" in domain_name.lower(): ax.set_ylabel(r"$k_{\eta, \nabla\times\mathbf{B}} / k_{\nu, {\rm vel}}$", fontsize=20)
    ## save plot
    fig_name = f"fig_dependance_{self.plot_name}_{domain_name}_Pm.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")
    print(" ")

  def plotDependance_kp_scale(self, k_stats_group_sim, domain_name):
    ## check for valid input paramaters
    if (
        not("vel" in domain_name.lower()) and
        not("kin" in domain_name.lower()) and
        not("mag" in domain_name.lower()) and
        not("cur" in domain_name.lower()) and
        not("rho" in domain_name.lower())):
      raise Exception(f"Error: '{domain_name}' is an invalid input")
    ## plot data
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for sim_index in range(len(self.Pm_group_sim)):
      plotScale(
        ax       = ax,
        x_median = k_stats_group_sim[sim_index][0],
        x_1sig   = k_stats_group_sim[sim_index][1],
        y_median = self.k_p_mag_stats_group_sim[sim_index][0],
        y_1sig   = self.k_p_mag_stats_group_sim[sim_index][1],
        color    = self.color_group_sim[sim_index],
        marker   = self.marker_group_sim[sim_index]
      )
    ## plot reference lines
    ax.axhline(y=5.0, ls=":", c="black")
    x = np.linspace(10**(-1), 10**(5), 10**4)
    PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.41*x, ls="--")
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
    # if self.bool_supersonic is not None:
    #   if   "mag" in domain_name.lower(): ax.set_xlim([ 5, 200 ])
    #   elif "cur" in domain_name.lower(): ax.set_xlim([ 5, 70 ])
    # if self.bool_supersonic is not None: ax.set_ylim([ 1, 30 ])
    if   "mag"     in domain_name.lower(): ax.set_xlabel(r"$k_{\eta, \mathbf{B}}$", fontsize=20)
    elif "cur"     in domain_name.lower(): ax.set_xlabel(r"$k_{\eta, \nabla\times\mathbf{B}}$", fontsize=20)
    elif "rho"     in domain_name.lower(): ax.set_xlabel(r"$k_{{\rm p}, \rho}$", fontsize=20)
    elif "vel_tot" in domain_name.lower(): ax.set_xlabel(r"$k_{\nu, {\rm vel}}$", fontsize=20)
    ax.set_ylabel(r"$k_{{\rm p}, \mathbf{B}}$", fontsize=20)
    ## save plot
    fig_name = f"fig_dependance_{self.plot_name}_kp_{domain_name}.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")
    print(" ")

  def plotDependance_kp_number(self, number_group_sim, domain_name):
    ## check for valid input paramaters
    if (
        not("Re".lower() in domain_name.lower()) and
        not("Rm".lower() in domain_name.lower()) and
        not("Pm".lower() in domain_name.lower())):
      raise Exception(f"Error: '{domain_name}' is an invalid input")
    ## plot data
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for sim_index in range(len(self.Pm_group_sim)):
      plotScale(
        ax       = ax,
        x_median = number_group_sim[sim_index],
        y_median = self.k_p_mag_stats_group_sim[sim_index][0],
        y_1sig   = self.k_p_mag_stats_group_sim[sim_index][1],
        color    = self.color_group_sim[sim_index],
        marker   = self.marker_group_sim[sim_index]
      )
    ## plot reference lines
    if "Pm".lower() in domain_name.lower():
      x = np.linspace(10**(-1), 10**(5), 10**4)
      PlotFuncs.plotData_noAutoAxisScale(ax, x, 3*x**(1/4), ls=":")
      PlotFuncs.plotData_noAutoAxisScale(ax, x, 3*x**(1/6), ls="-.")
      PlotFuncs.plotData_noAutoAxisScale(ax, x, 3*x**(1/8), ls="--")
      PlotFuncs.addLegend_fromArtists(
        ax,
        list_artists       = [ ":", "-.", "--" ],
        list_legend_labels = [ "Pm$^{1/4}$", "Pm$^{1/6}$", "Pm$^{1/8}$" ],
        list_marker_colors = [ "k" ],
        label_color        = "black",
        loc                = "lower right",
        bbox               = (1.0, 0.0),
        lw                 = 1
      )
    else: ax.axhline(y=5.0, ls=":", c="black")
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
    # if self.bool_supersonic is not None: ax.set_ylim([ 1, 30 ])
    if   "Re".lower() in domain_name.lower(): ax.set_xlabel(r"Re", fontsize=20)
    elif "Rm".lower() in domain_name.lower(): ax.set_xlabel(r"Rm", fontsize=20)
    elif "Pm".lower() in domain_name.lower(): ax.set_xlabel(r"Pm", fontsize=20)
    ax.set_ylabel(r"$k_{{\rm p}, \mathbf{B}}$", fontsize=20)
    ## save plot
    fig_name = f"fig_dependance_{self.plot_name}_kp_{domain_name}.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")
    print(" ")

  def plotDependance_krho_knu(self):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for sim_index in range(len(self.Pm_group_sim)):
      plotScale(
        ax       = ax,
        x_median = self.k_nu_vel_lgt_stats_group_sim[sim_index][0],
        x_1sig   = self.k_nu_vel_lgt_stats_group_sim[sim_index][1],
        y_median = self.k_p_rho_stats_group_sim[sim_index][0],
        y_1sig   = self.k_p_rho_stats_group_sim[sim_index][1],
        color    = self.color_group_sim[sim_index],
        marker   = self.marker_group_sim[sim_index]
      )
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
    # if self.bool_supersonic is not None: ax.set_xlim([ 0.9, 220 ])
    # if self.bool_supersonic is not None: ax.set_ylim([ 0.9, 5 ])
    ax.set_xlabel(r"$k_{\nu, {\rm vel}}$", fontsize=20)
    ax.set_ylabel(r"$k_{{\rm p}, \rho}$", fontsize=20)
    ## save plot
    fig_name = f"fig_dependance_{self.plot_name}_krho_knu.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")
    print(" ")

  def plotDependance_kp_Mach(self):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for sim_index in range(len(self.Pm_group_sim)):
      plotScale(
        ax       = ax,
        x_median = self.Mach_group_sim[sim_index],
        y_median = self.k_p_mag_stats_group_sim[sim_index][0],
        y_1sig   = self.k_p_mag_stats_group_sim[sim_index][1],
        color    = self.color_group_sim[sim_index],
        marker   = self.marker_group_sim[sim_index]
      )
    ## adjust axis
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathcal{M}$", fontsize=20)
    ax.set_ylabel(r"$k_{\rm p}$", fontsize=20)
    ## save plot
    fig_name = f"fig_dependance_{self.plot_name}_kp_Mach.png"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{fig_name}")
    print(" ")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  plot_obj = PlotSimScales(f"{PATH_SCRATCH}/vis_folder/")
  plot_obj.plotRoutines()


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
PATH_SCRATCH = "/scratch/ek9/nk7952/"
# PATH_SCRATCH = "/scratch/jh2/nk7952/"

# ## PLASMA PARAMETER SET
# LIST_MACH_REGIMES = [ "Mach5" ]
# LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm3000" ]
# LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]

## MACH NUMBER SET
LIST_SUITE_FOLDERS = [ "Rm3000" ]
LIST_MACH_REGIMES = [ "Mach0.3" ] # , "Mach1", "Mach5", "Mach10"
LIST_SIM_FOLDERS   = [ "Pm1", "Pm5", "Pm10", "Pm125" ]

## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
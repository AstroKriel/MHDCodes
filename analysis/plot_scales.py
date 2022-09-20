#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from TheUsefulModule import WWObjs, WWFnF
from TheFittingModule import FitMHDScales
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## LOAD + PLOT CONVERGED SCALES
## ###############################################################
class PlotSimScales():
  def __init__(self, filepath_plots):
    self.filepath_plots = filepath_plots
    print("Saving figures in:", self.filepath_plots)
    print(" ")
    ## INITIALISE CONTAINERS FOR DATA
    ## ------------------------------
    ## simulation parameters
    self.Re_group             = []
    self.Rm_group             = []
    self.Pm_group             = []
    self.color_group          = []
    self.marker_group         = []
    ## predicted scales
    self.k_nu_relation_group  = []
    self.k_eta_relation_group = []
    ## measured (converged) scales
    self.k_nu_group           = []
    self.k_eta_group          = []
    self.k_p_group            = []
    self.k_max_group          = []
    self.alpha_kin_group      = []
    self.alpha_mag_1_group    = []
    self.alpha_mag_2_group    = []

    ## LOOK AT EACH SIMULATION GROUPED BY RESOLUTION
    ## ---------------------------------------------
    ## loop over the simulation suites
    for suite_folder in [
        "Re10", "Re500", "Rm3000"
      ]: # "Re10", "Re500", "Rm3000", "keta"

      ## CHECK THE SUITE'S FIGURE FOLDER EXISTS
      ## --------------------------------------
      filepath_suite_output = WWFnF.createFilepath([ BASEPATH, suite_folder, SONIC_REGIME ])
      if not os.path.exists(filepath_suite_output):
        print("{} does not exist.".format(filepath_suite_output))
        continue
      str_message = "Loading datasets from suite: {}".format(suite_folder)
      print(str_message)
      print("=" * len(str_message))

      ## LOOP OVER SIMULATIONS
      ## ---------------------
      ## loop over the simulation folders
      for sim_folder in [
          "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"
        ]: # "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"

        ## check that the fitted spectra data exists for the Nres=288 simulation setup
        filepath_sim = WWFnF.createFilepath([ BASEPATH, suite_folder, "288", SONIC_REGIME, sim_folder ])
        dataset_name = f"{suite_folder}_{sim_folder}_dataset.json"
        filepath_dataset = f"{filepath_sim}/{dataset_name}"
        if not os.path.isfile(filepath_dataset): continue

        ## load scales
        print(f"\t> Loading '{sim_folder}' dataset.")
        if suite_folder == "Re10": self.marker_group.append("s")
        elif suite_folder == "Re500": self.marker_group.append("D")
        elif suite_folder == "Rm3000": self.marker_group.append("o")
        self.getParams(filepath_sim, dataset_name)

      ## create empty space
      print(" ")

  def getParams(self, filepath_data, dataset_name):
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
    ## extract measured scales
    dict_params_kin  = dict["dict_params_kin"]
    self.alpha_kin_group.append(dict_params_kin["alpha_kin"])
    self.k_nu_group.append(dict_params_kin["k_nu"])
    dict_params_mag  = dict["dict_params_mag"]
    self.alpha_mag_1_group.append(dict_params_mag["alpha_mag_1"])
    self.alpha_mag_2_group.append(dict_params_mag["alpha_mag_2"])
    self.k_eta_group.append(dict_params_mag["k_eta"])
    self.k_p_group.append(dict_params_mag["k_p"])
    self.k_max_group.append(dict_params_mag["k_max"])
    ## calculate predicted dissipatin scales
    k_nu_relation  = 2 * (Re)**(2/3) # super-sonic
    k_eta_relation = k_nu_relation * (Pm)**(1/2)
    self.k_nu_relation_group.append(k_nu_relation)
    self.k_eta_relation_group.append(k_eta_relation)

  def plotScaleRelations(self):
    ## initialise figure
    fig, axs = plt.subplots(1, 2, figsize=(7*2/1.1, 4/1.1))
    fig.subplots_adjust(wspace=0.225)
    ## plot scales
    for sim_index in range(len(self.Pm_group)):
      ## k_nu
      axs[0].plot(
        self.k_nu_relation_group[sim_index],
        self.k_nu_group[sim_index],
        color  = self.color_group[sim_index],
        marker = self.marker_group[sim_index],
        markersize=8, markeredgecolor="black", linestyle="", zorder=10
      )
      ## k_eta
      axs[1].plot(
        self.k_eta_relation_group[sim_index],
        self.k_eta_group[sim_index],
        color  = self.color_group[sim_index],
        marker = self.marker_group[sim_index],
        markersize=8, markeredgecolor="black", linestyle="", zorder=10
      )
    ## plot reference lines
    x = np.linspace(10, 1000, 100)
    axs[0].plot(x, x/10, "k:")
    axs[1].plot(x, x/50, "k:")
    ## label axis
    axs[0].set_xlabel(r"$k_{\nu, \mathrm{theory}} = \mathrm{Re}^{2/3}$", fontsize=20)
    axs[1].set_xlabel(r"$k_{\eta, \mathrm{theory}} = k_{\nu, \mathrm{theory}} \; \mathrm{Pm}^{1/2}$", fontsize=20)
    axs[0].set_ylabel(r"$k_\nu$", fontsize=20)
    axs[1].set_ylabel(r"$k_\eta$", fontsize=20)
    ## adjust axis
    axs[0].set_xscale("log")
    axs[1].set_xscale("log")
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    ## save plot
    fig_name = "fig_scale_relation.png"
    fig_filepath = WWFnF.createFilepath([ self.filepath_plots, fig_name ])
    plt.savefig(fig_filepath)
    print("Figure saved:", fig_name)

  def plotScaleDependance(self):
    ## initialise figure
    fig, axs = plt.subplots(1, 2, figsize=(14/1.1, 4/1.1))
    fig.subplots_adjust(wspace=0.225)
    ## plot scales
    for sim_index in range(len(self.Pm_group)):
      ## k_p vs k_nu
      axs[0].plot(
        self.k_nu_group[sim_index],
        self.k_p_group[sim_index],
        color  = self.color_group[sim_index],
        marker = self.marker_group[sim_index],
        markersize=8, markeredgecolor="black", linestyle="", zorder=10
      )
      ## k_p vs k_eta
      axs[1].plot(
        self.k_eta_group[sim_index],
        self.k_p_group[sim_index],
        color  = self.color_group[sim_index],
        marker = self.marker_group[sim_index],
        markersize=8, markeredgecolor="black", linestyle="", zorder=10
      )
    ## plot reference lines
    x = np.linspace(1, 100, 100)
    axs[0].plot(x, x, "k:")
    axs[1].plot(x, x, "k:")
    ## label axis
    axs[0].set_xlabel(r"$k_\nu$", fontsize=20)
    axs[1].set_xlabel(r"$k_\eta$", fontsize=20)
    axs[0].set_ylabel(r"$k_{\rm p}$", fontsize=20)
    axs[1].set_ylabel(r"$k_{\rm p}$", fontsize=20)
    ## adjust axis
    axs[0].set_xscale("log")
    axs[1].set_xscale("log")
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    ## save plot
    fig_name = "fig_scale_dependance.png"
    fig_filepath = WWFnF.createFilepath([ self.filepath_plots, fig_name ])
    plt.savefig(fig_filepath)
    print("Figure saved:", fig_name)

  def plotPmDependance(self):
    ## initialise figure
    fig, ax = plt.subplots(1, 1, figsize=(7/1.1, 4/1.1))
    ## plot scales
    for sim_index in range(len(self.Pm_group)):
      ax.plot(
        self.Pm_group[sim_index],
        self.k_max_group[sim_index],
        color  = self.color_group[sim_index],
        marker = self.marker_group[sim_index],
        markersize=8, markeredgecolor="black", linestyle="", zorder=10
      )
    ## plot reference lines
    # x = np.linspace(10, 1000, 100)
    # ax.plot(x, x/10, "k:")
    ## label axis
    ax.set_xlabel(r"$\mathrm{Pm}$", fontsize=20)
    ax.set_ylabel(r"$k_{\rm p}$",   fontsize=20)
    ## adjust axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    ## save plot
    fig_name = "fig_Pm_dependance.png"
    fig_filepath = WWFnF.createFilepath([ self.filepath_plots, fig_name ])
    plt.savefig(fig_filepath)
    print("Figure saved:", fig_name)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
BASEPATH           = "/scratch/ek9/nk7952/"
SONIC_REGIME       = "super_sonic"

def main():
  plot_obj = PlotSimScales(BASEPATH)
  # plot_obj.plotScaleRelations()
  # plot_obj.plotScaleDependance()
  plot_obj.plotPmDependance()


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
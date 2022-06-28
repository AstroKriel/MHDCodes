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
class SimScales():
  def __init__(self, filepath_plots):
    self.filepath_plots = filepath_plots
    ## ##############################
    ## INITIALISE CONTAINERS FOR DATA
    ## ##############################
    ## simulation parameters
    self.Re_group                  = []
    self.Rm_group                  = []
    self.Pm_group                  = []
    self.color_group               = []
    self.marker_group              = []
    ## predicted scales
    self.k_nu_relation_group       = []
    self.k_eta_relation_group      = []
    ## measured (converged) scales
    self.k_nu_converged_group      = []
    self.k_eta_converged_group     = []
    self.k_p_converged_group       = []
    self.k_nu_std_group            = []
    self.k_eta_std_group           = []
    self.k_p_std_group             = []
    self.alpha_kin_converged_group = []
    self.alpha_mag_converged_group = []

    ## #############################################
    ## LOOK AT EACH SIMULATION GROUPED BY RESOLUTION
    ## #############################################
    ## loop over the simulation suites
    for suite_folder in [
        "Re10", "Re500", "Rm3000"
      ]: # "Re10", "Re500", "Rm3000", "keta"

      ## ######################################
      ## CHECK THE SUITE'S FIGURE FOLDER EXISTS
      ## ######################################
      filepath_suite_output = WWFnF.createFilepath([ 
        BASEPATH, suite_folder, SONIC_REGIME
      ])
      if not os.path.exists(filepath_suite_output):
        print("{} does not exist.".format(filepath_suite_output))
        continue
      str_message = "Loading data from suite: {}".format(suite_folder)
      print(str_message)
      print("=" * len(str_message))

      ## #####################
      ## LOOP OVER SIMULATIONS
      ## #####################
      ## loop over the simulation folders
      for sim_folder in [
          "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"
        ]: # "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"

        ## check that the fitted spectra data exists for the Nres=288 simulation setup
        filepath_spect = WWFnF.createFilepath([ BASEPATH, suite_folder, "288", SONIC_REGIME, sim_folder, "spect" ])
        if not os.path.isfile(f"{filepath_spect}/{FILENAME_SPECTRA}"): continue

        ## load scales
        print(f"\t> Loading '{sim_folder}' simulation data.")
        if suite_folder == "Re10": self.marker_group.append("s")
        elif suite_folder == "Re500": self.marker_group.append("D")
        elif suite_folder == "Rm3000": self.marker_group.append("o")
        self.getSimParams(filepath_spect)
        self.getConvergedScales(filepath_suite_output, f"{sim_folder}_{FILENAME_CONVERGED}")

      ## create empty space
      print(" ")

  def getSimParams(self, filepath_sim):
    ## load spectra-fit data as a dictionary
    dict = WWObjs.loadJson2Dict(
      filepath          = filepath_sim,
      filename          = FILENAME_SPECTRA,
      bool_hide_updates = True
    )
    ## store dictionary data in spectra-fit object
    obj = FitMHDScales.SpectraFit(**dict)
    ## load plasma Reynolds numbers
    Re = int(obj.Re)
    Rm = int(obj.Rm)
    Pm = int(obj.Pm)
    self.Re_group.append(Re)
    self.Rm_group.append(Rm)
    self.Pm_group.append(Pm)
    self.color_group.append( "cornflowerblue" if Re < 100 else "orangered" )
    ## calculate predicted dissipatin scales
    k_nu_relation  = 2 * (Re)**(2/3) # super-sonic
    k_eta_relation = k_nu_relation * (Pm)**(1/2)
    self.k_nu_relation_group.append(k_nu_relation)
    self.k_eta_relation_group.append(k_eta_relation)

  def getConvergedScales(
      self,
      filepath_suite_output,
      filename
    ):
    ## load converged spectra-fit data as a dictionary
    dict = WWObjs.loadJson2Dict(
      filepath          = filepath_suite_output,
      filename          = filename,
      bool_hide_updates = True
    )
    ## store dictionary data in spectra-fit object
    obj = FitMHDScales.SpectraConvergedScales(**dict)    
    self.k_nu_converged_group.append(obj.k_nu_converged)
    self.k_eta_converged_group.append(obj.k_eta_converged)
    self.k_p_converged_group.append(obj.k_p_converged)
    self.k_nu_std_group.append(obj.k_nu_std)
    self.k_eta_std_group.append(obj.k_eta_std)
    self.k_p_std_group.append(obj.k_p_std)

  def plotScaleRelations(self):
    ## initialise figure
    fig, axs = plt.subplots(1, 2, figsize=(7*2/1.1, 4/1.1))
    fig.subplots_adjust(wspace=0.225)
    ## plot scale distributions
    for sim_index in range(len(self.Pm_group)):
      ## k_nu
      axs[0].errorbar(
        self.k_nu_relation_group[sim_index],
        self.k_nu_converged_group[sim_index],
        yerr   = self.k_nu_std_group[sim_index],
        color  = self.color_group[sim_index],
        fmt    = self.marker_group[sim_index],
        markersize=8, markeredgecolor="black", capsize=7.5, elinewidth=2, linestyle="None", zorder=10
      )
      ## k_eta
      axs[1].errorbar(
        self.k_eta_relation_group[sim_index],
        self.k_eta_converged_group[sim_index],
        yerr   = self.k_eta_std_group[sim_index],
        color  = self.color_group[sim_index],
        fmt    = self.marker_group[sim_index],
        markersize=8, markeredgecolor="black", capsize=7.5, elinewidth=2, linestyle="None", zorder=10
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
    # axs[1].set_xlim([ 10, 1000 ])
    # axs[1].set_xlim([ 10, 1000 ])
    # axs[0].set_ylim([ 10**0, 10**2 ])
    # axs[1].set_ylim([ 10**0, 10**2 ])
    ## save plot
    fig_name = "fig_scale_relation.pdf"
    fig_filepath = WWFnF.createFilepath([ self.filepath_plots, fig_name ])
    plt.savefig(fig_filepath)
    print("Figure saved:", fig_name)

  def plotScaleDependance(self):
    ## initialise figure
    fig, axs = plt.subplots(1, 2, figsize=(14/1.1, 4/1.1))
    fig.subplots_adjust(wspace=0.225)
    ## plot scale distributions
    for sim_index in range(len(self.Pm_group)):
      ## k_p vs k_nu
      axs[0].errorbar(
        self.k_nu_converged_group[sim_index],
        self.k_p_converged_group[sim_index],
        xerr   = self.k_nu_std_group[sim_index],
        yerr   = self.k_p_std_group[sim_index],
        color  = self.color_group[sim_index],
        fmt    = self.marker_group[sim_index],
        markersize=8, markeredgecolor="black", capsize=7.5, elinewidth=2, linestyle="None", zorder=10
      )
      ## k_p vs k_eta
      axs[1].errorbar(
        self.k_eta_converged_group[sim_index],
        self.k_p_converged_group[sim_index],
        xerr   = self.k_eta_std_group[sim_index],
        yerr   = self.k_p_std_group[sim_index],
        color  = self.color_group[sim_index],
        fmt    = self.marker_group[sim_index],
        markersize=8, markeredgecolor="black", capsize=7.5, elinewidth=2, linestyle="None", zorder=10
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
    # axs[0].set_xlim([ 1, 40 ])
    # axs[0].set_ylim([ 1, 40 ])
    # axs[1].set_xlim([ 1, 20 ])
    # axs[1].set_ylim([ 1, 20 ])
    ## save plot
    fig_name = "fig_scale_dependance.pdf"
    fig_filepath = WWFnF.createFilepath([ self.filepath_plots, fig_name ])
    plt.savefig(fig_filepath)
    print("Figure saved:", fig_name)

  def plotAlphaExponent(self, str_var):
    ## initialise figure
    fig_scaling = 1.45
    _, ax = plt.subplots(figsize=(8/fig_scaling, 5/fig_scaling))
    ## plot points
    for sim_index in range(len(self.Pm_group)):
        
      PlotFuncs.plotErrorBar(
        ax,
        data_x = self.Re_group[sim_index],
        data_y = (
          self.alpha_kin_converged_group[sim_index] if str_var == "kin" else
          self.alpha_mag_converged_group[sim_index]
        ),
        color  = self.color_group[sim_index],
        marker = self.marker_group[sim_index],
        ms = 9
      )
    ## label axis
    ax.set_xlabel(r"Re", fontsize=22)
    ax.set_ylabel(r"$\alpha_{\mathrm{"+str_var+"}}$", fontsize=22)
    ## adjust axis
    ax.set_xscale("log")
    ## save plot
    fig_name = f"fig_exponent_{str_var}.pdf"
    fig_filepath = WWFnF.createFilepath([ self.filepath_plots, fig_name ])
    plt.savefig(fig_filepath)
    print("Figure saved:", fig_name)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
BASEPATH           = "/scratch/ek9/nk7952/"
SONIC_REGIME       = "super_sonic"
FILENAME_SPECTRA   = "spectra_fits.json"
FILENAME_CONVERGED = "spectra_converged.json"
FILENAME_TAG       = ""

def main():
  sim_data_obj = SimScales(BASEPATH)
  sim_data_obj.plotScaleRelations()
  sim_data_obj.plotScaleDependance()


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
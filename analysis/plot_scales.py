#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

## load old user defined modules
from OldModules import the_fitting_library
sys.modules["the_fitting_library"] = the_fitting_library

## load new user modules
from TheUsefulModule import WWObjs, WWLists, WWFnF
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
##################################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend
# plt.style.use('dark_background')


## ###############################################################
## FUNCTIONS
## ###############################################################
def funcLoadData_sim(
    ## input: simulation directory and name
    filepath_sim,
    ## output: simulation parameters
    list_Re, list_Rm, list_Pm,
    ## output: predicted scales
    list_relation_k_nu, list_relation_k_eta,
    ## output: measured sclaes
    list_k_nu_group, list_k_eta_group, list_k_max_group,
    list_alpha_kin_group, list_alpha_mag_group
  ):
  ## #########################
  ## GET SIMULATION PARAMETERS
  ## ########
  spectra_obj = WWObjs.loadPickleObject(
    filepath_sim,
    SPECTRA_NAME,
    bool_hide_updates = True
  )
  ## check that a time range has been defined to collect statistics about
  sim_times = WWLists.getCommonElements(spectra_obj.kin_sim_times, spectra_obj.mag_sim_times)
  ## find indices of magnetic fit time range
  kin_index_start = WWLists.getIndexClosestValue(sim_times, 2)
  kin_index_end   = WWLists.getIndexClosestValue(sim_times, 10)
  mag_index_start = WWLists.getIndexClosestValue(sim_times, 2)
  mag_index_end   = WWLists.getIndexClosestValue(sim_times, 10)
  ## load parameters
  Re = int(spectra_obj.Re)
  Rm = int(spectra_obj.Rm)
  Pm = int(Rm / Re)
  ## calculate predicted scales
  relation_k_nu  = 2 * (Re)**(2/3) # super-sonic
  relation_k_eta = relation_k_nu * (Pm)**(1/2)
  ## load kazantsev exponent
  list_alpha_kin = [
    -abs(sub_list[1])
    for sub_list in spectra_obj.kin_list_fit_params_group_t[
      kin_index_start : kin_index_end
    ]
  ]
  list_alpha_mag = [
    sub_list[1]
    for sub_list in spectra_obj.mag_list_fit_params_group_t[
      mag_index_start : mag_index_end
    ]
  ]
  ## load measured scales
  list_k_nu  = spectra_obj.k_nu_group_t[kin_index_start : kin_index_end]
  list_k_eta = spectra_obj.k_eta_group_t[mag_index_start : mag_index_end]
  list_k_max = spectra_obj.k_max_group_t[mag_index_start : mag_index_end]
  ## save data
  list_Re.append(Re)
  list_Rm.append(Rm)
  list_Pm.append(Pm)
  list_relation_k_nu.append(relation_k_nu)
  list_relation_k_eta.append(relation_k_eta)
  list_alpha_kin_group.append(list_alpha_kin)
  list_alpha_mag_group.append(list_alpha_mag)
  list_k_nu_group.append(list_k_nu)
  list_k_eta_group.append(list_k_eta)
  list_k_max_group.append(list_k_max)

def funcLoadData(
    ## where simulation suites are
    filepath_data,
    ## output: simulation parameters
    list_Re, list_Rm, list_Pm,
    ## output: predicted scales
    list_relation_k_nu, list_relation_k_eta,
    ## output: measured scales
    list_k_nu_group, list_k_eta_group, list_k_max_group,
    list_alpha_kin_group, list_alpha_mag_group,
    ## output: simulation markers
    list_markers
  ):
  print("Loading simulation data...")
  ## simulation folders
  sim_folders_Re10   = [ "Pm25", "Pm50" ]
  sim_folders_Re500  = [ "Pm1", "Pm2", "Pm4" ]
  sim_folders_Rm3000 = [ "Pm1", "Pm2", "Pm5", "Pm10", "Pm25", "Pm50" ]
  ## Re = 10
  for sim_label in sim_folders_Re10:
    ## store simulation marker
    list_markers.append("s")
    ## load simulation data
    funcLoadData_sim(
      ## input: simulation directory and name
      WWFnF.createFilepath([ filepath_data, "Re10", "288", sim_label ]),
      ## output: simulation parameters
      list_Re = list_Re,
      list_Rm = list_Rm,
      list_Pm = list_Pm,
      ## output: predicted scales
      list_relation_k_nu  = list_relation_k_nu,
      list_relation_k_eta = list_relation_k_eta,
      ## output: measured (converged) sclaes
      list_k_nu_group  = list_k_nu_group,
      list_k_eta_group = list_k_eta_group,
      list_k_max_group = list_k_max_group,
      ## output: fitted power-law exponents
      list_alpha_kin_group = list_alpha_kin_group,
      list_alpha_mag_group = list_alpha_mag_group
    )
  ## Re = 500
  for sim_label in sim_folders_Re500:
    ## store simulation marker
    list_markers.append("D")
    ## load simulation data
    funcLoadData_sim(
      ## input: simulation directory and name
      WWFnF.createFilepath([ filepath_data, "Re500", "288", sim_label ]),
      ## output: simulation parameters
      list_Re = list_Re,
      list_Rm = list_Rm,
      list_Pm = list_Pm,
      ## output: predicted scales
      list_relation_k_nu  = list_relation_k_nu,
      list_relation_k_eta = list_relation_k_eta,
      ## output: measured (converged) sclaes
      list_k_nu_group  = list_k_nu_group,
      list_k_eta_group = list_k_eta_group,
      list_k_max_group = list_k_max_group,
      ## output: fitted power-law exponents
      list_alpha_kin_group = list_alpha_kin_group,
      list_alpha_mag_group = list_alpha_mag_group
    )
  ## Rm = 3000
  for sim_label in sim_folders_Rm3000:
    ## store simulation marker
    list_markers.append("o")
    ## load simulation data
    funcLoadData_sim(
      ## input: simulation directory and name
      WWFnF.createFilepath([ filepath_data, "Rm3000", "288", sim_label ]),
      ## output: simulation parameters
      list_Re = list_Re,
      list_Rm = list_Rm,
      list_Pm = list_Pm,
      ## output: predicted scales
      list_relation_k_nu  = list_relation_k_nu,
      list_relation_k_eta = list_relation_k_eta,
      ## output: measured (converged) sclaes
      list_k_nu_group  = list_k_nu_group,
      list_k_eta_group = list_k_eta_group,
      list_k_max_group = list_k_max_group,
      ## output: fitted power-law exponents
      list_alpha_kin_group = list_alpha_kin_group,
      list_alpha_mag_group = list_alpha_mag_group
    )
  print(" ")

def funcPlotScaleRelations(
    ## where to save figure
    filepath_plot,
    ## data point colors
    list_colors, list_markers,
    ## simulation parameters
    list_Re, list_Rm, list_Pm,
    ## predicted scales
    list_relation_k_nu, list_relation_k_eta,
    ## measured scales
    list_k_nu_group, list_k_eta_group
  ):
  ## #################
  ## INITIALISE FIGURE
  ## ##########
  fig, axs = plt.subplots(1, 2, figsize=(7*2/1.1, 4/1.1))
  fig.subplots_adjust(wspace=0.225)
  ## plot scale distributions
  for sim_index in range(len(list_relation_k_nu)):
    PlotFuncs.plotErrorBar(
      axs[0],
      data_x = list_relation_k_nu[sim_index],
      data_y = list_k_nu_group[sim_index],
      color  = "black",
      marker = list_markers[sim_index],
      ms = 9
    )
    PlotFuncs.plotErrorBar(
      axs[1],
      data_x = list_relation_k_eta[sim_index],
      data_y = list_k_eta_group[sim_index],
      color  = "black",
      marker = list_markers[sim_index],
      ms = 9
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
  axs[1].set_xlim([ 10, 1000 ])
  axs[1].set_xlim([ 10, 1000 ])
  axs[0].set_ylim([ 10**0, 10**2 ])
  axs[1].set_ylim([ 10**0, 10**2 ])
  ## save plot
  fig_name = "fig_scale_relation.pdf"
  fig_filepath = WWFnF.createFilepath([filepath_plot, fig_name])
  plt.savefig(fig_filepath)
  print("\t> Figure saved: " + fig_name)

def funcPlotScaleDependance(
    ## where to save figure
    filepath_plot,
    ## data point colors
    list_colors, list_markers,
    ## simulation parameters
    list_Re, list_Rm, list_Pm,
    ## measured scales
    list_k_nu_group, list_k_eta_group, list_k_max_group
  ):
  ## #################
  ## INITIALISE FIGURE
  ## ##########
  fig, axs = plt.subplots(1, 2, figsize=(14/1.1, 4/1.1))
  fig.subplots_adjust(wspace=0.225)
  ## plot scale distributions
  for sim_index in range(len(list_k_nu_group)):
    ## plot dependance on k_nu
    PlotFuncs.plotErrorBar(
      axs[0],
      data_x = list_k_nu_group[sim_index],
      data_y = list_k_max_group[sim_index],
      color  = "black",
      marker = list_markers[sim_index],
      ms = 9
    )
    ## plot dependance on k_eta
    PlotFuncs.plotErrorBar(
      axs[1],
      data_x = list_k_eta_group[sim_index],
      data_y = list_k_max_group[sim_index],
      color  = "black",
      marker = list_markers[sim_index],
      ms = 9
    )
  ## plot reference lines
  x = np.linspace(1, 100, 100)
  axs[0].plot(x, x, "k:")
  axs[1].plot(x, x, "k:")
  ## label axis
  axs[0].set_xlabel(r"$k_\nu$", fontsize=20)
  axs[1].set_xlabel(r"$k_\eta$", fontsize=20)
  axs[0].set_ylabel(r"$k_p$", fontsize=20)
  axs[1].set_ylabel(r"$k_p$", fontsize=20)
  ## adjust axis
  axs[0].set_xscale("log")
  axs[1].set_xscale("log")
  axs[0].set_yscale("log")
  axs[1].set_yscale("log")
  axs[0].set_xlim([ 1, 40 ])
  axs[0].set_ylim([ 1, 40 ])
  axs[1].set_xlim([ 1, 20 ])
  axs[1].set_ylim([ 1, 20 ])
  ## save plot
  fig_name = "fig_scale_dependance.pdf"
  fig_filepath = WWFnF.createFilepath([filepath_plot, fig_name])
  plt.savefig(fig_filepath)
  print("\t> Figure saved: " + fig_name)

def funcPlotExponent(
    ## where to save figure
    filepath_plot,
    ## data point colors
    list_colors, list_markers,
    ## simulation parameters
    list_Re, list_Rm, list_Pm,
    ## measured scales
    list_alpha_group,
    str_var
  ):
  ## #################
  ## INITIALISE FIGURE
  ## ##########
  factor = 1.45
  _, ax = plt.subplots(figsize=(8/factor, 5/factor))
  ## plot points
  for sim_index in range(len(list_alpha_group)):
    ## plot dependance on exponent on Re
    PlotFuncs.plotErrorBar(
      ax,
      data_x = list_Re[sim_index],
      data_y = list_alpha_group[sim_index],
      color  = "black",
      marker = list_markers[sim_index],
      ms = 9
    )
  ## label axis
  ax.set_xlabel(r"Re", fontsize=22)
  ax.set_ylabel(r"$\alpha_{\mathrm{"+str_var+"}}$", fontsize=22)
  ## adjust axis
  ax.set_xscale("log")
  ## save plot
  fig_name = "fig_exponent_{}.pdf".format(str_var)
  fig_filepath = WWFnF.createFilepath([filepath_plot, fig_name])
  plt.savefig(fig_filepath)
  print("\t> Figure saved: " + fig_name)


SPECTRA_NAME = "spectra_obj_full.pkl"
SONIC_REGIME = "super_sonic"
## ###############################################################
## DEFINE MAIN PROGRAM
## ###############################################################
def main():
  ## ####################
  ## INITIALISE VARIABLES
  ## ####################
  filepath_base = "/Users/dukekriel/Documents/Studies/TurbulentDynamo/"
  filepath_data = filepath_base + "data/" + SONIC_REGIME
  filepath_plot = filepath_base + "figures/" + SONIC_REGIME

  ## ####################
  ## LOAD SIMULATION DATA
  ## ####################
  ## simulation parameters
  list_Re = []
  list_Rm = []
  list_Pm = []
  ## predicted scales
  list_relation_k_nu   = []
  list_relation_k_eta  = []
  ## measured (convereged) scales
  list_k_nu_group  = []
  list_k_eta_group = []
  list_k_max_group = []
  ## kazantsev exponent
  list_alpha_kin_group = []
  list_alpha_mag_group = []
  ## list of simlation markers
  list_markers = []
  ## load data
  funcLoadData(
    filepath_data,
    list_Re, list_Rm, list_Pm,
    list_relation_k_nu, list_relation_k_eta,
    list_k_nu_group, list_k_eta_group, list_k_max_group,
    list_alpha_kin_group, list_alpha_mag_group,
    list_markers
  )
  ## define simulation points color
  list_colors = [
    "cornflowerblue" if Re < 100
    else "orangered"
    for Re in list_Re
  ]

  ## ####################
  ## PLOT SIMULATION DATA
  ## ####################
  print("Saving figures in: " + filepath_plot)

  ## plot measured vs predicted scales
  funcPlotScaleRelations(
    filepath_plot,
    list_colors, list_markers,
    list_Re, list_Rm, list_Pm,
    list_relation_k_nu, list_relation_k_eta,
    list_k_nu_group, list_k_eta_group,
  )

  ## plot peak scale dependance on dissipation scales
  funcPlotScaleDependance(
    filepath_plot,
    list_colors, list_markers,
    list_Re, list_Rm, list_Pm,
    list_k_nu_group, list_k_eta_group, list_k_max_group,
  )

  ## plot dependance of powerlaw exponent on Re
  funcPlotExponent(
    filepath_plot,
    list_colors, list_markers,
    list_Re, list_Rm, list_Pm,
    list_alpha_kin_group,
    str_var = "vel"
  )

  ## plot dependance of powerlaw exponent on Re
  funcPlotExponent(
    filepath_plot,
    list_colors, list_markers,
    list_Re, list_Rm, list_Pm,
    list_alpha_mag_group,
    str_var = "mag"
  )


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
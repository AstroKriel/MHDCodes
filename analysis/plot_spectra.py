#!/usr/bin/env python3

##################################################################
## MODULES
##################################################################
import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

## load old user defined modules
from OldModules.the_useful_library import *
from OldModules.the_loading_library import *
from OldModules.the_plotting_library import *


##################################################################
## PREPARE WORKSPACE
#################################################################
## work in a non-interactive environment
mpl.use("Agg")
plt.ioff()


##################################################################
## INPUT COMMAND LINE ARGUMENTS
##################################################################
args_input = argparse.ArgumentParser(description="A bunch of input arguments:") 
## ------------------- DEFINE OPTIONAL ARGUMENTS
optional_bool_args = {"required":False, "type":str2bool, "nargs":"?", "const":True}
args_opt = args_input.add_argument_group(description='Optional processing arguments:') # optional argument group
## program workflow parameters
args_opt.add_argument("-hide_updates", default=False, **optional_bool_args) # hide progress bar
args_opt.add_argument("-plot_spectra",      default=True,  **optional_bool_args) # plot spectra frames
args_opt.add_argument("-animate_spectra",   default=True,  **optional_bool_args) # animate spectra frames
## directory information
args_opt.add_argument("-vis_folder", type=str, default="vis_folder", required=False) # where figures are saved
args_opt.add_argument("-dat_folder", type=str, default="spect", required=False)      # subfolder where spectras are stored
## simulation information
args_opt.add_argument("-plots_per_eddy", type=float, default=10, required=False) # plot files per T_eddy
## ------------------- DEFINE REQUIRED ARGUMENTS
args_req = args_input.add_argument_group(description='Required processing arguments:') # required argument group
args_req.add_argument("-base_path",   type=str, required=True)
args_req.add_argument("-sim_folders", type=str, required=True, nargs="+")
args_req.add_argument("-sim_names",   type=str, required=True, nargs="+")
## ---------------------------- OPEN ARGUMENTS
args = vars(args_input.parse_args())
## ---------------------------- SAVE PARAMETERS
## program workflow parameters
bool_hide_updates   = args["hide_updates"] # should the progress bar be displayed?
bool_plot_spectra    = args["plot_spectra"]      # should evolution of the spectra be plotted?
bool_animate_spectra = args["animate_spectra"]   # should spectra frames be animated?
## fitting & simulation parameters
plots_per_eddy = args["plots_per_eddy"] # number of plot files in eddy turnover time
## ---------------------------- DIRECTORY PARAMETERS
filepath_base = args["base_path"]   # home directory
folders_sims  = args["sim_folders"] # list of subfolders where each simulation's data is stored
folders_data  = args["dat_folder"]  # subfolder where data is stored in sim_folders
folder_vis    = args["vis_folder"]  # subfolder where animation and plots will be saved
sim_names     = args["sim_names"]


##################################################################
## PREPARE WORKSPACE
#################################################################
if not(bool_hide_updates):
  os.system("clear") # clear terminal window
  plt.close("all")   # close all pre-existing plots


##################################################################
## INITIALISING VARIABLES
##################################################################
## check there are enough simulation names defined
if len(sim_names) < len(folders_sims):
  raise Exception("You need to define a figure name for each simulation.")
## folders where spectra data is
filepaths_data = []
for sim_index in range(len(folders_sims)):
  filepaths_data.append( createFilePath([filepath_base, folders_sims[sim_index], folders_data]) )
## folder where visualisations will be saved
filepath_vis = createFilePath([filepath_base, folder_vis])
createFolder(filepath_vis)
## folder where spectra plots will be saved
filepath_frames = createFilePath([filepath_vis, "plotSpectra"])
createFolder(filepath_frames)
## print filepath information to the console
printInfo("Base filepath:", filepath_base)
printInfo("Figure folder:", filepath_vis)
for sim_index in range(len(filepaths_data)):
  printInfo("({:d}) sim directory:".format(sim_index), filepaths_data[sim_index], 23)
  printInfo("\t> Sim name:", sim_names[sim_index])
print(" ")


##################################################################
## LOAD & PLOT SPECTRA DATA
##################################################################
## loop over each simulation dataset
for filepath_data, sim_index in zip(filepaths_data, range(len(filepaths_data))):
  ## load spectra data
  print("Loading data from:", filepath_data)
  kin_k, kin_power, kin_sim_times = loadSpectra(
    filepath_data,
    str_spectra_type  = "vel",
    plots_per_eddy    = plots_per_eddy,
    bool_hide_updates = bool_hide_updates
  )
  mag_k, mag_power, mag_sim_times = loadSpectra(
    filepath_data,
    str_spectra_type  = "mag",
    plots_per_eddy    = plots_per_eddy,
    bool_hide_updates = bool_hide_updates
  )
  sim_times = getCommonElements(
    kin_sim_times,
    mag_sim_times
  )
  print(" ")
  ## initialise plot object
  plot_obj = PlotSpectra(
    kin_k, kin_power, mag_k, mag_power,
    sim_times, sim_names[sim_index],
    filepath_frames, filepath_vis
  )
  ## plot spectra data
  if bool_plot_spectra:
    print("Plotting spectra...")
    plot_obj.plotSpectra(bool_hide_updates)
  ## animate spectra
  if bool_animate_spectra:
    plot_obj.aniSpectra()
  print(" ")


## END OF PROGRAM
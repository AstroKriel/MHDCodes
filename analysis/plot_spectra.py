#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import matplotlib as mpl

## 'tmpfile' needs to be loaded before 'matplotlib'.
## This is so matplotlib stores its cache in a temporary directory.
## (Useful for plotting parallel)
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
import matplotlib.pyplot as plt

## load user defined modules
from TheUsefulModule import WWArgparse, P2Term, WWFnF, WWLists
from TheLoadingModule import LoadFlashData
from ThePlottingModule import PlotSpectra


## ##############################################################
## PREPARE WORKSPACE
## ##############################################################
os.system("clear") # clear terminal window
## work in a non-interactive environment
plt.ioff()
mpl.use("Agg")


## ##############################################################
## DEFINE MAIN PROGRAM
## ##############################################################
def main():
  ## ############################
  ## INPUT COMMAND LINE ARGUMENTS
  ## ############################
  parser = WWArgparse.MyParser(description="Plot kinetic and magnetic energy spectra.")
  ## ------------------- DEFINE OPTIONAL ARGUMENTS
  args_opt = parser.add_argument_group(description="Optional processing arguments:")
  args_opt.add_argument("-v", "--verbose", **WWArgparse.opt_bool_arg)
  args_opt.add_argument("-vis_folder",     **WWArgparse.opt_arg, type=str, default="vis_folder")
  args_opt.add_argument("-data_folder",    **WWArgparse.opt_arg, type=str, default="spect")
  ## ------------------- DEFINE REQUIRED ARGUMENTS
  args_req = parser.add_argument_group(description="Required processing arguments:")
  args_req.add_argument("-suite_path", type=str, required=True, help="type: %(type)s")
  args_req.add_argument("-sim_folder", type=str, required=True, help="type: %(type)s")
  ## ---------------------------- OPEN ARGUMENTS
  args = vars(parser.parse_args())
  ## ---------------------------- SAVE PARAMETERS
  bool_hide_updates = not(args["verbose"])
  filepath_suite    = args["suite_path"]
  folder_sim        = args["sim_folder"]
  folder_data       = args["data_folder"]
  folder_vis        = args["vis_folder"]

  ## #####################
  ## PREPARING DIRECTORIES
  ## #####################
  ## filepath to where spectra data is stored
  filepath_sim   = WWFnF.createFilepath([filepath_suite, folder_sim])
  filepath_spect = WWFnF.createFilepath([filepath_sim, folder_data])
  ## filepath to where visualisations will be saved
  sub_folder_vis      = "plotSpectraData"
  filepath_vis        = WWFnF.createFilepath([filepath_suite, folder_vis])
  filepath_vis_frames = WWFnF.createFilepath([filepath_vis, sub_folder_vis])
  
  ## ##############
  ## CREATE FOLDERS
  ## ##############
  WWFnF.createFolder(filepath_vis)
  WWFnF.createFolder(filepath_vis_frames)
  
  ## ######################################
  ## PRINT SIMULATION PARAMETERS TO CONSOLE
  ## ######################################
  P2Term.printInfo("Base directory:", filepath_suite)
  P2Term.printInfo("Data directory:", filepath_spect)
  P2Term.printInfo("Vis. directory:", filepath_vis)
  print(" ")

  ## ############################
  ## LOAD SIMULATION PLT PER EDDY
  ## ############################
  plots_per_eddy = LoadFlashData.getPlotsPerEddy(filepath_sim, bool_hide_updates=False)
  if plots_per_eddy is None:
    raise Exception("ERROR: # plt-files could not be read from 'Turb.log'.")
  
  ## ##########################
  ## LOAD AND PLOT SPECTRA DATA
  ## ##########################
  print("Loading kinetic energy spectra...")
  kin_k, kin_power, kin_sim_times = LoadFlashData.loadListSpectra(
    filepath_data     = filepath_spect,
    str_spectra_type  = "vel",
    plots_per_eddy    = plots_per_eddy,
    bool_hide_updates = bool_hide_updates
  )
  print("Loading magnetic energy spectra...")
  mag_k, mag_power, mag_sim_times = LoadFlashData.loadListSpectra(
    filepath_data     = filepath_spect,
    str_spectra_type  = "mag",
    plots_per_eddy    = plots_per_eddy,
    bool_hide_updates = bool_hide_updates
  )
  sim_times = WWLists.getCommonElements(
    kin_sim_times,
    mag_sim_times
  )
  ## initialise plot object
  plot_obj = PlotSpectra.PlotSpectra(
    kin_k              = kin_k,
    kin_power          = kin_power,
    mag_k              = mag_k,
    mag_power          = mag_power,
    sim_times          = sim_times,
    fig_name           = folder_sim,
    filepath_frames    = filepath_vis_frames,
    filepath_ani_movie = filepath_vis
  )
  ## plot spectra data
  print("Plotting energy spectra...")
  plot_obj.plotSpectra(bool_hide_updates)
  plot_obj.aniSpectra(bool_hide_updates)
  print(" ")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()


## END OF PROGRAM
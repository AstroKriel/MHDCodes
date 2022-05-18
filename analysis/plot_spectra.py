#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import matplotlib as mpl
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
  parser = WWArgparse.MyParser()
  ## ------------------- DEFINE OPTIONAL ARGUMENTS
  args_opt = parser.add_argument_group(description="Optional processing arguments:")
  args_opt.add_argument("-hide_updates", type=WWArgparse.str2bool, default=False, required=False, nargs="?", const=True)
  args_opt.add_argument("-vis_folder",   type=str, default="vis_folder", required=False)
  args_opt.add_argument("-spect_folder", type=str, default="spect",      required=False)
  ## ------------------- DEFINE REQUIRED ARGUMENTS
  args_req = parser.add_argument_group(description="Required processing arguments:")
  args_req.add_argument("-base_path",    type=str, required=True)
  args_req.add_argument("-sim_folder",   type=str, required=True)
  ## ---------------------------- OPEN ARGUMENTS
  args = vars(parser.parse_args())
  ## ---------------------------- SAVE PARAMETERS
  filepath_base     = args["base_path"]
  folder_sim        = args["sim_folder"]
  folder_spect      = args["spect_folder"]
  folder_vis        = args["vis_folder"]
  bool_hide_updates = args["hide_updates"]

  ## ###############################
  ## INITIALISING FILEPATH VARIABLES
  ## ###############################
  ## filepath to where spectra data is stored
  filepath_sim   = WWFnF.createFilepath([filepath_base, folder_sim])
  filepath_spect = WWFnF.createFilepath([filepath_sim, folder_spect])
  ## filepath to where visualisations will be saved
  filepath_vis    = WWFnF.createFilepath([filepath_base, folder_vis])
  filepath_frames = WWFnF.createFilepath([filepath_vis, "plotSpectra"])
  WWFnF.createFolder(filepath_vis)
  WWFnF.createFolder(filepath_frames)
  ## print filepath information to the console
  P2Term.printInfo("Base directory:", filepath_base)
  P2Term.printInfo("Data directory:", filepath_spect)
  P2Term.printInfo("Vis. directory:", filepath_vis)
  print(" ")

  ## ############################
  ## LOAD SIMULATION PLT PER EDDY
  ## ############################
  str_plot_every_eddy = WWFnF.readLineFromFile(
    filepath = WWFnF.createFilepath([filepath_sim, "flash.par"]),
    des_str  = "plotFileIntervalTime"
  ).split(" # ")[1]
  if "T" not in str_plot_every_eddy:
    raise Exception("Could not read eddy-turnover-time from 'flash.par'.")
  plots_per_eddy = 1 / float(
    str_plot_every_eddy.split("T")[0] # plot interval in t_turb
  )
  print("Number of plt files per t_turb = {} from 'flash.par'.".format( plots_per_eddy ))
  print(" ")
  
  ## ##########################
  ## LOAD AND PLOT SPECTRA DATA
  ## ##########################
  kin_k, kin_power, kin_sim_times = LoadFlashData.loadListSpectra(
    filepath_data     = filepath_spect,
    str_spectra_type  = "vel",
    plots_per_eddy    = plots_per_eddy,
    bool_hide_updates = bool_hide_updates
  )
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
    kin_k           = kin_k,
    kin_power       = kin_power,
    mag_k           = mag_k,
    mag_power       = mag_power,
    sim_times       = sim_times,
    fig_name        = folder_sim,
    filepath_frames = filepath_frames,
    filepath_ani    = filepath_vis
  )
  ## plot spectra data
  plot_obj.plotSpectra(bool_hide_updates)
  plot_obj.aniSpectra(bool_hide_updates)
  print(" ")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()


## END OF PROGRAM
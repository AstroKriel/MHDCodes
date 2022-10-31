#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys
from TheUsefulModule import WWFnF

## ###############################################################
## PREPARE WORKSPACE
##################################################################
os.system("clear") # clear terminal window


## ###############################################################
## HELPER PROGRAM
## ###############################################################
def downloadFromGadi(filepath_gadi, filepath_mac, filename):
  print("GADI:", filepath_gadi)
  print("MAC:", filepath_mac)
  os.system(f"scp gadi:{filepath_gadi}/{filename} {filepath_mac}/.")
  print(" ")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## LOOK AT EACH SIMULATION FOLDER
  ## ------------------------------
  for suite_folder in LIST_SUITE_FOLDER:
    for sim_folder in LIST_SIM_FOLDER:

      ## COMMUNICATE PROGRESS
      ## --------------------
      ## check that the filepath exists (MAC)
      filepath_mac_sim = WWFnF.createFilepath([
        BASEPATH_MAC, suite_folder, SONIC_REGIME, sim_folder
      ])
      if not os.path.exists(filepath_mac_sim): continue
      str_message = f"Downloading from suite: {suite_folder}, sim: {sim_folder}"
      print(str_message)
      print("=" * len(str_message))
      print(" ")

      if BOOL_GET_PLOTS_CONVERGED:
        filepath_gadi_plot = WWFnF.createFilepath([
          BASEPATH_GADI, suite_folder, SONIC_REGIME, sim_folder, "vis_folder"
        ])
        filepath_mac_plot = WWFnF.createFilepath([
          BASEPATH_MAC, suite_folder, SONIC_REGIME
        ])
        downloadFromGadi(
          filepath_gadi = filepath_gadi_plot,
          filepath_mac  = filepath_mac_plot,
          filename      = FILENAME_PLOTS_CONVERGED
        )

      ## loop over the different resolution runs
      if BOOL_GET_PLOTS_NRES or BOOL_GET_VIDEOS_NRES:
        ## for aesthetic reasons
        if BOOL_GET_PLOTS_CONVERGED:
          print(" ")
        for sim_res in LIST_SIM_RES:

          ## CREATE FILEPATH TO FOLDER ON MAC
          ## --------------------------------
          ## create filepath to resolution dependent plots (MAC)
          filepath_mac_nres_plot = WWFnF.createFilepath([
            BASEPATH_MAC, suite_folder, SONIC_REGIME, sim_folder, sim_res
          ])
          ## check that the filepath exists (MAC)
          if not os.path.exists(filepath_mac_nres_plot): continue

          ## DOWNLOAD FIT FIGURES FROM GADI
          ## ------------------------------
          if BOOL_GET_PLOTS_NRES:
            filepath_gadi_nres_plots = WWFnF.createFilepath([
              BASEPATH_GADI, suite_folder, SONIC_REGIME, sim_folder, sim_res, "vis_folder"
            ])
            downloadFromGadi(
              filepath_gadi = filepath_gadi_nres_plots,
              filepath_mac  = filepath_mac_nres_plot,
              filename      = FILENAME_PLOTS_NRES
            )

          ## DOWNLOAD FIT VIDEOS FROM GADI
          ## -----------------------------
          if BOOL_GET_VIDEOS_NRES:
            ## download animated video (from GADI)
            filepath_gadi_nres_videos = WWFnF.createFilepath([
              BASEPATH_GADI, suite_folder, SONIC_REGIME, sim_folder, sim_res
            ])
            downloadFromGadi(
              filepath_gadi = filepath_gadi_nres_videos,
              filepath_mac  = filepath_mac_nres_plot,
              filename      = FILENAME_VIDEOS_NRES
            )

    ## print empty space
    print(" ")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BASEPATH_MAC             = "/Users/necoturb/Documents/Studies/MHDScales"
BASEPATH_GADI            = "/scratch/ek9/nk7952"
SONIC_REGIME             = "super_sonic"
LIST_SUITE_FOLDER        = [ "Re10", "Re500", "Rm3000" ]
LIST_SIM_FOLDER          = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
# LIST_SIM_RES             = [ "18", "36", "72", "144", "288", "576" ]
LIST_SIM_RES             = [ "18", "36", "72" ]
## animated spectra at a Nres
BOOL_GET_VIDEOS_NRES     = 0
FILENAME_VIDEOS_NRES     = "*fit*.mp4"
## plots of simulations at a Nres
BOOL_GET_PLOTS_NRES      = 0
FILENAME_PLOTS_NRES      = "*.png"
## plots of convergence data
BOOL_GET_PLOTS_CONVERGED = 1
FILENAME_PLOTS_CONVERGED = "*.png"


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
    main()
    sys.exit()


## END OF PROGRAM
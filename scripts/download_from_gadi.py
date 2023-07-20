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
## HELPER FUNCTION
##################################################################
def downloadFromGadiToMac(filepath_gadi, filepath_mac, filename):
  print("GADI:", filepath_gadi)
  print("MAC:",  filepath_mac)
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

      ## check that the filepath exists (MAC)
      filepath_mac_sim = WWFnF.createFilepath([
        BASEPATH_MAC, suite_folder, SONIC_REGIME, sim_folder
      ])
      filepath_gadi_sim = WWFnF.createFilepath([
        BASEPATH_GADI, suite_folder, SONIC_REGIME, sim_folder
      ])

      ## CHECK THE SIMULATION EXISTS
      ## ---------------------------
      if not os.path.exists(filepath_mac_sim): continue

      ## COMMUNICATE PROGRESS
      ## --------------------
      str_message = f"Downloading from suite: {suite_folder}, sim: {sim_folder}, regime: {SONIC_REGIME}"
      print(str_message)
      print("=" * len(str_message))
      print(" ")

      if BOOL_GET_PLOTS_CONVERGED:
        downloadFromGadiToMac(
          filepath_gadi = f"{filepath_gadi_sim}/vis_folder/",
          ## group everything into the sonic regime (for comparing)
          filepath_mac  = f"{filepath_mac_sim}/../",
          filename      = FILENAME_PLOTS_CONVERGED
        )

      if BOOL_GET_PLOTS_NRES or BOOL_GET_VIDEOS_NRES:
        ## for aesthetic reasons
        if BOOL_GET_PLOTS_CONVERGED:
          print(" ")

        ## loop over the different resolution runs
        for sim_res in LIST_SIM_RES:

          ## CREATE FILEPATH TO FOLDER ON MAC
          ## --------------------------------
          ## create filepath to store plots at Nres (MAC)
          filepath_mac_nres_vis  = f"{filepath_mac_sim}/{sim_res}/" # no need for extra subfolder
          filepath_gadi_nres_vis = f"{filepath_gadi_sim}/{sim_res}/vis_folder/"
          ## check that the filepath exists (MAC)
          if not os.path.exists(filepath_mac_nres_vis): continue
          print(f"Looking at Nres: {sim_res}...")

          ## DOWNLOAD FIT FIGURES FROM GADI
          ## ------------------------------
          if BOOL_GET_PLOTS_NRES:
            ## download plots (from GADI)
            downloadFromGadiToMac(
              filepath_gadi = filepath_gadi_nres_vis,
              filepath_mac  = filepath_mac_nres_vis,
              filename      = FILENAME_PLOTS_NRES
            )

          ## DOWNLOAD FIT VIDEOS FROM GADI
          ## -----------------------------
          if BOOL_GET_VIDEOS_NRES:
            ## download animated video (from GADI)
            downloadFromGadiToMac(
              filepath_gadi = filepath_gadi_nres_vis,
              filepath_mac  = filepath_mac_nres_vis,
              filename      = FILENAME_VIDEOS_NRES
            )

      ## print empty space
      print(" ")
    print(" ")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BASEPATH_MAC             = "/Users/necoturb/Documents/Studies/MHDScales"
BASEPATH_GADI            = "/scratch/ek9/nk7952"
## simulation paarmeters
SONIC_REGIME             = "Mach5"
LIST_SUITE_FOLDER        = [ "Re10", "Re500", "Rm3000" ]
LIST_SIM_FOLDER          = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
LIST_SIM_RES             = [ "18", "36", "72", "144", "288", "576" ]
## simulations plots
BOOL_GET_PLOTS_NRES      = 0
FILENAME_PLOTS_NRES      = "*.png"
## converged data
BOOL_GET_PLOTS_CONVERGED = 1
FILENAME_PLOTS_CONVERGED = "*_nres_scales.png"
## animated spectra
BOOL_GET_VIDEOS_NRES     = 0
FILENAME_VIDEOS_NRES     = "*.mp4"

# LIST_SUITE_FOLDER        = [ "Rm3000" ]
# LIST_SIM_RES             = [ "18", "36", "72" ]
# LIST_SIM_RES             = [ "144", "288", "576" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
    main()
    sys.exit()


## END OF PROGRAM
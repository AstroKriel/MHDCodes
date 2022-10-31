#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys, re


## ###############################################################
## PREPARE WORKSPACE
##################################################################
os.system("clear") # clear terminal window


## ###############################################################
## HELPER FUNCTIONS
##################################################################
def createFilepath(list_filepaths):
  return re.sub('/+', '/', "/".join([
    filepath for filepath in list_filepaths if not(filepath == "")
  ]))


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
      filepath_mac_sim = createFilepath([
        BASEPATH_MAC, suite_folder, SONIC_REGIME, sim_folder
      ])
      if not os.path.exists(filepath_mac_sim): continue
      str_message = f"Downloading from suite: {suite_folder}, sim: {sim_folder}"
      print(str_message)
      print("=" * len(str_message))
      print(" ")

      if BOOL_GET_PLOTS_CONVERGED:
        ## create filepath (MAC)
        filepath_mac_plot = createFilepath([
          BASEPATH_MAC, suite_folder, SONIC_REGIME
        ])
        ## download plots (from GADI)
        filepath_gadi_plot = createFilepath([
          BASEPATH_GADI, suite_folder, SONIC_REGIME, sim_folder, "vis_folder"
        ])
        print("GADI:", filepath_gadi_plot)
        print("MAC:", filepath_mac_plot)
        os.system(f"scp gadi:{filepath_gadi_plot}/{FILENAME_PLOTS_CONVERGED} {filepath_mac_plot}/.")
        print(" ")

      ## loop over the different resolution runs
      if BOOL_GET_PLOTS_NRES or BOOL_GET_VIDEOS_NRES:
        ## for aesthetic reasons
        if BOOL_GET_PLOTS_CONVERGED:
          print(" ")
        for sim_res in LIST_SIM_RES:

          ## CREATE FILEPATH TO FOLDER ON MAC
          ## --------------------------------
          ## create filepath to resolution dependent plots (MAC)
          filepath_mac_nres_plot = createFilepath([
            BASEPATH_MAC, suite_folder, SONIC_REGIME, sim_folder, sim_res
          ])
          ## check that the filepath exists (MAC)
          if not os.path.exists(filepath_mac_nres_plot): continue

          ## DOWNLOAD FIT FIGURES FROM GADI
          ## ------------------------------
          if BOOL_GET_PLOTS_NRES:
            ## download plots (from GADI)
            filepath_gadi_nres_plot = createFilepath([
              BASEPATH_GADI, suite_folder, SONIC_REGIME, sim_folder, sim_res, "vis_folder"
            ])
            print("GADI:", filepath_gadi_nres_plot)
            print("MAC:", filepath_mac_nres_plot)
            os.system(f"scp gadi:{filepath_gadi_nres_plot}/{FILENAME_PLOTS_NRES} {filepath_mac_nres_plot}/.")
            print(" ")

          ## DOWNLOAD FIT VIDEOS FROM GADI
          ## -----------------------------
          if BOOL_GET_VIDEOS_NRES:
            ## download animated video (from GADI)
            filepath_gadi_nres_videos = createFilepath([
              BASEPATH_GADI, suite_folder, SONIC_REGIME, sim_folder, sim_res
            ])
            print("GADI:", filepath_gadi_nres_videos)
            print("MAC:", filepath_mac_nres_plot)
            os.system(f"scp gadi:{filepath_gadi_nres_videos}/{FILENAME_PLOTS_NRES} {filepath_mac_nres_plot}/.")
            print(" ")

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
LIST_SIM_RES             = [ "72" ]
## animated spectra at a Nres
BOOL_GET_VIDEOS_NRES     = 0
FILENAME_VIDEOS_NRES     = "*fit*.mp4"
## plots of simulations at a Nres
BOOL_GET_PLOTS_NRES      = 1
FILENAME_PLOTS_NRES      = "*.png"
## plots of convergence data
BOOL_GET_PLOTS_CONVERGED = 0
FILENAME_PLOTS_CONVERGED = "*.png"


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
    main()
    sys.exit()


## END OF PROGRAM
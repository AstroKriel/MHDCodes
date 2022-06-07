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
## DEFINE MAIN PROGRAM
## ###############################################################
BASEPATH_MAC    = "/Users/necoturb/Documents/Studies/Dynamo/MHD-scales"
BASEPATH_GADI   = "/scratch/ek9/nk7952/"
SONIC_REGIME    = "super_sonic"
BOOL_GET_PLOTS  = 1
FILENAME_FIGS   = "*_check.png"
BOOL_GET_VIDEOS = 0
FILENAME_VIDS   = "*_ani_spectra.png"

def main():
  ## #################################
  ## LOOK AT EACH SUITE'S PLOTS FOLDER
  ## #################################
  ## loop over the simulation suites
  for suite_folder in [
      "Re10", "Re500", "Rm3000"
    ]: # "Re10", "Re500", "Rm3000", "keta"

    ## loop over the different resolution runs
    for sim_res in [
        "72", "144", "288"
      ]: # "18", "36", "72", "144", "288", "576"

      ## ################################
      ## CREATE FILEPATH TO FOLDER ON MAC
      ## ################################
      filepath_mac_vis = WWFnF.createFilepath([
        BASEPATH_MAC, suite_folder, sim_res, SONIC_REGIME, "vis_folder"
      ])
      ## check that the filepath exists on MAC
      if not os.path.exists(filepath_mac_vis):
        print("{} does not exist.".format( filepath_mac_vis ))
        continue
      str_message = "Downloading from suite: {}, Nres = {}".format(
        suite_folder,
        sim_res
      )
      print(str_message)
      print("=" * len(str_message))

      ## ##############################
      ## DOWNLOAD FIT FIGURES FROM GADI
      ## ##############################
      if BOOL_GET_PLOTS:
        filepath_gadi_vis = WWFnF.createFilepath([
          "gadi:"+BASEPATH_GADI, suite_folder, sim_res, SONIC_REGIME, "vis_folder"
        ])
        print("\t> From:", filepath_gadi_vis)
        print("\t> To:", filepath_mac_vis)
        ## download plots checking fits
        os.system("scp {}/{} {}/.".format(
          filepath_gadi_vis, FILENAME_FIGS,
          filepath_mac_vis
        ))

      ## #############################
      ## DOWNLOAD FIT VIDEOS FROM GADI
      ## #############################
      if BOOL_GET_VIDEOS:
        filepath_gadi_videos = WWFnF.createFilepath([
            "gadi:"+BASEPATH_GADI, suite_folder, sim_res, SONIC_REGIME, "vis_folder"
        ])
        print("\t> From:", filepath_gadi_videos)
        print("\t> To:", filepath_mac_vis)
        ## download plots + animations
        os.system("scp {}/*.mp4 {}/.".format(
          filepath_gadi_videos,
          filepath_mac_vis
        ))
      print(" ")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
    main()
    sys.exit()


## END OF PROGRAM
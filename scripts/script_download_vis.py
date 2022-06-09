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
BASEPATH_MAC            = "/Users/necoturb/Documents/Studies/Dynamo/MHD-scales"
BASEPATH_GADI           = "/scratch/ek9/nk7952/"
SONIC_REGIME            = "super_sonic"
BOOL_GET_PLOTS_RES      = 0
FILENAME_FIGS_RES       = "*_check.png"
# FILENAME_FIGS_RES       = "*_check_fk_fm.png"
BOOL_GET_PLOTS_CONVERGE = 1
FILENAME_FIGS_CONVERGE  = "*.png"
BOOL_GET_VIDEOS_RES     = 1
FILENAME_VIDS_RES       = "*.mp4"

def main():
  ## #################################
  ## LOOK AT EACH SUITE'S PLOTS FOLDER
  ## #################################
  ## loop over the simulation suites
  for suite_folder in [
      "Re10", "Re500", "Rm3000"
    ]: # "Re10", "Re500", "Rm3000", "keta"

    if BOOL_GET_PLOTS_CONVERGE:
      ## create filepath to convergence plots (MAC)
      filepath_mac_vis_converge = WWFnF.createFilepath([
        BASEPATH_MAC, suite_folder, SONIC_REGIME
      ])
      str_message = "Downloading from suite: {}".format(
        suite_folder
      )
      ## check that the filepath exists (MAC)
      if not os.path.exists(filepath_mac_vis_converge):
        print("{} does not exist.".format( filepath_mac_vis_converge ))
        continue
      print(str_message)
      print("=" * len(str_message))
      ## download convergence plots (from GADI)
      filepath_gadi_vis_converge = WWFnF.createFilepath([
        "gadi:"+BASEPATH_GADI, suite_folder, SONIC_REGIME
      ])
      print("\t> From:", filepath_gadi_vis_converge)
      print("\t> To:", filepath_mac_vis_converge)
      os.system("scp {}/{} {}/.".format(
        filepath_gadi_vis_converge, FILENAME_FIGS_CONVERGE,
        filepath_mac_vis_converge
      ))

    ## loop over the different resolution runs
    if BOOL_GET_PLOTS_RES or BOOL_GET_VIDEOS_RES:
      ## for aesthetic reasons
      if BOOL_GET_PLOTS_CONVERGE:
        print(" ")

      for sim_res in [
          "72", "144", "288"
        ]: # "18", "36", "72", "144", "288", "576"

        ## ################################
        ## CREATE FILEPATH TO FOLDER ON MAC
        ## ################################
        ## create filepath to resolution dependent plots (MAC)
        filepath_mac_vis = WWFnF.createFilepath([
          BASEPATH_MAC, suite_folder, sim_res, SONIC_REGIME, "vis_folder"
        ])
        str_message = "Downloading from suite: {}, Nres = {}".format(
          suite_folder,
          sim_res
        )
        ## check that the filepath exists (MAC)
        if not os.path.exists(filepath_mac_vis):
          print("{} does not exist.".format( filepath_mac_vis ))
          continue
        print(str_message)
        print("=" * len(str_message))

        ## ##############################
        ## DOWNLOAD FIT FIGURES FROM GADI
        ## ##############################
        if BOOL_GET_PLOTS_RES:
          ## download plots (from GADI)
          filepath_gadi_vis = WWFnF.createFilepath([
            "gadi:"+BASEPATH_GADI, suite_folder, sim_res, SONIC_REGIME, "vis_folder"
          ])
          print("\t> From:", filepath_gadi_vis)
          print("\t> To:", filepath_mac_vis)
          os.system("scp {}/{} {}/.".format(
            filepath_gadi_vis, FILENAME_FIGS_RES,
            filepath_mac_vis
          ))

        ## #############################
        ## DOWNLOAD FIT VIDEOS FROM GADI
        ## #############################
        if BOOL_GET_VIDEOS_RES:
          ## download animated video (from GADI)
          filepath_gadi_videos = WWFnF.createFilepath([
              "gadi:"+BASEPATH_GADI, suite_folder, sim_res, SONIC_REGIME, "vis_folder"
          ])
          print("\t> From:", filepath_gadi_videos)
          print("\t> To:", filepath_mac_vis)
          os.system("scp {}/{} {}/.".format(
            filepath_gadi_videos, FILENAME_VIDS_RES,
            filepath_mac_vis
          ))
        ## print empty line between looking at different resoluitions
        print(" ")
    ## print empty line between looking at different suites
    print(" ")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
    main()
    sys.exit()


## END OF PROGRAM
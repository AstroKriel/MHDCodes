#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import matplotlib.pyplot as plt

from os import path


## ###############################################################
## PREPARE WORKSPACE
##################################################################
os.system("clear") # clear terminal window
plt.ioff()
plt.switch_backend("agg") # use a non-interactive plotting backend


BASEPATH_MAC  = "/Users/dukekriel/Documents/Studies/TurbulentDynamo/data/"
BASEPATH_GADI = "/scratch/ek9/nk7952/"
SONIC_REGIME  = "super_sonic"
FILENAME      = "*_check.png"
## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  for suite_folder in [
      "Re10", "Re500", "Rm3000"
    ]: # "Re10", "Re500", "Rm3000", "keta"

    for sim_res in [
        "18", "36", "72", "144", "288"
      ]: # "72", "144", "288", "576"

      ## ################################
      ## CREATE FILEPATH TO FOLDER ON MAC
      ## ################################
      filepath_mac_figures = createFilepath([
        BASEPATH_MAC, SONIC_REGIME, suite_folder, sim_res, "vis_folder"
      ])
      ## check that the filepath exists on MAC
      if not path.exists(filepath_mac_figures):
        print("{} does not exist.".format( filepath_mac_figures ))
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
      filepath_gadi_figures = createFilepath([
        "gadi:"+BASEPATH_GADI, suite_folder, sim_res, SONIC_REGIME, "vis_folder"
      ])
      print("Downloading from:", filepath_gadi_figures)
      ## download plots checking fits
      os.system("scp {}/{} {}/.".format(
        filepath_gadi_figures,
        FILENAME,
        filepath_mac_figures
      ))

      ## #############################
      ## DOWNLOAD FIT VIDEOS FROM GADI
      ## #############################
      filepath_gadi_videos = createFilepath([
          "gadi:"+BASEPATH_GADI, suite_folder, sim_res, SONIC_REGIME, "vis_folder"
      ])
      print("Downloading from:", filepath_gadi_videos)
      ## download plots + animations
      os.system("scp {}/*.mp4 {}/.".format(
        filepath_gadi_videos,
        filepath_mac_figures
      ))
      print(" ")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
    main()
    sys.exit()


## END OF PROGRAM
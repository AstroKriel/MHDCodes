#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import matplotlib.pyplot as plt

from os import path

## load old user defined modules
from OldModules.the_useful_library import *


## ###############################################################
## PREPARE WORKSPACE
##################################################################
os.system("clear") # clear terminal window
plt.ioff()
plt.switch_backend("agg") # use a non-interactive plotting backend


SONIC_REGIME = "super_sonic"
GADI_BASEPATH = "/scratch/ek9/nk7952/"
FILENAME = "Turb.dat"
## ###############################################################
## DEFINE MAIN PROGRAM
## ###############################################################
def main():
  my_filepath_base = "/Users/dukekriel/Documents/Studies/TurbulentDynamo/data/" + SONIC_REGIME
  for suite_folder in [
      "Re10", "Re500", "Rm3000"
    ]: # "Re10", "Re500", "keta", "Rm3000"

    for sim_res in [
        "18", "36", "72", "144", "288"
      ]: # "72", "144", "288", "576"
      ## ################################
      ## CREATE FILEPATH TO FOLDER ON MAC
      ## ################################
      mac_filepath_figures = createFilepath([
        my_filepath_base, suite_folder, sim_res, "vis_folder"
      ])
      ## check that the filepath exists on MAC
      if not path.exists(mac_filepath_figures):
        print("{} does not exist.".format( mac_filepath_figures ))
        continue
      str_message = "Downloading from suite: {}, Nres = {}".format( suite_folder, sim_res )
      print(str_message)
      print("=" * len(str_message))

      ## ##############################
      ## DOWNLOAD FIT FIGURES FROM GADI
      ## ##############################
      gadi_filepath_figures = createFilepath([
        "gadi:"+GADI_BASEPATH, suite_folder, sim_res, SONIC_REGIME, "vis_folder"
      ])
      print("Downloading from:", gadi_filepath_figures)
      ## download plots checking fits
      os.system("scp " + gadi_filepath_figures + "/*_check_MeasuredScales.pdf " + mac_filepath_figures + "/.")
      os.system("scp " + gadi_filepath_figures + "/check_fits/*.pdf " + mac_filepath_figures + "/check_fits/.")

      ## #############################
      ## DOWNLOAD FIT VIDEOS FROM GADI
      ## #############################
      gadi_filepath_figures = createFilepath([
          "gadi:"+GADI_BASEPATH, suite_folder, sim_res, SONIC_REGIME, "vis_folder"
      ])
      print("Downloading from:", gadi_filepath_figures)
      ## download plots + animations
      os.system("scp " + gadi_filepath_figures + "/*.mp4 " + mac_filepath_figures + "/.")
      print(" ")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
    main()
    sys.exit()


## END OF PROGRAM
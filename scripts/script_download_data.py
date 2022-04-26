#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import matplotlib.pyplot as plt

from os import path

## load user defined modules
from TheUsefulModule import WWFnF


## ###############################################################
## PREPARE WORKSPACE
##################################################################
os.system("clear") # clear terminal window
plt.ioff()
plt.switch_backend("agg") # use a non-interactive plotting backend


SONIC_REGIME  = "super_sonic"
BASEPATH_MAC  = "/Users/dukekriel/Documents/Studies/TurbulentDynamo/data/"
BASEPATH_GADI = "/scratch/ek9/nk7952/"
FILENAME = "Turb.dat"
## ###############################################################
## DEFINE MAIN PROGRAM
## ###############################################################
def main():
  my_filepath_base = BASEPATH_MAC + SONIC_REGIME
  for suite_folder in [
      "Re10", "Re500", "Rm3000"
    ]: # "Re10", "Re500", "keta", "Rm3000"

    for sim_res in [
        "18", "36", "72", "144", "288"
      ]: # "72", "144", "288", "576"
      ## ################################
      ## CREATE FILEPATH TO FOLDER ON MAC
      ## ################################
      mac_filepath_figures = WWFnF.createFilepath([
        my_filepath_base, suite_folder, sim_res, "vis_folder"
      ])
      ## check that the filepath exists on MAC
      if not path.exists(mac_filepath_figures):
        print("{} does not exist.".format( mac_filepath_figures ))
        continue
      str_message = "Downloading from suite: {}, Nres = {}".format( suite_folder, sim_res )
      print(str_message)
      print("=" * len(str_message))

      ## ###################
      ## DOWNLOAD DATA FILES
      ## ###################
      for sim_folder in [
          "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"
        ]:
        ## create filepath to folder on MAC
        mac_filepath_data = WWFnF.createFilepath([ my_filepath_base, suite_folder, sim_res, sim_folder ])
        ## check if the filepath exists on MAC
        if not path.exists(mac_filepath_data):
          continue
        ## create filepath to data folder on GADI
        gadi_filepath_data = WWFnF.createFilepath([
          "gadi:"+BASEPATH_GADI, suite_folder, sim_res, SONIC_REGIME, sim_folder
        ])
        print("Downloading from:", gadi_filepath_data)
        ## download data
        os.system("scp {}/{} {}/.".format(
          gadi_filepath_data,
          FILENAME,
          mac_filepath_data
        ))
      print(" ")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
    main()
    sys.exit()


## END OF PROGRAM
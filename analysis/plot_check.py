#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

## load user defined modules
from TheUsefulModule import WWFnF


## ###############################################################
## PREPARE WORKSPACE
##################################################################
os.system("clear") # clear terminal window
plt.ioff()
plt.switch_backend("agg") # use a non-interactive plotting backend


## FUNCTIONS
def funcPlotSimData(filepath_data):
  ## check that the data files exist
  if not os.path.exists(WWFnF.createFilepath([filepath_data, "Turb.dat"])):
    return False
  if not os.path.exists(WWFnF.createFilepath([filepath_data, "spectra_obj_full.pkl"])):
    return False
  


SONIC_REGIME = "super_sonic"
BASEPATH = "/Users/dukekriel/Documents/Studies/TurbulentDynamo/data/"
## ###############################################################
## DEFINE MAIN PROGRAM
## ###############################################################
def main():
  filepath_base = BASEPATH + SONIC_REGIME

  ## #######################
  ## LOOK AT EACH SIMULATION
  ## #######################
  for suite_folder in [
      "Re10", "Re500", "Rm3000"
    ]: # "Re10", "Re500", "Rm3000", "keta"

    for sim_res in [
        "18", "36", "72", "144", "288"
      ]: # "18", "36", "72", "144", "288", "576"
      ## ####################################
      ## CREATE FILEPATH TO SIMULATION FOLDER
      ## ####################################
      mac_filepath_figures = WWFnF.createFilepath([
        filepath_base, suite_folder, sim_res, "vis_folder"
      ])
      ## check that the filepath exists on MAC
      if not os.path.exists(mac_filepath_figures):
        print("{} does not exist.".format( mac_filepath_figures ))
        continue
      str_message = "Looking at suite: {}, Nres = {}".format( suite_folder, sim_res )
      print(str_message)
      print("=" * len(str_message))

      ## ####################
      ## PLOT SIMULATION DATA
      ## ####################
      for sim_folder in [
          "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"
        ]: # "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"
        ## create filepath to the simulation folder
        filepath_data = WWFnF.createFilepath([ filepath_base, suite_folder, sim_res, sim_folder ])
        ## check that the filepath exists
        if not os.path.exists(filepath_data):
          continue
        ## plot simulation data
        funcPlotSimData(filepath_data)


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
    main()
    sys.exit()


## END OF PROGRAM
#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys

## load user defined modules
from TheUsefulModule import WWFnF


## ###############################################################
## MAIN PROGRAM
## ###############################################################
BASEPATH     = "/scratch/ek9/nk7952"
SONIC_REGIME = "super_sonic"

def main():
  ## LOOK AT EACH SIMULATION FOLDER
  ## ------------------------------
  ## loop over each simulation suite
  for suite_folder in [
      "Re10", "Re500", "Rm3000"
    ]:

    ## loop over each resolution
    for sim_res in [
        "72", "144", "288"
      ]:

      ## print to the terminal which suite is being looked at
      str_message = f"Looking at suite: {suite_folder}, Nres = {sim_res}"
      print(str_message)
      print("=" * len(str_message))

      ## loop over each simulation folder
      for sim_folder in [
          "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"
        ]: # "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"

        ## CHECK THAT THE SIMULATION FOLDER EXISTS
        ## ---------------------------------------
        ## create filepath to simulation folder
        sim_filepath_plt = WWFnF.createFilepath([
          BASEPATH, suite_folder, sim_res, SONIC_REGIME, sim_folder, "plt"
        ])
        ## check that the filepath exists
        if not os.path.exists(sim_filepath_plt):
          print(sim_filepath_plt, "does not exist.")
          continue

        ## archive data
        os.chdir(sim_filepath_plt) # change the directory
        os.system("pwd") # check the directory
        os.system("archive.py -i Turb_hdf5_plt_cnt_*") # archive data

      print(" ")
    print(" ")

## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
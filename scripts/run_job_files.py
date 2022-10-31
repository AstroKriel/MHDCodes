#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys
import subprocess

## load old user defined modules
from TheUsefulModule import WWFnF


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## LOOK AT EACH SIMULATION FOLDER
  ## ------------------------------
  ## loop over the simulation suites
  for suite_folder in LIST_SUITE_FOLDER:

    ## loop over the simulation folders
    for sim_folder in LIST_SIM_FOLDER:

      ## CHECK THE SIMULATION EXISTS
      ## ---------------------------
      filepath_sim = WWFnF.createFilepath([
        BASEPATH, suite_folder, SONIC_REGIME, sim_folder
      ])
      if not os.path.exists(filepath_sim): continue
      str_message = f"Looking at suite: {suite_folder}, sim: {sim_folder}"
      print(str_message)
      print("=" * len(str_message))
      print(" ")

      ## loop over the different resolution runs
      for sim_res in LIST_SIM_RES:

        ## CHECK THE RESOLUTION RUN EXISTS
        ## -------------------------------
        filepath_sim_res = WWFnF.createFilepath([
          filepath_sim, sim_res, DATA_SUBFOLDER
        ])
        ## check that the filepath exists
        if not os.path.exists(filepath_sim_res): continue

        ## CHECK THE JOB FILE EXISTS
        ## -------------------------
        if not os.path.isfile(f"{filepath_sim_res}/{JOB_NAME}"):
          print(f"\t> {JOB_NAME} does not exist in:\n\t", filepath_sim_res)
          continue
        ## indicate which folder is being worked on
        print("Looking at:", filepath_sim_res)
        print("Submitting job:", JOB_NAME)
        p = subprocess.Popen([ "qsub", JOB_NAME ], cwd=filepath_sim_res)
        p.wait()

        ## create an empty line after each suite
        print(" ")
      print(" ")
    print(" ")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BASEPATH          = "/scratch/ek9/nk7952/"
SONIC_REGIME      = "super_sonic"
JOB_NAME          = "job_calc_spect.sh"
LIST_SUITE_FOLDER = [ "Re10", "Re500", "Rm3000" ]
LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
# LIST_SIM_RES      = [ "18", "36", "72", "144", "288", "576" ]
LIST_SIM_RES      = [ "72" ]
DATA_SUBFOLDER    = "plt"


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()

## END OF PROGRAM
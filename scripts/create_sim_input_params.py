#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys

## load user defined module
from TheUsefulModule import WWFnF
from TheJobModule import SimInputParams


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## LOOK AT EACH SIMULATION
  ## -----------------------
  ## loop over each simulation suite
  for suite_folder in LIST_SUITE_FOLDER:

    ## loop over each simulation folder
    for sim_folder in LIST_SIM_FOLDER:

      ## CHECK THE SIMULATION EXISTS
      ## ---------------------------
      sonic_regime = SimInputParams.getSonicRegime(DES_MACH)
      filepath_sim = WWFnF.createFilepath([
        BASEPATH, suite_folder, sonic_regime, sim_folder
      ])
      if not os.path.exists(filepath_sim): continue
      str_message = f"Looking at suite: {suite_folder}, sim: {sim_folder}, regime: {sonic_regime}"
      print(str_message)
      print("=" * len(str_message))
      print(" ")

      ## loop over each resolution
      for sim_res in LIST_SIM_RES:

        ## CHECK THE RESOLUTION RUN EXISTS
        ## -------------------------------
        filepath_sim_res = f"{filepath_sim}/{sim_res}/"
        ## check that the filepath exists
        if not os.path.exists(filepath_sim_res): continue

        ## create and save simulation input parameters file
        SimInputParams.makeSimInputParams(
          filepath_sim_res, suite_folder, sim_folder, sim_res, K_TURB, DES_MACH
        )

        ## create empty space
        print(" ")
      print(" ")
    print(" ")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BASEPATH          = "/scratch/ek9/nk7952/"
K_TURB            = 1.0
DES_MACH          = 0.3
# LIST_SUITE_FOLDER = [ "Re10", "Re500", "Rm3000" ]
LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
# LIST_SIM_RES      = [ "18", "36", "72", "144", "288", "576" ]

LIST_SUITE_FOLDER = [ "Rm3000" ]
LIST_SIM_RES      = [ "576" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
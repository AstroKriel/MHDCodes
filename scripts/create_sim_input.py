#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys

## load user defined module
from TheUsefulModule import WWFnF
from TheJobModule import SimInputParams
from TheLoadingModule import LoadFlashData


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear")


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def makeSimInputParams(filepath_sim, suite_folder, sim_folder, sim_res):
  if sim_res in [ "144", "288", "576" ]:
    num_blocks = [ 36, 36, 48 ]
  elif sim_res in [ "36", "72" ]:
    num_blocks = [ 12, 12, 18 ]
  elif sim_res in [ "18" ]:
    num_blocks = [ 6, 6, 6 ]
  obj_sim_params = SimInputParams.SimInputParams()
  obj_sim_params.defineParams(
    suite_folder = suite_folder,
    sim_folder   = sim_folder,
    sim_res      = sim_res,
    num_blocks   = num_blocks,
    k_turb       = K_TURB,
    desired_Mach = DES_MACH,
    Re           = LoadFlashData.getPlasmaNumbers_fromName(suite_folder, "Re"),
    Rm           = LoadFlashData.getPlasmaNumbers_fromName(suite_folder, "Rm"),
    Pm           = LoadFlashData.getPlasmaNumbers_fromName(sim_folder,   "Pm")
  )
  SimInputParams.saveSimInputParams(obj_sim_params, filepath_sim)


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
      filepath_sim = WWFnF.createFilepath([
        BASEPATH, suite_folder, SONIC_REGIME, sim_folder
      ])
      if not os.path.exists(filepath_sim): continue
      str_message = f"Looking at suite: {suite_folder}, sim: {sim_folder}"
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

        # ## create and save simulation input parameters file
        # makeSimInputParams(filepath_sim_res, suite_folder, sim_folder, sim_res)

        ## create empty space
        print(" ")
      print(" ")
    print(" ")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BASEPATH          = "/scratch/ek9/nk7952/"
SONIC_REGIME      = "super_sonic"
K_TURB            = 2.0
DES_MACH          = 5.0
LIST_SUITE_FOLDER = [ "Re10", "Re500", "Rm3000" ]
LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
LIST_SIM_RES      = [ "18", "36", "72", "144", "288", "576" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
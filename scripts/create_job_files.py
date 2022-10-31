#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys

## load old user defined modules
from TheUsefulModule import WWFnF
from TheJobModule.PrepSimJob import PrepSimJob
from TheJobModule.CalcSpectraJob import CalcSpectraJob
from TheJobModule.PlotSpectraJob import PlotSpectraJob
from TheJobModule.SimParams import SimParams, getPlasmaNumberFromName


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
        filepath_sim_res = f"{filepath_sim}/{sim_res}/"
        ## check that the filepath exists
        if not os.path.exists(filepath_sim_res): continue

        obj_sim_params = SimParams(
          suite_folder = suite_folder,
          sim_folder   = sim_folder,
          sim_res      = int(sim_res),
          num_blocks   = NUM_BLOCKS,
          k_turb       = K_TURB,
          Mach         = MACH,
          Re           = getPlasmaNumberFromName(suite_folder, "Re"),
          Rm           = getPlasmaNumberFromName(suite_folder, "Rm"),
          Pm           = getPlasmaNumberFromName(sim_folder,   "Pm")
        )

        ## CREATE JOB FILE TO RUN SIMULATION
        ## ---------------------------------
        if BOOL_PREP_SIM:
          obj_prep_sim = PrepSimJob(
            filepath_ref   = f"{BASEPATH}/backup_files/",
            filepath_sim   = filepath_sim_res,
            obj_sim_params = obj_sim_params
          )
          obj_prep_sim.fromLowerNres(f"{filepath_sim}/36/")

        ## CREATE JOB FILE TO CALCULATE SPECTRA
        ## ------------------------------------
        if BOOL_CALC_SPECTRA:
          CalcSpectraJob(
            filepath_plt = f"{filepath_sim_res}/plt/",
            suite_folder = suite_folder,
            obj_sim_params = obj_sim_params
          )

        ## CREATE JOB FILE TO PLOT SPECTRA
        ## -------------------------------
        if BOOL_PLOT_SPECTRA:
          PlotSpectraJob(
            filepath_scratch = BASEPATH,
            filepath_sim     = filepath_sim_res,
            obj_sim_params   = obj_sim_params
          )

        ## create an empty line after each suite
        print(" ")
      print(" ")
    print(" ")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
## Simulation parameters
EMAIL_ADDRESS      = "neco.kriel@anu.edu.au"
BASEPATH           = "/scratch/ek9/nk7952/"
SONIC_REGIME       = "super_sonic"
MACH               = 5.0
K_TURB             = 2.0
NUM_BLOCKS         = [
  # 36, 36, 48  # Nres = 144, 288, 576
  12, 12, 18  # Nres = 36, 72
  # 6,  6,  6   # Nres = 18
]
BOOL_PREP_SIM      = 1
BOOL_CALC_SPECTRA  = 0
BOOL_PLOT_SPECTRA  = 0
LIST_SUITE_FOLDER = [ "Re10", "Re500", "Rm3000" ]
LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
LIST_SIM_RES      = [ "72" ]
# LIST_SIM_RES      = [ "18", "36", "72", "144", "288", "576" ]


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys

## load old user defined modules
from TheUsefulModule import WWFnF
from TheJobModule.PrepSimJob import PrepSimJob
from TheJobModule.PrepSpectCalcJob import PrepSpectCalcJob
from TheJobModule.PrepSpectPlotJob import PrepPlotSpectra


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## ##############################
  ## LOOK AT EACH SIMULATION FOLDER
  ## ##############################
  ## loop over the simulation suites
  for suite_folder in [
      "Re10",
      # "Re500",
      "Rm3000"
    ]:

    ## loop over the different resolution runs
    for sim_res in [
        "576"
      ]:

      ## print to the terminal what suite is being looked at
      str_msg = "Looking at suite: {}, Nres = {}".format( suite_folder, sim_res )
      print(str_msg)
      print("=" * len(str_msg))

      ## loop over the simulation folders
      for sim_folder in [
          "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"
        ]: # "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"

        ## ##################################
        ## CHECK THE SIMULATION FOLDER EXISTS
        ## ##################################
        ## create filepath to simulation directory
        filepath_sim = WWFnF.createFilepath([
          BASEPATH, suite_folder, sim_res, SONIC_REGIME, sim_folder
        ])
        ## check that the simulation directory exists
        if not os.path.exists(filepath_sim):
          print(filepath_sim, "does not exist.")
          continue
        ## indicate which folder is being worked on
        print("Looking at: {}".format(filepath_sim))

        ## #################################
        ## CREATE JOB FILE TO RUN SIMULATION
        ## #################################
        if BOOL_PREP_SIM:
          PrepSimJob(
            filepath_home = BASEPATH,
            filepath_sim  = filepath_sim,
            suite_folder  = suite_folder,
            sim_res       = sim_res,
            sim_folder    = sim_folder
          )

        ## ####################################
        ## CREATE JOB FILE TO CALCULATE SPECTRA
        ## ####################################
        if BOOL_CALC_SPECTRA:
          PrepSpectCalcJob(
            filepath_plt = filepath_sim,
            suite_folder = suite_folder,
            sonic_regime = SONIC_REGIME,
            sim_folder   = sim_folder,
            sim_res      = sim_res
          )

        ## ###############################
        ## CREATE JOB FILE TO PLOT SPECTRA
        ## ###############################
        if BOOL_PLOT_SPECTRA:
          PrepPlotSpectra(
            filepath_scratch = BASEPATH,
            filepath_sim     = filepath_sim,
            suite_folder     = suite_folder,
            sim_res          = sim_res,
            sim_folder       = sim_folder
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
  36, 36, 48  # Nres = 144, 288, 576
  # 12, 12, 18  # Nres = 36, 72
  # 6,  6,  6   # Nres = 18
]
BOOL_PREP_SIM      = 0
BOOL_CALC_SPECTRA  = 1
BOOL_PLOT_SPECTRA  = 0


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
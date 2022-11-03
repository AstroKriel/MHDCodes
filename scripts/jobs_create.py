#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys

## load user defined modules
from TheUsefulModule import WWFnF
from TheJobModule import SimInputParams
from TheJobModule import PrepSimJob
from TheJobModule import CalcSpectraJob
from TheJobModule import PlotSpectraJob


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
      sonic_regime = SimInputParams.getSonicRegime(DES_MACH)
      filepath_sim = WWFnF.createFilepath([
        BASEPATH, suite_folder, sonic_regime, sim_folder
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

        ## READ/CREATE SIMULATION INPUT REFERENCE FILE
        ## -------------------------------------------
        if os.path.isfile(f"{filepath_sim_res}/sim_inputs.json"):
          obj_sim_params = SimInputParams.readSimInputParams(filepath_sim_res)
        else: obj_sim_params = SimInputParams.makeSimInputParams(
          filepath_sim_res, suite_folder, sim_folder, sim_res, K_TURB, DES_MACH
        )
        dict_sim_params = obj_sim_params.getSimParams()

        ## CREATE JOB FILE TO RUN SIMULATION
        ## ---------------------------------
        if BOOL_PREP_SIM:
          obj_prep_sim = PrepSimJob.PrepSimJob(
            filepath_ref    = f"{BASEPATH}/backup_files/",
            filepath_sim    = filepath_sim_res,
            dict_sim_params = dict_sim_params
          )
          obj_prep_sim.fromLowerNres(f"{filepath_sim}/36/")

        ## CREATE JOB FILE TO CALCULATE SPECTRA
        ## ------------------------------------
        if BOOL_CALC_SPECTRA:
          CalcSpectraJob.CalcSpectraJob(
            filepath_plt    = f"{filepath_sim_res}/plt/",
            dict_sim_params = dict_sim_params
          )

        ## CREATE JOB FILE TO PLOT SPECTRA
        ## -------------------------------
        if BOOL_PLOT_SPECTRA:
          PlotSpectraJob.PlotSpectraJob(
            filepath_sim    = filepath_sim_res,
            dict_sim_params = dict_sim_params
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
DES_MACH           = 5.0
K_TURB             = 2.0
BOOL_PREP_SIM      = 0
BOOL_CALC_SPECTRA  = 1
BOOL_PLOT_SPECTRA  = 0
LIST_SUITE_FOLDER = [ "Re10", "Re500", "Rm3000" ]
LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
LIST_SIM_RES      = [ "18", "36", "72", "144", "288", "576" ]


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
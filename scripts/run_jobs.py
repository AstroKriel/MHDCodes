#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys, subprocess

## load user defined modules
from TheUsefulModule import WWFnF
from TheLoadingModule import FileNames


## ###############################################################
## SUBMIT JOB SCRIPT
## ###############################################################
def runJob(filepath_job):
  print("Looking at:", filepath_job)
  print("Submitting job:", JOB_NAME)
  p = subprocess.Popen([ "qsub", JOB_NAME ], cwd=filepath_job)
  p.wait()


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def loopOverMachSet():
  for suite_folder in LIST_SUITE_FOLDERS:
    ## LOOK AT EACH SIMULATION FOLDER
    ## -----------------------------
    for sim_folder in LIST_SIM_FOLDERS:
      ## CHECK THE SUITE + SIMULATION CONFIG EXISTS
      ## ------------------------------------------
      filepath_sim = WWFnF.createFilepath([
        BASEPATH, suite_folder, sim_folder
      ])
      if not os.path.exists(filepath_sim): continue
      ## loop over the different resolution runs
      for sim_res in LIST_SIM_RES:
        ## CHECK THE RESOLUTION RUN EXISTS
        ## -------------------------------
        filepath_sim_res = f"{filepath_sim}/{sim_res}/"
        if not os.path.exists(filepath_sim_res): continue
        ## CHECK THE JOB FILE EXISTS
        ## -------------------------
        filepath_job = f"{filepath_sim_res}/{SUBFOLDER}/"
        if not os.path.isfile(f"{filepath_job}/{JOB_NAME}"):
          print("ERROR: job file does not exist.")
          print("\t> Job name:", JOB_NAME)
          print("\t> Directory:", filepath_job)
          continue
        ## submit job script
        runJob(filepath_job)
        ## create empty space
        print(" ")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BASEPATH  = "/scratch/ek9/nk7952/"
SUBFOLDER = ""
# JOB_NAME  = FileNames.FILENAME_JOB_RUN_SIM

# LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm3000" ]
# LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
# LIST_SIM_RES       = [ "18", "36", "72", "144", "288", "576" ]

LIST_SUITE_FOLDERS = [ "Mach" ]
LIST_SIM_FOLDERS   = [ "0.3", "1", "10" ]
LIST_SIM_RES       = [ "144" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  loopOverMachSet()
  sys.exit()


## END OF PROGRAM
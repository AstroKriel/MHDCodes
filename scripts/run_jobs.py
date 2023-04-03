#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import sys, subprocess

## load user defined modules
from TheFlashModule import SimParams, FileNames
from TheUsefulModule import WWFnF


## ###############################################################
## SUBMIT JOB SCRIPT
## ###############################################################
def runJob(filepath_sim_res, **kwargs):
  filepath_file = WWFnF.createFilepath([ filepath_sim_res, SUBFOLDER ])
  print("Looking at:", filepath_file)
  print("Submitting job:", JOB_NAME)
  p = subprocess.Popen([ "qsub", JOB_NAME ], cwd=filepath_file)
  p.wait()
  print(" ")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  SimParams.callFuncForAllSimulations(
    func               = runJob,
    basepath           = BASEPATH,
    list_suite_folders = LIST_SUITE_FOLDERS,
    list_sonic_regimes = LIST_SONIC_REGIMES,
    list_sim_folders   = LIST_SIM_FOLDERS,
    list_sim_res       = LIST_SIM_RES
  )


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BASEPATH  = "/scratch/ek9/nk7952/"
# JOB_NAME  = FileNames.FILENAME_PROCESS_PLT_JOB
SUBFOLDER = "plt"

# ## PLASMA PARAMETER SET
# LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm3000" ]
# LIST_SONIC_REGIMES = [ "Mach5" ]
# LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
# LIST_SIM_RES       = [ "18", "36", "72", "144", "288", "576" ]

# ## MACH NUMBER SET
# LIST_SUITE_FOLDERS = [ "Re300" ]
# # LIST_SONIC_REGIMES = [ "Mach0.3", "Mach1", "Mach5", "Mach10" ]
# LIST_SONIC_REGIMES = [ "Mach5" ]
# LIST_SIM_FOLDERS   = [ "Pm4" ]
# # LIST_SIM_RES       = [ "36", "72", "144", "288" ]
# LIST_SIM_RES       = [ "36", "72", "144" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
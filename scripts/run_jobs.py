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
def runJob(filepath_sim_res):
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
  list_sim_filepaths = SimParams.getListOfSimFilepaths(
    list_base_paths    = LIST_BASE_PATHS,
    list_suite_folders = LIST_SUITE_FOLDERS,
    list_mach_regimes  = LIST_MACH_REGIMES,
    list_sim_folders   = LIST_SIM_FOLDERS,
    list_sim_res       = LIST_SIM_RES
  )
  index_job = 0
  for filepath_sim_res in list_sim_filepaths:
    print(f"({index_job})")
    runJob(filepath_sim_res)
    index_job += 1


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
JOB_NAME        = FileNames.FILENAME_RUN_SIM_JOB
SUBFOLDER       = ""

# LIST_BASE_PATHS    = [ "/scratch/jh2/nk7952/" ]
# LIST_MACH_REGIMES  = [ ]
# LIST_SUITE_FOLDERS = [ ]
# LIST_SIM_FOLDERS   = [ ]
# LIST_SIM_RES       = [ ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
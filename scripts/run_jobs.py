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
    basepath           = BASEPATH,
    list_suite_folders = LIST_SUITE_FOLDERS,
    list_sonic_regimes = LIST_SONIC_REGIMES,
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
BASEPATH  = "/scratch/ek9/nk7952/"
JOB_NAME  = FileNames.FILENAME_RUN_SIM_JOB
SUBFOLDER = ""

# ## PLASMA PARAMETER SET
# LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm3000" ]
# LIST_SONIC_REGIMES = [ "Mach5" ]
# LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
# LIST_SIM_RES       = [ "18", "36", "72", "144", "288", "576" ]

# ## RERUN RE=RM=3000
# LIST_SUITE_FOLDERS = [ "Rm3000" ]
# LIST_SONIC_REGIMES = [ "Mach5" ]
# LIST_SIM_FOLDERS   = [ "Pm1" ]
# LIST_SIM_RES       = [ "18", "36", "72", "144" ] # 288

# ## MACH NUMBER SET
# LIST_SUITE_FOLDERS = [ "Rm3000" ]
# LIST_SONIC_REGIMES = [ "Mach0.3", "Mach1", "Mach10" ]
# LIST_SIM_FOLDERS   = [ "Pm1", "Pm5", "Pm10", "Pm125" ]
# LIST_SIM_RES       = [ "18", "36", "72" ]

# ## BOTTLENECK RUN
# LIST_SUITE_FOLDERS = [ "Rm3000" ]
# LIST_SONIC_REGIMES = [ "Mach0.3", "Mach5" ]
# LIST_SIM_FOLDERS   = [ "Pm1" ]
# LIST_SIM_RES       = [ "576", "1152" ]



## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
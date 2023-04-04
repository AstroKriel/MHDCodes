#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys

## load user defined modules
from TheFlashModule import FileNames, SimParams, RunSimJob, ProcessPltFilesJob, LoadFlashData
from TheUsefulModule import WWFnF


## ###############################################################
## CREATE JOB SCRIPT
## ###############################################################
def createJobs(filepath_sim_res):
  ## read/create simulation input reference file
  dict_sim_inputs = SimParams.readSimInputs(filepath_sim_res)
  ## --------------
  if BOOL_PREP_SIM_RUN:
  ## --------------
    obj_prep_sim = RunSimJob.RunSimJob(
      filepath_sim    = filepath_sim_res,
      dict_sim_inputs = dict_sim_inputs
    )
    if len(PREP_FROM_LOWER_NRES) > 0:
      ## prepare simulation from a different resolution run
      filepath_ref_sim = filepath_sim_res.replace(
        dict_sim_inputs["sim_res"],
        PREP_FROM_LOWER_NRES
      )
      if not os.path.exists(filepath_ref_sim):
        raise Exception("Error: reference simulation does not exist:", filepath_ref_sim)
      obj_prep_sim.fromLowerNres(filepath_ref_sim)
    else:
      ## prepare simulation from template files
      filepath_ref_folder = f"{BASEPATH}/backup_files/"
      if not os.path.exists(filepath_ref_folder):
        raise Exception("Error: reference folder does not exist:", filepath_ref_folder)
      obj_prep_sim.fromTemplate(filepath_ref_folder)
  ## ------------------
  if BOOL_PROCESS_PLT_FILES:
  ## ------------------
    filepath_plt = f"{filepath_sim_res}/plt/"
    if not os.path.exists(filepath_plt):
      raise Exception("Error: plt sub-folder does not exist")
    ProcessPltFilesJob.ProcessPltFilesJob(
      filepath_plt    = filepath_plt,
      dict_sim_inputs = dict_sim_inputs
    )


## ###############################################################
## LOOP OVER AND GET ALL SIMULATION DETAILS
## ###############################################################
def getSimInputDetails():
  dicts_grouped_sim = []
  ## LOOK AT EACH SIMULATION SUITE
  ## -----------------------------
  for suite_folder in LIST_SUITE_FOLDERS:
    ## LOOK AT EACH SONIC REGIME
    ## -------------------------
    for sonic_regime in LIST_SONIC_REGIMES:
      ## LOOK AT EACH SIMULATION FOLDER
      ## -----------------------------
      for sim_folder in LIST_SIM_FOLDERS:
        ## CHECK THE SIMULATION CONFIGURATION EXISTS
        ## -----------------------------------------
        filepath_sim = WWFnF.createFilepath([
          BASEPATH, suite_folder, sonic_regime, sim_folder
        ])
        if not os.path.exists(filepath_sim): continue
        ## loop over the different resolution runs
        for sim_res in LIST_SIM_RES:
          ## CHECK THE RESOLUTION RUN EXISTS
          ## -------------------------------
          filepath_sim_res = f"{filepath_sim}/{sim_res}/"
          if not os.path.exists(filepath_sim_res): continue
          ## store for looking at later
          dicts_grouped_sim.append({
            "filepath_sim_res" : filepath_sim_res,
            "suite_folder"     : suite_folder,
            "sim_folder"       : sim_folder,
            "sim_res"          : sim_res,
            "desired_Mach"     : LoadFlashData.getNumberFromString(sonic_regime, "Mach"),
            "Re"               : LoadFlashData.getNumberFromString(suite_folder, "Re"),
            "Rm"               : LoadFlashData.getNumberFromString(suite_folder, "Rm"),
            "Pm"               : LoadFlashData.getNumberFromString(sim_folder,   "Pm")
          })
  return dicts_grouped_sim


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## loop over simulation directories
  index_job = 0
  for dict_sim in getSimInputDetails():
    filepath_sim_res = dict_sim["filepath_sim_res"]
    print(f"({index_job})")
    print("Looking at:", filepath_sim_res)
    index_job += 1
    bool_printed_something = False
    ## create simulation input parameter file if it doesn't exist
    bool_sim_inputs_exists = os.path.isfile(f"{filepath_sim_res}/{FileNames.FILENAME_SIM_INPUTS}")
    if BOOL_CREATE_SIM_INPUTS or not(bool_sim_inputs_exists):
      bool_printed_something = True
      SimParams.createSimInputs(
        filepath      = filepath_sim_res,
        suite_folder  = dict_sim["suite_folder"],
        sim_folder    = dict_sim["sim_folder"],
        sim_res       = dict_sim["sim_res"],
        k_turb        = K_TURB,
        desired_Mach  = dict_sim["desired_Mach"],
        Re            = dict_sim["Re"],
        Rm            = dict_sim["Rm"],
        Pm            = dict_sim["Pm"]
      )
    ## create job script
    if BOOL_PREP_SIM_RUN or BOOL_PROCESS_PLT_FILES:
      bool_printed_something = True
      createJobs(filepath_sim_res)
    ## create empty space
    if bool_printed_something:
      print(" ")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BASEPATH               = "/scratch/ek9/nk7952/"
# K_TURB                 = 2.0
BOOL_CREATE_SIM_INPUTS = 0
BOOL_PREP_SIM_RUN      = 0
PREP_FROM_LOWER_NRES   = ""
BOOL_PROCESS_PLT_FILES = 0

# ## PLASMA PARAMETER SET
# LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm3000" ]
# LIST_SONIC_REGIMES = [ "Mach5" ]
# LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
# LIST_SIM_RES       = [ "18", "36", "72", "144", "288", "576" ]

# ## MACH NUMBER SET
# LIST_SUITE_FOLDERS = [ "Re300" ]
# LIST_SONIC_REGIMES = [ "Mach0.3", "Mach1", "Mach5", "Mach10" ]
# LIST_SIM_FOLDERS   = [ "Pm4" ]
# LIST_SIM_RES       = [ "36", "72", "144", "288" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
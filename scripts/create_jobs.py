#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys

## load user defined modules
from TheSimModule import SimParams
from TheSimModule import PrepSimJob
from TheSimModule import CalcSpectraJob
from TheSimModule import PlotSpectraJob
from TheUsefulModule import WWFnF
from TheLoadingModule import LoadFlashData


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear")


## ###############################################################
## CREATE JOB SCRIPT
## ###############################################################
def createJob(filepath_sim_res, filepath_sim_lower_res=None):
  ## read/create simulation input reference file
  dict_sim_inputs = SimParams.readSimInputs(filepath_sim_res)
  ## --------------
  if BOOL_PREP_SIM:
  ## --------------
    obj_prep_sim = PrepSimJob.PrepSimJob(
      filepath_ref    = f"{BASEPATH}/backup_files/",
      filepath_sim    = filepath_sim_res,
      dict_sim_inputs = dict_sim_inputs
    )
    if filepath_sim_lower_res is not None:
      obj_prep_sim.fromLowerNres(filepath_sim_lower_res)
    else: obj_prep_sim.fromTemplate()
  ## ------------------
  if BOOL_CALC_SPECTRA:
  ## ------------------
    CalcSpectraJob.CalcSpectraJob(
      filepath_plt    = f"{filepath_sim_res}/plt/",
      dict_sim_inputs = dict_sim_inputs
    )


## ###############################################################
## MAIN PROGRAM: loop over plasma number set of simulations
## ###############################################################
def getSimInputDetails_plasmaSet(desired_Mach):
  dicts_grouped_sim = []
  ## LOOK AT EACH SIMULATION SUITE
  ## -----------------------------
  for suite_folder in LIST_SUITE_FOLDERS:
    ## LOOK AT EACH SIMULATION FOLDER
    ## -----------------------------
    for sim_folder in LIST_SIM_FOLDERS:
      ## CHECK THE SUITE + SIMULATION CONFIG EXISTS
      ## ------------------------------------------
      sonic_regime = SimParams.getSonicRegime(desired_Mach)
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
        dict_sim = {
          "filepath_sim_res" : filepath_sim_res,
          "suite_folder"     : suite_folder,
          "sim_folder"       : sim_folder,
          "sim_res"          : sim_res,
          "Re" : LoadFlashData.getNumberFromString(suite_folder, "Re"),
          "Rm" : LoadFlashData.getNumberFromString(suite_folder, "Rm"),
          "Pm" : LoadFlashData.getNumberFromString(sim_folder,   "Pm")
        }
        dicts_grouped_sim.append(dict_sim)
  ## return all simulation parameters
  return dicts_grouped_sim

def loopOverPlasmaSet(desired_Mach):
  for dict_sim in getSimInputDetails_plasmaSet(desired_Mach):
    filepath_sim_res = dict_sim["filepath_sim_res"]
    ## create and save simulation input parameters
    if not os.path.isfile(f"{filepath_sim_res}/{SimParams.FILENAME_SIM_INPUTS}"):
      SimParams.createSimInputs(
        filepath      = filepath_sim_res,
        suite_folder  = dict_sim["suite_folder"],
        sim_folder    = dict_sim["sim_folder"],
        sim_res       = dict_sim["sim_res"],
        k_turb        = K_TURB,
        desired_Mach  = desired_Mach,
        Re            = dict_sim["Re"],
        Rm            = dict_sim["Rm"],
        Pm            = dict_sim["Pm"]
      )


## ###############################################################
## MAIN PROGRAM: loop over mach number set of simulations
## ###############################################################
def getSimInputDetails_machSet():
  suite_folder = "Mach"
  dicts_grouped_sim = []
  ## LOOK AT EACH SIMULATION FOLDER
  ## -----------------------------
  for sim_mach in LIST_SIM_MACH:
    ## CHECK THE SUITE + SIMULATION CONFIG EXISTS
    ## ------------------------------------------
    filepath_sim = WWFnF.createFilepath([
      BASEPATH, suite_folder, sim_mach
    ])
    if not os.path.exists(filepath_sim): continue
    ## loop over the different resolution runs
    for sim_res in LIST_SIM_RES:
      ## CHECK THE RESOLUTION RUN EXISTS
      ## -------------------------------
      filepath_sim_res = f"{filepath_sim}/{sim_res}/"
      if not os.path.exists(filepath_sim_res): continue
      dicts_grouped_sim.append({
        "filepath_sim_res" : filepath_sim_res,
        "suite_folder"     : suite_folder,
        "sim_folder"       : sim_mach,
        "sim_res"          : sim_res,
        "desired_Mach"         : float(sim_mach)
      })
  ## return all simulation parameters
  return dicts_grouped_sim

def loopOverMachSet():
  for dict_sim in getSimInputDetails_machSet():
    filepath_sim_res = dict_sim["filepath_sim_res"]
    ## create simulation input parameter file if it doesn't exist
    if not os.path.isfile(f"{filepath_sim_res}/{SimParams.FILENAME_SIM_INPUTS}"):
      SimParams.createSimInputs(
        filepath      = filepath_sim_res,
        suite_folder  = dict_sim["suite_folder"],
        sim_folder    = dict_sim["sim_folder"],
        sim_res       = dict_sim["sim_res"],
        k_turb        = K_TURB,
        desired_Mach  = dict_sim["desired_Mach"],
        Re            = 300.0,
        Pm            = 4.0
      )
    ## create job script
    createJob(filepath_sim_res)
    ## create empty space
    print(" ")
    print(" ")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BASEPATH          = "/scratch/ek9/nk7952/"
K_TURB            = 2.0

BOOL_PREP_SIM     = 1
BOOL_CALC_SPECTRA = 0

BOOL_LOOP_OVER_PLASMA_SET = 0
FIXED_MACH         = 5.0
# LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm3000" ]
# LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
# LIST_SIM_RES       = [ "18", "36", "72", "144", "288", "576" ]

BOOL_LOOP_OVER_MACH_SET = 1
LIST_SIM_MACH      = [ "0.3", "1", "10" ]
LIST_SIM_RES       = [ "144" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  if BOOL_LOOP_OVER_PLASMA_SET: loopOverPlasmaSet(FIXED_MACH)
  if BOOL_LOOP_OVER_MACH_SET: loopOverMachSet()
  sys.exit()


## END OF PROGRAM
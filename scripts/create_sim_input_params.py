#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys

## load user defined module
from TheSimModule import SimParams
from TheUsefulModule import WWFnF
from TheLoadingModule import LoadFlashData


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear")


## ###############################################################
## CREATE LIST OF SIMULATION DIRECTORIES TO ANALYSE
## ###############################################################
def getSimInputDetails_allSims():
  dicts_grouped_sim = []
  ## LOOK AT EACH SIMULATION SUITE
  ## -----------------------------
  for suite_folder in LIST_SUITE_FOLDER:
    ## LOOK AT EACH SIMULATION FOLDER
    ## -----------------------------
    for sim_folder in LIST_SIM_FOLDER:
      ## CHECK THE SUITE + SIMULATION CONFIG EXISTS
      ## ------------------------------------------
      sonic_regime = SimParams.getSonicRegime(DES_MACH)
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
          "sim_res"          : sim_res
        }
        dicts_grouped_sim.append(dict_sim)
  ## return all simulation parameters
  return dicts_grouped_sim


## ###############################################################
## LOOP OVER MAIN SET OF SIMULATIONS
## ###############################################################
def loopOverMainSimulations():
  for dict_sim in getSimInputDetails_allSims():
    ## create and save simulation input parameters
    SimParams.createSimInputs(
      filepath_sim_res = dict_sim["filepath_sim_res"],
      suite_folder     = dict_sim["suite_folder"],
      sim_folder       = dict_sim["sim_folder"],
      sim_res          = dict_sim["sim_res"],
      k_turb           = K_TURB,
      des_mach         = DES_MACH,
      Re               = LoadFlashData.getPlasmaNumbers_fromName(dict_sim["suite_folder"], "Re"),
      Rm               = LoadFlashData.getPlasmaNumbers_fromName(dict_sim["suite_folder"], "Rm"),
      Pm               = LoadFlashData.getPlasmaNumbers_fromName(dict_sim["sim_folder"],   "Pm")
    )


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## create and save simulation input parameters
  for des_mach in [ 0.3, 1, 10]:
    sim_folder = str(des_mach)
    for sim_res in [ "144", "288" ]:
      SimParams.createSimInputs(
        filepath_sim_res = f"/scratch/ek9/nk7952/Mach/{sim_folder}/{sim_res}",
        suite_folder     = "Mach",
        sim_folder       = sim_folder,
        sim_res          = sim_res,
        k_turb           = 2.0,
        des_mach         = des_mach,
        Re               = 300,
        Pm               = 4
      )


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BASEPATH          = "/scratch/ek9/nk7952/"
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
#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys

## load user defined modules
from TheSimModule import SimParams
from TheSimModule import PrepSimJob
from TheSimModule import CalcSpectraJob
from TheSimModule import PlotSpectraJob


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window


## ###############################################################
## MODULES
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
  ## ------------------
  if BOOL_PLOT_SPECTRA:
  ## ------------------
    PlotSpectraJob.PlotSpectraJob(
      filepath_sim    = filepath_sim_res,
      dict_sim_inputs = dict_sim_inputs
    )


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  for des_mach in [ 0.3, 10 ]:
    for sim_res in [ "144" ]:
      createJob(f"/scratch/ek9/nk7952/Mach/{des_mach}/{sim_res}")
      print(" ")
      print(" ")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BASEPATH                = "/scratch/ek9/nk7952/"

BOOL_PREP_SIM           = 1
BOOL_CALC_SPECTRA       = 0
BOOL_PLOT_SPECTRA       = 0

# LIST_SUITE_FOLDER       = [ "Re10", "Re500", "Rm3000" ]
# LIST_SIM_FOLDER         = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
# LIST_SIM_RES            = [ "18", "36", "72", "144", "288", "576" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
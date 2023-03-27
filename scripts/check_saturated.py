#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys, subprocess
import numpy as np
from datetime import datetime

## load user defined routines
from organise_folders import removeFiles

## load user defined modules
from TheFlashModule import SimParams, LoadFlashData, FileNames
from TheUsefulModule import WWLists
from TheFittingModule import FitFuncs


## ###############################################################
## READ / UPDATE DRIVING PARAMETERS
## ###############################################################
def readDrivingAmplitude(filepath):
  filepath_file = f"{filepath}/{FileNames.FILENAME_DRIVING_INPUT}"
  ## check the file exists
  if not os.path.isfile(filepath_file):
    raise Exception("ERROR: turbulence generator input file does not exist:", FileNames.FILENAME_DRIVING_INPUT)
  ## open file
  with open(filepath_file) as fp:
    for line in fp.readlines():
      list_line_elems = line.split()
      ## ignore empty lines
      if len(list_line_elems) == 0: continue
      ## read driving amplitude
      if list_line_elems[0] == "ampl_factor":
        return float(list_line_elems[2])
  raise Exception(f"ERROR: could not read 'ampl_factor' in the turbulence generator")

def updateDrivingAmplitude(filepath, driving_amplitude):
  filepath_file = f"{filepath}/{FileNames.FILENAME_DRIVING_INPUT}"
  ## read previous driving parameters
  list_lines = []
  with open(filepath_file, "r") as fp:
    for line in fp.readlines():
      if "ampl_factor" in line:
        new_line = line.replace(line.split("=")[1].split()[0], str(driving_amplitude))
        list_lines.append(new_line)
      else: list_lines.append(line)
  ## write updated driving paramaters
  with open(filepath_file, "w") as output_file:
    output_file.writelines(list_lines)

def updateDrivingHistory(filepath, current_time, measured_Mach, old_driving_amplitude, new_driving_amplitude):
  filepath_file = f"{filepath}/{FileNames.FILENAME_DRIVING_HISTORY}"
  with open(filepath_file, "a") as fp:
    fp.write(f"{current_time} {measured_Mach} {old_driving_amplitude} {new_driving_amplitude}\n")


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class CheckDynamoSaturated():
  def __init__(self, filepath_sim):
    self.filepath_sim = filepath_sim
    dict_sim_inputs   = SimParams.readSimInputs(self.filepath_sim, bool_verbose=False)
    self.num_t_turb   = dict_sim_inputs["num_t_turb"]

  def performRoutine(self):
    print("Checking dynamo has saturated in:", self.filepath_sim)
    ## check whether tmax needs to be updated
    self._loadData()
    self._measureSatERatio()
    if not(self.bool_converged):
      self._updateTmax()
      self._removeOldData()
      self._reRunSimulation()
    else: print(f"\t> Energy ratio has already converged")



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
  for sim_filepath in list_sim_filepaths:
    obj_tune_driving = CheckDynamoSaturated(sim_filepath)
    obj_tune_driving.performRoutine()
    print(" ")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BASEPATH = "/scratch/ek9/nk7952/"

# ## PLASMA PARAMETER SET
# LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm3000" ]
# LIST_SONIC_REGIMES = [ "Mach0.3", "Mach5" ]
# LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
# LIST_SIM_RES       = [ "18", "36", "72", "144", "288", "576" ]

## MACH NUMBER SET
LIST_SUITE_FOLDERS = [ "Re300" ]
LIST_SONIC_REGIMES = [ "Mach0.3", "Mach1", "Mach10" ]
LIST_SIM_FOLDERS   = [ "Pm4" ]
LIST_SIM_RES       = [ "18", "36", "72" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
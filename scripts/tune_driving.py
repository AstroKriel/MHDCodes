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
from TheSimModule import SimParams
from TheLoadingModule import LoadFlashData, FileNames


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
  a = 10


## ###############################################################
## TUNE TUTBULENCE DRIVING FILE
## ###############################################################
class TurbDrvingFile():
  def __init__(self, filepath_sim):
    self.filepath_sim = filepath_sim
    dict_sim_inputs   = SimParams.readSimInputs(self.filepath_sim, bool_verbose=False)
    self.k_turb       = dict_sim_inputs["k_turb"]
    self.t_turb       = dict_sim_inputs["t_turb"]
    self.desired_Mach = dict_sim_inputs["desired_Mach"]
    self.current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    ## initialise program parameters, measured quantities, and flags
    self.relative_tol           = 0.05
    self.num_decimals_rounded   = 5
    self.list_old_Mach          = []
    self.list_old_coef          = []
    self.ave_Mach               = None
    self.std_Mach               = None
    self.old_driving_amplitude  = None
    self.new_driving_amplitude  = None
    self.bool_history_converged = False
    self.bool_Mach_converged    = False
    self.bool_repeating         = False

  def performRoutine(self):
    print("Tuning driving parameters in:", self.filepath_sim)
    ## make sure a driving history file exists
    if os.path.isfile(f"{self.filepath_sim}/{FileNames.FILENAME_DRIVING_HISTORY}"):
      self._checkIfAlreadyConverged()
    else: self._createDrivingHistory()
    ## check whether the driving paramaters need to be updated
    if not(self.bool_history_converged):
      self._measureMach()
      if self.bool_repeating:
        print("\t> Measured Mach number is too similar to a previous entry.")
        return
      if not(self.bool_Mach_converged):
        self._tuneDriving()
        self._reRunSimulation()
    else: print(f"\t> Driving parameters have already converged on the desired Mach = {self.desired_Mach}")

  def __round(self, value):
    return round(value, self.num_decimals_rounded)

  def __relErrLessTol(self, value_ref, value):
    return abs(value_ref - value) / value_ref < self.relative_tol

  def _createDrivingHistory(self):
    with open(f"{self.filepath_sim}/{FileNames.FILENAME_DRIVING_HISTORY}", "w") as fp:
      fp.write("## (0: DATE) (1: TIME) (2: PREV MACH) (3: PREV AMPLITUDE) (4: NEW AMPLITUDE)\n")
    print("\t> Created a driving history file")

  def _checkIfAlreadyConverged(self):
    with open(f"{self.filepath_sim}/{FileNames.FILENAME_DRIVING_HISTORY}", "r") as fp:
      for line in fp.readlines():
        if "#" in line: continue
        if len(line) == 0: continue
        old_Mach = float(line.split()[2])
        old_coef = float(line.split()[4])
        self.list_old_Mach.append(old_Mach)
        self.list_old_coef.append(old_coef)
        if ("converged" in line.lower()): self.bool_history_converged = True
    print("\t> Previous Mach numbers:",       self.list_old_Mach)
    print("\t> Previous driving amplitudes:", self.list_old_coef)

  def _measureMach(self):
    data_time, data_Mach = LoadFlashData.loadTurbData(
      filepath   = self.filepath_sim,
      quantity   = "Mach",
      t_turb     = self.t_turb,
      time_start = 0.1,
      time_end   = np.inf
    )
    ## check that there is sufficient data to look at
    if (len(data_time) > 0) and (data_time[-1] < 4):
      raise Exception("ERROR: time range is insufficient to tune driving parameters")
    elif len(data_time) == 0:
      raise Exception("ERROR: no simulation data")
    self.ave_Mach = self.__round(np.mean(data_Mach))
    self.std_Mach = self.__round(np.std(data_Mach))
    print(f"\t> Measured Mach = {self.ave_Mach} +/- {self.std_Mach}")
    self.bool_Mach_converged = self.__relErrLessTol(self.desired_Mach, self.ave_Mach)
    self.bool_repeating = any([
      self.__relErrLessTol(self.ave_Mach, old_Mach)
      for old_Mach in self.list_old_Mach
    ])

  def _tuneDriving(self):
    self.old_driving_amplitude = readDrivingAmplitude(self.filepath_sim)
    self.new_driving_amplitude = self.__round(self.old_driving_amplitude * self.desired_Mach / self.ave_Mach)
    print(f"\t> Tuning driving amplitude to achieve Mach = {self.desired_Mach}")
    print(f"\t\t Prev: {self.old_driving_amplitude}")
    print(f"\t\t New:  {self.new_driving_amplitude}")
    updateDrivingAmplitude(self.filepath_sim, self.new_driving_amplitude)
    updateDrivingHistory(
      filepath              = self.filepath_sim,
      current_time          = self.current_time,
      measured_Mach         = self.ave_Mach,
      old_driving_amplitude = self.old_driving_amplitude,
      new_driving_amplitude = self.new_driving_amplitude
    )

  def _removeOldData(self):
    removeFiles(self.filepath_sim, "Turb")
    removeFiles(self.filepath_sim, "stir.dat")
    removeFiles(self.filepath_sim, "sim_outputs.json")
    removeFiles(self.filepath_sim, "*.o")
    removeFiles(self.filepath_sim, "shell_sim.out00")

  def _reRunSimulation(self):
    ## submit simulation PBS job script
    print("\t> Submitting job to run simulation:")
    p = subprocess.Popen([ "qsub", FileNames.FILENAME_JOB_RUN_SIM ], cwd=self.filepath_sim)
    p.wait()


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  obj_tune_driving = TurbDrvingFile("/scratch/ek9/nk7952/Mach/10/144/")
  obj_tune_driving.performRoutine()


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
FILEPATH_BASE = "/scratch/ek9/nk7952/"


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
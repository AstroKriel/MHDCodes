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
class TurbDrvingFile():
  def __init__(self, filepath_sim):
    self.filepath_sim = filepath_sim
    dict_sim_inputs   = SimParams.readSimInputs(self.filepath_sim, bool_verbose=False)
    self.k_turb       = dict_sim_inputs["k_turb"]
    self.t_turb       = dict_sim_inputs["t_turb"]
    self.desired_Mach = dict_sim_inputs["desired_Mach"]
    self.current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    ## initialise program parameters, measured quantities, and flags
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
      self._loadData()
      self._measureMach()
      if self.bool_repeating:
        print("\t> Measured Mach number is too similar to a previous entry.")
        return
      if not(self.bool_Mach_converged):
        self._tuneDriving()
        self._removeOldData()
        self._reRunSimulation()
    else: print(f"\t> Driving parameters have already converged on the desired Mach = {self.desired_Mach}")

  def __round(self, value):
    return round(value, self.num_decimals_rounded)

  def __relErr(self, value_ref, value):
    return abs(value_ref - value) / value_ref

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

  def _loadData(self):
    ## load Mach data
    data_time, self.data_Mach = LoadFlashData.loadVIData(
      filepath   = self.filepath_sim,
      field_name = "mach",
      t_turb     = self.t_turb,
      time_start = 0.1,
      time_end   = np.inf
    )
    ## check that there is sufficient data to look at
    if (len(data_time) > 100) and (data_time[-1] < 4):
      raise Exception("ERROR: time range is insufficient to tune driving parameters")
    elif len(data_time) == 0:
      raise Exception("ERROR: no simulation data")
    ## load kinetic energy
    _, data_kin_energy = LoadFlashData.loadVIData(
      filepath   = self.filepath_sim,
      field_name = "kin",
      t_turb     = self.t_turb,
      time_start = 0.1,
      time_end   = np.inf
    )
    ## load magnetic energy
    _, data_mag_energy = LoadFlashData.loadVIData(
      filepath   = self.filepath_sim,
      field_name = "mag",
      t_turb     = self.t_turb,
      time_start = 0.1,
      time_end   = np.inf
    )
    ## compute energy ratio
    data_E_ratio = [
      mag_energy / kin_energy
      for mag_energy, kin_energy in zip(
        data_mag_energy,
        data_kin_energy
      )
    ]
    ## find saturated energy ratio
    time_start_sat  = 0.75 * data_time[-1]
    index_start_sat = WWLists.getIndexClosestValue(data_time, time_start_sat)
    index_end_sat   = len(data_time)-1
    E_ratio_sat     = np.mean(data_E_ratio[index_start_sat : index_end_sat])
    ## find indices associated with kinematic phase
    t_start_index    = WWLists.getIndexClosestValue(data_time, 5.0)
    E_growth_percent = E_ratio_sat / data_E_ratio[t_start_index]
    if E_growth_percent > 10**2:
      index_E_lo = WWLists.getIndexClosestValue(data_E_ratio, 10**(-8))
      index_E_hi = WWLists.getIndexClosestValue(data_E_ratio, E_ratio_sat/100)
      self.index_start_Mach = max([ t_start_index, min([ index_E_lo, index_E_hi ]) ])
      self.index_end_Mach   = max([ index_E_lo, index_E_hi ])
    else:
      self.index_start_Mach = index_start_sat
      self.index_end_Mach   = index_end_sat

  def _measureMach(self):
    ## measure Mach number statistics in kinematic phase
    self.ave_Mach = self.__round(np.mean(self.data_Mach[self.index_start_Mach : self.index_end_Mach]))
    self.std_Mach = self.__round(np.std(self.data_Mach[self.index_start_Mach : self.index_end_Mach]))
    print(f"\t> Measured Mach = {self.ave_Mach} +/- {self.std_Mach}")
    rel_Mach_err = self.__relErr(self.desired_Mach, self.ave_Mach)
    self.bool_Mach_converged = rel_Mach_err < 0.05
    print(f"\t> Measured Mach {100*rel_Mach_err:.3f}% off from desired Mach = {self.desired_Mach:.1f}")
    self.bool_repeating = any([
      self.__relErr(self.ave_Mach, old_Mach) < 0.01
      for old_Mach in self.list_old_Mach
    ])

  def _tuneDriving(self):
    self.old_driving_amplitude = readDrivingAmplitude(self.filepath_sim)
    self.new_driving_amplitude = self.__round(self.old_driving_amplitude * self.desired_Mach / self.ave_Mach)
    print(f"\t> Tuning driving amplitude to achieve Mach = {self.desired_Mach}")
    print(f"\t\t Prev: {self.old_driving_amplitude}")
    print(f"\t\t New:  {self.new_driving_amplitude}")
    if BOOL_CHECK_ONLY: return
    updateDrivingAmplitude(self.filepath_sim, self.new_driving_amplitude)
    updateDrivingHistory(
      filepath              = self.filepath_sim,
      current_time          = self.current_time,
      measured_Mach         = self.ave_Mach,
      old_driving_amplitude = self.old_driving_amplitude,
      new_driving_amplitude = self.new_driving_amplitude
    )

  def _removeOldData(self):
    if BOOL_CHECK_ONLY: return
    removeFiles(self.filepath_sim, "Turb")
    removeFiles(self.filepath_sim, "stir.dat")
    removeFiles(self.filepath_sim, "sim_outputs.json")
    removeFiles(self.filepath_sim, "shell_sim.out00")

  def _resetFlashInputFile(self):
    a = 10 # TODO

  def _reRunSimulation(self):
    if BOOL_CHECK_ONLY: return
    ## submit simulation PBS job script
    print("\t> Submitting job to run simulation:")
    p = subprocess.Popen([ "qsub", FileNames.FILENAME_JOB_RUN_SIM ], cwd=self.filepath_sim)
    p.wait()


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
    obj_tune_driving = TurbDrvingFile(sim_filepath)
    obj_tune_driving.performRoutine()
    print(" ")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BOOL_CHECK_ONLY = 0
BASEPATH        = "/scratch/ek9/nk7952/"

# ## PLASMA PARAMETER SET
# LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm3000" ]
# LIST_SONIC_REGIMES = [ "Mach0.3", "Mach5" ]
# LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
# LIST_SIM_RES       = [ "18", "36", "72", "144", "288", "576" ]

## MACH NUMBER SET
LIST_SUITE_FOLDERS = [ "Re300" ]
LIST_SONIC_REGIMES = [ "Mach5" ]
LIST_SIM_FOLDERS   = [ "Pm4" ]
LIST_SIM_RES       = [ "288" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
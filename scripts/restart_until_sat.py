#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np

## load user defined modules
from TheFlashModule import JobRunSim, SimParams, FileNames, LoadData
from TheUsefulModule import WWFnF


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class RestartSim():
  def __init__(self, filepath_sim_res):
    self.filepath_sim_res = filepath_sim_res
    self.dict_sim_inputs  = SimParams.readSimInputs(self.filepath_sim_res, False)
    self.max_num_t_turb   = self.dict_sim_inputs["max_num_t_turb"]
    self.t_turb           = self.dict_sim_inputs["t_turb"]
  
  def performRoutine(self):
    self._readLastFileIndex()
    print(self.last_chk_index, self.last_plt_index)
    self._readLastTimePoint()
    bool_reach_max_num_t_turb = (self.last_time_point / self.max_num_t_turb) > 0.95
    print(self.last_time_point, bool_reach_max_num_t_turb)
    # if bool_reach_max_num_t_turb: self._updateFiles()

  def _readLastFileIndex(self):
    ## define helper function
    def getLastFileIndex(filename_starts_with):
      list_filenames = WWFnF.getFilesInDirectory(
        directory             = self.filepath_sim_res,
        filename_starts_with  = filename_starts_with,
        filename_not_contains = "spect",
        loc_file_index        = -1,
      )
      last_index = max([
        int(filename.split("_")[-1])
        for filename in list_filenames
      ])
      return last_index
    ## get last file indices
    self.last_chk_index = getLastFileIndex(FileNames.FILENAME_FLASH_CHK_FILES)
    self.last_plt_index = getLastFileIndex(FileNames.FILENAME_FLASH_PLT_FILES)
    ## get simulation job index
    list_sim_outputs = [
      int(file.split(".out")[1])
      if len(file.split(".out")) > 1 else
      np.nan
      for file in os.listdir(self.filepath_sim_res)
      if file.startswith(FileNames.FILENAME_RUN_SIM_OUTPUT)
    ]
    if len(list_sim_outputs) > 0:
      self.last_run_index = np.nanmax(list_sim_outputs)
    else: self.last_run_index = 0

  def _readLastTimePoint(self):
    data_time, _ = LoadData.loadVIData(
      directory  = self.filepath_sim_res,
      field_name = "mach",
      t_turb     = self.t_turb,
      time_start = 2.0,
      time_end   = np.inf
    )
    self.last_time_point = max(data_time)

  def _updateFiles(self):
    ## update simulation job index
    obj_prep_sim = JobRunSim.JobRunSim(
      filepath_sim    = self.filepath_sim_res,
      dict_sim_inputs = self.dict_sim_inputs,
      run_index       = self.last_run_index
    )
    ## update chk and plt file indices in flash.par file


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
    print("Looking at:", sim_filepath)
    obj_tune_driving = RestartSim(sim_filepath)
    obj_tune_driving.performRoutine()
    print(" ")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BOOL_CHECK_ONLY = 1
BASEPATH        = "/scratch/ek9/nk7952/"

## PLASMA PARAMETER SET
LIST_SUITE_FOLDERS = [ "Rm3000" ]
LIST_SONIC_REGIMES = [ "Mach5" ]
LIST_SIM_FOLDERS   = [ "Pm1" ]
LIST_SIM_RES       = [ "18" ] # , "36", "72", "144"


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
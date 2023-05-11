#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np

## load user defined modules
from TheFlashModule import JobRunSim, SimParams, FileNames, LoadData
from TheUsefulModule import WWFnF, WWLists


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
    self._checkEmagSaturated()
    ## resimulate if unsaturated and the simulation has not yet completed 100 turnover-times
    if not(self.Emag_converged):
      self._updateFiles()
      self._restartSim()

  def _readLastFileIndex(self):
    ## define helper function
    def getLastFileIndex(filename_starts_with, sub_folder=""):
      list_filenames = WWFnF.getFilesInDirectory(
        directory             = f"{self.filepath_sim_res}/{sub_folder}/",
        filename_starts_with  = filename_starts_with,
        filename_not_contains = "spect",
        loc_file_index        = -1,
      )
      if len(list_filenames) == 0: return np.nan
      last_index = max([
        int(filename.split("_")[-1])
        for filename in list_filenames
      ])
      return last_index
    ## get last output file indices
    self.last_chk_index = getLastFileIndex(FileNames.FILENAME_FLASH_CHK_FILES)
    last_plt_index_sim  = getLastFileIndex(FileNames.FILENAME_FLASH_PLT_FILES)
    last_plt_index_plt  = getLastFileIndex(FileNames.FILENAME_FLASH_PLT_FILES, "plt")
    self.last_plt_index = int(np.nanmax([
      last_plt_index_sim,
      last_plt_index_plt
    ]))
    ## get latest simulation job index
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

  def _checkEmagSaturated(self):
    self.data_time, data_Emag = LoadData.loadVIData(
      directory  = self.filepath_sim_res,
      field_name = "mag",
      t_turb     = self.t_turb,
      time_start = 2.0,
      time_end   = np.inf
    )
    data_log_Eratio = np.log10(data_Emag)
    list_std = []
    for time_start in range(10, int(max(self.data_time)), 10):
      index_start = WWLists.getIndexClosestValue(self.data_time, time_start)
      index_end   = WWLists.getIndexClosestValue(self.data_time, time_start+5)
      data_window = data_log_Eratio[index_start : index_end]
      list_std.append(np.std(data_window))
    ## check that Emag has been saturated for 30 t_turb
    self.Emag_converged = all([
      val < 5e-2
      for val in list_std[-3:]
    ])


  def _updateFiles(self):
    ## update simulation job index
    obj_prep_sim = JobRunSim.JobRunSim(
      filepath_sim    = self.filepath_sim_res,
      dict_sim_inputs = self.dict_sim_inputs,
      run_index       = self.last_run_index+1
    )
    ## update chk and plt file indices in flash.par file
    ## make sure that the simulation can run for another 100 t_turb
    (max(self.data_time) / self.max_num_t_turb) > 0.95


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  list_sim_filepaths = SimParams.getListOfSimFilepaths(
    basepath           = PATH_SCRATCH,
    list_suite_folders = LIST_SUITE_FOLDERS,
    list_sonic_regimes = LIST_MACH_REGIMES,
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
PATH_SCRATCH    = "/scratch/ek9/nk7952/"
# PATH_SCRATCH    = "/scratch/jh2/nk7952/"

## PLASMA PARAMETER SET
LIST_SUITE_FOLDERS = [ "Rm3000" ]
LIST_MACH_REGIMES = [ "Mach5" ]
LIST_SIM_FOLDERS   = [ "Pm250" ]
LIST_SIM_RES       = [ "18", "144" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
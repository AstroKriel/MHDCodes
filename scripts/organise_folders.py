#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np

## load user defined modules
from TheUsefulModule import WWFnF, WWTerminal
from TheFlashModule import FileNames, SimParams, JobProcessFiles


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def runCommand(command):
  if BOOL_CHECK_ONLY:
    print(command)
  else: os.system(command)

def removeFiles(directory, filename_starts_with):
  list_files_in_directory = WWFnF.getFilesInDirectory(
    directory             = directory,
    filename_starts_with  = filename_starts_with
  )
  if len(list_files_in_directory) > 0:
    if BOOL_CHECK_ONLY:
      print(f"Note: there are {len(list_files_in_directory)} '{filename_starts_with}*' file(s) to remove")
    else:
      runCommand(f"rm {directory}/{filename_starts_with}*")
      print(f"\t> Removed {len(list_files_in_directory)} '{filename_starts_with}*' file(s)")
  else: print(f"\t> There are no '{filename_starts_with}*' files in:\n\t", directory)

def moveFiles(
    directory_from, directory_to,
    filename_contains     = None,
    filename_not_contains = None
  ):
  list_files_in_directory= WWFnF.getFilesInDirectory(
    directory             = directory_from,
    filename_starts_with  = "Turb",
    filename_contains     = filename_contains,
    filename_not_contains = filename_not_contains
  )
  if len(list_files_in_directory) > 0:
    runCommand(f"mv {directory_from}/*{filename_contains}* {directory_to}/.")
    print(f"\t> Moved {len(list_files_in_directory)} '*{filename_contains}*' files")
    print("\t\tFrom:", directory_from)
    print("\t\tTo:", directory_to)
  else: print(f"\t> There are no '*{filename_contains}*' files in:\n\t", directory_from)

def countFiles(
    directory,
    filename_contains     = None,
    filename_not_contains = None
  ):
  list_files_in_directory= WWFnF.getFilesInDirectory(
    directory             = directory,
    filename_starts_with  = "Turb",
    filename_contains     = filename_contains,
    filename_not_contains = filename_not_contains
  )
  num_files = len(list_files_in_directory)
  print(f"\t> There are {num_files} '*{filename_contains}*' files in:\n\t", directory)
  return num_files, list_files_in_directory

def renameFiles(directory, old_phrase, new_phrase):
  list_files_in_directory= WWFnF.getFilesInDirectory(
    directory         = directory,
    filename_contains = old_phrase
  )
  if len(list_files_in_directory) > 0:
    if BOOL_CHECK_ONLY: command_arg = "-n"
    else: command_arg = ""
    WWTerminal.runCommand(
      command   = f"rename {command_arg} {old_phrase} {new_phrase} *",
      directory = directory
    )
    print(f"\t> Renamed {len(list_files_in_directory)} '*{old_phrase}*' file(s) to '*{new_phrase}*'")
  else: print(f"\t> There are no '*{old_phrase}*' files in:\n\t", directory)


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class ReorganiseSimFolder():
  def __init__(self, filepath_sim):
    self.filepath_sim   = filepath_sim
    self.filepath_plt   = WWFnF.createFilepath([filepath_sim, "plt"])
    self.filepath_spect = WWFnF.createFilepath([filepath_sim, "spect"])
    ## create sub-folders
    WWFnF.createFolder(self.filepath_plt,   bool_verbose=False)
    WWFnF.createFolder(self.filepath_spect, bool_verbose=False)

  def removeExtraFiles(self):
    print("Removing extraneous files...")
    ## remove extraneous files
    removeFiles(self.filepath_sim, "core.flash4_nxb_")
    removeFiles(self.filepath_sim, "Turb_proj_")
    removeFiles(self.filepath_sim, "Turb_slice_")
    ## count number of chk-files
    list_chk_files = WWFnF.getFilesInDirectory(
      directory            = self.filepath_sim,
      filename_starts_with = "Turb_hdf5_chk_"
    )
    ## if there are many chk-files
    num_chk_files_to_keep = 1
    num_chk_files_removed = 0
    if len(list_chk_files) > num_chk_files_to_keep:
      ## cull all but the final few chk-files
      for file_index in range(len(list_chk_files) - num_chk_files_to_keep):
        runCommand(f"rm {self.filepath_sim}/{list_chk_files[file_index]}")
        num_chk_files_removed += 1
      ## indicate the number of chk-files removed
      print(f"\t> Removed {num_chk_files_removed} 'chk' files from:\n\t", self.filepath_sim)

  def movePltFiles(self):
    print("Working with plt-files...")
    if not os.path.exists(self.filepath_plt):
      raise Exception("Error: 'plt' sub-folder does not exist")
    ## move plt-files from simulation folder to plt sub-folder
    moveFiles(
      directory_from        = self.filepath_sim,
      directory_to          = self.filepath_plt,
      filename_contains     = "plt_",
      filename_not_contains = "spect_"
    )
    ## count number of plt-files in the plt sub-folder
    self.num_plt_files_in_plt, self.list_files_in_plt = countFiles(
      directory             = self.filepath_plt,
      filename_contains     = "plt_",
      filename_not_contains = "spect_"
    )

  def moveSpectFiles(self):
    print("Working with spect-files...")
    if not os.path.exists(self.filepath_spect):
      raise Exception("Error: 'spect' sub-folder does not exist")
    ## check that there are spectra files to move
    self.num_spect_files_in_plt, _ = countFiles(
      directory         = self.filepath_plt,
      filename_contains = "spect_"
    )
    if self.num_spect_files_in_plt == 0: return
    ## move spectra from the simulation folder to spect sub-folder
    moveFiles(
      directory_from    = self.filepath_sim,
      directory_to      = self.filepath_spect,
      filename_contains = "spect_"
    )
    ## move spectra from plt sub-folder to spect sub-folder
    moveFiles(
      directory_from    = self.filepath_plt,
      directory_to      = self.filepath_spect,
      filename_contains = "spect_"
    )
    ## count number of spectra in the spect sub-folder
    countFiles(
      directory         = self.filepath_spect,
      filename_contains = "spect_"
    )
    ## rename current spectra files
    renameFiles(
      directory  = self.filepath_spect,
      old_phrase = "dset_curx_cury_curz",
      new_phrase = "current"
    )

  def checkNumFiles(self, list_filepaths):
    print("Checking processed files...")
    bool_submit_job = False
    file_start_index = np.nan
    for spect_field in [ "_current", "_mag", "_vel", "_sqrtrho", "_varrho" ]:
      num_files_in_spect, list_files_in_spect = countFiles(
        directory         = self.filepath_spect,
        filename_contains = spect_field
      )
      if num_files_in_spect < self.num_plt_files_in_plt:
        print(f"Note: {num_files_in_spect} of {self.num_plt_files_in_plt} {spect_field} files have been computed and moved")
        bool_submit_job = True
        if num_files_in_spect > 0:
          list_spect_subfilenames = [
            filename.split("_spect")[0]
            for filename in list_files_in_spect
          ]
          list_files_unprocessed = [
            int(filename.split("_")[-1]) # filename: Turb_hdf5_plt_cnt_NUMBER
            if filename not in list_spect_subfilenames else
            np.nan
            for filename in self.list_files_in_plt
          ]
          min_file_indices = np.nanmin(list_files_unprocessed)
          file_start_index = np.nanmin([ file_start_index, min_file_indices ])
    if bool_submit_job and not(BOOL_CHECK_ONLY):
      list_filepaths.append(self.filepath_sim)
      if np.isnan(file_start_index): file_start_index = 0
      dict_sim_inputs = SimParams.readSimInputs(self.filepath_sim, False)
      JobProcessFiles.JobProcessFiles(
        filepath_plt     = self.filepath_plt,
        dict_sim_inputs  = dict_sim_inputs,
        file_start_index = int(file_start_index)
      )
      job_name = FileNames.FILENAME_PROCESS_PLT_JOB
      print("Submitting job:", job_name)
      WWTerminal.runCommand(
        command   = f"qsub {job_name}",
        directory = self.filepath_plt
      )


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  if BOOL_CHECK_ONLY: print("Running in debug mode.")
  list_sim_filepaths = SimParams.getListOfSimFilepaths(
    basepath           = PATH_SCRATCH,
    list_suite_folders = LIST_SUITE_FOLDERS,
    list_sonic_regimes = LIST_SONIC_REGIMES,
    list_sim_folders   = LIST_SIM_FOLDERS,
    list_sim_res       = LIST_SIM_RES
  )
  list_filepaths = []
  for filepath_sim in list_sim_filepaths:
    print("Reorganising:", filepath_sim)
    obj_sim_folder = ReorganiseSimFolder(filepath_sim)
    obj_sim_folder.removeExtraFiles()
    obj_sim_folder.movePltFiles()
    obj_sim_folder.moveSpectFiles()
    obj_sim_folder.checkNumFiles(list_filepaths)
    print(" ")
  if len(list_filepaths) > 0:
    print("Jobs have been submitted in the following simulation diectories:")
    print("\t> " + "\n\t> ".join(list_filepaths))


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
BOOL_CHECK_ONLY = 1
PATH_SCRATCH    = "/scratch/ek9/nk7952/"
# PATH_SCRATCH    = "/scratch/jh2/nk7952/"

# ## PLASMA PARAMETER SET
# LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm3000" ]
# LIST_SONIC_REGIMES = [ "Mach5" ]
# LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
# LIST_SIM_RES       = [ "18", "36", "72", "144", "288", "576" ]

# ## RERUN RM=3000, PM=1
# LIST_SUITE_FOLDERS = [ "Rm3000" ]
# LIST_SONIC_REGIMES = [ "Mach5" ]
# LIST_SIM_FOLDERS   = [ "Pm1" ]
# LIST_SIM_RES       = [ "18", "36", "72", "144", "288" ]

# ## MACH NUMBER SET
# LIST_SUITE_FOLDERS = [ "Rm3000" ]
# LIST_SONIC_REGIMES = [ "Mach0.3", "Mach1", "Mach10" ]
# LIST_SIM_FOLDERS   = [ "Pm1", "Pm5", "Pm10", "Pm125" ]
# LIST_SIM_RES       = [ "18", "36", "72", "144", "288" ]

# ## BOTTLENECK RUN
# LIST_SUITE_FOLDERS = [ "Re2000" ]
# LIST_SONIC_REGIMES = [ "Mach0.3", "Mach5" ]
# LIST_SIM_FOLDERS   = [ "Pm5" ]
# LIST_SIM_RES       = [ "18", "36", "72", "144", "288" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
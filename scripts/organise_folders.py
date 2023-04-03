#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys

## load user defined modules
from TheFlashModule import SimParams
from TheUsefulModule import WWFnF


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def runCommand(command):
  if BOOL_CHECK_ONLY: print(command)
  else: os.system(command)

def removeFiles(filepath, filename_starts_with):
  list_files_in_filepath = WWFnF.getFilesFromFilepath(
    filepath             = filepath,
    filename_starts_with = filename_starts_with
  )
  if len(list_files_in_filepath) > 0:
    runCommand(f"rm {filepath}/{filename_starts_with}*")
    print(f"\t> Removed {len(list_files_in_filepath)} '{filename_starts_with}*' file(s)")
  else: print(f"\t> There are no '{filename_starts_with}*' files in:\n\t", filepath)

def moveFiles(
    filepath_from, filepath_to,
    filename_contains     = None,
    filename_not_contains = None
  ):
  list_files_in_filepath = WWFnF.getFilesFromFilepath(
    filepath              = filepath_from,
    filename_starts_with  = "Turb",
    filename_contains     = filename_contains,
    filename_not_contains = filename_not_contains
  )
  if len(list_files_in_filepath) > 0:
    runCommand(f"mv {filepath_from}/*{filename_contains}* {filepath_to}/.")
    print(f"\t> Moved {len(list_files_in_filepath)} '*{filename_contains}*' files")
    print("\t\tFrom:", filepath_from)
    print("\t\tTo:", filepath_to)
  else: print(f"\t> There are no '*{filename_contains}*' files in:\n\t", filepath_from)

def countFiles(
    filepath,
    filename_contains     = None,
    filename_not_contains = None
  ):
  list_files_in_filepath = WWFnF.getFilesFromFilepath(
    filepath              = filepath,
    filename_starts_with  = "Turb",
    filename_contains     = filename_contains,
    filename_not_contains = filename_not_contains
  )
  num_files = len(list_files_in_filepath)
  print(f"\t> There are {num_files} '*{filename_contains}*' files in:\n\t", filepath)
  return num_files

def renameFiles(filepath, old_filename_ends_with, new_filename_ends_with):
  list_files_in_filepath = WWFnF.getFilesFromFilepath(
    filepath           = filepath,
    filename_ends_with = old_filename_ends_with
  )
  if len(list_files_in_filepath) > 0:
    for old_filename in list_files_in_filepath:
      new_filename = old_filename.replace(old_filename_ends_with, new_filename_ends_with)
      runCommand(f"rm {filepath}/{old_filename} {filepath}/{new_filename}")
    print(f"\t> Renamed {len(list_files_in_filepath)} '*{old_filename_ends_with}' file(s) to '*{new_filename_ends_with}'")
  else: print(f"\t> There are no '*{old_filename_ends_with}' files in:\n\t", filepath)


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
    list_chk_files = WWFnF.getFilesFromFilepath(
      filepath             = self.filepath_sim,
      filename_starts_with = "Turb_hdf5_chk_"
    )
    ## if there are many chk-files
    num_chk_files_to_keep = 3
    num_chk_files_removed = 0
    if len(list_chk_files) > num_chk_files_to_keep:
      ## cull chk-files at early simulation times
      for file_index in range(len(list_chk_files) - num_chk_files_to_keep):
        runCommand(f"rm {self.filepath_sim}/{list_chk_files[file_index]}")
        num_chk_files_removed += 1
      ## reflect the number of files removed
      print(f"\t> Removed {num_chk_files_removed} 'chk' files from:\n\t", self.filepath_sim)

  def movePltFiles(self):
    print("Working with plt-files...")
    if not os.path.exists(self.filepath_plt):
      raise Exception("ERROR: 'plt' sub-folder does not exist")
    ## move plt-files from simulation folder to plt sub-folder
    moveFiles(
      filepath_from         = self.filepath_sim,
      filepath_to           = self.filepath_plt,
      filename_contains     = "plt",
      filename_not_contains = "spect"
    )
    ## count number of plt-files in the plt sub-folder
    countFiles(
      filepath              = self.filepath_plt,
      filename_contains     = "plt",
      filename_not_contains = "spect"
    )

  def moveSpectFiles(self):
    print("Working with spect-files...")
    if not os.path.exists(self.filepath_plt):
      raise Exception("ERROR: 'spect' sub-folder does not exist")
    ## move spect-files from simulation folder to spect sub-folder
    moveFiles(
      filepath_from     = self.filepath_sim,
      filepath_to       = self.filepath_spect,
      filename_contains = "spect"
    )
    ## move spect-files from plt sub-folder to spect sub-folder
    moveFiles(
      filepath_from     = self.filepath_plt,
      filepath_to       = self.filepath_spect,
      filename_contains = "spect"
    )
    ## count number of spect-files in the spect sub-folder
    countFiles(
      filepath          = self.filepath_spect,
      filename_contains = "spect"
    )
    ## rename current spectra
    renameFiles(
      filepath               = self.filepath_spect,
      old_filename_ends_with = "dset_curx_cury_curz.dat",
      new_filename_ends_with = "current.dat"
    )


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def reorganiseSimFolder(filepath_sim_res, **kwargs):
  print("Reorganising:", filepath_sim_res)
  obj_sim_folder = ReorganiseSimFolder(filepath_sim_res)
  obj_sim_folder.removeExtraFiles()
  obj_sim_folder.movePltFiles()
  obj_sim_folder.moveSpectFiles()
  print(" ")

def main():
  if BOOL_CHECK_ONLY: print("Running in debug mode.")
  SimParams.callFuncForAllSimulations(
    func               = reorganiseSimFolder,
    basepath           = BASEPATH,
    list_suite_folders = LIST_SUITE_FOLDERS,
    list_sonic_regimes = LIST_SONIC_REGIMES,
    list_sim_folders   = LIST_SIM_FOLDERS,
    list_sim_res       = LIST_SIM_RES
  )


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
BOOL_CHECK_ONLY = 1
BASEPATH         = "/scratch/ek9/nk7952/"

# ## PLASMA PARAMETER SET
# LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm3000" ]
# LIST_SONIC_REGIMES = [ "Mach0.3", "Mach5" ]
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
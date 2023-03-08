#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys

## load user defined modules
from TheUsefulModule import WWFnF


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def removeFiles(filepath_files, file_name_starts_with):
  list_files_in_filepath = WWFnF.getFilesFromFilepath(
    filepath            = filepath_files,
    filename_startswith = file_name_starts_with
  )
  if len(list_files_in_filepath) > 0:
    os.system(f"rm {filepath_files}/{file_name_starts_with}*")
    print(f"\t> Removed {len(list_files_in_filepath)} '{file_name_starts_with}*' files.")
  else: print(f"\t> There are no '{file_name_starts_with}*' files in:\n\t", filepath_files)
  print(" ")

def moveFiles(
    filepath_files_from, filepath_files_to,
    filename_contains     = None,
    filename_not_contains = None
  ):
  list_files_in_filepath = WWFnF.getFilesFromFilepath(
    filepath              = filepath_files_from,
    filename_startswith   = "Turb",
    filename_contains     = filename_contains,
    filename_not_contains = filename_not_contains
  )
  if len(list_files_in_filepath) > 0:
    os.system(f"mv {filepath_files_from}/*_{filename_contains}* {filepath_files_to}/.")
    print(f"\t> Moved {len(list_files_in_filepath)} '*_{filename_contains}*' files to:\n\t", filepath_files_to)
  else: print(f"\t> There are no '*_{filename_contains}*' files in:\n\t", filepath_files_from)
  print(" ")

def countFiles(
    filepath_files,
    filename_contains     = None,
    filename_not_contains = None
  ):
  list_files_in_filepath = WWFnF.getFilesFromFilepath(
    filepath              = filepath_files,
    filename_startswith   = "Turb",
    filename_contains     = filename_contains,
    filename_not_contains = filename_not_contains
  )
  num_files = len(list_files_in_filepath)
  print(f"\t> There are {num_files} '*_{filename_contains}*' files in:\n\t", filepath_files)
  print(" ")
  return num_files


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class ReorganiseSimFolder():
  def __init__(self, filepath_sim):
    self.filepath_sim   = filepath_sim
    self.filepath_plt   = f"{filepath_sim}/plt/"
    self.filepath_spect = f"{filepath_sim}/spect/"

  def removeExtraFiles(self):
    print("Removing extraneous files...")
    ## remove extraneous files
    removeFiles(self.filepath_sim, "core.flash4_nxb_")
    removeFiles(self.filepath_sim, "Turb_proj_")
    removeFiles(self.filepath_sim, "Turb_slice_")
    ## count number of chk-files
    list_chk_files = WWFnF.getFilesFromFilepath(
      filepath            = self.filepath_sim,
      filename_startswith = "Turb_hdf5_chk_"
    )
    ## if there are many chk-files
    num_chk_files_keep = 3
    num_files_removed  = 0
    if len(list_chk_files) > num_chk_files_keep:
      ## cull chk-files at early simulation times
      for file_index in range(len(list_chk_files) - num_chk_files_keep):
        os.system(f"rm {self.filepath_sim}/{list_chk_files[file_index]}")
        num_files_removed += 1
      ## reflect the number of files removed
      print(f"\t> Removed {num_files_removed} 'chk' files from:\n\t", self.filepath_sim)
      print(" ")

  def movePltFiles(self):
    print("Working with plt-files...")
    ## create plt sub-folder
    WWFnF.createFilepath(self.filepath_plt)
    ## move plt-files from simulation folder to plt sub-folder
    moveFiles(
      filepath_files_from   = self.filepath_sim,
      filepath_files_to     = self.filepath_plt,
      filename_contains     = "plt",
      filename_not_contains = "spect"
    )
    ## count number of plt-files in the plt sub-folder
    countFiles(
      filepath_files        = self.filepath_plt,
      filename_contains     = "plt",
      filename_not_contains = "spect"
    )

  def moveSpectFiles(self):
    print("Working with spect-files...")
    ## create spect sub-folder
    WWFnF.createFilepath(self.filepath_spect)
    ## move spect-files from simulation folder to spect sub-folder
    moveFiles(
      filepath_files_from = self.filepath_sim,
      filepath_files_to   = self.filepath_spect,
      filename_contains   = "spect"
    )
    ## move spect-files from plt sub-folder to spect sub-folder
    moveFiles(
      filepath_files_from = self.filepath_plt,
      filepath_files_to   = self.filepath_spect,
      filename_contains   = "spect"
    )
    ## count number of spect-files in the spect sub-folder
    countFiles(
      filepath_files    = self.filepath_spect,
      filename_contains = "spect"
    )


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## LOOK AT EACH SIMULATION FOLDER
  ## ------------------------------
  ## loop over the simulation suites
  for suite_folder in LIST_SUITE_FOLDER:

    ## loop over the simulation folders
    for sim_folder in LIST_SIM_FOLDER:

      ## CHECK THE SIMULATION EXISTS
      ## ---------------------------
      filepath_sim = WWFnF.createFilepath([
        BASEPATH, suite_folder, SONIC_REGIME, sim_folder
      ])
      if not os.path.exists(filepath_sim): continue
      str_message = f"Looking at suite: {suite_folder}, sim: {sim_folder}, regime: {SONIC_REGIME}"
      print(str_message)
      print("=" * len(str_message))
      print(" ")

      ## loop over the different resolution runs
      for sim_res in LIST_SIM_RES:

        ## CHECK THE RESOLUTION RUN EXISTS
        ## -------------------------------
        filepath_sim_res = f"{filepath_sim}/{sim_res}/"
        ## check that the filepath exists
        if not os.path.exists(filepath_sim_res): continue
        print(f"Looking at Nres = {sim_res}")

        ## evaluate function
        obj_sim_folder = ReorganiseSimFolder(filepath_sim_res)
        # obj_sim_folder.removeExtraFiles()
        # obj_sim_folder.movePltFiles()
        # obj_sim_folder.moveSpectFiles()

        ## create an empty line after each suite
        print(" ")
      print(" ")
    print(" ")


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
BOOL_DEBUG        = 0
BASEPATH          = "/scratch/ek9/nk7952/"
SONIC_REGIME      = "super_sonic"

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
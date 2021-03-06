#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys

from TheUsefulModule import WWFnF


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def removeFiles(filepath_sim, file_name_starts_with):
  ## get instances of file category
  list_files = WWFnF.getFilesFromFolder(
    folder_directory = filepath_sim,
    str_startswith   = file_name_starts_with
  )
  ## if there are any files, remove them
  if len(list_files) > 0:
    ## remove files
    os.system("rm {}".format(
      filepath_sim + "/{}*".format(file_name_starts_with)
    ))
    print("\t> Removed {} '{}' files.".format(
      len(list_files),
      file_name_starts_with
    ))
  else: print("\t> There are no '{}' files.".format(
    file_name_starts_with
  ))


def createFolderNMoveFiles(
    filepath_sim,
    file_n_folder_name,
    file_name_not_conatins = None
  ):
  ## create filepath for sub-folder
  sim_filepath_sub_folder = WWFnF.createFilepath([filepath_sim, file_n_folder_name])
  ## get the list of file instances in the base simulation folder
  list_files_in_mainfolder = WWFnF.getFilesFromFolder(
    filepath         = filepath_sim,
    str_startswith   = "Turb",
    str_contains     = file_n_folder_name,
    str_not_contains = file_name_not_conatins
  )
  ## if the sub-folder does not exist
  if not os.path.exists(sim_filepath_sub_folder):
    ## create the sub-folder
    os.system("mkdir {}".format(
      sim_filepath_sub_folder
    ))
  else: print("\t> The sub-folder '{}' already exists.".format(
    file_n_folder_name
  ))
  ## check inside the sub-folder
  if os.path.exists(sim_filepath_sub_folder):
    ## get the list of files instances in the sub-folder
    list_files_in_subfolder = WWFnF.getFilesFromFolder(
      filepath         = sim_filepath_sub_folder,
      str_startswith   = "Turb",
      str_contains     = file_n_folder_name,
      str_not_contains = file_name_not_conatins
    )
  else: list_files_in_subfolder = []
  ## if the sub-folder is empty (assumes that the files have not been transferred yet)
  if len(list_files_in_subfolder) == 0:
    ## if there are files in the base simulation folder, then move them into the sub-folder
    if len(list_files_in_mainfolder) > 0:
      os.system("mv {} {}".format(
        filepath_sim + "/*_"+file_n_folder_name+"*",
        sim_filepath_sub_folder
      ))
      print("\t> Moved {} '{}' files.".format(
        len(list_files_in_mainfolder),
        file_n_folder_name
      ))
    ## if there are no files of this type, then notify the user
    else: print("\t> There are no '{}' files in: {}".format(
      file_n_folder_name,
      sim_filepath_sub_folder
    ))


## ###############################################################
## MAIN PROGRAM
## ###############################################################
BASEPATH     = "/scratch/ek9/nk7952/"
SONIC_REGIME = "super_sonic"

def main():
  ## ##############################
  ## LOOK AT EACH SIMULATION FOLDER
  ## ##############################
  ## loop over the simulation suites
  for suite_folder in [
      "Re10", "Re500", "Rm3000"
    ]: # "Re10", "Re500", "Rm3000", "keta"

    ## loop over the different resolution runs
    for sim_res in [
        "18", "36", "72", "144", "288", "576"
      ]: # "18", "36", "72", "144", "288", "576"

      ## print to the terminal what suite is being looked at
      str_msg = "Looking at suite: {}, Nres = {}".format( suite_folder, sim_res )
      print(str_msg)
      print("=" * len(str_msg))

      ## loop over the simulation folders
      for sim_folder in [
          "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"
        ]: # "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"

        ## #######################################
        ## CHECK THAT THE SIMULATION FOLDER EXISTS
        ## #######################################
        ## check that the simulation filepath exists
        filepath_sim = WWFnF.createFilepath([
          BASEPATH, suite_folder, sim_res, SONIC_REGIME, sim_folder
        ])
        if not os.path.exists(filepath_sim):
          print(filepath_sim, "does not exist.")
          continue
        print("Looking at:", filepath_sim)

        ## ################################
        ## REMOVE UNUSED SIMULATION OUTPUTS
        ## ################################
        ## remove 'core.flash4_nxb...' files
        removeFiles(
          filepath_sim,
          file_name_starts_with = "core.flash4_nxb"
        )
        ## remove 'proj' files
        removeFiles(
          filepath_sim,
          file_name_starts_with = "Turb_proj"
        )
        ## remove 'slice' files
        removeFiles(
          filepath_sim,
          file_name_starts_with = "Turb_slice"
        )

        ## #########################################
        ## CREATE SUB-FOLDERS + MOVE FILES INTO THEM
        ## #########################################
        ## create 'spect' folder if it does not exist
        createFolderNMoveFiles(
          filepath_sim,
          file_n_folder_name = "spect"
        )
        ## create 'plt' folder if it does not exist
        createFolderNMoveFiles(
          filepath_sim,
          file_n_folder_name     = "plt",
          file_name_not_conatins = "spect"
        )
        ## create filepath to sub-folders
        filepath_plt   = filepath_sim + "/plt"
        filepath_spect = filepath_sim + "/spect"
        ## get the number of 'spect' files in 'plt' folder
        list_spect_files_in_plt_folder = WWFnF.getFilesFromFolder(
          filepath     = filepath_plt,
          str_contains = "spect",
          str_endswith = ".dat"
        )
        ## move 'spect' files from 'plt' to 'spect' folder if there are any
        if len(list_spect_files_in_plt_folder) > 0:
          os.system("mv {} {}".format(
            filepath_plt + "/*_spect*",
            filepath_spect + "/."
          ))
          print("\t> Moved {} 'spect' files to 'spect' sub-folder.".format(
            len(list_spect_files_in_plt_folder)
          ))

        ## ######################################
        ## CULL EXTRA SIMULATION CHECKPOINT FILES
        ## ######################################
        ## get number of 'chk' files
        list_chk_files = WWFnF.getFilesFromFolder(
          filepath       = filepath_sim,
          str_startswith = "Turb_hdf5_chk_"
        )
        ## if there are many 'chk' files
        if len(list_chk_files) > 3:
          ## cull 'chk' files at early simulation times
          num_files_removed = 0
          for file_index in range(len(list_chk_files) - 3):
            ## don't remove 'chk' files with index beyond 97
            if int(list_chk_files[file_index].split("_")[-1]) > 96:
              break
            ## remove file
            os.system("rm {}".format(
              filepath_sim + "/" + list_chk_files[file_index]
            ))
            ## increment the number of files removed
            num_files_removed += 1
          ## indicate the number of files removed
          print("\t> Removed {} '{}' files.".format(
            num_files_removed,
            "chk"
          ))

        ## create an empty line after each suite
        print(" ")
      print(" ")
    print(" ")

## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()

## END OF PROGRAM
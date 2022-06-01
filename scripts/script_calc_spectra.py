#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import numpy as np

## load new user defined modules
from TheUsefulModule import WWArgparse, WWFnF


#################################################################
## PREPARE TERMINAL/WORKSPACE/CODE
#################################################################
os.system("clear")  # clear terminal window


## ###############################################################
## PROCESSING FUNCTION
## ###############################################################
def funcProcessPltFile(filenames, num_proc):
  for file_name in filenames:
    print("--------- Looking at: " + file_name + " -----------------------------------")
    os.system("mpirun -np {} spectra_mpi {} -vels_spect -mags_spect".format(
      num_proc,
      file_name
    ))
    print(" ")


## ###############################################################
## DEFINE MAIN PROGRAM
## ###############################################################
def main():
  ## ###########################
  ## COMMAND LINE ARGUMENT INPUT
  ## ###########################
  parser = WWArgparse.MyParser()
  ## ------------------- DEFINE OPTIONAL ARGUMENTS
  args_opt = parser.add_argument_group(description="Optional processing arguments:")
  ## ------------------- DEFINE OPTIONAL ARGUMENTS
  args_opt.add_argument("-check_only", type=WWArgparse.str2bool, default=False, required=False, nargs="?", const=True)
  args_opt.add_argument("-file_start", type=int, default=0,      required=False)
  args_opt.add_argument("-file_end",   type=int, default=np.Inf, required=False)
  args_opt.add_argument("-num_proc",   type=str, default="8",    required=False)
  ## ------------------- DEFINE REQUIRED ARGUMENTS
  args_req = parser.add_argument_group(description="Required processing arguments:")
  args_req.add_argument("-data_path",  type=str, required=True)
  ## ---------------------------- OPEN ARGUMENTS
  args = vars(parser.parse_args())
  ## ---------------------------- SAVE PARAMETERS
  bool_check_only = args["check_only"] # only check for and process unprocessed (plt) spectra files
  directory_data  = args["data_path"]  # directory where data is stored
  file_start      = args["file_start"] # first file to process
  file_end        = args["file_end"]   # last file to process
  num_proc        = args["num_proc"]   # number of processors

  ## ###############################
  ## PRINT JOB PARAMETERS TO CONSOLE
  ## ###############################
  print("Began running the spectra code in folder: " + directory_data)
  print("First file index to process: "              + str(file_start))
  print("Last file index to process: "               + str(file_end))
  print("Number of processors: "                     + str(num_proc))
  print(" ")

  ## #########################
  ## PROCESS ALL THE PLT-FILES
  ## #########################
  # loop over the data directory and compute spectra files
  list_filenames = WWFnF.getFilesFromFolder(
    filepath           = directory_data,
    str_contains       = "Turb_hdf5_plt_cnt_",
    str_not_contains   = "spect",
    file_index_placing = -1,
    file_start_index   = file_start,
    file_end_index     = file_end
  )
  if not(bool_check_only):
    print("There are {} files to process.".format( len(list_filenames) ))
    if len(list_filenames) > 0:
      print("These files are:")
      print("\t> " + "\n\t> ".join(list_filenames))
      print(" ")
    ## loop over and process file names
      funcProcessPltFile(list_filenames, num_proc)

  ## ########################################
  ## CHECK WHICH PLT-FILES WERE NOT PROCESSED
  ## ########################################
  ## now check which spectra files have successfully been computed
  list_filenames_spect_mag = WWFnF.getFilesFromFolder(
    filepath           = directory_data,
    str_contains       = "Turb_hdf5_plt_cnt_",
    str_endswith       = "spect_mags.dat",
    file_index_placing = -3,
    file_start_index   = file_start,
    file_end_index     = file_end
  )
  list_filenames_spect_vel = WWFnF.getFilesFromFolder(
    filepath           = directory_data,
    str_contains       = "Turb_hdf5_plt_cnt_",
    str_endswith       = "spect_vels.dat",
    file_index_placing = -3,
    file_start_index   = file_start,
    file_end_index     = file_end
  )
  ## initialise list of files to (re)process
  list_filenames_redo = []
  ## check if there are any files that were not been processed
  for file_name in list_filenames:
    ## for each plt file
    bool_mags_exists = False
    bool_vels_exists = False
    ## do not (re)process the plt file if both the velocity and magnetic spectra files have already been processed
    if (file_name + "_spect_mags.dat") in list_filenames_spect_mag:
      bool_mags_exists = True
    if (file_name + "_spect_vels.dat") in list_filenames_spect_vel:
      bool_vels_exists = True
    ## (re)process the plt file if either the magnetic or velocity spectra files are missing
    if not(bool_mags_exists) or not(bool_vels_exists):
      list_filenames_redo.append(file_name)
  ## if there are any plt files to (re)process
  if len(list_filenames_redo) > 0:
    print("There are {} plt files to (re)process.".format( len(list_filenames_redo) ))
    print("These files are:")
    print("\t" + "\n\t".join(list_filenames_redo)) # print file names
    print(" ")
    ## loop over plt file names and processes them
    print("Processing these plt files again...")
    funcProcessPltFile(list_filenames_redo, num_proc)
  else:
    print("There are no more plt files to process.")
  print(" ")
  print("Finished running the spectra code.")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()


## END OF PROGRAM
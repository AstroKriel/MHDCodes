#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import argparse
import numpy as np

## load new user defined modules
from OldModules.the_useful_library import *


#################################################################
## PREPARE TERMINAL/WORKSPACE/CODE
#################################################################
os.system("clear")  # clear terminal window


## ###############################################################
## FUNCTIONS
## ###############################################################
def processHDF5(filenames):
  for file_name in filenames:
    print("--------- Looking at: " + file_name + " -----------------------------------")
    os.system("mpirun -np " + num_proc + " spectra_mpi_sp " + file_name)
    print(" ")


## ###############################################################
## COMMAND LINE ARGUMENT INPUT
## ###############################################################
ap = argparse.ArgumentParser(description="A bunch of input arguments")
## ------------------- DEFINE OPTIONAL ARGUMENTS
ap.add_argument("-check_only", type=str2bool, default=False, required=False, nargs="?", const=True)
ap.add_argument("-file_start", type=int, default=0,      required=False)
ap.add_argument("-file_end",   type=int, default=np.Inf, required=False)
ap.add_argument("-num_proc",   type=str, default="8",    required=False)
## ------------------- DEFINE REQUIRED ARGUMENTS
ap.add_argument("-base_path",  type=str, required=True)
## ---------------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ---------------------------- SAVE PARAMETERS
bool_check_only  = args["check_only"] # only check for & process missing (hdf5) spectra files
folder_directory = args["base_path"]  # base folder_directory to data
file_start       = args["file_start"] # first file to process
file_end         = args["file_end"]   # last file to process
num_proc         = args["num_proc"]   # number of processors
## ---------------------------- ADJUST ARGUMENTS
## remove the trailing "/" from the input folder_directory
if folder_directory.endswith("/"): folder_directory = folder_directory[:-1]
## replace any "//" with "/"
folder_directory = folder_directory.replace("//", "/")
## ---------------------------- START CODE
print("Began running the spectra code in folder: " + folder_directory)
print("First file index to process: "              + str(file_start))
print("Last file index to process: "               + str(file_end))
print("Number of processors: "                     + str(num_proc))
print(" ")


## ###############################################################
## PROCESS DATA
## ###############################################################
# loop over folder_directory and execute spectra for each file
filenames = getFilesFromFolder(
  folder_directory,
  str_contains       = "Turb_hdf5_plt_cnt_",
  str_not_contains   = "spect",
  file_index_placing = -1,
  file_start_index   = file_start,
  file_end_index     = file_end
)
print("There are " + str(len(filenames)) + " files to process.")
if len(filenames) > 0:
  print("These files are:")
  print("\t> " + "\n\t> ".join(filenames))
  print(" ")
  ## loop over and process file names
  if not(bool_check_only):
    processHDF5(filenames)


## ###############################################################
## CHECK EVERYTHING WAS PROCESSED OKAY
## ###############################################################
## now check which spectra files exist in the directory
filenames_spect_mag = getFilesFromFolder(
  folder_directory,
  str_contains       = "Turb_hdf5_plt_cnt_",
  str_endswith       = "spect_mags.dat",
  file_index_placing = -3,
  file_start_index   = file_start,
  file_end_index     = file_end
)
filenames_spect_vel = getFilesFromFolder(
  folder_directory,
  str_contains       = "Turb_hdf5_plt_cnt_",
  str_endswith       = "spect_vels.dat",
  file_index_placing = -3,
  file_start_index   = file_start,
  file_end_index     = file_end
)
## initialise list of files to re-process
filenames_redo = []
## check if there are any files that haven"t been processed properly
for file_name in filenames:
  ## for each hdf5 file, check if there exists magnetic and velocity spectra output files
  bool_mags_exists = False
  bool_vels_exists = False
  ## check if the file file has been analysed then don"t look at it again
  if (file_name + "_spect_mags.dat") in filenames_spect_mag:
    bool_mags_exists = True
  if (file_name + "_spect_vels.dat") in filenames_spect_vel:
    bool_vels_exists = True
  ## if either the magnetic or velocity files don"t exist, then re-process the hdf5 file
  if (not(bool_mags_exists) or not(bool_vels_exists)):
    filenames_redo.append(file_name)
## if there are any files to process
if len(filenames_redo) > 0:
  print("There were " + str(len(filenames_redo)) + " files processed incorrectly.")
  print("These files were:")
  print("\t" + "\n\t".join(filenames_redo)) # print file names
  print(" ")
  ## loop over file names and processes them
  print("Processing these files again...")
  processHDF5(filenames_redo)
else: print("There are no more spectra files to process.")
print(" ")
print("Finished running the spectra code.")


## END OF PROGRAM
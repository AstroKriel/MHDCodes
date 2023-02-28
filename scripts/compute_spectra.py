#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import numpy as np

## load new user defined modules
from TheUsefulModule import WWArgparse, WWFnF


## ###############################################################
## PREPARE TERMINAL/WORKSPACE/CODE
## ###############################################################
os.system("clear")  # clear terminal window


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def processAllPltFiles(list_filenames_plt, num_proc):
  for filename_plt in list_filenames_plt:
    print(f"--------- Looking at: {filename_plt} -----------------------------------")
    os.system(f"mpirun -np {num_proc} spectra_mpi {filename_plt} -types 1 2")
    print(" ")


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class CalcSpectraFiles():
  def __init__(
      self,
      filepath_data, num_proc, file_start, file_end, bool_check_only
    ):
    self.filepath_data   = filepath_data
    self.num_proc        = num_proc
    self.file_start      = file_start
    self.file_end        = file_end
    self.bool_check_only = bool_check_only
    
  def performRoutines(self):
    self.__getPltFiles()
    if not(self.bool_check_only): self.__processFiles()
    self.__checkAllFilesProcessed()
    self.__reprocessFiles()
    print("Finished running the spectra code")

  def __getPltFiles(self):
    self.list_filenames_plt = WWFnF.getFilesFromFilepath(
      filepath              = self.filepath_data,
      filename_contains     = "plt",
      filename_not_contains = "spect",
      loc_file_index        = -1,
      file_start_index      = self.file_start,
      file_end_index        = self.file_end
    )

  def __processFiles(self):
    ## loop over and compute spectra for plt-files in the data directory
    print(f"There are {len(self.list_filenames_plt)} files to process")
    if len(self.list_filenames_plt) > 0:
      print("\t> " + "\n\t> ".join(self.list_filenames_plt), "\n")
      print("Processing plt-files...")
      processAllPltFiles(self.list_filenames_plt, self.num_proc)

  def __checkAllFilesProcessed(self):
    ## check all spectra files have been successfully computed
    list_filenames_spect_mag = WWFnF.getFilesFromFilepath(
      filepath          = self.filepath_data,
      filename_contains = "plt",
      filename_endswith = "spect_mags.dat",
      loc_file_index    = -3,
      file_start_index  = self.file_start,
      file_end_index    = self.file_end
    )
    list_filenames_spect_vel = WWFnF.getFilesFromFilepath(
      filepath          = self.filepath_data,
      filename_contains = "plt",
      filename_endswith = "spect_vels.dat",
      loc_file_index    = -3,
      file_start_index  = self.file_start,
      file_end_index    = self.file_end
    )
    ## initialise list of files to (re)process
    self.list_filenames_to_process = []
    ## check if there are any files that were not been processed
    for filename_plt in self.list_filenames_plt:
      bool_mags_exists = False
      bool_vels_exists = False
      if f"{filename_plt}_spect_mags.dat" in list_filenames_spect_mag:
        bool_mags_exists = True
      if f"{filename_plt}_spect_vels.dat" in list_filenames_spect_vel:
        bool_vels_exists = True
      ## (re)process plt-file if either spectra files are missing
      if not(bool_mags_exists) or not(bool_vels_exists):
        self.list_filenames_to_process.append(filename_plt)

  def __reprocessFiles(self):
    ## if there are any plt-files to (re)process
    if len(self.list_filenames_to_process) > 0:
      print(f"There are {len(self.list_filenames_to_process)} plt-files to (re)process:")
      print("\t" + "\n\t> ".join(self.list_filenames_to_process)) # print file names
      print(" ")
      ## loop over processes plt-files
      print("(Re)processing plt-files...")
      processAllPltFiles(self.list_filenames_to_process, self.num_proc)
    else: print("There are no more plt-files to process.")
    print(" ")


## ################################
## GET COMMAND LINE INPUT ARGUMENTS
## ################################
def getInputArgs():
  parser = WWArgparse.MyParser(description="Calculate kinetic and magnetic energy spectra.")
  ## ------------------- DEFINE OPTIONAL ARGUMENTS
  args_opt = parser.add_argument_group(description="Optional processing arguments:")
  ## ------------------- DEFINE OPTIONAL ARGUMENTS
  args_opt.add_argument("-check_only", **WWArgparse.opt_bool_arg, default=False)
  args_opt.add_argument("-file_start", **WWArgparse.opt_arg, type=int, default=0)
  args_opt.add_argument("-file_end",   **WWArgparse.opt_arg, type=int, default=np.inf)
  args_opt.add_argument("-num_proc",   **WWArgparse.opt_arg, type=int, default=8)
  ## ------------------- DEFINE REQUIRED ARGUMENTS
  args_req = parser.add_argument_group(description="Required processing arguments:")
  args_req.add_argument("-data_path",  type=str, required=True, help="type: %(type)s")
  ## ---------------------------- OPEN ARGUMENTS
  args = vars(parser.parse_args())
  ## ---------------------------- SAVE PARAMETERS
  filepath_data   = args["data_path"]
  file_start      = args["file_start"]
  file_end        = args["file_end"]
  num_proc        = args["num_proc"]
  bool_check_only = args["check_only"]
  ## ---------------------------- START CODE
  print("Began running the spectra code in folder: " + filepath_data)
  print("First file index to process: "              + str(file_start))
  print("Last file index to process: "               + str(file_end))
  print("Number of processors: "                     + str(num_proc))
  print("Only process unprocessed files: "           + str(bool_check_only))
  print(" ")
  ## ---------------------------- RETURN ARGS
  return filepath_data, file_start, file_end, num_proc, bool_check_only


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  filepath_data, file_start, file_end, num_proc, bool_check_only = getInputArgs()
  ## #################
  ## CALCULATE SPECTRA
  ## #################
  obj_calc_spectra = CalcSpectraFiles(filepath_data, num_proc, file_start, file_end, bool_check_only)
  obj_calc_spectra.performRoutines()


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF PROGRAM
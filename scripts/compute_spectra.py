#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import numpy as np

## load new user defined modules
from TheUsefulModule import WWArgparse, WWFnF


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def processAllPltFiles(list_filenames_plt, num_proc):
  for filename_plt in list_filenames_plt:
    print(f"--------- Looking at: {filename_plt} -----------------------------------", flush=True)
    ## current components
    print("> Computing current components...", flush=True)
    os.system(f"mpirun -np {num_proc} derivative_var {filename_plt} -current")
    ## other key components
    print("\n> Computing other components...", flush=True)
    os.system(f"mpirun -np {num_proc} derivative_var {filename_plt} -MHD_scales -divv -vort -dissipation")
    ## velocity, magnetic, and kinetic energy spectra
    print("\n> Computing velocity + magnetic + kinetic energy spectra...", flush=True)
    os.system(f"mpirun -np {num_proc} spectra_mpi {filename_plt} -types 1 2 7") # vels, mags, sqrtrho
    ## current spectrum
    print("\n> Computing current spectrum...", flush=True)
    os.system(f"mpirun -np {num_proc} spectra_mpi {filename_plt} -types 0 -dsets curx cury curz")
    print("\n", flush=True)


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
    self._getPltFiles()
    if not(self.bool_check_only): self._processFiles()
    self._checkAllFilesProcessed()
    self._reprocessFiles()
    print("Finished running the spectra code")

  def _getPltFiles(self):
    self.list_filenames_plt = WWFnF.getFilesFromFilepath(
      filepath              = self.filepath_data,
      filename_contains     = "plt",
      filename_not_contains = "spect",
      loc_file_index        = -1,
      file_start_index      = self.file_start,
      file_end_index        = self.file_end
    )

  def _processFiles(self):
    ## loop over and compute spectra for plt-files in the data directory
    print(f"There are {len(self.list_filenames_plt)} files to process")
    if len(self.list_filenames_plt) > 0:
      print("\t> " + "\n\t> ".join(self.list_filenames_plt), "\n")
      print("Processing plt-files...")
      processAllPltFiles(self.list_filenames_plt, self.num_proc)

  def _checkAllFilesProcessed(self):
    ## check all spectra files have been successfully computed
    list_filenames_spect_mag = WWFnF.getFilesFromFilepath(
      filepath           = self.filepath_data,
      filename_contains  = "plt",
      filename_ends_with = "spect_mags.dat",
      loc_file_index     = -3,
      file_start_index   = self.file_start,
      file_end_index     = self.file_end
    )
    list_filenames_spect_vel = WWFnF.getFilesFromFilepath(
      filepath           = self.filepath_data,
      filename_contains  = "plt",
      filename_ends_with = "spect_vels.dat",
      loc_file_index     = -3,
      file_start_index   = self.file_start,
      file_end_index     = self.file_end
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

  def _reprocessFiles(self):
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


## ###############################################################
## GET COMMAND LINE INPUT ARGUMENTS
## ###############################################################
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
  num_proc        = args["num_proc"]
  file_start      = args["file_start"]
  file_end        = args["file_end"]
  bool_check_only = args["check_only"]
  ## ---------------------------- START CODE
  print("Running spectra code in folder: "                    + filepath_data)
  print("First file index to process: "                       + str(file_start))
  print("Last file index to process: "                        + str(file_end))
  print("Number of processors: "                              + str(num_proc))
  print("Should the program only process unprocessed files: " + str(bool_check_only))
  print(" ", flush=True)
  ## ---------------------------- RETURN ARGS
  return filepath_data, num_proc, file_start, file_end, bool_check_only


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  filepath_data, num_proc, file_start, file_end, bool_check_only = getInputArgs()
  obj_calc_spectra = CalcSpectraFiles(filepath_data, num_proc, file_start, file_end, bool_check_only)
  obj_calc_spectra.performRoutines()


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF PROGRAM
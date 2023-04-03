#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import subprocess
import numpy as np

## load user defined routines
from h5del import h5del

## load new user defined modules
from TheUsefulModule import WWArgparse, WWFnF


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def printLine(mssg):
  print(mssg, flush=True)


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class ProcessPltFiles():
  def __init__(self):
    self._getInputArgs()

  def performRoutines(self):
    self._getPltFiles()
    if not(self.bool_check_only) and (len(self.list_filenames_to_process) > 0):
      self._processPltFiles(self.list_filenames_to_process)
    else: printLine("There are no plt-files to process.\n")
    self._checkAllPltFilesProcessed()
    if len(self.list_filenames_to_reprocess) > 0:
      self._processPltFiles(self.list_filenames_to_reprocess)
    else: printLine("There are no plt-files to re-process.\n")
    printLine("Finished processing files.")

  def _getInputArgs(self):
    parser = WWArgparse.MyParser(description="Calculate kinetic and magnetic energy spectra.")
    ## ------------------- DEFINE OPTIONAL ARGUMENTS
    args_opt = parser.add_argument_group(description="Optional processing arguments:")
    args_opt.add_argument("-check_only",        **WWArgparse.opt_bool_arg, default=False)
    args_opt.add_argument("-h5del_dsets",       **WWArgparse.opt_bool_arg, default=False)
    args_opt.add_argument("-compute_all_dsets", **WWArgparse.opt_bool_arg, default=False)
    args_opt.add_argument("-file_start",        **WWArgparse.opt_arg, type=int, default=0)
    args_opt.add_argument("-file_end",          **WWArgparse.opt_arg, type=int, default=np.inf)
    args_opt.add_argument("-num_procs",         **WWArgparse.opt_arg, type=int, default=8)
    ## ------------------- DEFINE REQUIRED ARGUMENTS
    args_req = parser.add_argument_group(description="Required processing arguments:")
    args_req.add_argument("-data_path",  type=str, required=True, help="type: %(type)s")
    ## open arguments
    args = vars(parser.parse_args())
    ## save parameters
    self.bool_check_only        = args["check_only"]
    self.bool_h5del_dsets       = args["h5del_dsets"]
    self.bool_compute_all_dsets = args["compute_all_dsets"]
    self.file_start             = args["file_start"]
    self.file_end               = args["file_end"]
    self.num_procs              = args["num_procs"]
    self.filepath_data          = args["data_path"]
    ## report input parameters
    printLine("Processing in directory: "    + self.filepath_data)
    printLine("Processing from file index: " + str(self.file_start))
    printLine("Processing upto file index: " + str(self.file_end))
    printLine("Number of processors: "       + str(self.num_procs))
    if self.bool_check_only:        printLine("Will only process unprocessed files.")
    if self.bool_compute_all_dsets: printLine("Will compute extended list of datasets.")
    if self.bool_h5del_dsets:       printLine("Will cull extraneous datasets from hdf5-files")
    printLine(" ")

  def _getPltFiles(self):
    self.list_filenames_to_process = WWFnF.getFilesFromFilepath(
      filepath              = self.filepath_data,
      filename_contains     = "plt",
      filename_not_contains = "spect",
      loc_file_index        = 4,
      file_start_index      = self.file_start,
      file_end_index        = self.file_end
    )

  def _checkAllPltFilesProcessed(self):
    ## filename structure: Turb_hdf5_plt_cnt_NUMBER
    ## check all spectra files have been successfully computed
    list_filenames_spect_mag = WWFnF.getFilesFromFilepath(
      filepath           = self.filepath_data,
      filename_contains  = "plt",
      filename_ends_with = "spect_mags.dat",
      loc_file_index     = 4,
      file_start_index   = self.file_start,
      file_end_index     = self.file_end
    )
    list_filenames_spect_vel = WWFnF.getFilesFromFilepath(
      filepath           = self.filepath_data,
      filename_contains  = "plt",
      filename_ends_with = "spect_vels.dat",
      loc_file_index     = 4,
      file_start_index   = self.file_start,
      file_end_index     = self.file_end
    )
    list_filenames_spect_current = WWFnF.getFilesFromFilepath(
      filepath           = self.filepath_data,
      filename_contains  = "plt",
      filename_ends_with = "spect_dset_curx_cury_curz.dat",
      loc_file_index     = 4,
      file_start_index   = self.file_start,
      file_end_index     = self.file_end
    )
    ## initialise list of files to (re)process
    self.list_filenames_to_reprocess = []
    ## check if there are any files that were not been processed
    for filename_plt in self.list_filenames_to_process:
      bool_mags_exists    = f"{filename_plt}_spect_mags.dat" in list_filenames_spect_mag
      bool_vels_exists    = f"{filename_plt}_spect_vels.dat" in list_filenames_spect_vel
      bool_current_exists = f"{filename_plt}_spect_dset_curx_cury_curz.dat" in list_filenames_spect_current
      ## (re)process plt-file if any spectra files are missing
      if any([ not(bool_mags_exists), not(bool_vels_exists), not(bool_current_exists) ]):
        self.list_filenames_to_reprocess.append(filename_plt)

  def _processPltFiles(self, list_filenames):
    ## helper function
    def runCommand(command):
      p = subprocess.Popen(
        [ f"mpirun -np {self.num_procs} {command}" ],
        shell=True, cwd=self.filepath_data
      )
      p.wait()
    ## process each plt-file
    printLine(f"There are {len(list_filenames)} files to (re)process")
    printLine("\t> " + "\n\t> ".join(list_filenames))
    printLine("Processing plt-files...")
    for filename in list_filenames:
      printLine(f"--------- Looking at: {filename} -----------------------------------")
      ## compute current components
      printLine("> Processing current components (J = curl of B)...")
      runCommand(f"derivative_var {filename} -current")
      ## compute current spectrum
      printLine("\n> Processing current (J) spectrum...")
      runCommand(f"spectra_mpi {filename} -types 0 -dsets curx cury curz")
      ## compute velocity, magnetic, and kinetic energy spectra
      printLine("\n> Processing velocity and magnetic power spectra + kinetic energy spectrum...")
      runCommand(f"spectra_mpi {filename} -types 1 2 7") # vels, mags, sqrtrho
      ## compute other interesting datasets
      if self.bool_compute_all_dsets:
        printLine("\n> Processing (B cross J), (B dot J), magnetic tension, (div of U), vorticity, viscous dissipation...")
        runCommand(f"derivative_var {filename} -MHD_scales -divv -vort -dissipation")
      ## delete unused components in plt-file
      if self.bool_h5del_dsets:
        list_dsets = [
          ## B cross J (MHD scales)
          "mXcx", "mXcy", "mXcz",
          ## B dot J (MHD scales)
          "mdc",
          # magnetic tension (MHD scales)
          "tenx", "teny", "tenz",
          ## div of vel field
          "divv",
          ## vorticity
          "vorticity_x", "vorticity_y", "vorticity_z",
          ## viscous dissipation
          "diss_rate"
        ]
        h5del(filename, list_dsets, self.filepath_data)
      ## add empty space
      printLine("\n")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  obj_calc_spectra = ProcessPltFiles()
  obj_calc_spectra.performRoutines()


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF PROGRAM
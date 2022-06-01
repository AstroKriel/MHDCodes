#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import subprocess

from os import path

## load old user defined modules
from TheUsefulModule import WWArgparse, WWFnF


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## #############################
  ## DEFINE COMMAND LINE ARGUMENTS
  ## #############################
  parser = WWArgparse.MyParser(description="Fit kinetic and magnetic energy spectra.")
  ## ------------------- DEFINE OPTIONAL ARGUMENTS
  args_opt = parser.add_argument_group(description='Optional processing arguments:')
  ## define typical input requirements
  opt_bool_arg = {
    "required":False, "default":False, "action":"store_true",
    "help":"type: bool, default: %(default)s"
  }
  opt_arg = {
    "required":False, "metavar":"",
    "help":"type: %(type)s, default: %(default)s",
  }
  req_arg = {
    "required":True, "help":"type: %(type)s"
  }
  ## define directory inputs
  args_opt.add_argument("-sub_folder",   type=str, default="",            **opt_arg)
  args_opt.add_argument("-sonic_regime", type=str, default="super_sonic", **opt_arg)
  ## ------------------- DEFINE REQUIRED ARGUMENTS
  args_req = parser.add_argument_group(description='Required processing arguments:')
  ## define inputs
  args_req.add_argument("-base_path",   type=str,            **req_arg)
  args_req.add_argument("-sim_suites",  type=str, nargs="+", **req_arg)
  args_req.add_argument("-sim_res",     type=str, nargs="+", **req_arg)
  args_req.add_argument("-sim_folders", type=str, nargs="+", **req_arg)
  ## define job name input
  args_req.add_argument("-job_name", type=str, required=True)

  ## #########################
  ## INTERPRET INPUT ARGUMENTS
  ## #########################
  ## ---------------------------- OPEN ARGUMENTS
  args = vars(parser.parse_args())
  ## ---------------------------- SAVE PARAMETERS
  ## job name
  job_name           = args["job_name"]
  ## directory information
  filepath_base      = args["base_path"]
  list_suite_folders = args["sim_suites"]
  list_sim_res       = args["sim_res"]
  sonic_regime       = args["sonic_regime"]
  list_sim_folders   = args["sim_folders"]
  sub_folder         = args["sub_folder"]

  ## ####################
  ## PROCESS MAIN PROGRAM
  ## ####################
  ## loop over the simulation suites
  for suite_folder in list_suite_folders:
    ## loop over the different resolution runs
    for sim_res in list_sim_res:
      ## print to the terminal what suite is being looked at
      str_msg = "Looking at suite: {}, Nres = {}".format( suite_folder, sim_res )
      print(str_msg)
      print("=" * len(str_msg))
      ## loop over the simulation folders
      for sim_folder in list_sim_folders:

        ## ##################################
        ## CHECK THE SIMULATION FOLDER EXISTS
        ## ##################################
        ## create filepath to simulation folder (on GADI)
        filepath_sim = WWFnF.createFilepath([
          filepath_base, suite_folder, sim_res, sonic_regime, sim_folder, sub_folder
        ])
        ## check that the simulation filepath exists
        if not path.exists(filepath_sim):
          print(filepath_sim, "does not exist.")
          continue

        ## #########################
        ## CHECK THE JOB FILE EXISTS
        ## #########################
        ## check that the job exists in the folder
        if not path.isfile(filepath_sim + "/" + job_name):
          print(job_name, "does not exist in", filepath_sim)
          continue
        ## indicate which folder is being worked on
        print("Looking at: {}".format(filepath_sim))
        print("\t> Submitting the simulation job:")
        p = subprocess.Popen([ "qsub", job_name ], cwd=filepath_sim)
        p.wait()

        ## clear line if things have been printed
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
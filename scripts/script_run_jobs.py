#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import subprocess

from os import path

## load old user defined modules
from OldModules.the_useful_library import *


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
    ## #############################
    ## DEFINE COMMAND LINE ARGUMENTS
    ## #############################
    parser = MyParser()
    ## ------------------- DEFINE OPTIONAL ARGUMENTS
    args_opt = parser.add_argument_group(description='Optional processing arguments:')
    ## define directory inputs
    args_opt.add_argument("-sub_folder",   required=False, type=str, default="")
    args_opt.add_argument("-sonic_regime", required=False, type=str, default="super_sonic")
    ## ------------------- DEFINE REQUIRED ARGUMENTS
    args_req = parser.add_argument_group(description='Required processing arguments:')
    ## define inputs
    args_req.add_argument("-base_path",   type=str, required=True)
    args_req.add_argument("-sim_suites",  type=str, required=True, nargs="+")
    args_req.add_argument("-sim_res",     type=str, required=True, nargs="+")
    args_req.add_argument("-sim_folders", type=str, required=True, nargs="+")
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
                filepath_sim = createFilepath([
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
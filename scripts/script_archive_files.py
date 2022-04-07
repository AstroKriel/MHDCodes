#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys

from os import path

## user defined libraries
from OldModules.the_useful_library import *

def main():
    parser   = MyParser()
    args_req = parser.add_argument_group(description="Required processing arguments:")
    args_req.add_argument("-suite_folder", type=str, required=True, nargs="+")
    args = vars(parser.parse_args())
    list_suite_folder = args["suite_folder"]

    list_sim_res = [
        "288", "576"
    ]
    list_sim_folders = [
        "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm25", "Pm50", "Pm125", "Pm250"
    ]

    for suite_folder in list_suite_folder:
        for sim_res in list_sim_res:
            for sim_folder in list_sim_folders:
                ## create filepath to simulation folder (on GADI)
                sim_filepath_data = createFilepath(["/scratch/ek9/nk7952", suite_folder, sim_res, "sub_sonic", sim_folder, "plt"])
                ## check that the filepath exists
                if not path.exists(sim_filepath_data):
                    print(sim_filepath_data, "does not exist.")
                    continue
                # ## create TAR file for data
                # tar_filename = suite_folder+"_"+sim_res+"_"+sim_folder+".tar"
                # os.system("tar -cvf {:s} {:s}/Turb_hdf5_plt_cnt_*".format(
                #     tar_filename,
                #     sim_filepath_data
                # ))
                ## archive data
                os.chdir(sim_filepath_data) # change the directory
                os.system("pwd") # check the directory
                os.system("archive.py -i Turb_hdf5_plt_cnt_*") # archive data
            print(" ")
        print(" ")

## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
    main()
    sys.exit()

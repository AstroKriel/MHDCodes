#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys

from os import path

## user defined libraries
from OldModules.the_useful_library import *


## ###############################################################
## USER DEFINED FUNCTIONS
## ###############################################################
def removeFiles(sim_filepath_data, file_name_starts_with):
    ## get instances of file category
    list_files = getFilesFromFolder(
        folder_directory = sim_filepath_data,
        str_startswith   = file_name_starts_with
    )
    ## if there are any files, remove them
    if len(list_files) > 0:
        ## remove files
        os.system("rm {}".format(
            sim_filepath_data + "/{}*".format(file_name_starts_with)
        ))
        print("\t> Removed {} '{}' files.".format(
            len(list_files),
            file_name_starts_with
        ))
    else: print("\t> There are no '{}' files.".format(file_name_starts_with))

def createFolderNMoveFiles(
        sim_filepath_data,
        file_n_folder_name,
        file_name_not_conatins = None
    ):
    ## create filepath for sub-folder
    sim_filepath_sub_folder = createFilepath([sim_filepath_data, file_n_folder_name])
    ## get the list of file instances in the base simulation folder
    list_files_in_mainfolder = getFilesFromFolder(
        folder_directory = sim_filepath_data,
        str_startswith   = "Turb",
        str_contains     = file_n_folder_name,
        str_not_contains = file_name_not_conatins
    )
    ## if the sub-folder does not exist
    if not path.exists(sim_filepath_sub_folder):
        ## create the sub-folder
        os.system("mkdir {}".format(
            sim_filepath_sub_folder
        ))
    else: print("\t> The sub-folder '{}' already exists.".format(file_n_folder_name))
    ## check inside the sub-folder
    if os.path.exists(sim_filepath_sub_folder):
        ## get the list of files instances in the sub-folder
        list_files_in_subfolder = getFilesFromFolder(
            folder_directory = sim_filepath_sub_folder,
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
                sim_filepath_data + "/*_"+file_n_folder_name+"*",
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
def main():
    ## #############################
    ## DEFINE COMMAND LINE ARGUMENTS
    ## #############################
    parser = MyParser()
    ## ------------------- DEFINE INPUT ARGUMENTS
    args_input = parser.add_argument_group(description="Processing arguments:")
    args_input.add_argument("-base_path",    type=str, required=True)
    args_input.add_argument("-sim_suites",   type=str, required=True, nargs="+")
    args_input.add_argument("-sim_res",      type=str, required=True, nargs="+")
    args_input.add_argument("-sonic_regime", type=str, required=False, default="super_sonic")
    args_input.add_argument("-sim_folders",  type=str, required=True, nargs="+")

    ## #########################
    ## INTERPRET INPUT ARGUMENTS
    ## #########################
    ## ---------------------------- OPEN ARGUMENTS
    args = vars(parser.parse_args())
    ## ---------------------------- SAVE PARAMETERS
    filepath_base      = args["base_path"]
    list_suite_folders = args["sim_suites"]
    list_sim_res       = args["sim_res"]
    sonic_regime       = args["sonic_regime"]
    list_sim_folders   = args["sim_folders"]

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
                sim_filepath_data = createFilepath([
                    filepath_base, suite_folder, sim_res, sonic_regime, sim_folder
                ])
                ## check that the filepath exists
                if not path.exists(sim_filepath_data):
                    print(sim_filepath_data, "does not exist.")
                    continue
                ## indicate which folder is being worked on
                print("Looking at: " + sim_filepath_data)

                ## ##################################
                ## REMOVING UNUSED SIMULATION OUTPUTS
                ## ##################################
                ## remove 'core.flash4_nxb...' files
                removeFiles(
                    sim_filepath_data,
                    file_name_starts_with = "core.flash4_nxb"
                )
                ## remove 'proj' files
                removeFiles(
                    sim_filepath_data,
                    file_name_starts_with = "Turb_proj"
                )
                ## remove 'slice' files
                removeFiles(
                    sim_filepath_data,
                    file_name_starts_with = "Turb_slice"
                )

                ## #############################################
                ## CREATING NECESSARY SUB-FOLDERS + MOVING FILES
                ## #############################################
                ## create 'spect' folder if it does not exist
                createFolderNMoveFiles(
                    sim_filepath_data,
                    file_n_folder_name = "spect"
                )
                ## create 'plt' folder if it does not exist
                createFolderNMoveFiles(
                    sim_filepath_data,
                    file_n_folder_name = "plt",
                    file_name_not_conatins = "spect"
                )
                ## get the number of 'spect' files in 'plt' folder
                list_spect_files_in_plt_folder = getFilesFromFolder(
                    folder_directory = sim_filepath_data + "/plt",
                    str_contains = "spect",
                    str_endswith = ".dat"
                )
                ## move 'spect' files from 'plt' folder if there are any
                if len(list_spect_files_in_plt_folder) > 0:
                    os.system("mv {} {}".format(
                        sim_filepath_data + "/plt/*_spect*",
                        sim_filepath_data + "/spect/."
                    ))
                    print("\t> Moved {} 'spect' files to 'spect' sub-folder.".format(len(list_spect_files_in_plt_folder)))

                ## ##################################
                ## REMOVE SIMULATION CHECKPOINT FILES
                ## ##################################
                ## get number of 'chk' files
                list_chk_files = getFilesFromFolder(
                    folder_directory = sim_filepath_data,
                    str_startswith = "Turb_hdf5_chk_"
                )
                ## if there are many 'chk' files
                if len(list_chk_files) > 3:
                    ## cull 'chk' files at early simulation times
                    num_files_removed = 0
                    for file_index in range(len(list_chk_files) - 3):
                        ## don't remove 'chk' files with time index beyond 97
                        if int(list_chk_files[file_index].split("_")[-1]) > 96:
                            break
                        ## remove file
                        os.system("rm {}".format(
                            sim_filepath_data + "/" + list_chk_files[file_index]
                        ))
                        ## increment the number of files removed
                        num_files_removed += 1
                    print("\t> Removed {} '{}' files.".format(
                        num_files_removed,
                        "chk"
                    ))

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

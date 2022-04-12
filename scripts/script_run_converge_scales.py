#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys

## plotting stuff
import matplotlib.pyplot as plt

from os import path

## load old user defined modules
from OldModules.the_useful_library import *


## ###############################################################
## PREPARE WORKSPACE
##################################################################
os.system("clear") # clear terminal window
plt.ioff()
plt.switch_backend("agg") # use a non-interactive plotting backend

SPECTRA_NAME = "spectra_obj_mixed.pkl"
MAC_VIS_FOLDER = "vis_mixed"

BOOL_DOWNLOAD_MP4 = False
BOOL_DOWNLOAD_SPECTRA = True


## ###############################################################
## FUNCTIONS
## ###############################################################
def funcCheckSpectraObjNeedsUpdate(mac_filepath_data):
    ## load spectra object
    spectra_obj = loadPickleObject(
        mac_filepath_data,
        SPECTRA_NAME,
        bool_check = True,
        bool_hide_updates = True
    )
    ## if no file
    if spectra_obj == -1:
        return True
    # ## if file is not up to date
    # if "01/12/2021" not in spectra_obj.date_analysed:
    #     return True
    ## otherwise, do not update
    else: return False


## ###############################################################
## DEFINE MAIN PROGRAM
## ###############################################################
def main():
    filepath_base = "/scratch/ek9/nk7952"
    sub_folder = "spect"

    ## keta
    os.system("plot_spectra_2_resolution.py -base_path {}/{}/ -sim_res {} -sim_folder {} -sub_folder {}".format(
        filepath_base,
        "keta",
        "72 144 288",
        "Pm25",
        sub_folder
    ))
    os.system("plot_spectra_2_resolution.py -base_path {}/{}/ -sim_res {} -sim_folder {} -sub_folder {}".format(
        filepath_base,
        "keta",
        "72 144 288",
        "Pm50",
        sub_folder
    ))
    os.system("plot_spectra_2_resolution.py -base_path {}/{}/ -sim_res {} -sim_folder {} -sub_folder {}".format(
        filepath_base,
        "keta",
        "72 144 288",
        "Pm125",
        sub_folder
    ))
    os.system("plot_spectra_2_resolution.py -base_path {}/{}/ -sim_res {} -sim_folder {} -sub_folder {}".format(
        filepath_base,
        "keta",
        "72 144 288",
        "Pm250",
        sub_folder
    ))

    ## Re10
    os.system("plot_spectra_2_resolution.py -base_path {}/{}/ -sim_res {} -sim_folder {} -sub_folder {}".format(
        filepath_base,
        "Re10",
        "72 144 288 576",
        "Pm25",
        sub_folder
    ))
    os.system("plot_spectra_2_resolution.py -base_path {}/{}/ -sim_res {} -sim_folder {} -sub_folder {}".format(
        filepath_base,
        "Re10",
        "72 144 288 576",
        "Pm50",
        sub_folder
    ))
    os.system("plot_spectra_2_resolution.py -base_path {}/{}/ -sim_res {} -sim_folder {} -sub_folder {}".format(
        filepath_base,
        "Re10",
        "72 144 288 576",
        "Pm125",
        sub_folder
    ))
    os.system("plot_spectra_2_resolution.py -base_path {}/{}/ -sim_res {} -sim_folder {} -sub_folder {}".format(
        filepath_base,
        "Re10",
        "72 144 288 576",
        "Pm250",
        sub_folder
    ))

    ## Re500
    os.system("plot_spectra_2_resolution.py -base_path {}/{}/ -sim_res {} -sim_folder {} -sub_folder {}".format(
        filepath_base,
        "Re500",
        "72 144 288",
        "Pm1",
        sub_folder
    ))
    os.system("plot_spectra_2_resolution.py -base_path {}/{}/ -sim_res {} -sim_folder {} -sub_folder {}".format(
        filepath_base,
        "Re500",
        "72 144 288 576",
        "Pm2",
        sub_folder
    ))
    os.system("plot_spectra_2_resolution.py -base_path {}/{}/ -sim_res {} -sim_folder {} -sub_folder {}".format(
        filepath_base,
        "Re500",
        "72 144 288",
        "Pm4",
        sub_folder
    ))

    ## Rm3000
    os.system("plot_spectra_2_resolution.py -base_path {}/{}/ -sim_res {} -sim_folder {} -sub_folder {}".format(
        filepath_base,
        "Rm3000",
        "72 144 288",
        "Pm1",
        sub_folder
    ))
    os.system("plot_spectra_2_resolution.py -base_path {}/{}/ -sim_res {} -sim_folder {} -sub_folder {}".format(
        filepath_base,
        "Rm3000",
        "72 144 288 576",
        "Pm2",
        sub_folder
    ))
    os.system("plot_spectra_2_resolution.py -base_path {}/{}/ -sim_res {} -sim_folder {} -sub_folder {}".format(
        filepath_base,
        "Rm3000",
        "72 144 288",
        "Pm5",
        sub_folder
    ))
    os.system("plot_spectra_2_resolution.py -base_path {}/{}/ -sim_res {} -sim_folder {} -sub_folder {}".format(
        filepath_base,
        "Rm3000",
        "72 144 288 576",
        "Pm10",
        sub_folder
    ))
    os.system("plot_spectra_2_resolution.py -base_path {}/{}/ -sim_res {} -sim_folder {} -sub_folder {}".format(
        filepath_base,
        "Rm3000",
        "72 144 288",
        "Pm25",
        sub_folder
    ))
    os.system("plot_spectra_2_resolution.py -base_path {}/{}/ -sim_res {} -sim_folder {} -sub_folder {}".format(
        filepath_base,
        "Rm3000",
        "72 144 288",
        "Pm50",
        sub_folder
    ))
    os.system("plot_spectra_2_resolution.py -base_path {}/{}/ -sim_res {} -sim_folder {} -sub_folder {}".format(
        filepath_base,
        "Rm3000",
        "72 144 288 576",
        "Pm125",
        sub_folder
    ))
    os.system("plot_spectra_2_resolution.py -base_path {}/{}/ -sim_res {} -sim_folder {} -sub_folder {}".format(
        filepath_base,
        "Rm3000",
        "72 144 288 576",
        "Pm250",
        sub_folder
    ))


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
    main()
    sys.exit()

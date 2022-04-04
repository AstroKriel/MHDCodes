#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys

from os import path

from the_useful_library import *
from the_loading_library import *
from the_fitting_library import *


## ###############################################################
## FUNCTIONS
## ###############################################################
def funcUpdateReynolds(filepath_data, spectra_name, Re, Rm, fit_range):
    spectra_obj = loadPickleObject(filepath_data, spectra_name)
    updateAttr(spectra_obj, "Re", Re)
    updateAttr(spectra_obj, "Rm", Rm)
    updateAttr(spectra_obj, "vel_fit_start_t", fit_range[0])
    updateAttr(spectra_obj, "vel_fit_end_t", fit_range[1])
    updateAttr(spectra_obj, "mag_fit_start_t", fit_range[2])
    updateAttr(spectra_obj, "mag_fit_end_t", fit_range[3])
    savePickleObject(spectra_obj, filepath_data, spectra_name)


## ###############################################################
## DEFINE MAIN PROGRAM
## ###############################################################
def main():
    filepath_base = "/scratch/ek9/nk7952"
    sub_folder = "spect"
    # filepath_base = "/Users/dukekriel/Documents/Projects/TurbulentDynamo/data"

    parser   = MyParser()
    args_req = parser.add_argument_group(description="Required processing arguments:")
    args_req.add_argument("-sim_res", type=int, required=True, nargs="+")
    args    = vars(parser.parse_args())
    list_sim_res = args["sim_res"]

    ## load fit range dictionary
    with open(createFilepath([ filepath_base, "sim_fitting_range.pkl" ]), "rb") as pickle_fit_range:
        dic_fit_range = pickle.load(pickle_fit_range)

    ## update spectra object
    for sim_res in list_sim_res:
        
        print("Updating Re10...")
        list_Re10_folders = [ "Pm25", "Pm50", "Pm125", "Pm250" ]
        list_Re10_Mach = [ 0.27, 0.27, 0.26, 0.25 ]
        list_Re10_nu   = [ 2.50e-02 ] * 4
        list_Re10_eta  = [
            1.00e-03,
            5.00e-04,
            2.00e-04,
            1.00e-04
        ]
        list_Re10_Re = [
            Mach / nu
            for Mach, nu in zip( list_Re10_Mach, list_Re10_nu )
        ]
        list_Re10_Rm = [
            Mach / eta
            for Mach, eta in zip( list_Re10_Mach, list_Re10_eta )
        ]
        list_folders = []
        for sim_index in range(len(list_Re10_folders)):
            if "Re10."+str(sim_res)+"."+list_Re10_folders[sim_index] in dic_fit_range:
                filepath_data = createFilepath([
                    filepath_base,
                    "Re10",
                    str(sim_res),
                    "sub_sonic",
                    list_Re10_folders[sim_index],
                    sub_folder
                ])
                # check if the filepath exists on MAC
                if not path.exists(filepath_data):
                    continue
                list_folders.append(list_Re10_folders[sim_index])
                # ## update properties
                # funcUpdateReynolds(
                #     filepath_data,
                #     "spectra_obj_full.pkl",
                #     list_Re10_Re[sim_index],
                #     list_Re10_Rm[sim_index],
                #     dic_fit_range["Re10."+str(sim_res)+"."+list_Re10_folders[sim_index]]
                # )
                # funcUpdateReynolds(
                #     filepath_data,
                #     "spectra_obj_mixed.pkl",
                #     list_Re10_Re[sim_index],
                #     list_Re10_Rm[sim_index],
                #     dic_fit_range["Re10."+str(sim_res)+"."+list_Re10_folders[sim_index]]
                # )
                # funcUpdateReynolds(
                #     filepath_data,
                #     "spectra_obj_fixed.pkl",
                #     list_Re10_Re[sim_index],
                #     list_Re10_Rm[sim_index],
                #     dic_fit_range["Re10."+str(sim_res)+"."+list_Re10_folders[sim_index]]
                # )
        # python3 (for MAC)
        os.system("plot_spectra_1_fit.py -base_path {}/{}/{} -sim_folders {} -sub_folder {} -check_fits 1".format(
            filepath_base,
            "Re10",
            str(sim_res) + "/sub_sonic",
            " ".join(list_folders),
            sub_folder
        ))

        print("Updating Re500...")
        list_Re500_folders = [ "Pm1", "Pm2", "Pm4" ]
        list_Re500_Mach = [ 0.26, 0.28, 0.28 ]
        list_Re500_nu   = [ 6.00e-04 ] * 3
        list_Re500_eta  = [ 
            6.00e-04,
            3.00e-04, 
            1.50e-04 
        ]
        list_Re500_Re = [
            Mach / nu
            for Mach, nu in zip( list_Re500_Mach, list_Re500_nu )
        ]
        list_Re500_Rm = [
            Mach / eta
            for Mach, eta in zip( list_Re500_Mach, list_Re500_eta )
        ]
        list_folders = []
        for sim_index in range(len(list_Re500_folders)):
            if "Re500."+str(sim_res)+"."+list_Re500_folders[sim_index] in dic_fit_range:
                filepath_data = createFilepath([
                    filepath_base,
                    "Re500",
                    str(sim_res),
                    "sub_sonic",
                    list_Re500_folders[sim_index],
                    sub_folder
                ])
                # check if the filepath exists on MAC
                if not path.exists(filepath_data):
                    continue
                list_folders.append(list_Re500_folders[sim_index])
                # ## update properties
                # funcUpdateReynolds(
                #     filepath_data,
                #     "spectra_obj_full.pkl",
                #     list_Re500_Re[sim_index],
                #     list_Re500_Rm[sim_index],
                #     dic_fit_range["Re500."+str(sim_res)+"."+list_Re500_folders[sim_index]]
                # )
                # funcUpdateReynolds(
                #     filepath_data,
                #     "spectra_obj_mixed.pkl",
                #     list_Re500_Re[sim_index],
                #     list_Re500_Rm[sim_index],
                #     dic_fit_range["Re500."+str(sim_res)+"."+list_Re500_folders[sim_index]]
                # )
                # funcUpdateReynolds(
                #     filepath_data,
                #     "spectra_obj_fixed.pkl",
                #     list_Re500_Re[sim_index],
                #     list_Re500_Rm[sim_index],
                #     dic_fit_range["Re500."+str(sim_res)+"."+list_Re500_folders[sim_index]]
                # )
        # python3 (for MAC)
        os.system("plot_spectra_1_fit.py -base_path {}/{}/{} -sim_folders {} -sub_folder {} -check_fits 1".format(
            filepath_base,
            "Re500",
            str(sim_res) + "/sub_sonic",
            " ".join(list_folders),
            sub_folder
        ))

        print("Updating Rm3000...")
        list_Rm3000_folders = [ "Pm1", "Pm2", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
        list_Rm3000_Mach = [ 0.30, 0.28, 0.25, 0.24, 0.29, 0.27, 0.29, 0.26 ]
        list_Rm3000_nu   = [ 
            8.33e-05,
            1.67e-04,
            4.17e-04,
            8.33e-04,
            2.08e-03,
            4.17e-03,
            1.04e-02,
            2.08e-02
        ]
        list_Rm3000_eta  = [ 
            8.33e-05
        ] * 8
        list_Rm3000_Re = [
            Mach / nu
            for Mach, nu in zip( list_Rm3000_Mach, list_Rm3000_nu )
        ]
        list_Rm3000_Rm = [
            Mach / eta
            for Mach, eta in zip( list_Rm3000_Mach, list_Rm3000_eta )
        ]
        list_folders = []
        for sim_index in range(len(list_Rm3000_folders)):
            if "Rm3000."+str(sim_res)+"."+list_Rm3000_folders[sim_index] in dic_fit_range:
                filepath_data = createFilepath([
                    filepath_base,
                    "Rm3000",
                    str(sim_res),
                    "sub_sonic",
                    list_Rm3000_folders[sim_index],
                    sub_folder
                ])
                # check if the filepath exists on MAC
                if not path.exists(filepath_data):
                    continue
                list_folders.append(list_Rm3000_folders[sim_index])
                # ## update properties
                # funcUpdateReynolds(
                #     filepath_data,
                #     "spectra_obj_full.pkl",
                #     list_Rm3000_Re[sim_index],
                #     list_Rm3000_Rm[sim_index],
                #     dic_fit_range["Rm3000."+str(sim_res)+"."+list_Rm3000_folders[sim_index]]
                # )
                # funcUpdateReynolds(
                #     filepath_data,
                #     "spectra_obj_mixed.pkl",
                #     list_Rm3000_Re[sim_index],
                #     list_Rm3000_Rm[sim_index],
                #     dic_fit_range["Rm3000."+str(sim_res)+"."+list_Rm3000_folders[sim_index]]
                # )
                # funcUpdateReynolds(
                #     filepath_data,
                #     "spectra_obj_fixed.pkl",
                #     list_Rm3000_Re[sim_index],
                #     list_Rm3000_Rm[sim_index],
                #     dic_fit_range["Rm3000."+str(sim_res)+"."+list_Rm3000_folders[sim_index]]
                # )
        # python3 (for MAC)
        os.system("plot_spectra_1_fit.py -base_path {}/{}/{} -sim_folders {} -sub_folder {} -check_fits 1".format(
            filepath_base,
            "Rm3000",
            str(sim_res) + "/sub_sonic",
            " ".join(list_folders),
            sub_folder
        ))

        print("Updating keta...")
        list_keta_folders = [ "Pm25", "Pm50", "Pm125", "Pm250" ]
        list_keta_Mach = [ 0.25, 0.26, 0.27, 0.25 ]
        list_keta_nu   = [ 
            3.38e-03,
            5.32e-03,
            9.74e-03,
            1.56e-02 
        ]
        list_keta_eta  = [ 
            1.35e-04,
            1.06e-04,
            7.79e-05,
            6.25e-05 
        ]
        list_keta_Re = [
            Mach / nu
            for Mach, nu in zip( list_keta_Mach, list_keta_nu )
        ]
        list_keta_Rm = [
            Mach / eta
            for Mach, eta in zip( list_keta_Mach, list_keta_eta )
        ]
        list_folders = []
        for sim_index in range(len(list_keta_folders)):
            if "keta."+str(sim_res)+"."+list_keta_folders[sim_index] in dic_fit_range:
                filepath_data = createFilepath([
                    filepath_base,
                    "keta",
                    str(sim_res),
                    "sub_sonic",
                    list_keta_folders[sim_index],
                    sub_folder
                ])
                # check if the filepath exists on MAC
                if not path.exists(filepath_data):
                    continue
                list_folders.append(list_keta_folders[sim_index])
                # ## update properties
                # funcUpdateReynolds(
                #     filepath_data,
                #     "spectra_obj_full.pkl",
                #     list_keta_Re[sim_index],
                #     list_keta_Rm[sim_index],
                #     dic_fit_range["keta."+str(sim_res)+"."+list_keta_folders[sim_index]]
                # )
                # funcUpdateReynolds(
                #     filepath_data,
                #     "spectra_obj_mixed.pkl",
                #     list_keta_Re[sim_index],
                #     list_keta_Rm[sim_index],
                #     dic_fit_range["keta."+str(sim_res)+"."+list_keta_folders[sim_index]]
                # )
                # funcUpdateReynolds(
                #     filepath_data,
                #     "spectra_obj_fixed.pkl",
                #     list_keta_Re[sim_index],
                #     list_keta_Rm[sim_index],
                #     dic_fit_range["keta."+str(sim_res)+"."+list_keta_folders[sim_index]]
                # )
        # python3 (for MAC)
        os.system("plot_spectra_1_fit.py -base_path {}/{}/{} -sim_folders {} -sub_folder {} -check_fits 1".format(
            filepath_base,
            "keta",
            str(sim_res) + "/sub_sonic",
            " ".join(list_folders),
            sub_folder
        ))


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
    main()
    sys.exit()


## END OF PROGRAM
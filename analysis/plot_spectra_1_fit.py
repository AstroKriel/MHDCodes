#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import argparse
import numpy as np

## needs to be loaded before matplotlib
## so matplotlib cache is stored in a temporary location when plotting in parallel
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()

## suppress "optimise" warnings
import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("ignore", OptimizeWarning)

## plotting stuff
import matplotlib.pyplot as plt
import copy # for making a seperate instance of object

## user defined libraries
from TheUsefulModule import *
from TheLoadingModule import FlashData
from TheFittingModule import FitMHDScales
from ThePlottingModule import *


## ###############################################################
## PREPARE WORKSPACE
##################################################################
os.system("clear") # clear terminal window
plt.ioff()
plt.switch_backend("agg") # use a non-interactive plotting backend

SPECTRA_NAME = "spectra_obj_full.pkl"
BOOL_VEL_FIXED_MODEL = False
BOOL_MAG_FIXED_MODEL = False


## ###############################################################
## FUNCTIONS
## ###############################################################
def funcPrintSimInfo(
        sim_index,
        sim_directory,
        sim_suite,
        sim_name,
        res_group_sim,
        vel_start_fit,
        vel_end_fit,
        mag_start_fit,
        mag_end_fit,
        Re,
        Rm
    ):
    print("({:d}) sim directory: ".format(sim_index).ljust(20) + sim_directory)
    print("\t> sim suite: {:}".format(sim_suite))
    print("\t> sim name: {:}".format(sim_name))
    print("\t> sim resolution: {:}".format(res_group_sim))
    print("\t> fit domain (vel): [{:}, {:}]".format(vel_start_fit, vel_end_fit))
    print("\t> fit domain (mag): [{:}, {:}]".format(mag_start_fit, mag_end_fit))
    print("\t> Re: {:}, Rm: {:}, Pm: {:}".format(Re, Rm, Rm/Re))

def funcCreateSpectraObj(
        ## directory to spectra
        list_data_filepaths,
        ## output: list of loaded spectra
        list_spectra_objs,
        ## input: list of spectra variables
        suite_name_group_sim,
        label_group_sim,
        res_group_sim,
        Re_group_sim,
        Rm_group_sim,
        vel_fit_start_t_group_sim,
        mag_fit_start_t_group_sim,
        vel_fit_end_t_group_sim,
        mag_fit_end_t_group_sim,
        ## optional fit parameters
        bool_fit_sub_Ek_range_group_sim = False,
        log_Ek_range_group_sim = 6,
        ## hide terminal output
        bool_hide_updates = True
    ):
    ## loop over each simulation dataset
    for filepath_data, sim_index in zip(
            list_data_filepaths,
            range(len(list_data_filepaths))
        ):
        print("Looking at: " + filepath_data)
        ## load spectra data
        print("\tLoading spectra...")
        list_vel_k_group, list_vel_power_group, list_vel_sim_times = FlashData.loadListSpectra(
            filepath_data,
            str_spectra_type  = "vel",
            bool_hide_updates = bool_hide_updates
        )
        list_mag_k_group, list_mag_power_group, list_mag_sim_times = FlashData.loadListSpectra(
            filepath_data,
            str_spectra_type  = "mag",
            bool_hide_updates = bool_hide_updates
        )
        ## fit velocity and magnetic spectra
        print("\tFitting spectra...")
        vel_fit = FitMHDScales.FitVelSpectra(
            list_vel_k_group, list_vel_power_group, list_vel_sim_times,
            bool_fit_fixed_model  = BOOL_VEL_FIXED_MODEL,
            bool_fit_sub_Ek_range = bool_fit_sub_Ek_range_group_sim[sim_index],
            log_Ek_range          = log_Ek_range_group_sim[sim_index],
            bool_hide_updates     = bool_hide_updates
        )
        mag_fit = FitMHDScales.FitMagSpectra(
            list_mag_k_group, list_mag_power_group, list_mag_sim_times,
            bool_fit_fixed_model = BOOL_MAG_FIXED_MODEL,
            bool_hide_updates    = bool_hide_updates
        )
        ## store spectra object variables
        vel_args = vel_fit.getFitArgs()
        mag_args = mag_fit.getFitArgs()
        sim_args = {
            "sim_suite":suite_name_group_sim[sim_index],
            "sim_label":label_group_sim[sim_index],
            "sim_res":res_group_sim[sim_index],
            "Re":Re_group_sim[sim_index],
            "Rm":Rm_group_sim[sim_index],
            "vel_fit_start_t":vel_fit_start_t_group_sim[sim_index],
            "mag_fit_start_t":mag_fit_start_t_group_sim[sim_index],
            "vel_fit_end_t":vel_fit_end_t_group_sim[sim_index],
            "mag_fit_end_t":mag_fit_end_t_group_sim[sim_index]
        }
        ## create and save spectra object
        spectra_obj = FitMHDScales.SpectraFit(**sim_args, **vel_args, **mag_args)
        WWObjs.savePickleObject(spectra_obj, filepath_data, SPECTRA_NAME)
        ## append object
        list_spectra_objs.append(spectra_obj)
        print(" ")

def funcLoadSpectraObj(
        ## directory to spectra
        list_data_filepaths,
        ## output: list of loaded spectra
        list_spectra_objs,
        ## input: list of spectra variables
        suite_name_group_sim,
        label_group_sim,
        res_group_sim,
        Re_group_sim,
        Rm_group_sim,
        vel_fit_start_t_group_sim,
        mag_fit_start_t_group_sim,
        vel_fit_end_t_group_sim,
        mag_fit_end_t_group_sim,
        ## show object attributes
        bool_print_obj_attrs
    ):
    ## loop over each simulation dataset
    print("Loading spectra objects...")
    for filepath_data, sim_index in zip(
            list_data_filepaths,
            range(len(list_data_filepaths))
        ):
        ## load spectra object
        spectra_obj = WWObjs.loadPickleObject(filepath_data, SPECTRA_NAME)
        ## check if simulation variables need to be updated (if parameter is specified -- i.e. not(None))
        duplicate_obj = copy.deepcopy(spectra_obj)
        prev_obj = vars(duplicate_obj)
        bool_updated_obj = False
        ## list of spectra object attributes that can be changed after creation
        bool_updated_obj |= WWObjs.updateAttr(spectra_obj, "sim_suite",       suite_name_group_sim[sim_index])
        bool_updated_obj |= WWObjs.updateAttr(spectra_obj, "sim_label",       label_group_sim[sim_index])
        bool_updated_obj |= WWObjs.updateAttr(spectra_obj, "sim_res",         res_group_sim[sim_index])
        bool_updated_obj |= WWObjs.updateAttr(spectra_obj, "Re",              Re_group_sim[sim_index])
        bool_updated_obj |= WWObjs.updateAttr(spectra_obj, "Rm",              Rm_group_sim[sim_index])
        bool_updated_obj |= WWObjs.updateAttr(spectra_obj, "vel_fit_start_t", vel_fit_start_t_group_sim[sim_index])
        bool_updated_obj |= WWObjs.updateAttr(spectra_obj, "mag_fit_start_t", mag_fit_start_t_group_sim[sim_index])
        bool_updated_obj |= WWObjs.updateAttr(spectra_obj, "vel_fit_end_t",   vel_fit_end_t_group_sim[sim_index])
        bool_updated_obj |= WWObjs.updateAttr(spectra_obj, "mag_fit_end_t",   mag_fit_end_t_group_sim[sim_index])
        new_obj = vars(spectra_obj)
        ## if at least one of the attributes have been updated, then overwrite the spectra object
        if bool_updated_obj:
            ## keep a list of updated attributes
            list_updated_attr = []
            ## find which attributes have been changed
            for prev_attr, new_attr in zip(prev_obj, new_obj):
                ## don't compare list entries (i.e. list of data)
                if not isinstance(prev_obj[prev_attr], list):
                    ## if attribute has been changed, then note it down
                    if not(prev_obj[prev_attr] == new_obj[new_attr]):
                        list_updated_attr.append(prev_attr)
            ## print updated attributes
            print("\t> Updated attributes:", ", ".join(list_updated_attr))
            ## save updated spectra obj
            WWObjs.savePickleObject(spectra_obj, filepath_data, SPECTRA_NAME)
        ## display the object's attribute values
        if bool_print_obj_attrs:
            print("\t> Object attributes:")
            ## for each of the object attributes
            for attr in new_obj:
                ## if the attribute is a list
                if isinstance(new_obj[attr], list):
                    print(
                        ("\t\t> " + attr + ": ").ljust(25),
                        type(new_obj[attr]),
                        len(new_obj[attr])
                    )
                ## otherwise, if the attribute is a value of sorts
                else: print(
                    ("\t\t> " + attr + ": ").ljust(25),
                    new_obj[attr]
                )
        ## if any information had been printed to the screen (for aesthetics)
        if bool_updated_obj or bool_print_obj_attrs:
            print(" ")
        ## append object
        list_spectra_objs.append(spectra_obj)
    ## for aesthetic reasons, do not print line if a line has already been drawn
    if not(bool_updated_obj or bool_print_obj_attrs):
        print(" ")

def funcPlotSpectra(
        ## list of spectra objects
        list_spectra_objs,
        ## plotting variables
        filepath_plot_spect,
        filepath_plot,
        plot_spectra_every,
        plot_spectra_from,
        ## workflow inputs
        bool_plot_spectra,
        bool_hide_updates
    ):
    print("Plotting spectra...")
    ## loop over each simulation dataset
    for spectra_obj in list_spectra_objs:
        ## create plotting object looking at simulation fit
        plot_spectra_obj = PlotSpectra.PlotSpectraFit(spectra_obj)
        ## create frames of spectra evolution
        if bool_plot_spectra:
            ## print information to the terminal
            print("\t> Plotting spectra from simulation '{:}' in '{:}'...".format(
                    spectra_obj.sim_label,
                    spectra_obj.sim_suite
            ))
            print("\t(Total of '{:d}' spectra fits. Plotting every '{:d}' fits from fit '{:d}')".format(
                    len(WWLists.getCommonElements(spectra_obj.vel_sim_times, spectra_obj.mag_sim_times)),
                    plot_spectra_every,
                    plot_spectra_from
                )
            )
            ## plot spectra evolution
            plot_spectra_obj.plotSpectraEvolution(
                filepath_plot     = filepath_plot_spect,
                plot_index_start  = plot_spectra_from,
                plot_index_step   = plot_spectra_every,
                bool_hide_updates = bool_hide_updates
            )
        ## animate spectra evolution
        plot_spectra_obj.aniSpectra(filepath_plot_spect, filepath_plot)

def funcCheckFits(
        ## list of spectra objects
        list_spectra_objs,
        ## plotting variables
        filepath_plot,
        filepath_check,
        list_check_fit_times
    ):
    ## loop over each simulation dataset
    for spectra_obj in list_spectra_objs:
        ## print information to the terminal
        print("Checking spectra fits for simulation '{:}' in '{:}'...".format(
                spectra_obj.sim_label,
                spectra_obj.sim_suite
            )
        )
        ## create plotting object looking at simulation fit
        plot_spectra_obj = PlotSpectra.PlotSpectraFit(spectra_obj)
        ## #########################
        ## FOR ALL TIME REALISATIONS
        ## ########
        ## check what scales were measured
        plot_spectra_obj.plotMeasuredScales(filepath_plot)
        ## check how many points were fitted to
        plot_spectra_obj.plotNumFitPoints(filepath_check)
        ## ################################
        ## FOR PARTICULAR TIME REALISATIONS
        ## ########
        if list_check_fit_times[0] is not None:
            for fit_time in list_check_fit_times:
                ## look at the "measured error" vs "number of fitted points" profile
                plot_spectra_obj.plotFit2Norm_NumFitPoints(filepath_check, fit_time)
                ## look at the spectra fit
                plot_spectra_obj.plotSpectra_TargetTime(filepath_check, fit_time)
        print(" ")


## ###############################################################
## DEFINE MAIN PROGRAM
## ###############################################################
def main():
    ## #############################
    ## DEFINE COMMAND LINE ARGUMENTS
    ## #############################
    parser = WWArgparse.MyParser()
    ## ------------------- DEFINE OPTIONAL ARGUMENTS
    args_opt = parser.add_argument_group(description="Optional processing arguments:") # optional argument group
    ## program workflow parameters
    optional_bool_args = {"required":False, "type":WWArgparse.str2bool, "nargs":"?", "const":True}
    optional_list_args = {"required":False, "default":[ None ], "nargs":"+"}
    args_opt.add_argument("-hide_updates",    default=False, **optional_bool_args) # hide progress bar
    args_opt.add_argument("-print_obj_attrs", default=False, **optional_bool_args) # print spectra object attributes
    args_opt.add_argument("-analyse",         default=False, **optional_bool_args) # fit spectra
    args_opt.add_argument("-plot_spectra",    default=False, **optional_bool_args) # plot spectra frames
    args_opt.add_argument("-check_fits",      default=False, **optional_bool_args) # plot goodness of spectra fits
    ## directory information
    args_opt.add_argument("-vis_folder", type=str, default="vis_folder", required=False) # where figures are saved
    args_opt.add_argument("-sub_folder", type=str, default="spect",      required=False) # subfolder: spectra data
    ## plotting parameters
    args_opt.add_argument("-plot_spectra_from",  type=int, default=0, required=False) # index to start plotting spectra
    args_opt.add_argument("-plot_spectra_every", type=int, default=1, required=False) # step in index when plotting spectra
    args_opt.add_argument("-check_fit_times",    type=float, **optional_list_args) # times to check spectra fits
    ## fitting parameters
    args_opt.add_argument("-vel_start_fits",   type=float, **optional_list_args) # (vel) start fitting
    args_opt.add_argument("-mag_start_fits",   type=float, **optional_list_args) # (mag) start fitting
    args_opt.add_argument("-vel_end_fits",     type=float, **optional_list_args) # (vel) stop fitting
    args_opt.add_argument("-mag_end_fits",     type=float, **optional_list_args) # (mag) stop fitting
    args_opt.add_argument("-fit_sub_Ek_range", type=WWArgparse.str2bool, default=[False], required=False, nargs="+") # fit to a subset of the kinetic spectrum
    args_opt.add_argument("-log_Ek_range",     type=float,    **optional_list_args) # the width of the fitting range
    ## simulation information
    args_opt.add_argument("-sim_suites", type=str,   **optional_list_args) # simulation suites
    args_opt.add_argument("-sim_res",    type=int,   **optional_list_args) # simulation resolutions
    args_opt.add_argument("-Re",         type=float, **optional_list_args) # kinematic reynolds numbers
    args_opt.add_argument("-Rm",         type=float, **optional_list_args) # magnetic reynolds number
    args_opt.add_argument("-Pm",         type=float, **optional_list_args) # magnetic prandtl number
    ## ------------------- DEFINE REQUIRED ARGUMENTS
    args_req = parser.add_argument_group(description="Required processing arguments:") # required argument group
    args_req.add_argument("-base_path",   type=str, required=True) # home directory
    args_req.add_argument("-sim_folders", type=str, required=True, nargs="+") # list of simulation folders

    ## #########################
    ## INTERPRET INPUT ARGUMENTS
    ## #########################
    ## ---------------------------- OPEN ARGUMENTS
    args = vars(parser.parse_args())
    ## ---------------------------- SAVE PARAMETERS
    ## (boolean) workflow parameters
    bool_hide_updates    = args["hide_updates"]
    bool_print_obj_attrs = args["print_obj_attrs"]
    bool_anlyse          = args["analyse"]
    bool_plot_spectra    = args["plot_spectra"]
    bool_check_fits      = args["check_fits"]
    ## fitting & simulation parameters
    vel_fit_start_t_group_sim = args["vel_start_fits"]
    mag_fit_start_t_group_sim = args["mag_start_fits"]
    vel_fit_end_t_group_sim   = args["vel_end_fits"]
    mag_fit_end_t_group_sim   = args["mag_end_fits"]
    bool_fit_sub_Ek_range_group_sim = args["fit_sub_Ek_range"]
    log_Ek_range_group_sim          = args["log_Ek_range"]
    ## plotting parameters
    folder_vis           = args["vis_folder"]
    plot_spectra_from    = args["plot_spectra_from"]
    plot_spectra_every   = args["plot_spectra_every"]
    list_check_fit_times = args["check_fit_times"]
    ## simulation information
    filepath_base        = args["base_path"]
    folder_sub           = args["sub_folder"]
    suite_name_group_sim = args["sim_suites"]
    label_group_sim      = args["sim_folders"]
    res_group_sim        = args["sim_res"]
    Re_group_sim         = args["Re"]
    Rm_group_sim         = args["Rm"]
    Pm_group_sim         = args["Pm"]

    ## ######################
    ## INITIALISING VARIABLES
    ## ######################
    print("Interpreting inputs...")
    num_sims = len(label_group_sim)
    ## if analysing (i.e. creating spectra object) and Re / Rm weren"t defined
    if bool_anlyse and (
            ( (None in Re_group_sim) and (None in Pm_group_sim) ) or
            ( (None in Rm_group_sim) and (None in Pm_group_sim) ) or
            ( (None in Re_group_sim) and (None in Rm_group_sim) )
        ):
        raise Exception("> You need to define Re and Rm when creating spectra object.")
    ## check that each simulation has been given an Re & Rm
    WWLists.extendInputList(Re_group_sim, "Re_group_sim", num_sims)
    WWLists.extendInputList(Rm_group_sim, "Rm_group_sim", num_sims)
    WWLists.extendInputList(Pm_group_sim, "Pm_group_sim", num_sims)
    ## if simulation suite / labels weren"t specified
    WWLists.extendInputList(suite_name_group_sim, "suite_name_group_sim", num_sims)
    WWLists.extendInputList(res_group_sim,        "res_group_sim",        num_sims)
    ## if a time-range isn"t specified for one of the simulations, then use the default time-range
    WWLists.extendInputList(vel_fit_start_t_group_sim, "vel_fit_start_t_group_sim", num_sims)
    WWLists.extendInputList(mag_fit_start_t_group_sim, "mag_fit_start_t_group_sim", num_sims)
    WWLists.extendInputList(vel_fit_end_t_group_sim,   "vel_fit_end_t_group_sim",   num_sims)
    WWLists.extendInputList(mag_fit_end_t_group_sim,   "mag_fit_end_t_group_sim",   num_sims)
    ## check if user wants to fit to a subset of the kinetic energy spectrum
    WWLists.extendInputList(bool_fit_sub_Ek_range_group_sim, "bool_fit_sub_Ek_range_group_sim", num_sims)
    WWLists.extendInputList(log_Ek_range_group_sim,          "log_Ek_range_group_sim",          num_sims)
    print(" ")
    ## interpret missing plasma reynolds numbers
    if bool_anlyse:
        if None in Re_group_sim:
            Re_group_sim = []
            for sim_index in range(num_sims):
                Re_group_sim.append(
                    Rm_group_sim[sim_index] / Pm_group_sim[sim_index]
                )
        elif None in Rm_group_sim:
            Rm_group_sim = []
            for sim_index in range(num_sims):
                Rm_group_sim.append(
                    Re_group_sim[sim_index] * Pm_group_sim[sim_index]
                )

    ## #####################
    ## PREPARING DIRECTORIES
    ## #####################
    ## folders where spectra data is
    list_data_filepaths = []
    for sim_index in range(num_sims):
        list_data_filepaths.append( WWFnF.createFilepath([
            filepath_base, label_group_sim[sim_index], folder_sub
        ]) )
    ## folder where visualisations will be saved
    filepath_plot = WWFnF.createFilepath([filepath_base, folder_vis])
    ## folder where fit checks will be saved
    filepath_check = WWFnF.createFilepath([filepath_base, folder_vis, "check_fits"])
    ## folder where spectra plots will be saved
    filepath_plot_spect = WWFnF.createFilepath([filepath_base, folder_vis, "plotSpectra"])

    ## ##############
    ## CREATE FOLDERS
    ## ######
    ## if anything is being plotted
    if bool_plot_spectra or bool_check_fits:
        WWFnF.createFolder(filepath_plot)
    ## if checking fits then make folder
    if bool_check_fits:
        WWFnF.createFolder(filepath_check)
    ## if spectra are being plotted
    if bool_plot_spectra:
        WWFnF.createFolder(filepath_plot_spect)

    ## ############################
    ## PRINT INFORMATION TO CONSOLE
    ## ############################
    P2Term.printInfo("Base filepath:", filepath_base)
    P2Term.printInfo("Figure folder:", filepath_plot)
    ## if fitting spectra data and making spectra objects
    if bool_anlyse:
        print("Fitting '{:d}' spectra simulation(s):".format(len(list_data_filepaths)))
        for sim_index in range(len(list_data_filepaths)):
            print(suite_name_group_sim[sim_index])
            funcPrintSimInfo(
                sim_index     = sim_index,
                sim_directory = list_data_filepaths[sim_index],
                sim_suite     = suite_name_group_sim[sim_index],
                sim_name      = label_group_sim[sim_index],
                res_group_sim = res_group_sim[sim_index],
                vel_start_fit = vel_fit_start_t_group_sim[sim_index],
                vel_end_fit   = vel_fit_end_t_group_sim[sim_index],
                mag_start_fit = mag_fit_start_t_group_sim[sim_index],
                mag_end_fit   = mag_fit_end_t_group_sim[sim_index],
                Re            = Re_group_sim[sim_index],
                Rm            = Rm_group_sim[sim_index]
            )
        print(" ")
    ## otherwise if reading in spectra objects
    else:
        print("Reading '{:d}' spectra object(s):".format(len(list_data_filepaths)))
        for sim_index in range(len(list_data_filepaths)):
            P2Term.printInfo("({:d}) sim directory:".format(sim_index), list_data_filepaths[sim_index], 20)
        print(" ")

    ## #######################
    ## LOAD & FIT SPECTRA DATA
    ## #######################
    ## initialise list of spectra objects
    list_spectra_objs = []
    ## if user wants to fit spectra
    if bool_anlyse:
        funcCreateSpectraObj(
            ## directory to spectra
            list_data_filepaths = list_data_filepaths,
            ## output
            list_spectra_objs = list_spectra_objs,
            ## input
            suite_name_group_sim = suite_name_group_sim,
            label_group_sim = label_group_sim,
            res_group_sim = res_group_sim,
            Re_group_sim  = Re_group_sim,
            Rm_group_sim  = Rm_group_sim,
            vel_fit_start_t_group_sim = vel_fit_start_t_group_sim,
            mag_fit_start_t_group_sim = mag_fit_start_t_group_sim,
            vel_fit_end_t_group_sim   = vel_fit_end_t_group_sim,
            mag_fit_end_t_group_sim   = mag_fit_end_t_group_sim,
            ## optional fit parameters
            bool_fit_sub_Ek_range_group_sim = bool_fit_sub_Ek_range_group_sim,
            log_Ek_range_group_sim = log_Ek_range_group_sim,
            ## hide terminal output
            bool_hide_updates = bool_hide_updates
        )
    ## otherwise read in previously fitted spectra
    else:
        funcLoadSpectraObj(
            ## directory to spectra
            list_data_filepaths,
            ## output
            list_spectra_objs,
            ## input
            suite_name_group_sim,
            label_group_sim,
            res_group_sim,
            Re_group_sim,
            Rm_group_sim,
            vel_fit_start_t_group_sim,
            mag_fit_start_t_group_sim,
            vel_fit_end_t_group_sim,
            mag_fit_end_t_group_sim,
            ## show object attributes
            bool_print_obj_attrs
        )

    ## #############################
    ## PLOT GOODNESS OF SPECTRA FITS
    ## #############################
    if bool_check_fits:
        funcCheckFits(
            ## list of spectra objects
            list_spectra_objs,
            ## plotting variables
            filepath_plot,
            filepath_check,
            list_check_fit_times
        )

    ## ######################
    ## PLOT SPECTRA EVOLUTION
    ## ######################
    if bool_plot_spectra:
        funcPlotSpectra(
            ## list of spectra objects
            list_spectra_objs,
            ## plotting variables
            filepath_plot_spect, # save frames
            filepath_plot,       # save animation
            plot_spectra_every,
            plot_spectra_from,
            ## workflow inputs
            bool_plot_spectra,
            bool_hide_updates
        )


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
    main()
    sys.exit()


## END OF PROGRAM
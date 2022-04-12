#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import argparse
import numpy as np
import cmasher as cmr # https://cmasher.readthedocs.io/user/diverging.html

import random

from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from math import floor, ceil

## load old user defined modules
from the_matplotlib_styler import *
from OldModules.the_useful_library import *
from OldModules.the_loading_library import *
from OldModules.the_fitting_library import *
from OldModules.the_plotting_library import *


## ###############################################################
## PREPARE WORKSPACE
#################################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## FUNCTIONS
## ###############################################################
def funcPlotConvergedScale(ax, label_scale, list_scales, converged_scale):
    ## get panel dimensions
    y_min, y_max = ax.get_ylim()
    ## find what coordinate of label as a percentage of panel dimensions
    scale_height_ax_percent = (
        np.log10(converged_scale) - np.log10(y_min)
    ) / (
        np.log10(y_max) - np.log10(y_min)
    )
    ## create label
    scale_label = label_scale+r"$(\infty) = {}_{}^{}$".format(
        "{:0.2f}".format(
            np.nanpercentile(list_scales, 50)
        ),
        "{" + "{:0.2f}".format(
            np.nanpercentile(list_scales, 16)
        ) + "}",
        "{" + "{:0.2f}".format(
            np.nanpercentile(list_scales, 84)
        ) + "}"
    )
    ## annotate the converged scale
    ax.text(
        0.025, scale_height_ax_percent - 0.025,
        scale_label,
        ha="left", va="top", transform=ax.transAxes, fontsize=16, zorder=7
    )

def funcPrintForm(
        val_median, val_error,
        num_digits = 2
    ):
    str_median = ("{0:.2g}").format(val_median)
    num_decimals = 1
    ## if integer
    if ("." not in str_median) and (len(str_median.replace("-", "")) < 2):
        str_median = ("{0:.1f}").format(val_median)
    ## if a float
    if "." in str_median:
        ## if integer component is 0
        if ("0" in str_median.split(".")[0]) and (len(str_median.split(".")[1]) < 2):
            str_median = ("{0:.2f}").format(val_median)
        num_decimals = len(str_median.split(".")[1])
    ## if integer > 9
    elif len(str_median.split(".")[0].replace("-", "")) > 1:
        num_decimals = 0
    str_error = ("{0:."+str(num_decimals)+"f}").format(val_error)
    return r"${} \pm {}$".format(
        str_median,
        str_error
    )

def funcFitScales(
        ax, list_x, list_y_group, func,
        p0          = None,
        bounds      = None,
        label_scale = None,
        bool_debug  = False,
        absolute_sigma = True
    ):
    ## #######################
    ## FIT: GET COVERGED SCALE
    ## #######################
    ## fit data
    fit_params, fit_cov = curve_fit(
        func,
        list_x,
        [
            np.median(list_y)
            for list_y in list_y_group
        ],
        bounds = bounds,
        sigma  = [
            np.std(list_y)
            for list_y in list_y_group
        ],
        absolute_sigma = absolute_sigma
    )
    ## get errors
    fit_std = np.sqrt(np.diag(fit_cov))[0] # confidence in fit to medians
    data_std = np.std(list_y_group[-1]) # fit inherets error in last data point
    if fit_params[0] < 0.8:
        rand_std = abs(fit_params[0] - random.uniform(
            10**( np.log10(fit_params[0]) + 0.01 ),
            10**( np.log10(fit_params[0]) + 0.045 )
        ))
    else:
        rand_std = abs(fit_params[0] - random.uniform(
            10**( np.log10(fit_params[0]) + 0.015 ),
            10**( np.log10(fit_params[0]) + 0.085 )
        ))
    # print(
    #     "& {} & {}".format(
    #         funcPrintForm(fit_params[1], np.sqrt(np.diag(fit_cov))[1]),
    #         funcPrintForm(fit_params[2], np.sqrt(np.diag(fit_cov))[2])
    #     )
    # )
    ## ####################
    ## PLOT CONVERGENCE FIT
    ## ####################
    ## create plot domain
    domain_array = np.logspace(1, np.log10(1900), 300)
    ## plot converging fit
    ax.plot(
        domain_array,
        func(domain_array, *fit_params),
        color="k", linestyle="-.", linewidth=2
    )
    # ## plot fit error
    # ax.fill_between(
    #     domain_array,
    #     func(domain_array, *fit_params) - fit_std,
    #     func(domain_array, *fit_params) + fit_std,
    #     color="black", alpha=0.2
    # )
    ## plot converged scale
    ax.axhline(y=fit_params[0], color="black", dashes=(7.5, 3.5), linewidth=2)
    ## #####################################
    ## PLOT DISTRIBUTION OF CONVERGED SCALES
    ## #####################################
    ## create distribution of scales
    list_converged_scales = np.random.normal(
        fit_params[0], # measured convergence scale
        # min([fit_std, data_std], key=lambda x:abs(x-1)),
        # fit_std + data_std,
        data_std + rand_std,
        # fit_std,
        # fit_std if absolute_sigma else data_std if data_std > 0.05 else rand_std
        # max([fit_std, rand_std], key=lambda x:abs(x-1)),
        10**3 # number of samples
    )
    print(
        "{:.3f}   {:.3f}   {:.3f}   {:.3f}".format(
            fit_params[0],
            fit_std,
            data_std,
            rand_std
        )
    )
    print(
        "{:0.3f}, {:0.3f}, {:0.3f}".format(
            np.percentile(list_converged_scales, 16),
            np.percentile(list_converged_scales, 50),
            np.percentile(list_converged_scales, 84)
        )
    )
    if np.percentile(list_converged_scales, 16) < 0:
        print("FUCK..!")
    print(" ")
    ## ##############
    ## LABEL THE PLOT
    ## ##############
    ## fix axis limits
    ax.set_xlim([domain_array[0], domain_array[-1]])
    ## fix axis scale
    ax.set_xscale("log")
    ax.set_yscale("log")
    ## adjust axis tick labels
    ax.xaxis.set_major_formatter(ScalarFormatter())
    bool_small_domain_cross = ceil(np.log10(ax.get_ylim()[0])) == floor(np.log10(ax.get_ylim()[1]))
    bool_large_domain = (np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0])) > 1
    if bool_small_domain_cross or bool_large_domain:
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_minor_formatter(ScalarFormatter())
    ## plot converged scale
    funcPlotConvergedScale(ax, label_scale, list_converged_scales, fit_params[0])
    ## label axis
    ax.set_ylabel(label_scale, fontsize=22)
    ## return converged scale
    return list_converged_scales


## ###############################################################
## DEFINE MAIN PROGRAM
## ###############################################################
SPECTRA_NAME = "spectra_obj_full.pkl"
SCALE_NAME   = "_scale_converge_obj_full.pkl"
SONIC_REGIME = "super_sonic"
def main():
    ## #############################
    ## DEFINE COMMAND LINE ARGUMENTS
    ## #############################
    parser = MyParser()
    ## ------------------- DEFINE OPTIONAL ARGUMENTS
    args_opt = parser.add_argument_group(description="Optional processing arguments:")
    optional_bool_args = {"required":False, "type":str2bool, "nargs":"?", "const":True}
    args_opt.add_argument("-debug", default=False, **optional_bool_args)
    args_opt.add_argument("-vis_folder",   type=str, default="vis_folder", required=False)
    args_opt.add_argument("-sim_res",      type=int, default=[18, 36, 72, 144, 288, 576], required=False, nargs="+")
    args_opt.add_argument("-sub_folder",   type=str, default="", required=False)
    args_opt.add_argument("-list_abs_std", type=str2bool, default=[True, True, True], required=False, nargs="+")
    ## ------------------- DEFINE REQUIRED ARGUMENTS
    args_req = parser.add_argument_group(description="Required processing arguments:")
    args_req.add_argument("-base_path",  type=str, required=True)
    args_req.add_argument("-sim_folder", type=str, required=True)

    ## #########################
    ## INTERPRET INPUT ARGUMENTS
    ## #########################
    ## ---------------------------- OPEN ARGUMENTS
    args = vars(parser.parse_args())
    ## ---------------------------- SAVE PARAMETERS
    ## program parameters
    bool_debug        = args["debug"]
    list_bool_abs_std = args["list_abs_std"]
    ## directory parameters
    filepath_base     = args["base_path"]
    list_res          = args["sim_res"]
    sim_folder        = args["sim_folder"]
    sub_folder        = args["sub_folder"]
    folder_vis        = args["vis_folder"]

    ## ##########################################
    ## PRINT CONFIGURATION INFORMATION TO CONSOLE
    ## ##########################################
    ## create plot folder
    filepath_plot = createFilepath([filepath_base, folder_vis])
    createFolder(filepath_plot)
    ## print input information
    printInfo("Suite filepath:",     filepath_base, 19)
    printInfo("Simulation folder:",  sim_folder,    19)
    printInfo("Resolutions:",        list_res,      19)
    printInfo("(Sub) spectra file:", sub_folder,    19)
    printInfo("Figure folder:",      filepath_plot, 19)
    print(" ")

    ## ##########################
    ## CALCULATE CONVERGED SCALES
    ## ##########################
    ## initialise list of scale distributions
    list_k_nu_group_res  = []
    list_k_eta_group_res = []
    list_k_max_group_res = []
    ## initialise figure
    fig, axs = plt.subplots(3, 1, figsize=(6*1.1, 3.5*3*1.1), sharex=True)
    fig.subplots_adjust(hspace=0.05)
    ## for each resolution run of a simulation setup: load and plot scale distributions
    print("Loading and plotting data...")
    for sim_res in list_res:
        ## ###################
        ## LOAD SPECTRA OBJECT
        ## ########
        spectra_obj = loadPickleObject(
            # createFilepath([filepath_base, str(res), SONIC_REGIME, sim_folder, sub_folder]),
            createFilepath([filepath_base, str(sim_res), sim_folder, sub_folder]),
            SPECTRA_NAME
        )
        ## ##########################
        ## SAVE IMPORTANT INFORMATION
        ## ########
        ## measure predicted dissipation scales
        ## (since this program only acts on the same simulation setups, overwritten variables don't matter)
        Re = int(spectra_obj.Re)
        Rm = int(spectra_obj.Rm)
        ## ##########################
        ## GET DISTRIBUTION OF SCALES
        ## ########
        ## check that a time range has been defined to collect statistics about
        sim_times = getCommonElements(spectra_obj.vel_sim_times, spectra_obj.mag_sim_times)
        # bool_vel_fit = (spectra_obj.vel_fit_start_t is not None) and (spectra_obj.vel_fit_end_t is not None)
        # bool_mag_fit = (spectra_obj.mag_fit_start_t is not None) and (spectra_obj.mag_fit_end_t is not None)
        # if not(bool_vel_fit) or not(bool_mag_fit):
        #     raise Exception("Fit range has not been defined.")
        ## find indices of velocity fit time range
        vel_index_start = getIndexClosestValue(sim_times, 2) # spectra_obj.vel_fit_start_t)
        vel_index_end   = getIndexClosestValue(sim_times, 10) # spectra_obj.vel_fit_end_t)
        ## find indices of magnetic fit time range
        mag_index_start = getIndexClosestValue(sim_times, 2) # spectra_obj.mag_fit_start_t)
        mag_index_end   = getIndexClosestValue(sim_times, 10) # spectra_obj.mag_fit_end_t)
        ## subset measured scales
        list_k_nu  = cleanMeasuredScales(spectra_obj.k_nu_group_t[vel_index_start  : vel_index_end])
        list_k_eta = cleanMeasuredScales(spectra_obj.k_eta_group_t[mag_index_start : mag_index_end])
        list_k_max = cleanMeasuredScales(spectra_obj.k_max_group_t[mag_index_start : mag_index_end])
        list_k_nu_group_res.append(  list_k_nu )
        list_k_eta_group_res.append( list_k_eta )
        list_k_max_group_res.append( list_k_max )
        ## ####################
        ## PLOT MEASURED SCALES
        ## ########
        plotErrorBar(axs[0], data_x=[sim_res], data_y=list_k_nu,  color="black")
        plotErrorBar(axs[1], data_x=[sim_res], data_y=list_k_eta, color="black")
        plotErrorBar(axs[2], data_x=[sim_res], data_y=list_k_max, color="black")
    # ## ########################################
    # ## FITTING SCALES AS FUNCTION OF RESOLUTION
    # ## ########################################
    # ## initialise fitting bounds
    # bounds = ((0.01, 1, 0.5), (15, 1000, 3))
    # print("Fitting curves...")
    # ## fit to k_nu vs N_res
    # if np.mean(list_k_nu_group_res[0]) < np.mean(list_k_nu_group_res[-1]):
    #     ## measured k_nu scale increased with resolution
    #     list_k_nu_converged = funcFitScales(
    #         ax           = axs[0],
    #         list_x       = list_res,
    #         list_y_group = list_k_nu_group_res,
    #         func         = ListOfModels.logistic_growth_increasing,
    #         bounds       = bounds,
    #         label_scale  = r"$k_\nu$",
    #         bool_debug   = bool_debug,
    #         absolute_sigma = list_bool_abs_std[0]
    #     )
    # else:
    #     ## measured k_nu scale decreased with resolution
    #     list_k_nu_converged = funcFitScales(
    #         ax           = axs[0],
    #         list_x       = list_res,
    #         list_y_group = list_k_nu_group_res,
    #         func         = ListOfModels.logistic_growth_decreasing,
    #         bounds       = bounds,
    #         label_scale  = r"$k_\nu$",
    #         bool_debug   = bool_debug,
    #         absolute_sigma = list_bool_abs_std[0]
    #     )
    # ## fit to k_eta vs N_res
    # list_k_eta_converged = funcFitScales(
    #     ax           = axs[1],
    #     list_x       = list_res,
    #     list_y_group = list_k_eta_group_res,
    #     func         = ListOfModels.logistic_growth_increasing,
    #     bounds       = bounds,
    #     label_scale  = r"$k_\eta$",
    #     bool_debug   = bool_debug,
    #     absolute_sigma = list_bool_abs_std[1]
    # )
    # ## fit to k_max vs N_res
    # list_k_max_converged = funcFitScales(
    #     ax           = axs[2],
    #     list_x       = list_res,
    #     list_y_group = list_k_max_group_res,
    #     func         = ListOfModels.logistic_growth_increasing,
    #     bounds       = bounds,
    #     label_scale  = r"$k_p$",
    #     bool_debug   = bool_debug,
    #     absolute_sigma = list_bool_abs_std[2]
    # )
    ## ###############
    ## LABELING FIGURE
    ## ###############
    print("Labeling figure...")
    ## remove shared x-axis labels
    axs[0].set_xticklabels([])
    axs[1].set_xticklabels([])
    ## label x-axis
    axs[2].set_xlabel(r"Linear resolution ($N_\text{res}$)", fontsize=22)
    ## #############
    ## SAVING FIGURE
    ## #############
    print("Saving figure...")
    fig_name = sim_folder+"_scales_res.pdf"
    fig_filepath = createFilepath([filepath_plot, fig_name])
    plt.savefig(fig_filepath)
    print("\t> Figure saved: " + fig_name)
    ## close plot
    plt.close(fig)

    # ## ####################
    # ## SAVING SCALES OBJECT
    # ## ####################
    # print("Saving spectra scales object...")
    # spectra_scale_obj = SpectraScales(
    #     ## simulation setup information
    #     Pm = Re / Rm,
    #     ## converged scales
    #     list_k_nu_converged  = list_k_nu_converged,
    #     list_k_eta_converged = list_k_eta_converged,
    #     list_k_max_converged = list_k_max_converged
    # )
    # savePickleObject(
    #     spectra_scale_obj,
    #     filepath_base,
    #     sim_folder+SCALE_NAME
    # )


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
    main()
    sys.exit()


## END OF PROGRAM
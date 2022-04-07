#!/usr/bin/env python3

##################################################################
## MODULES
##################################################################
## import modules
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
## use a non-interactive plotting backend
plt.switch_backend('agg')

## import user defined libraries
from OldModules.the_useful_library import *
from OldModules.the_loading_library import *
from OldModules.the_fitting_library import *
from OldModules.the_plotting_library import *


##################################################################
## PREPARE TERMINAL / WORKSPACE / CODE
#################################################################
os.system("clear") # clear terminal window


##################################################################
## FUNCTIONS
#################################################################
def funcPlotFrame(
        ax,
        filepath_figures,
        time_val, time_index,
        k = None, power = None,
        fit_k = None, fit_power = None, k_scale = None
    ):
    ## domain bounds
    y_min = 1e-30 # -5
    y_max = 1e-20 # +6
    x_min = 1e-1
    x_max = 3e+2
    ## plot spectra
    ax.plot(
        k, power,
        label=r"spectra data", color="blue", ls="", marker=".", markersize=8
    )

    x_data = np.linspace(1, 300, 300)

    # y_data = SpectraModels.kinetic_linear(x_data, 300, -1.5, 1/10)
    # ax.plot(x_data, y_data, label=r"referrence", color="black", ls="-")

    y_data = SpectraModels.magnetic_linear(x_data, 5*10**(-25), 3, 1/1.75)
    ax.plot(x_data, y_data, label=r"referrence", color="black", ls="-")

    # ## plot fitted spectra
    # ax.plot(
    #     fit_k, fit_power,
    #     label=r"spectra fit", color="red", ls="-", marker="", markersize=8
    # )
    # ## plot measured scales
    # ax.axvline(x=k_scale, color="green",  ls="--", label=r"$k_{\mathrm{scale}}$")
    ## add legend
    ax.legend(frameon=True, loc="upper left", facecolor="white", framealpha=0.5, fontsize=12)
    ## adjust figure axes
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ## label axes
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\mathcal{P}$")
    ax.text(
        0.95, 0.95,
        r"$t / T = {}$".format(str(time_val)),
        ha="right", va="top", transform=ax.transAxes
    )
    ## save image
    fig_name = createFilepath([
        filepath_figures,
        "spectra={0:04}.png".format(time_index)
    ])
    plt.savefig(fig_name)
    ## clear axis
    ax.clear()

def funcPlotError(
        ax, filepath_figures,
        time_val, time_index,
        list_fit_k_range, list_fit_2norm_group_t, fit_k_index_group_t
    ):
    ## plot errors
    ax.plot(list_fit_k_range, list_fit_2norm_group_t, "k.")
    ax.axvline(x=fit_k_index_group_t, ls="--", color="k")
    ## label axes
    ax.set_xlabel(r"Number of fitted points")
    ax.set_ylabel(r"Error")
    ## number of fitted points
    ax.text(
        0.05, 0.95,
        r"$n = {}$".format(str(fit_k_index_group_t)),
        ha="left", va="top", transform=ax.transAxes
    )
    ## time point
    ax.text(
        0.95, 0.95,
        r"$t / T = {}$".format(str(time_val)),
        ha="right", va="top", transform=ax.transAxes
    )
    ## save figure
    fig_name = createFilepath([
        filepath_figures,
        "errors={0:04}.png".format(time_index)
    ])
    plt.savefig(fig_name)
    ## clear axis
    ax.clear()


##################################################################
## INPUT COMMAND LINE ARGUMENTS
##################################################################
ap = argparse.ArgumentParser(description="A bunch of input arguments")
## ------------------- DEFINE ARGUMENTS
ap.add_argument("-vis_folder",  type=str, required=False, default="vis_folder")
ap.add_argument("-base_path",   type=str, required=True)
ap.add_argument("-sim_folders", type=str, required=True, nargs="+")
## ---------------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ---------------------------- SAVE FILEPATH PARAMETERS
filepath_base = args["base_path"]
sim_folders   = args["sim_folders"]
folder_vis    = args["vis_folder"]


bool_hide_updates = False
##################################################################
## MAIN PROGRAM
##################################################################
for sim_index in range(len(sim_folders)):
    print("Fitting: " + sim_folders[sim_index])
    ## get path to data folder
    filepath_data = createFilepath([filepath_base, sim_folders[sim_index]])
    ## create folder where spectra plots will be saved
    filepath_figures = createFilepath([filepath_base, folder_vis, sim_folders[sim_index]])
    createFolder(filepath_figures, bool_hide_updates)
    ## ###########################
    ## LOAD SPECTRA DATA
    ## ###########################
    print("\t> Loading spectra...")
    str_field = "mag"
    k_group_times, power_group_times, sim_times = loadListSpectra(
        filepath_data,
        str_spectra_type  = str_field,
        file_start_index  = 145,
        file_end_index    = 150,
        bool_hide_updates = bool_hide_updates
    )
    # ## ###########################
    # ## FIT SPECTRA DATA
    # ## ###########################
    # print("\t> Fitting spectra...")
    # ## fit spectra
    # fit_args = FitVelSpectra(
    #     k_group_times, power_group_times, sim_times,
    #     bool_fixed_model = bool_hide_updates
    # ).getFitArgs() # get fit arguments
    # ## extract fit arguments
    # sim_times               = fit_args[str_field+"_sim_times"]
    # list_fit_k_group_t      = fit_args[str_field+"_list_fit_k_group_t"]
    # list_fit_power_group_t  = fit_args[str_field+"_list_fit_power_group_t"]
    # k_scale_group_t         = fit_args["k_nu_group_t"]
    # fit_k_index_group_t     = fit_args[str_field+"_fit_k_index_group_t"]
    # list_fit_k_range        = fit_args[str_field+"_list_fit_k_range"]
    # list_fit_2norm_group_t  = fit_args[str_field+"_list_fit_2norm_group_t"]
    # list_fit_params_group_t = fit_args[str_field+"_list_fit_params_group_t"]
    # for sub_list in list_fit_params_group_t:
    #     print(*sub_list)
    # ## ###########################
    # ## PLOT FIT ERRORS
    # ## ############
    # print("\t> Plotting errors...")
    # ## initialise figure
    # fig, ax = plt.subplots(constrained_layout=True)
    # ## extract fit parameters (as function of time realisations)
    # for time_val, time_index in loopListWithUpdates(sim_times[::2]):
    #     funcPlotError(
    #         ax, filepath_figures,
    #         time_val, time_index,
    #         list_fit_k_range,
    #         list_fit_2norm_group_t[time_index],
    #         fit_k_index_group_t[time_index]
    #     )
    # ## close figure
    # plt.close(fig)
    # ## ############
    # ## ANIMATE ERROR FRAMES
    # ## ###########################
    # if len(sim_times) > 3:
    #     aniEvolution(
    #         filepath_frames    = filepath_figures,
    #         filepath_ani_movie = filepath_figures,
    #         input_name         = "errors=%*.png",
    #         output_name        = "ani_errors.mp4",
    #         bool_hide_updates  = True
    #     )
    ## ###########################
    ## PLOT SPECTRA FRAMES
    ## ############
    ## initialise figure
    fig, ax = plt.subplots(constrained_layout=True)
    ## loop over spectra
    print("\t> Plotting spectra...")
    for time_val, time_index in loopListWithUpdates(sim_times[::2]):
        ## plot and save the time frame
        funcPlotFrame(
            ax, filepath_figures,
            time_val, time_index,
            k_group_times[time_index],
            power_group_times[time_index],
            # list_fit_k_group_t[time_index],
            # list_fit_power_group_t[time_index],
            # k_scale_group_t[time_index]
        )
    ## close figure
    plt.close(fig)
    ## ############
    ## ANIMATE SPLECTRA FRAMES
    ## ###########################
    if len(sim_times) > 3:
        aniEvolution(
            filepath_frames    = filepath_figures,
            filepath_ani_movie = filepath_figures,
            input_name         = "spectra=%*.png",
            output_name        = "ani_spectra.mp4",
            bool_hide_updates  = True
        )
    print(" ")


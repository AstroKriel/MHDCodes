#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import argparse
import numpy as np
import cmasher as cmr # https://cmasher.readthedocs.io/user/diverging.html

from matplotlib.gridspec import GridSpec

## load old user defined modules
from the_matplotlib_styler import *
from OldModules.the_useful_library import *
from OldModules.the_loading_library import *
from OldModules.the_fitting_library import *
from OldModules.the_plotting_library import *


## ###############################################################
## PREPARE TERMINAL/WORKSPACE/CODE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend('agg') # use a non-interactive plotting backend


## ###############################################################
## FUNCTIONS
## ###############################################################
def funcPlotParams_sim( 
        ax, label_var,
        list_Pm, list_params, param_index, y_label,
        p0         = None,
        bounds     = None,
        bool_debug = False
    ):
    ## #############
    ## PLOT DATA
    ## ######
    ## for each simulation (Pm)
    for group_index in range(len(list_Pm)):
        plotErrorBar(
            ax,
            data_x = [ list_Pm[group_index] ],
            ## characteristic Nres (Nc) [1], convergence rate [2]
            data_y = list_params[group_index][param_index],
            color  = "black"
        )
    ## ################
    ## FIT DISTRIBUTION
    ## #######
    list_fit_params = plotKDEFit(
        ax      = ax,
        var_str = r"Pm",
        input_x = list_Pm,
        input_y = [
            ## characteristic Nres (Nc) [1], convergence rate [2]
            list_params[group_index][param_index]
            for group_index in range(len(list_params))
        ],
        func_label = "PowerLaw",
        func_fit   = ListOfModels.powerlaw_log10,
        func_plot  = ListOfModels.powerlaw_linear,
        bool_log_fit = True,
        list_indices_unlog = [0],
        p0         = p0,
        bounds     = bounds,
        num_resamp = 10**3,
        maxfev     = 10**3,
        num_digits = 2,
        bool_debug = bool_debug,
        plot_args  = { "x":0.95, "y":0.95, "va":"top", "ha":"right", "color":"black" }
    )
    ## print statistics of the fit
    print("p1=0.16, p2=0.50, x1={:0.5f}, x2={:0.5f}".format(
        np.percentile(list_fit_params[0], 16),
        np.percentile(list_fit_params[0], 84)
    ))
    print("p1=0.16, p2=0.50, x1={:0.5f}, x2={:0.5f}".format(
        np.percentile(list_fit_params[1], 16),
        np.percentile(list_fit_params[1], 84)
    ))
    print(" ")
    ## ############
    ## LABEL FIGURE
    ## ########
    ax.text(
        0.05, 0.05, label_var,
        va="bottom", ha="left", fontsize=16, transform=ax.transAxes,
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round', alpha=0.85)
    )
    ax.set_ylabel(y_label, fontsize=20)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ## check that the y-axis domain is appropriate
    FixLogAxis(ax, bool_fix_x_axis=True, bool_fix_y_axis=True)

def funcPlotParams(
        filepath_plot, fig_name,
        list_Pm, param_index, y_label,
        list_k_nu_converge_params, list_k_eta_converge_params, list_k_max_converge_params,
        bool_debug
    ):
    ## initialise figure
    fig, axs = plt.subplots(3, 1, figsize=(6, 12), sharex=True)
    fig.subplots_adjust(hspace=0.05)
    ## #############
    ## PLOTTING DATA
    ## ######
    if len(list_Pm) > 1:
        bounds = ((np.log(1e-3), -2), (np.log(300), 2))
        ## plot k_nu data
        funcPlotParams_sim(
            axs[0], r"$k_\nu$",
            list_Pm     = list_Pm,
            list_params = list_k_nu_converge_params,
            param_index = param_index,
            y_label     = y_label,
            ## get guess by fitting to median of distributions
            p0 = curve_fit(
                f = ListOfModels.powerlaw_log10,
                xdata = np.log(list_Pm),
                ydata = np.log([
                    np.median(list_k_nu[param_index])
                    for list_k_nu in list_k_nu_converge_params
                ]),
                bounds = bounds
            )[0],
            bounds     = bounds,
            bool_debug = bool_debug 
        )
        ## plot k_eta data
        funcPlotParams_sim(
            axs[1], r"$k_\eta$",
            list_Pm     = list_Pm,
            list_params = list_k_eta_converge_params,
            param_index = param_index,
            y_label     = y_label,
            ## get guess by fitting to median of distributions
            p0 = curve_fit(
                f = ListOfModels.powerlaw_log10,
                xdata = np.log(list_Pm),
                ydata = np.log([
                    np.median(list_k_eta[param_index])
                    for list_k_eta in list_k_eta_converge_params
                ]),
                bounds = bounds
            )[0],
            bounds     = bounds,
            bool_debug = bool_debug
        )
        ## plot k_max data
        funcPlotParams_sim(
            axs[2], r"$k_p$",
            list_Pm     = list_Pm,
            list_params = list_k_max_converge_params,
            param_index = param_index,
            y_label     = y_label,
            ## get guess by fitting to median of distributions
            p0 = curve_fit(
                f = ListOfModels.powerlaw_log10,
                xdata = np.log(list_Pm[param_index]),
                ydata = np.log([
                    np.median(list_k_max)
                    for list_k_max in list_k_max_converge_params
                ]),
                bounds = bounds
            )[0],
            bounds     = bounds,
            bool_debug = bool_debug
        )
    ## label domain
    axs[2].set_xlabel(r"Pm", fontsize=20)
    ## #############
    ## SAVING FIGURE
    ## ######
    fig_filepath = createFilePath([filepath_plot, fig_name])
    plt.savefig(fig_filepath)
    print("\t> Figure saved:", fig_name)
    ## close plot
    plt.close(fig)


## ###############################################################
## DEFINE COMMAND LINE ARGUMENTS
## ###############################################################
parser = MyParser()
## ------------------- DEFINE OPTIONAL ARGUMENTS
args_opt = parser.add_argument_group(description='Optional processing arguments:')
optional_bool_args = {"required":False, "type":str2bool, "nargs":"?", "const":True}
args_opt.add_argument("-debug",      default=False, **optional_bool_args)
args_opt.add_argument("-vis_folder", type=str, default="vis_folder", required=False)
## ------------------- DEFINE REQUIRED ARGUMENTS
args_req = parser.add_argument_group(description='Required processing arguments:')
args_req.add_argument("-base_path",   type=str, required=True)
args_req.add_argument("-sim_folders", type=str, required=True, nargs="+")
args_req.add_argument("-sim_suite",   type=str, required=True)


## ###############################################################
## INTERPRET INPUT ARGUMENTS
## ###############################################################
## ---------------------------- OPEN ARGUMENTS
args = vars(parser.parse_args())
## ---------------------------- SAVE PARAMETERS
## required parameters
filepath_base    = args["base_path"]
sim_suite        = args["sim_suite"]
list_sim_folders = args["sim_folders"]
## optional parameters
bool_debug = args["debug"]
folder_vis = args["vis_folder"]


## ###############################################################
## PRINT CONFIGURATION INFORMATION TO CONSOLE
## ###############################################################
## create plot folder
filepath_plot = createFilePath([filepath_base, folder_vis])
createFolder(filepath_plot)
## print input information
printInfo("Suite filepath:",       filepath_base, 22)
printInfo("Simulation folder(s):", list_sim_folders, 22)
printInfo("Figure folder:",        filepath_plot, 22)
printInfo("Suite name:",           sim_suite,     22)
print(" ")


## ###############################################################
## LOADING DATA
## ###############################################################
## list of simulation Pm
list_Pm = []
## initialise list of simulation convergence parameters
list_k_nu_converge_params  = []
list_k_eta_converge_params = []
list_k_max_converge_params = []
## for each simulation setup: load and plot scales predicted by proxy vs theory
print("Loading data...")
for sim_folder, group_index in zip(list_sim_folders, range(len(list_sim_folders))):
    ## ###################
    ## LOAD SPECTRA OBJECT
    ## ############
    converge_obj = loadPickleObject(filepath_base, sim_folder+"_scale_converge_obj.pkl")
    ## ##########################
    ## SAVE IMPORTANT INFORMATION
    ## ##############
    ## load simulation's Pm
    list_Pm.append(converge_obj.Pm)
    ## load convergencing fit parameters
    list_k_nu_converge_params.append(converge_obj.k_nu_converge_params)
    list_k_eta_converge_params.append(converge_obj.k_eta_converge_params)
    list_k_max_converge_params.append(converge_obj.k_max_converge_params)
print(" ")


## ###############################################################
## PLOTTING
## ###############################################################
print("Saving figures in:", filepath_plot)

## #############################
## PLOTTING ALL 'Nc' DISTRIBUTIONS
## ############
funcPlotParams(
    filepath_plot = filepath_plot,
    fig_name      = sim_suite+"_converge_Nc.pdf",
    list_Pm       = list_Pm,
    param_index   = 1,
    y_label       = r"$N_c$",
    list_k_nu_converge_params  = list_k_nu_converge_params,
    list_k_eta_converge_params = list_k_eta_converge_params,
    list_k_max_converge_params = list_k_max_converge_params,
    bool_debug = bool_debug
)

funcPlotParams(
    filepath_plot = filepath_plot,
    fig_name      = sim_suite+"_converge_rate.pdf",
    list_Pm       = list_Pm,
    param_index   = 2,
    y_label       = r"$\alpha$",
    list_k_nu_converge_params  = list_k_nu_converge_params,
    list_k_eta_converge_params = list_k_eta_converge_params,
    list_k_max_converge_params = list_k_max_converge_params,
    bool_debug = bool_debug
)


## END OF PROGRAM
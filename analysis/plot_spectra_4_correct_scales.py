#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import argparse
import numpy as np
import cmasher as cmr # https://cmasher.readthedocs.io/user/diverging.html

from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection

## user defined libraries
from the_matplotlib_styler import *
from OldModules.the_useful_library import *
from OldModules.the_loading_library import *
from OldModules.the_fitting_library import *
from OldModules.the_plotting_library import *


## ###############################################################
## PREPARE TERMINAL/WORKSPACE/CODE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## FUNCTIONS
## ###############################################################
def funcAddLegend(
        ax, artists, legend_labels,
        colors         = ["k"],
        title          = None,
        bool_place_top = False
    ):
    ## check that the inputs are the correct length
    if len(artists) < len(legend_labels): artists.extend( artists[0] * len(legend_labels) )
    if len(colors) < len(legend_labels): colors.extend( colors[0] * len(legend_labels) )
    ## useful lists
    list_markers   = ["o", "s", "D", "^", "v"] # list of marker styles
    list_lines     = ["-", "--", "-."] # list of line styles
    ## iniialise list of artists for legend
    legend_artists = []
    ## create legend artists
    for artist, color in  zip(artists, colors):
        ## if the artist is a marker
        if artist in list_markers:
            legend_artists.append( Line2D([0], [0], marker=artist, color=color, linewidth=0) )
        ## if the artist is a line
        elif artist in list_lines:
            legend_artists.append( Line2D([0], [0], linestyle=artist, color=color, linewidth=2) )
        ## otherwise throw an error
        else: raise Exception("Artist '{}' is not a valid marker- or line-style.".format(artist))
    ## draw the legend
    if bool_place_top:
        legend = ax.legend(
            legend_artists, legend_labels, title=title,
            frameon=True, loc="lower center", bbox_to_anchor=(0.5, 1.0),
            columnspacing=1.0, handletextpad=0.4,
            facecolor="white", framealpha=0.5, fontsize=14, ncol=len(legend_artists)
        )
    else: legend = ax.legend(
            legend_artists, legend_labels, title=title,
            frameon=True, loc="lower left", bbox_to_anchor=(1.0, 0.0),
            labelspacing=1.2,
            facecolor="white", framealpha=0.5, fontsize=14, ncol=1
        )
    ## add legend
    ax.add_artist(legend)

def funcFitScale(fit_args_ax, bool_debug=False, list_bool_fits=[True, True, True]):
    list_bool_fits.append(
        [list_bool_fits[0]] * (3 - len(list_bool_fits))
    )
    ## fit with linear line
    if list_bool_fits[0]:
        plotKDEFit(
            **fit_args_ax,
            func_label = "linear",
            func       = ListOfModels.linear,
            num_resamp = 10**3,
            num_fit    = 10**3,
            bounds     = (0, 10),
            num_digits = 2,
            plot_args  = { "x":0.05, "y":0.95, "va":"top", "ha":"left", "color":"black" },
            bool_debug = bool_debug
        )
    ## fit with linear line + offset
    if list_bool_fits[1]:
        plotKDEFit(
            **fit_args_ax,
            func_label = "linear_offset",
            func       = ListOfModels.linear_offset,
            num_resamp = 10**3,
            num_fit    = 10**3,
            bounds     = (0, (5, 10)),
            num_digits = 2,
            plot_args  = { "x":0.95, "y":0.2, "va":"bottom", "ha":"right", "color":"red" }
        )
    ## fit with power-law
    if list_bool_fits[2]:
        plotKDEFit(
            **fit_args_ax,
            func_label = "PowerLaw",
            func       = ListOfModels.powerlaw_linear,
            num_resamp = 10**3,
            num_fit    = 10**3,
            bounds     = (0, (10, 3)),
            p0         = (0.01, 1),
            num_digits = 2,
            plot_args  = { "x":0.95, "y":0.05, "va":"bottom", "ha":"right", "color":"blue" }
        )

## normal distribution from percentiles:
## https://veryjoe.com/maths/2020/04/27/Parameter-Estimation.html

class FixedRe10():
    def N_crit_k_nu(Pm, num_samples):
        return np.random.normal(0.8, 0.15, num_samples)
    def N_crit_k_eta(Pm, num_samples):
        a0 = sampleGaussFromQuantiles(p1=0.16, p2=0.50, x1=1.07502, x2=4.38288, num_samples=num_samples)
        a1 = sampleGaussFromQuantiles(p1=0.16, p2=0.50, x1=0.32331, x2=0.53283, num_samples=num_samples)
        return a0 * (Pm)**(a1)
    def N_crit_k_max(Pm, num_samples):
        a0 = sampleGaussFromQuantiles(p1=0.16, p2=0.50, x1=1.32171, x2=5.54656, num_samples=num_samples)
        a1 = sampleGaussFromQuantiles(p1=0.16, p2=0.50, x1=0.32965, x2=0.53855, num_samples=num_samples)
        return a0 * (Pm)**(a1)

class FixedRm3000():
    def N_crit_k_nu(Pm, num_samples):
        a0 = sampleGaussFromQuantiles(p1=0.16, p2=0.50, x1=153.84873, x2=218.61353, num_samples=num_samples)
        a1 = sampleGaussFromQuantiles(p1=0.16, p2=0.50, x1=-1.05016, x2=-0.82917, num_samples=num_samples)
        return a0 * (Pm)**(a1)
    def N_crit_k_eta(Pm, num_samples):
        a0 = sampleGaussFromQuantiles(p1=0.16, p2=0.50, x1=182.19866, x2=229.31726, num_samples=num_samples)
        a1 = sampleGaussFromQuantiles(p1=0.16, p2=0.50, x1=-0.34942, x2=-0.22361, num_samples=num_samples)
        return a0 * (Pm)**(a1)
    def N_crit_k_max(Pm, num_samples):
        a0 = sampleGaussFromQuantiles(p1=0.16, p2=0.50, x1=52.92241, x2=67.39089, num_samples=num_samples)
        a1 = sampleGaussFromQuantiles(p1=0.16, p2=0.50, x1=-0.13657, x2=-0.04338, num_samples=num_samples)
        return a0 * (Pm)**(a1)

class MixedReRm():
    def N_crit_k_eta(Pm, num_samples):
        a0 = sampleGaussFromQuantiles(p1=0.16, p2=0.50, x1=15.58734, x2=41.36182, num_samples=num_samples)
        a1 = sampleGaussFromQuantiles(p1=0.16, p2=0.50, x1=-0.07554, x2=0.12164, num_samples=num_samples)
        return a0 * (Pm)**(a1)
    def N_crit_k_max(Pm, num_samples):
        a0 = sampleGaussFromQuantiles(p1=0.16, p2=0.50, x1=17.49434, x2=32.25419, num_samples=num_samples)
        a1 = sampleGaussFromQuantiles(p1=0.16, p2=0.50, x1=-0.07032, x2=0.06491, num_samples=num_samples)
        return a0 * (Pm)**(a1)

def funcGetData(
        filepath_data,
        list_Pm,
        list_relation_k_nu, list_relation_k_eta,
        list_k_nu_conv_group, list_k_eta_conv_group, list_k_max_conv_group,
        func_k_nu  = None,
        func_k_eta = None,
        func_k_max = None
    ):
    ## load spectra object
    spectra_obj = loadPickleObject(filepath_data, "spectra_obj.pkl")
    ## ##########################
    ## GET SIMULATION INFORMATION
    ## ########
    sim_res = spectra_obj.sim_res
    if sim_res is None:
        raise Exception("Simulation resolution has not been defined.")
    Pm = int(spectra_obj.Pm)
    Re = int(spectra_obj.Re)
    Rm = int(spectra_obj.Rm)
    if "sub" in spectra_obj.sim_suite.lower():
        relation_k_nu = 1 * (Re)**(3/4) # sub-sonic
    else: relation_k_nu = 1 * (Re)**(2/3) # super-sonic
    relation_k_eta = relation_k_nu * (Rm / Re)**(1/2)
    ## save information
    list_Pm.append(Pm)
    list_relation_k_nu.append(relation_k_nu)
    list_relation_k_eta.append(relation_k_eta)
    ## ##########################
    ## GET DISTRIBUTION OF SCALES
    ## ########
    ## check that a time range has been defined to collect statistics about
    sim_times = getCommonElements(spectra_obj.vel_sim_times, spectra_obj.mag_sim_times)
    bool_vel_fit = (spectra_obj.vel_start_fit is not None) and (spectra_obj.vel_end_fit is not None)
    bool_mag_fit = (spectra_obj.mag_start_fit is not None) and (spectra_obj.mag_end_fit is not None)
    if not(bool_vel_fit) or not(bool_mag_fit):
        raise Exception("Fit range has not been defined.")
    ## find indices of velocity fit time range
    vel_index_start = getIndexClosestValue(sim_times, spectra_obj.vel_start_fit)
    vel_index_end   = getIndexClosestValue(sim_times, spectra_obj.vel_end_fit)
    ## find indices of magnetic fit time range
    mag_index_start = getIndexClosestValue(sim_times, spectra_obj.mag_start_fit)
    mag_index_end   = getIndexClosestValue(sim_times, spectra_obj.mag_end_fit)
    ## subset measured scalesi
    list_k_nu  = cleanMeasuredScales(spectra_obj.k_nu[vel_index_start  : vel_index_end])
    list_k_eta = cleanMeasuredScales(spectra_obj.k_eta[mag_index_start : mag_index_end])
    list_k_max = cleanMeasuredScales(spectra_obj.k_max[mag_index_start : mag_index_end])
    ## correct measured k_nu
    if func_k_nu is not None:
        num_scales_k_nu = len(list_k_nu) # get number of samples
        list_N_crit_k_nu = func_k_nu(Pm, num_scales_k_nu) # get corrections
        list_N_crit_k_nu[list_N_crit_k_nu <= 0] = np.median(list_N_crit_k_nu[list_N_crit_k_nu > 0]) # make sure non-zero corrections
        list_k_nu_conv_group.append([
            list_k_nu[k_nu_index] / ( 1 - np.exp(-sim_res / list_N_crit_k_nu[k_nu_index]) )
            for k_nu_index in range(num_scales_k_nu)
        ])
    else: list_k_nu_conv_group.append(list_k_nu)
    ## correct measured k_eta
    if func_k_eta is not None:
        num_scales_k_eta = len(list_k_eta) # get number of samples
        list_N_crit_k_eta = func_k_eta(Pm, num_scales_k_eta) # get corrections
        list_N_crit_k_eta[list_N_crit_k_eta <= 0] = np.median(list_N_crit_k_eta[list_N_crit_k_eta > 0]) # make sure non-zero corrections
        list_k_eta_conv_group.append([
            list_k_eta[k_eta_index] / ( 1 - np.exp(-sim_res / list_N_crit_k_eta[k_eta_index]) )
            for k_eta_index in range(num_scales_k_eta)
        ])
    else: list_k_nu_conv_group.append(list_k_eta)
    ## correct measured k_max
    if func_k_max is not None:
        num_scales_k_max = len(list_k_max) # get number of samples
        list_N_crit_k_max = func_k_max(Pm, num_scales_k_max) # get corrections
        list_N_crit_k_max[list_N_crit_k_max <= 0] = np.median(list_N_crit_k_max[list_N_crit_k_max > 0]) # make sure non-zero corrections
        list_k_max_conv_group.append([
            list_k_max[k_max_index] / ( 1 - np.exp(-sim_res / list_N_crit_k_max[k_max_index]) )
            for k_max_index in range(num_scales_k_max)
        ])
    else: list_k_nu_conv_group.append(list_k_max)


## ###############################################################
## DEFINE COMMAND LINE ARGUMENTS
## ###############################################################
parser = MyParser()
## ------------------- DEFINE OPTIONAL ARGUMENTS
args_opt = parser.add_argument_group(description="Optional processing arguments:")
optional_bool_args = {"required":False, "type":str2bool, "nargs":"?", "const":True}
args_opt.add_argument("-debug",      default=False, **optional_bool_args)
args_opt.add_argument("-fit_ax0",    default=True,  **optional_bool_args)
args_opt.add_argument("-fit_ax1",    default=True,  **optional_bool_args)
args_opt.add_argument("-vis_folder", type=str, default="vis_folder", required=False)
args_opt.add_argument("-sub_folder", type=str, default="spect",      required=False)
## ------------------- DEFINE REQUIRED ARGUMENTS
args_req = parser.add_argument_group(description="Required processing arguments:")
args_req.add_argument("-base_path",   type=str, required=True)
args_req.add_argument("-sim_folders", type=str, required=True, nargs="+")
args_req.add_argument("-fig_name",    type=str, required=True)


## ###############################################################
## INTERPRET INPUT ARGUMENTS
## ###############################################################
## ---------------------------- OPEN ARGUMENTS
args = vars(parser.parse_args())
## ---------------------------- SAVE PARAMETERS
bool_debug       = args["debug"]
bool_fit_ax0     = args["fit_ax0"]
bool_fit_ax1     = args["fit_ax1"]
## simulation information
filepath_base    = args["base_path"]
list_sim_folders = args["sim_folders"]
folder_sub       = args["sub_folder"]
## plotting information
folder_vis       = args["vis_folder"]
fig_name         = args["fig_name"]


## ###############################################################
## PREPARING DIRECTORIES
## ###############################################################
## folders where spectra data is
filepaths_data = []
for sim_index in range(len(list_sim_folders)):
    filepaths_data.append(
        createFilePath([filepath_base, list_sim_folders[sim_index], folder_sub])
    )
## folder where visualisations will be saved
filepath_plot = createFilePath([filepath_base, folder_vis])
## create folder
createFolder(filepath_plot)


## ###############################################################
## PRINT CONFIGURATION INFORMATION TO CONSOLE
## ###############################################################
## print input information
printInfo("Base filepath:", filepath_base, 20)
for sim_index in range(len(filepaths_data)):
    printInfo(
        "({:d}) sim directory:".format(sim_index),
        filepaths_data[sim_index],
        20
    )
print(" ")


## ###############################################################
## LOADING DATA
## ###############################################################
## initialise fitting distributions
list_Pm = []
list_relation_k_nu  = []
list_relation_k_eta = []
list_k_nu_conv_group  = []
list_k_eta_conv_group = []
list_k_max_conv_group = []
## choose functions for correction
if "fixed_Re".lower() in fig_name.lower():
    func_k_nu  = FixedRe10.N_crit_k_nu
    func_k_eta = FixedRe10.N_crit_k_eta
    func_k_max = FixedRe10.N_crit_k_max
elif "fixed_Rm".lower() in fig_name.lower():
    func_k_nu  = FixedRm3000.N_crit_k_nu
    func_k_eta = FixedRm3000.N_crit_k_eta
    func_k_max = FixedRm3000.N_crit_k_max
else:
    func_k_nu  = None
    func_k_eta = MixedReRm.N_crit_k_eta
    func_k_max = MixedReRm.N_crit_k_max
## for each simulation: load and correct measured scales
print("Loading and correcting measured scales...")
for sim_index in range(len(list_sim_folders)):
    funcGetData(
        filepaths_data[sim_index],
        list_Pm,
        list_relation_k_nu, list_relation_k_eta,
        list_k_nu_conv_group, list_k_eta_conv_group, list_k_max_conv_group,
        func_k_nu  = func_k_nu,
        func_k_eta = func_k_eta,
        func_k_max = func_k_max
    )
print(" ")


## ###############################################################
## PLOTTING DATA
## ###############################################################
print("Plotting figures in:", filepath_plot)

## ##################################
## PLOTTING CONVERGED SCALE RELATIONS
## ##################
## extract colours from Cmasher"s colormap
num_sims = len(list_Pm)
cmasher_colormap = plt.get_cmap("cmr.tropical", num_sims)
my_colormap = cmasher_colormap(np.linspace(0, 1, num_sims))
## initialise figure
fig, axs = plt.subplots(2, 1, figsize=(7, 9))
## for each resolution run of a simulation setup: plot scale distributions
print("\tPlotting converged scales...")
for sim_index in range(num_sims):
    ## plot corrected scales
    plotErrorBar(
        axs[0],
        data_x = list_relation_k_nu[sim_index],
        data_y = list_k_nu_conv_group[sim_index],
        color  = my_colormap[sim_index]
    )
    plotErrorBar(
        axs[1],
        data_x = list_relation_k_eta[sim_index],
        data_y = list_k_eta_conv_group[sim_index],
        color  = my_colormap[sim_index]
    )
## fit k_nu scale relation
if reasonableFitDomain(list_relation_k_nu) and reasonableFitDomain(list_k_nu_conv_group):
    funcFitScale(
        {
            "ax":axs[0],
            "var_str":r"$k_\nu$",
            "input_x":list_relation_k_nu,
            "input_y":list_k_nu_conv_group
        },
        bool_debug = bool_debug
    )
## fit k_eta scale relation
if reasonableFitDomain(list_relation_k_eta) and reasonableFitDomain(list_k_eta_conv_group):
    funcFitScale(
        {
            "ax":axs[1],
            "var_str":r"$k_\eta$",
            "input_x":list_relation_k_eta,
            "input_y":list_k_eta_conv_group
        },
        bool_debug = bool_debug
    )
## #############
## LABELING
## ######
## add legend
funcAddLegend(
    ax             = axs[1],
    title          = r"Pm",
    artists        = ["o"],
    colors         = my_colormap[:num_sims],
    legend_labels  = [ r"${}$".format(str(Pm)) for Pm in list_Pm ],
    bool_place_top = False
)
## label axis
axs[0].set_xlabel(r"$k_\nu = {\rm Re}^{3/4}$")
axs[1].set_xlabel(r"$k_\eta = {\rm Pm}^{1/2} k_\nu$")
axs[0].set_ylabel(r"$k_\nu(N_{\rm{res}} = \infty)$ measured")
axs[1].set_ylabel(r"$k_\eta(N_{\rm{res}} = \infty)$ measured")
## adjust axis
axs[0].set_xscale("log")
axs[1].set_xscale("log")
axs[0].set_yscale("log")
axs[1].set_yscale("log")
## check axis ranges are appropriate
FixLogAxis(axs[0], bool_fix_x_axis=True, bool_fix_y_axis=True)
FixLogAxis(axs[1], bool_fix_x_axis=True, bool_fix_y_axis=True)
## #############
## SAVING FIGURE
## ######
this_fig_name = fig_name+"_scale_relations.pdf" # figure name
fig_filepath = createFilePath([filepath_plot, this_fig_name]) # filepath where figure is saved
plt.savefig(fig_filepath) # save the figure
plt.close(fig) # close the figure
print("\t> Figure saved:", this_fig_name)
print(" ")


## ############################################################
## PLOTTING CONVERGED "k_max" DEPENDANCE ON DISSIPATIONS SCALES
## ##################
## extract colours from Cmasher"s colormap
num_sims = len(list_Pm)
cmasher_colormap = plt.get_cmap("cmr.tropical", num_sims)
my_colormap = cmasher_colormap(np.linspace(0, 1, num_sims))
## initialise figure
fig, axs = plt.subplots(2, 1, figsize=(7, 9))
## for each resolution run of a simulation setup: load and plot scale distributions
print("\tPlotting 'k_max' dependance...")
for sim_index in range(num_sims):
    ## ####################
    ## PLOT CORRECTED SCALES
    ## ########
    plotErrorBar(
        axs[0],
        data_x = list_k_nu_conv_group[sim_index],
        data_y = list_k_max_conv_group[sim_index],
        color  = my_colormap[sim_index]
    )
    plotErrorBar(
        axs[1],
        data_x = list_k_eta_conv_group[sim_index],
        data_y = list_k_max_conv_group[sim_index],
        color  = my_colormap[sim_index]
    )
## ##################
## FITTING DEPENDANCE
## #######
## k_nu
if bool_fit_ax0:
    funcFitScale(
        {
            "ax":axs[0],
            "var_str":r"$k_\nu$",
            "input_x":list_k_nu_conv_group,
            "input_y":list_k_max_conv_group
        },
        bool_debug = bool_debug,
        list_bool_fits = [1, 0, 0]
    )
# ## add reference line
# x_domain = np.linspace(0.1, 20, 100)
# axs[0].add_collection(
#     LineCollection(
#         [
#             np.column_stack((
#                 x_domain,
#                 10 ** (0.35 * np.log10(x_domain) + np.log10(4.25))
#             ))
#         ],
#         linestyle="--", color="black", alpha=1
#     ),
#     autolim = False # ignore these points when setting the axis limits
# )
## k_eta
if bool_fit_ax1:
    funcFitScale(
        {
            "ax":axs[1],
            "var_str":r"$k_\eta$",
            "input_x":list_k_eta_conv_group,
            "input_y":list_k_max_conv_group
        }, 
        bool_debug = bool_debug,
        list_bool_fits = [1, 0, 1]
    )
## ###############
## LABELING FIGURE
## ########
## add legend
funcAddLegend(
    ax             = axs[1],
    title          = r"Pm",
    artists        = ["o"],
    colors         = my_colormap[:num_sims],
    legend_labels  = [ r"${}$".format(str(Pm)) for Pm in list_Pm ],
    bool_place_top = False
)
## label axis
axs[0].set_xlabel(r"$k_\nu(N_{\rm{res}} = \infty)$ measured")
axs[1].set_xlabel(r"$k_\eta(N_{\rm{res}} = \infty)$ measured")
axs[0].set_ylabel(r"$k_p(N_{\rm{res}} = \infty)$ measured")
axs[1].set_ylabel(r"$k_p(N_{\rm{res}} = \infty)$ measured")
## adjust axis
axs[0].set_xscale("log")
axs[1].set_xscale("log")
axs[0].set_yscale("log")
axs[1].set_yscale("log")
## check axis ranges are appropriate
FixLogAxis(axs[0], bool_fix_x_axis=True, bool_fix_y_axis=True)
FixLogAxis(axs[1], bool_fix_x_axis=True, bool_fix_y_axis=True)
## #############
## SAVING FIGURE
## ######
this_fig_name = fig_name+"_scale_dependance.pdf" # figure name
fig_filepath = createFilePath([filepath_plot, this_fig_name]) # filepath where figure is saved
plt.savefig(fig_filepath) # save the figure
plt.close(fig) # close the figure
print("\t> Figure saved:", this_fig_name)


## END OF PROGRAM
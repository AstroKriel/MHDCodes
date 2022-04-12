#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import numpy as np
import cmasher as cmr # https://cmasher.readthedocs.io/user/introduction.html#colormap-overview
import matplotlib as mpl

from matplotlib.collections import LineCollection

## load old user defined modules
from the_matplotlib_styler import *
from OldModules.the_useful_library import *
from OldModules.the_loading_library import *
from OldModules.the_fitting_library import *
from OldModules.the_plotting_library import *


## ###############################################################
## PREPARE WORKSPACE
##################################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend
# plt.style.use('dark_background')

SPECTRA_NAME = "spectra_obj_full.pkl"
SCALE_NAME = "_scale_converge_obj_full.pkl"

## ###############################################################
## FUNCTIONS
## ###############################################################
def funcLoadData_sim(
        ## input: simulation directory and name
        filepath_suite, sim_folder,
        ## output: simulation parameters
        list_Re, list_Rm, list_Pm,
        ## output: predicted scales
        list_relation_k_nu, list_relation_k_eta,
        ## output: converged (measured) sclaes
        list_k_nu_converged_group, list_k_eta_converged_group, list_k_max_converged_group,
        list_alpha_vel_group, list_alpha_mag_group
    ):
    ## #########################
    ## GET SIMULATION PARAMETERS
    ## ########
    spectra_obj = loadPickleObject(
        createFilepath([filepath_suite, "288", sim_folder]),
        SPECTRA_NAME,
        bool_hide_updates = True
    )
    ## check that a time range has been defined to collect statistics about
    sim_times = getCommonElements(spectra_obj.vel_sim_times, spectra_obj.mag_sim_times)
    bool_vel_fit = (spectra_obj.vel_fit_start_t is not None) and (spectra_obj.vel_fit_end_t is not None)
    bool_mag_fit = (spectra_obj.mag_fit_start_t is not None) and (spectra_obj.mag_fit_end_t is not None)
    if not(bool_vel_fit) or not(bool_mag_fit):
        raise Exception("Fit range has not been defined.")
    ## find indices of magnetic fit time range
    vel_index_start = getIndexClosestValue(sim_times, spectra_obj.vel_fit_start_t)
    vel_index_end   = getIndexClosestValue(sim_times, spectra_obj.vel_fit_end_t)
    mag_index_start = getIndexClosestValue(sim_times, spectra_obj.mag_fit_start_t)
    mag_index_end   = getIndexClosestValue(sim_times, spectra_obj.mag_fit_end_t)
    ## load parameters
    Re = int(spectra_obj.Re)
    Rm = int(spectra_obj.Rm)
    Pm = int(Rm / Re) # int(spectra_obj.Pm)
    ## calculate predicted scales
    relation_k_nu = 1 * (Re)**(3/4) # sub-sonic
    ## relation_k_nu = 1 * (Re)**(2/3) # super-sonic
    relation_k_eta = relation_k_nu * (Rm / Re)**(1/2)
    ## load kazantsev exponent
    list_alpha_kin, _ = cleanMeasuredScales(
        list_times  = [
            -abs(sub_list[1])
            for sub_list in spectra_obj.vel_list_fit_params_group_t[vel_index_start : vel_index_end]
        ],
        list_scales = spectra_obj.k_max_group_t[vel_index_start : vel_index_end]
    )
    list_alpha_mag, _ = cleanMeasuredScales(
        list_times  = [
            sub_list[1]
            for sub_list in spectra_obj.mag_list_fit_params_group_t[mag_index_start : mag_index_end]
        ],
        list_scales = spectra_obj.k_max_group_t[mag_index_start : mag_index_end]
    )
    ## save data
    list_Re.append(Re)
    list_Rm.append(Rm)
    list_Pm.append(Pm)
    list_relation_k_nu.append(relation_k_nu)
    list_relation_k_eta.append(relation_k_eta)
    list_alpha_vel_group.append(list_alpha_kin)
    list_alpha_mag_group.append(list_alpha_mag)
    ## #####################
    ## LOAD CONVERGED SCALES
    ## ############
    converge_obj = loadPickleObject(
        filepath_suite,
        sim_folder + SCALE_NAME,
        bool_hide_updates = True
    )
    ## load measured (converged) scales
    list_k_nu_converged_group.append(list(converge_obj.list_k_nu_converged))
    list_k_eta_converged_group.append(list(converge_obj.list_k_eta_converged))
    list_k_max_converged_group.append(list(converge_obj.list_k_max_converged))

def funcLoadData(
        ## where simulation suites are
        filepath_data,
        ## output: simulation parameters
        list_Re, list_Rm, list_Pm,
        ## output: predicted scales
        list_relation_k_nu, list_relation_k_eta,
        ## output: measured scales
        list_k_nu_converged_group, list_k_eta_converged_group, list_k_max_converged_group,
        list_alpha_vel_group, list_alpha_mag_group,
        ## output: simulation markers
        list_markers
    ):
    print("Loading simulation data...")
    ## simulation folders
    sim_folders_Re10   = [ "Pm25", "Pm50", "Pm125", "Pm250" ]
    sim_folders_Re500  = [ "Pm1", "Pm2", "Pm4" ]
    sim_folders_Rm3000 = [ "Pm1", "Pm2", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
    sim_folders_keta   = [ "Pm25", "Pm50", "Pm125", "Pm250" ]
    ## Re = 10
    for sim_index in range(len(sim_folders_Re10)):
        print("\t> Loading from '{}' in '{}'...".format(
            sim_folders_Re10[sim_index],
            "Re10"
        ))
        ## store simulation marker
        list_markers.append("s")
        ## load simulation data
        funcLoadData_sim(
            ## input: simulation directory and name
            filepath_suite = createFilepath([filepath_data, "Re10"]),
            sim_folder = sim_folders_Re10[sim_index],
            ## output: simulation parameters
            list_Re = list_Re,
            list_Rm = list_Rm,
            list_Pm = list_Pm,
            ## output: predicted scales
            list_relation_k_nu  = list_relation_k_nu,
            list_relation_k_eta = list_relation_k_eta,
            ## output: measured (converged) sclaes
            list_k_nu_converged_group  = list_k_nu_converged_group,
            list_k_eta_converged_group = list_k_eta_converged_group,
            list_k_max_converged_group = list_k_max_converged_group,
            ## output: fitted power-law exponents
            list_alpha_vel_group = list_alpha_vel_group,
            list_alpha_mag_group = list_alpha_mag_group
        )
    ## Re = 500
    for sim_index in range(len(sim_folders_Re500)):
        print("\t> Loading from '{}' in '{}'...".format(
            sim_folders_Re500[sim_index],
            "Re500"
        ))
        ## store simulation marker
        list_markers.append("D")
        ## load simulation data
        funcLoadData_sim(
            ## input: simulation directory and name
            filepath_suite = createFilepath([filepath_data, "Re500"]),
            sim_folder = sim_folders_Re500[sim_index],
            ## output: simulation parameters
            list_Re = list_Re,
            list_Rm = list_Rm,
            list_Pm = list_Pm,
            ## output: predicted scales
            list_relation_k_nu  = list_relation_k_nu,
            list_relation_k_eta = list_relation_k_eta,
            ## output: measured (converged) sclaes
            list_k_nu_converged_group  = list_k_nu_converged_group,
            list_k_eta_converged_group = list_k_eta_converged_group,
            list_k_max_converged_group = list_k_max_converged_group,
            ## output: fitted power-law exponents
            list_alpha_vel_group = list_alpha_vel_group,
            list_alpha_mag_group = list_alpha_mag_group
        )
    ## Rm = 3000
    for sim_index in range(len(sim_folders_Rm3000)):
        print("\t> Loading from '{}' in '{}'...".format(
            sim_folders_Rm3000[sim_index],
            "Rm3000"
        ))
        ## store simulation marker
        list_markers.append("o")
        ## load simulation data
        funcLoadData_sim(
            ## input: simulation directory and name
            filepath_suite = createFilepath([filepath_data, "Rm3000"]),
            sim_folder = sim_folders_Rm3000[sim_index],
            ## output: simulation parameters
            list_Re = list_Re,
            list_Rm = list_Rm,
            list_Pm = list_Pm,
            ## output: predicted scales
            list_relation_k_nu  = list_relation_k_nu,
            list_relation_k_eta = list_relation_k_eta,
            ## output: measured (converged) sclaes
            list_k_nu_converged_group  = list_k_nu_converged_group,
            list_k_eta_converged_group = list_k_eta_converged_group,
            list_k_max_converged_group = list_k_max_converged_group,
            ## output: fitted power-law exponents
            list_alpha_vel_group = list_alpha_vel_group,
            list_alpha_mag_group = list_alpha_mag_group
        )
    ## keta = 2.5
    for sim_index in range(len(sim_folders_keta)):
        print("\t> Loading from '{}' in '{}'...".format(
            sim_folders_keta[sim_index], 
            "keta"
        ))
        ## store simulation marker
        list_markers.append("v")
        ## load simulation data
        funcLoadData_sim(
            ## input: simulation directory and name
            filepath_suite = createFilepath([filepath_data, "keta"]),
            sim_folder = sim_folders_keta[sim_index],
            ## output: simulation parameters
            list_Re = list_Re,
            list_Rm = list_Rm,
            list_Pm = list_Pm,
            ## output: predicted scales
            list_relation_k_nu  = list_relation_k_nu,
            list_relation_k_eta = list_relation_k_eta,
            ## output: measured (converged) sclaes
            list_k_nu_converged_group  = list_k_nu_converged_group,
            list_k_eta_converged_group = list_k_eta_converged_group,
            list_k_max_converged_group = list_k_max_converged_group,
            ## output: fitted power-law exponents
            list_alpha_vel_group = list_alpha_vel_group,
            list_alpha_mag_group = list_alpha_mag_group
        )
    print(" ")

def funcPlotScaleRelations(
        ## where to save figure
        filepath_plot,
        ## data point colors
        list_colors, list_markers,
        ## simulation parameters
        list_Re, list_Rm, list_Pm,
        ## predicted scales
        list_relation_k_nu, list_relation_k_eta,
        ## measured scales
        list_k_nu_converged_group, list_k_eta_converged_group
    ):
    ## #################
    ## INITIALISE FIGURE
    ## ##########
    fig, axs = plt.subplots(1, 2, figsize=(7*2/1.1, 4/1.1))
    fig.subplots_adjust(wspace=0.225)
    ## plot scale distributions
    for sim_index in range(len(list_relation_k_nu)):
        plotErrorBar(
            axs[0],
            data_x = list_relation_k_nu[sim_index],
            data_y = list_k_nu_converged_group[sim_index],
            color  = list_colors[sim_index],
            marker = "o",
            ms = 9
        )
        plotErrorBar(
            axs[1],
            data_x = list_relation_k_eta[sim_index],
            data_y = list_k_eta_converged_group[sim_index],
            color  = list_colors[sim_index],
            marker = "o",
            ms = 9
        )
    ## ###############
    ## FIT k_nu SCALES
    ## ########
    ## fit Re > 100 for constant
    plotDistributionFit(
        ax = axs[0],
        var_str = r"$k_{\nu, \mathrm{theory}}$",
        input_x = [
            k_nu
            for Re, Rm, Pm, k_nu in zip(
                list_Re, list_Rm, list_Pm,
                list_relation_k_nu
            ) if (Re > 100)
        ],
        input_y = [
            list_k_nu
            for Re, Rm, Pm, list_k_nu in zip(
                list_Re, list_Rm, list_Pm,
                list_k_nu_converged_group
            ) if (Re > 100)
        ],
        func_label = "linear",
        func_fit   = ListOfModels.linear,
        func_plot  = ListOfModels.linear,
        maxfev     = 10**3,
        p0         = [ 0.03 ],
        bounds     = [ 0.01, 0.5 ],
        pre_label  = r"$k_\nu = \;$",
        num_digits = 2,
        bool_hide_coef = False,
        bool_show_label = False,
        plot_domain = np.linspace(1, 1000, 100),
        plot_args   = {
            "x":0.05,
            "y":0.95,
            "va":"top",
            "ha":"left",
            "color":"black",
            "ls":"-",
            "bool_box":False
        }
    )
    ## fit Re > 100 for exponent
    plotDistributionFit(
        ax = axs[0],
        var_str = r"$k_{\nu, \mathrm{theory}}$",
        input_x = [
            k_nu
            for Re, Rm, Pm, k_nu in zip(
                list_Re, list_Rm, list_Pm,
                list_relation_k_nu
            ) if (Re > 100)
        ],
        input_y = [
            list_k_nu
            for Re, Rm, Pm, list_k_nu in zip(
                list_Re, list_Rm, list_Pm,
                list_k_nu_converged_group
            ) if (Re > 100)
        ],
        func_label = "PowerLaw",
        func_fit   = ListOfModels.powerlaw_log10,
        func_plot  = ListOfModels.powerlaw_linear,
        bool_log_fit = True,
        list_func_indices_unlog = [0],
        maxfev = 10**3,
        p0     = [ np.log(0.03), 1 ],
        bounds = [
            ( np.log(0.01), 0.5 ),
            ( np.log(0.5),  1.5 )
        ],
        pre_label  = r"$k_\nu \propto \;$",
        num_digits = 2,
        bool_hide_coef = True,
        bool_show_label = False,
        plot_domain = np.linspace(1, 1000, 100),
        plot_args   = {
            "x":0.05,
            "y":0.95-0.125,
            "va":"top",
            "ha":"left",
            "color":"red",
            "ls":"--",
            "bool_box":False
        }
    )
    ## fit Re < 100
    plotDistributionFit(
        ax = axs[0],
        var_str = r"$k_{\nu, \mathrm{theory}}$",
        input_x = [
            k_nu
            for Re, Rm, Pm, k_nu in zip(
                list_Re, list_Rm, list_Pm,
                list_relation_k_nu
            ) if (Re < 100)
        ],
        input_y = [
            list_k_nu
            for Re, Rm, Pm, list_k_nu in zip(
                list_Re, list_Rm, list_Pm,
                list_k_nu_converged_group
            ) if (Re < 100)
        ],
        func_label = "PowerLaw",
        func_fit   = ListOfModels.powerlaw_log10,
        func_plot  = ListOfModels.powerlaw_linear,
        bool_log_fit = True,
        list_func_indices_unlog = [0],
        maxfev = 10**3,
        p0     = [ np.log(0.03), 1 ],
        bounds = [
            ( np.log(0.01), 0.5 ),
            ( np.log(0.5),  1.5 )
        ],
        pre_label  = r"$k_\nu \propto \;$",
        num_digits = 2,
        bool_hide_coef = True,
        bool_show_label = False,
        plot_domain = np.linspace(1, 1000, 100),
        plot_args   = {
            "x":0.05,
            "y":0.95-2*0.125,
            "va":"top",
            "ha":"left",
            "color":"blue",
            "ls":":",
            "bool_box":False
        }
    )
    ## ################
    ## FIT k_eta SCALES
    ## #########
    ## fit Re > 100 for constant
    plotDistributionFit(
        ax = axs[1],
        input_x = [
            k_eta
            for Re, Rm, Pm, k_eta in zip(
                list_Re, list_Rm, list_Pm,
                list_relation_k_eta
            ) if (Re > 100) and (Pm > 1)
        ],
        input_y = [
            list_k_eta
            for Re, Rm, Pm, list_k_eta in zip(
                list_Re, list_Rm, list_Pm,
                list_k_eta_converged_group
            ) if (Re > 100) and (Pm > 1)
        ],
        func_label = "linear",
        func_fit   = ListOfModels.linear,
        func_plot  = ListOfModels.linear,
        maxfev = 10**3,
        p0     = [ 0.02 ],
        bounds = [ 1e-2, 7.5e-2 ],
        var_str    = r"$k_{\eta, \mathrm{theory}}$",
        pre_label  = r"$k_\eta = \;$",
        num_digits = 3,
        bool_hide_coef = False,
        bool_show_label = False,
        plot_domain = np.linspace(1, 1000, 100),
        plot_args   = {
            "x":0.95,
            "y":0.05,
            "va":"bottom",
            "ha":"right",
            "color":"black",
            "ls":"-",
            "bool_box":False
        }
    )
    ## fit Re > 100 for exponent
    plotDistributionFit(
        ax = axs[1],
        input_x = [
            k_eta
            for Re, Rm, Pm, k_eta in zip(
                list_Re, list_Rm, list_Pm,
                list_relation_k_eta
            ) if (Re > 100)
        ],
        input_y = [
            list_k_eta
            for Re, Rm, Pm, list_k_eta in zip(
                list_Re, list_Rm, list_Pm,
                list_k_eta_converged_group
            ) if (Re > 100)
        ],
        # errors = [
        #     [ np.std(list_k_eta) ] * len(list_k_eta)
        #     for Re, Rm, Pm, list_k_eta in zip(
        #         list_Re, list_Rm, list_Pm,
        #         list_k_eta_converged_group
        #     ) if (Re > 100)
        # ],
        func_label = "PowerLaw",
        func_fit   = ListOfModels.powerlaw_log10,
        func_plot  = ListOfModels.powerlaw_linear,
        bool_log_fit = True,
        list_func_indices_unlog = [0],
        maxfev = 10**3,
        p0     = [ np.log(0.02), 1 ],
        bounds = [
            ( np.log(1e-2), 0.5 ),
            ( np.log(7.5e-2), 1.5 )
        ],
        var_str    = r"$k_{\eta, \mathrm{theory}}$",
        pre_label  = r"$k_\eta \propto \;$",
        num_digits = 2,
        bool_hide_coef = True,
        bool_show_label = False,
        plot_domain = np.linspace(1, 1000, 100),
        plot_args   = {
            "x":0.95,
            "y":0.05+2*0.125,
            "va":"bottom",
            "ha":"right",
            "color":"red",
            "ls":"--",
            "bool_box":False
        }
    )
    ## fit Re < 100
    plotDistributionFit(
        ax = axs[1],
        input_x = [
            k_eta
            for Re, Rm, Pm, k_eta in zip(
                list_Re, list_Rm, list_Pm,
                list_relation_k_eta
            ) if (Re < 100)
        ],
        input_y = [
            list_k_eta
            for Re, Rm, Pm, list_k_eta in zip(
                list_Re, list_Rm, list_Pm,
                list_k_eta_converged_group
            ) if (Re < 100)
        ],
        func_label = "PowerLaw",
        func_fit   = ListOfModels.powerlaw_log10,
        func_plot  = ListOfModels.powerlaw_linear,
        bool_log_fit = True,
        list_func_indices_unlog = [0],
        maxfev = 10**3,
        p0     = [ np.log(0.02), 1 ],
        bounds = [
            ( np.log(1e-2), 0.5 ),
            ( np.log(7.5e-2), 1.5 )
        ],
        var_str    = r"$k_{\eta, \mathrm{theory}}$",
        pre_label  = r"$k_\eta \propto \;$",
        num_digits = 2,
        bool_hide_coef = True,
        bool_show_label = False,
        plot_domain = np.linspace(1, 1000, 100),
        plot_args   = {
            "x":0.95,
            "y":0.05+0.125,
            "va":"bottom",
            "ha":"right",
            "color":"blue",
            "ls":":",
            "bool_box":False
        }
    )
    ## ############
    ## LABEL FIGURE
    ## #####
    axs[0].text(
        0.925, 0.225,
        r"Re $< 100$", color="blue",
        va="bottom", ha="right", transform=axs[0].transAxes, fontsize=15
    )
    axs[0].text(
        0.925, 0.1,
        r"Re $> 100$", color="red",
        va="bottom", ha="right", transform=axs[0].transAxes, fontsize=15
    )
    axs[1].text(
        0.05, 0.9,
        r"Re $< 100$", color="blue",
        va="top", ha="left", transform=axs[1].transAxes, fontsize=15
    )
    axs[1].text(
        0.05, 0.78,
        r"Re $> 100$", color="red",
        va="top", ha="left", transform=axs[1].transAxes, fontsize=15
    )

    ## add legend: equations
    addLegend(
        ax = axs[0],
        loc  = "upper left",
        bbox = (0.0, 1.0),
        artists = [ "-", "--", ":" ],
        colors  = [ "black", "red", "blue" ],
        legend_labels = [
            r"$k_{\nu} = 0.025_{-0.006}^{+0.005} \;k_{\nu, \mathrm{theory}}$",
            r"$k_{\nu} \propto k_{\nu, \mathrm{theory}}^{0.50_{-0.07}^{+0.04}}$",
            r"$k_{\nu} \propto k_{\nu, \mathrm{theory}}^{0.96_{-0.08}^{+0.06}}$"
        ],
        rspacing = 0.25,
        cspacing = 0.25,
        ncol = 1,
        fontsize = 15,
        labelcolor = "white",
        lw = 1.5
    )
    axs[0].text(
        0.15, 0.941,
        # r"$k_{\nu} = 0.031_{-0.007}^{+0.006} \;k_{\nu, \mathrm{theory}}$", # mixed
        r"$k_{\nu} = 0.025_{-0.006}^{+0.005} \;k_{\nu, \mathrm{theory}}$", # full
        color="black", va="top", ha="left", transform=axs[0].transAxes, fontsize=15, zorder=8
    )
    axs[0].text(
        0.15, 0.8375,
        # r"$k_{\nu} \propto k_{\nu, \mathrm{theory}}^{1.00_{-0.14}^{+0.13}}$", # mixed
        r"$k_{\nu} \propto k_{\nu, \mathrm{theory}}^{0.96_{-0.08}^{+0.06}}$", # full
        color="black", va="top", ha="left", transform=axs[0].transAxes, fontsize=15, zorder=8
    )
    axs[0].text(
        0.15, 0.695,
        # r"$k_{\nu} \propto k_{\nu, \mathrm{theory}}^{0.67_{-0.07}^{+0.08}}$", # mixed
        r"$k_{\nu} \propto k_{\nu, \mathrm{theory}}^{0.50_{-0.07}^{+0.04}}$", # full
        color="black", va="top", ha="left", transform=axs[0].transAxes, fontsize=15, zorder=8
    )
    addLegend(
        ax = axs[1],
        loc  = "lower right",
        bbox = (1.0, 0.125),
        artists = [ ":", "--" ],
        colors  = [ "blue", "red" ],
        legend_labels = [
            r"$k_{\eta} \propto k_{\eta, {\mathrm{theory}}}^{0.97_{-0.14}^{+0.17}}$",
            r"$k_{\eta} \propto k_{\eta, {\mathrm{theory}}}^{0.83_{-0.11}^{+0.09}}$"
        ],
        rspacing = 0.25,
        cspacing = 0.25,
        ncol = 1,
        fontsize = 15,
        labelcolor = "white",
        lw = 1.5
    )
    addLegend(
        ax = axs[1],
        loc  = "lower right",
        bbox = (1.0, 0.0),
        artists = [ "-" ],
        colors  = [ "black" ],
        legend_labels = [
            r"$k_{\eta} = 0.022_{-0.002}^{+0.003} \;k_{\eta, {\mathrm{theory}}}$"
        ],
        rspacing = 0.25,
        cspacing = 0.25,
        ncol = 1,
        fontsize = 15,
        labelcolor = "white",
        lw = 1.5
    )
    axs[1].text(
        0.95, 0.340,
        r"$k_{\eta} \propto k_{\eta, {\mathrm{theory}}}^{0.83_{-0.11}^{+0.09}}$",
        color="black", va="bottom", ha="right", transform=axs[1].transAxes, fontsize=15, zorder=8
    )
    axs[1].text(
        0.95, 0.200,
        r"$k_{\eta} \propto k_{\eta, {\mathrm{theory}}}^{0.97_{-0.14}^{+0.17}}$",
        color="black", va="bottom", ha="right", transform=axs[1].transAxes, fontsize=15, zorder=8
    )
    axs[1].text(
        0.95, 0.075,
        r"$k_{\eta} = 0.022_{-0.002}^{+0.003} \;k_{\eta, {\mathrm{theory}}}$",
        color="black", va="bottom", ha="right", transform=axs[1].transAxes, fontsize=15, zorder=8
    )

    ## label axis
    axs[0].set_xlabel(r"$k_{\nu, \mathrm{theory}} = \mathrm{Re}^{3/4}$", fontsize=20)
    axs[1].set_xlabel(r"$k_{\eta, \mathrm{theory}} = k_{\nu, \mathrm{theory}} \; \mathrm{Pm}^{1/2}$", fontsize=20)
    axs[0].set_ylabel(r"$k_\nu$", fontsize=20)
    axs[1].set_ylabel(r"$k_\eta$", fontsize=20)
    ## adjust axis
    axs[0].set_xscale("log")
    axs[1].set_xscale("log")
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlim([ 20, 1000 ])
    axs[0].set_ylim([ 10**(-1), 20 ])
    axs[1].set_ylim([ 3*10**(-1), 20 ])
    ## save plot
    fig_name = "fig_scale_relation_full.pdf"
    fig_filepath = createFilepath([filepath_plot, fig_name])
    plt.savefig(fig_filepath)
    print("\t> Figure saved: " + fig_name)

def funcPlotScaleDependance(
        ## where to save figure
        filepath_plot,
        ## data point colors
        list_colors, list_markers,
        ## simulation parameters
        list_Re, list_Rm, list_Pm,
        ## measured scales
        list_k_nu_converged_group, list_k_eta_converged_group, list_k_max_converged_group
    ):
    ## #################
    ## INITIALISE FIGURE
    ## ##########
    fig, axs = plt.subplots(1, 2, figsize=(14/1.1, 4/1.1))
    fig.subplots_adjust(wspace=0.225)
    ## plot scale distributions
    for sim_index in range(len(list_k_nu_converged_group)):
        ## plot dependance on k_nu
        plotErrorBar(
            axs[0],
            data_x = list_k_nu_converged_group[sim_index],
            data_y = list_k_max_converged_group[sim_index],
            color  = list_colors[sim_index],
            marker = list_markers[sim_index],
            ms = 9
        )
        ## plot dependance on k_eta
        plotErrorBar(
            axs[1],
            data_x = list_k_eta_converged_group[sim_index],
            data_y = list_k_max_converged_group[sim_index],
            color  = list_colors[sim_index],
            marker = list_markers[sim_index],
            ms = 9
        )
    ## ###############
    ## FIT k_nu SCALES
    ## ########
    k_nu_ref_domain = np.logspace(-2, 2, 100)
    k_nu_ref_line = [ np.column_stack((
        k_nu_ref_domain,
        10**(np.log10(4.75) + 1/3 * np.log10(k_nu_ref_domain))
    )) ]
    axs[0].add_collection(
        LineCollection(k_nu_ref_line, colors="black", ls=":", lw=1.5, zorder=9),
        autolim = False # ignore line when setting axis bounds
    )
    ## ################
    ## FIT k_eta SCALES
    ## #########
    ## fit Re > 100 for constant
    plotDistributionFit(
        ax = axs[1],
        input_x = [
            list_k_eta
            for Re, Rm, Pm, list_k_eta in zip(
                list_Re, list_Rm, list_Pm,
                list_k_eta_converged_group
            ) if (Re > 100)
        ],
        input_y = [
            list_k_max
            for Re, Rm, Pm, list_k_max in zip(
                list_Re, list_Rm, list_Pm,
                list_k_max_converged_group
            ) if (Re > 100)
        ],
        func_label = "linear",
        func_fit   = ListOfModels.linear,
        func_plot  = ListOfModels.linear,
        maxfev     = 10**4,
        p0         = [ 1.2 ],
        bounds     = [ 1, 1.7 ],
        var_str    = r"$k_\eta$",
        pre_label  = r"$k_p = \;$",
        num_digits = 2,
        bool_hide_coef = False,
        bool_show_label = False,
        plot_domain = np.linspace(0.1, 100, 100),
        plot_args   = {
            "x":0.95,
            "y":0.05,
            "va":"bottom",
            "ha":"right",
            "color":"black",
            "ls":"-",
            "bool_box":False
        }
    )
    ## fit Re > 100 for exponent
    plotDistributionFit(
        ax = axs[1],
        input_x = [
            list_k_eta
            for Re, Rm, Pm, list_k_eta in zip(
                list_Re, list_Rm, list_Pm,
                list_k_eta_converged_group
            ) if (Re > 100) and not (Rm == 3361.0) and not (Pm == 2.0)
        ],
        input_y = [
            list_k_max
            for Re, Rm, Pm, list_k_max in zip(
                list_Re, list_Rm, list_Pm,
                list_k_max_converged_group
            ) if (Re > 100) and not (Rm == 3361.0) and not (Pm == 2.0)
        ],
        func_label = "PowerLaw",
        func_fit   = ListOfModels.powerlaw_log10,
        func_plot  = ListOfModels.powerlaw_linear,
        bool_log_fit = True,
        list_func_indices_unlog = [0],
        maxfev = 10**4,
        p0     = [ np.log(1.2), 1 ],
        bounds = [
            ( np.log(0.5), 0 ),
            ( np.log(2),   2 )
        ],
        var_str    = r"$k_\eta$",
        pre_label  = r"$k_p \propto \;$",
        num_digits = 2,
        bool_hide_coef = True,
        bool_show_label = False,
        plot_domain = np.linspace(0.1, 100, 100),
        plot_args   = {
            "x":0.95,
            "y":0.05+0.125,
            "va":"bottom",
            "ha":"right",
            "color":"red",
            "ls":"--",
            "bool_box":False
        }
    )
    ## ############
    ## LABEL FIGURE
    ## #####
    axs[0].text(
        0.95, 0.325,
        r"Re $< 100$", color="blue",
        va="bottom", ha="right", transform=axs[0].transAxes, fontsize=15
    )
    axs[0].text(
        0.95, 0.2,
        r"Re $> 100$", color="red",
        va="bottom", ha="right", transform=axs[0].transAxes, fontsize=15
    )
    axs[1].text(
        0.05, 0.675,
        r"Re $< 100$", color="blue",
        va="top", ha="left", transform=axs[1].transAxes, fontsize=15
    )
    axs[1].text(
        0.05, 0.55,
        r"Re $> 100$", color="red",
        va="top", ha="left", transform=axs[1].transAxes, fontsize=15
    )
    ## add legend: equations
    addLegend(
        ax = axs[0],
        loc  = "lower right",
        bbox = (1.0, 0.0),
        artists = [ ":" ],
        colors  = [ "black" ],
        legend_labels = [
            r"$k_p \propto k_\nu^{1/3}$"
        ],
        rspacing = 0.25,
        cspacing = 0.25,
        ncol = 1,
        fontsize = 15,
        lw = 1.5
    )
    addLegend(
        ax = axs[1],
        loc  = "lower right",
        bbox = (1.0, 0.0),
        artists = [ "--", "-" ],
        colors  = [ "red", "black" ],
        legend_labels = [
            r"$k_p \propto \;k_\eta^{0.94_{-0.23}^{+0.36}}$",
            r"$k_p = 1.2_{-0.2}^{+0.2} \;k_\eta$"
        ],
        rspacing = 0.25,
        cspacing = 0.25,
        ncol = 1,
        fontsize = 15,
        lw = 1.5
    )
    ## add legend: simulation marker
    addLegend(
        ax  = axs[0],
        loc  = "upper left",
        bbox = (-0.025, 1.0),
        artists = [ "s", "D", "o", "v" ],
        colors  = [ "black" ] * 4,
        legend_labels  = [
            r"Re $= 10$",
            r"Re $\approx 450$",
            r"Rm $\approx 3300$",
            r"$k_{\eta, \mathrm{theory}} \approx 125$"
        ],
        ms = 9,
        tpad = 0.1,
        rspacing = 0.5,
        cspacing = 0.5,
        ncol = 2,
        fontsize = 15,
        labelcolor = "white"
    )
    axs[0].text(
        0.1, 0.93,
        r"Re $= 10$", color="black",
        va="top", ha="left", transform=axs[0].transAxes, fontsize=15, zorder=10
    )
    axs[0].text(
        0.425, 0.93,
        r"Rm $\approx 3300$", color="black",
        va="top", ha="left", transform=axs[0].transAxes, fontsize=15, zorder=10
    )
    axs[0].text(
        0.1, 0.82,
        r"Re $\approx 450$", color="black",
        va="top", ha="left", transform=axs[0].transAxes, fontsize=15, zorder=10
    )
    axs[0].text(
        0.425, 0.82,
        r"$k_{\eta, \mathrm{theory}} \approx 125$", color="black",
        va="top", ha="left", transform=axs[0].transAxes, fontsize=15, zorder=10
    )
    addLegend(
        ax  = axs[1],
        loc  = "upper left",
        bbox = (-0.025, 1.0),
        artists = [ "s", "D", "o", "v" ],
        colors  = [ "black" ] * 4,
        legend_labels  = [
            r"Re $= 10$",
            r"Re $\approx 450$",
            r"Rm $\approx 3300$",
            r"$k_{\eta, \mathrm{theory}} \approx 125$"
        ],
        ms = 9,
        tpad = 0.1,
        rspacing = 0.5,
        cspacing = 0.5,
        ncol = 2,
        fontsize = 15,
        labelcolor = "white"
    )
    axs[1].text(
        0.1, 0.93,
        r"Re $= 10$", color="black",
        va="top", ha="left", transform=axs[1].transAxes, fontsize=15, zorder=10
    )
    axs[1].text(
        0.425, 0.93,
        r"Rm $\approx 3300$", color="black",
        va="top", ha="left", transform=axs[1].transAxes, fontsize=15, zorder=10
    )
    axs[1].text(
        0.1, 0.82,
        r"Re $\approx 450$", color="black",
        va="top", ha="left", transform=axs[1].transAxes, fontsize=15, zorder=10
    )
    axs[1].text(
        0.425, 0.82,
        r"$k_{\eta, \mathrm{theory}} \approx 125$", color="black",
        va="top", ha="left", transform=axs[1].transAxes, fontsize=15, zorder=10
    )
    ## add boxes around knu fixed
    axs[0].add_patch(
        plt.Rectangle(
            (0.14, 0.15), 0.13, 0.525,
            ls="-", ec="k", fc="None",
            transform=axs[0].transAxes
        )
    )
    axs[0].add_patch(
        plt.Rectangle(
            (0.4625, 0.15), 0.147, 0.47,
            ls="-", ec="k", fc="None",
            transform=axs[0].transAxes
        )
    )
    axs[0].text(
        0.122 , 0.0825, r"Re $\approx$ 10",
        fontsize=15, transform=axs[0].transAxes
    )
    axs[0].text(
        0.44, 0.0825, r"Re $\approx$ 450",
        fontsize=15, transform=axs[0].transAxes
    )
    ## label axis
    axs[0].set_xlabel(r"$k_\nu$", fontsize=20)
    axs[1].set_xlabel(r"$k_\eta$", fontsize=20)
    axs[0].set_ylabel(r"$k_p$", fontsize=20)
    axs[1].set_ylabel(r"$k_p$", fontsize=20)
    ## adjust axis
    axs[0].set_xscale("log")
    axs[1].set_xscale("log")
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    axs[0].set_xlim([ 0.1, 30 ])
    axs[1].set_xlim([ 0.5, 15 ])
    axs[0].set_ylim([ 0.9, 25 ])
    axs[1].set_ylim([ 0.9, 25 ])
    ## save plot
    fig_name = "fig_scale_dependance_full.pdf"
    fig_filepath = createFilepath([filepath_plot, fig_name])
    plt.savefig(fig_filepath)
    print("\t> Figure saved: " + fig_name)

def funcPlotExponent(
        ## where to save figure
        filepath_plot,
        ## data point colors
        list_colors, list_markers,
        ## simulation parameters
        list_Re, list_Rm, list_Pm,
        ## measured scales
        list_alpha_mag_group
    ):
    ## #################
    ## INITIALISE FIGURE
    ## ##########
    factor = 1.45
    fig, ax = plt.subplots(figsize=(8/factor, 5/factor))
    ## plot points
    for sim_index in range(len(list_alpha_mag_group)):
        ## plot dependance on exponent on Re
        plotErrorBar(
            ax,
            data_x = list_Re[sim_index],
            data_y = list_alpha_mag_group[sim_index],
            color  = list_colors[sim_index],
            marker = list_markers[sim_index],
            ms = 9
        )
    ## fit to Re > 100 data
    fit_param, fit_cov = curve_fit(
        ListOfModels.constant,
        [
            Re
            for Re in list_Re
            if Re > 100
        ],
        [
            np.median(list_kaz_exp)
            for Re, list_kaz_exp in zip(
                list_Re,
                list_alpha_mag_group
            )
            if Re > 100
        ],
        sigma = [
            np.std(list_kaz_exp)
            for Re, list_kaz_exp in zip(
                list_Re,
                list_alpha_mag_group
            )
            if Re > 100
        ],
        absolute_sigma = True
    )
    fit_mean = fit_param[0]
    fit_std = np.sqrt(np.diag(fit_cov))[0]
    ## add average line
    ax.axhline(y=fit_mean, dashes=(5, 1.5), color="orangered", lw=1.5)
    ## add asymptotic line
    ax.axhline(y=3/2, ls=":", color="black", lw=1.5)
    ## add legend: simulation marker
    addLegend(
        ax  = ax,
        loc  = "upper right",
        bbox = (1.0, 1.0),
        artists = [ "s", "D", "o", "v" ],
        colors  = [ "black" ] * 4,
        legend_labels  = [
            r"Re $= 10$",
            r"Re $\approx 450$",
            r"Rm $\approx 3300$",
            r"$k_{\eta, \mathrm{theory}} \approx 125$"
        ],
        ms = 9,
        tpad = 0.1,
        rspacing = 0.5,
        cspacing = 0.5,
        ncol = 2,
        fontsize = 15,
        labelcolor = "white"
    )
    ax.text(
        0.35, 0.925,
        r"Re $= 10$", color="black",
        va="top", ha="left", transform=ax.transAxes, fontsize=15, zorder=10
    )
    ax.text(
        0.675, 0.925,
        r"Rm $\approx 3300$", color="black",
        va="top", ha="left", transform=ax.transAxes, fontsize=15, zorder=10
    )
    ax.text(
        0.35, 0.81,
        r"Re $\approx 450$", color="black",
        va="top", ha="left", transform=ax.transAxes, fontsize=15, zorder=10
    )
    ax.text(
        0.675, 0.81,
        r"$k_{\eta, \mathrm{theory}} \approx 125$", color="black",
        va="top", ha="left", transform=ax.transAxes, fontsize=15, zorder=10
    )
    ## add Reynolds number legend
    ax.text(
        0.935, 0.675,
        r"Re $< 100$", color="blue",
        va="top", ha="right", transform=ax.transAxes, fontsize=15
    )
    ax.text(
        0.935, 0.565,
        r"Re $> 100$", color="red",
        va="top", ha="right", transform=ax.transAxes, fontsize=15
    )
    ## add legend: lines
    ax.text(
        0.05, 0.175,
        r"$\alpha_{\mathrm{mag}} =$ " + r"${} \pm {}$".format(
            "{:0.1f}".format( fit_mean ),
            "{:0.1f}".format( fit_std )
        ),
        va="bottom", ha="left", transform=ax.transAxes, fontsize=15, color="orangered"
    )
    ax.text(
        0.05, 0.0475,
        r"$\alpha_{\mathrm{mag}} = 1.5$",
        va="bottom", ha="left", transform=ax.transAxes, fontsize=15, color="black"
    )
    ## label axis
    ax.set_xlabel(r"Re", fontsize=22)
    ax.set_ylabel(r"$\alpha_{\mathrm{mag}}$", fontsize=22)
    ## adjust axis
    ax.set_xscale("log")
    ## save plot
    fig_name = "fig_exponent_full.pdf"
    fig_filepath = createFilepath([filepath_plot, fig_name])
    plt.savefig(fig_filepath)
    print("\t> Figure saved: " + fig_name)

def funcPrintForm(
        distribution,
        num_digits = 2
    ):
    str_median = ("{0:.2g}").format(np.percentile(distribution, 50))
    num_decimals = 1
    ## if integer
    if ("." not in str_median) and (len(str_median.replace("-", "")) < 2):
        str_median = ("{0:.1f}").format(np.percentile(distribution, 50))
    ## if a float
    if "." in str_median:
        ## if integer component is 0
        if ("0" in str_median.split(".")[0]) and (len(str_median.split(".")[1]) < 2):
            str_median = ("{0:.2f}").format(np.percentile(distribution, 50))
        num_decimals = len(str_median.split(".")[1])
    ## if integer > 9
    elif len(str_median.split(".")[0].replace("-", "")) > 1:
        num_decimals = 0
    str_low  = ("-{0:."+str(num_decimals)+"f}").format(np.percentile(distribution, 50) - np.percentile(distribution, 16))
    str_high = ("+{0:."+str(num_decimals)+"f}").format(np.percentile(distribution, 84) - np.percentile(distribution, 50))
    return r"${}_{}^{}$".format(
        str_median,
        "{" + str_low  + "}",
        "{" + str_high + "}"
    )

class PlotSpectraAndScales():
    def __init__(
            self,
            filepath_plot,
            spectra_obj_0, spectra_obj_1
        ):
        ## initialise figure
        factor = 1.2
        fig, axs = plt.subplots(2, 1, figsize=(6/factor, 2*3.5/factor), sharex=True)
        fig.subplots_adjust(hspace=0.05)
        ## plot Re 500 simulation
        self.plotSpectraObj(
            axs,
            spectra_obj = spectra_obj_0,
            list_colors = [ "green", "green" ]
        )
        ## plot Re 1500 simulation
        self.plotSpectraObj(
            axs,
            spectra_obj = spectra_obj_1,
            list_colors = [ "orange", "orange" ]
        )
        ## label axes
        axs[1].set_xlabel(r"$k$", fontsize=20)
        axs[0].set_ylabel(r"$\widehat{\mathcal{P}}_{\mathrm{kin}}(k)$", fontsize=20)
        axs[1].set_ylabel(r"$\widehat{\mathcal{P}}_{\mathrm{mag}}(k)$", fontsize=20)
        ## add legend
        addLegend(
            ax = axs[0],
            loc  = "upper right",
            bbox = (0.95, 1.0),
            artists = [ "-", "-" ],
            colors  = [ "green", "orange" ],
            legend_labels = [ r"Re470Pm2", r"Re1700Pm2" ],
            rspacing = 0.5,
            cspacing = 0.25,
            ncol = 1,
            fontsize = 15,
            labelcolor = "white"
        )
        axs[0].text(
            0.64, 0.9125,
            r"Re470Pm2", color="black",
            va="top", ha="left", transform=axs[0].transAxes, fontsize=13, zorder=10
        )
        axs[0].text(
            0.64, 0.77125,
            r"Re1700Pm2", color="black",
            va="top", ha="left", transform=axs[0].transAxes, fontsize=13, zorder=10
        )
        ## initialise figure domain bounds
        y_max = 1e2
        y_min = 1e-9
        x_max = 120
        x_min = 1
        ## adjust top axis
        axs[0].set_xlim(x_min, x_max)
        axs[0].set_ylim(y_min, y_max)
        axs[0].set_xscale("log")
        axs[0].set_yscale("log")
        locmin = mpl.ticker.LogLocator(
            base=10.0,
            subs=np.arange(2, 10) * 0.1,
            numticks=100
        )
        axs[0].yaxis.set_minor_locator(locmin)
        axs[0].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ## adjust bottom axis
        axs[1].set_xlim(x_min, x_max)
        axs[1].set_ylim(y_min, y_max)
        axs[1].set_xscale("log")
        axs[1].set_yscale("log")
        locmin = mpl.ticker.LogLocator(
            base=10.0,
            subs=np.arange(2, 10) * 0.1,
            numticks=100
        )
        axs[1].yaxis.set_minor_locator(locmin)
        axs[1].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        y_major = mpl.ticker.LogLocator(base=10.0, numticks=6)
        axs[0].yaxis.set_major_locator(y_major)
        axs[1].yaxis.set_major_locator(y_major)
        ## save plot
        fig_name = "fig_spectra_full.pdf"
        fig_filepath = createFilepath([filepath_plot, fig_name])
        plt.savefig(fig_filepath)
        print("\t> Figure saved: " + fig_name)
    def plotSpectraObj(
            self,
            axs, spectra_obj, list_colors
        ):
        ## check that a time range has been defined to collect statistics about
        sim_times = getCommonElements(spectra_obj.vel_sim_times, spectra_obj.mag_sim_times)
        bool_vel_fit = (spectra_obj.vel_fit_start_t is not None) and (spectra_obj.vel_fit_end_t is not None)
        bool_mag_fit = (spectra_obj.mag_fit_start_t is not None) and (spectra_obj.mag_fit_end_t is not None)
        if not(bool_vel_fit) or not(bool_mag_fit):
            raise Exception("Fit range has not been defined.")
        ## find indices of velocity fit time range
        vel_index_start = getIndexClosestValue(sim_times, spectra_obj.vel_fit_start_t)
        vel_index_end   = getIndexClosestValue(sim_times, spectra_obj.vel_fit_end_t)
        ## find indices of magnetic fit time range
        mag_index_start = getIndexClosestValue(sim_times, spectra_obj.mag_fit_start_t)
        mag_index_end   = getIndexClosestValue(sim_times, spectra_obj.mag_fit_end_t)
        ## load spectra
        list_vel_power = spectra_obj.vel_list_power_group_t[vel_index_start : vel_index_end]
        list_mag_power = spectra_obj.mag_list_power_group_t[mag_index_start : mag_index_end]
        ## load measured scales
        list_k_nu  = cleanMeasuredScales(spectra_obj.k_nu_group_t[vel_index_start  : vel_index_end])
        list_k_eta = cleanMeasuredScales(spectra_obj.k_eta_group_t[mag_index_start : mag_index_end])
        list_kaz, list_k_max = cleanMeasuredScales(
            list_times  = [
                sub_list[1]
                for sub_list in spectra_obj.mag_list_fit_params_group_t[mag_index_start : mag_index_end]
            ],
            list_scales = spectra_obj.k_max_group_t[mag_index_start : mag_index_end]
        )
        ## plot velocity spectra
        self.plotSpectra(
            ax = axs[0],
            color  = list_colors[0],
            list_k = spectra_obj.vel_list_k_group_t[0],
            list_power = [
                np.array(vel_power) / np.sum(vel_power)
                for vel_power in list_vel_power
            ],
            list_fit_k = spectra_obj.vel_list_fit_k_group_t[0],
            list_fit_power = [
                np.array(vel_fit_power) / np.sum(vel_power)
                for vel_fit_power, vel_power in zip(
                    spectra_obj.vel_list_fit_power_group_t[vel_index_start : vel_index_end],
                    list_vel_power
                )
            ]
        )
        ## plot magnetic spectra
        self.plotSpectra(
            ax = axs[1],
            color  = list_colors[1],
            list_k = spectra_obj.mag_list_k_group_t[1],
            list_power = [
                np.array(mag_power) / np.sum(mag_power)
                for mag_power in list_mag_power
            ],
            list_fit_k = spectra_obj.mag_list_fit_k_group_t[0],
            list_fit_power = [
                np.array(mag_fit_power) / np.sum(mag_power)
                for mag_fit_power, mag_power in zip(
                    spectra_obj.mag_list_fit_power_group_t[mag_index_start : mag_index_end],
                    list_mag_power
                )
            ]
        )
        ## plot measured k_nu
        self.plotScale(
            axs[0], list_colors[0], r"$k_\nu$",
            list_fit_k = spectra_obj.vel_list_k_group_t[0],
            list_fit_power = [
                np.array(vel_fit_power) / np.sum(vel_power)
                for vel_fit_power, vel_power in zip(
                    spectra_obj.vel_list_power_group_t[vel_index_start : vel_index_end],
                    list_vel_power
                )
            ],
            list_scale   = list_k_nu,
            bool_pos_top = False
        )
        ## plot measured k_eta
        self.plotScale(
            axs[1], list_colors[1], r"$k_\eta$",
            list_fit_k = spectra_obj.mag_list_k_group_t[0],
            list_fit_power = [
                np.array(mag_fit_power) / np.sum(mag_power)
                for mag_fit_power, mag_power in zip(
                    spectra_obj.mag_list_power_group_t[mag_index_start : mag_index_end],
                    list_mag_power
                )
            ],
            list_scale   = list_k_eta,
            bool_pos_top = False
        )
        ## plot measured k_max
        self.plotScale(
            axs[1], list_colors[1], r"$k_p$",
            list_fit_k = spectra_obj.mag_list_fit_k_group_t[0],
            list_fit_power = [
                np.array(mag_fit_power) / np.sum(mag_power)
                for mag_fit_power, mag_power in zip(
                    spectra_obj.mag_list_fit_power_group_t[mag_index_start : mag_index_end],
                    list_mag_power
                )
            ],
            list_scale   = list_k_max,
            bool_pos_top = True
        )
    def plotSpectra(
            self,
            ax, color,
            list_k, list_power,
            list_fit_k, list_fit_power
        ):
        ## plot spectra data
        ax.plot(
            list_k,
            np.median(list_power, axis=0),
            color=color, ls="-", linewidth=2
        )
        ## plot spectra data error (in time average)
        ax.fill_between(
            list_k,
            1/1.2 * np.percentile(list_power, 16, axis=0),
            1.2 * np.percentile(list_power, 84, axis=0),
            facecolor=color, alpha=0.3, zorder=1
        )
        ## plot spectra fit
        ax.plot(
            list_fit_k,
            np.median(list_fit_power, axis=0),
            color="black", linestyle="-.", linewidth=2
        )
        list_median_fit_vals = list(np.median(list_fit_power, axis=0))
        list_median_vals = list(np.median(list_power, axis=0))
    def plotScale(
            self,
            ax, color, str_scale,
            list_fit_k, list_fit_power, list_scale,
            bool_pos_top = True
        ):
        ## annotate scale
        ax.text(
            0.5 * (np.percentile(list_scale, 16) + np.percentile(list_scale, 84)),
            2 if bool_pos_top else 5e-4,
            str_scale, ha="center",
            va = "bottom" if bool_pos_top else "top",
            fontsize = 18
        )
        ## indicate scale spread
        ax.annotate(
            text   = "",
            xy     = (
                np.percentile(list_scale, 16),
                1 if bool_pos_top else 1e-3
            ),
            xytext = (
                np.percentile(list_scale, 84),
                1 if bool_pos_top else 1e-3
            ),
            arrowprops=dict(
                linewidth  = 1.5,
                color      = color,
                arrowstyle = "|-|, widthA=0.25, widthB=0.25",
                shrinkA    = 0,
                shrinkB    = 0
            ), zorder = 5
        )
        ## connect scale spread to spectra curve
        ax.plot(
            [ np.percentile(list_scale, 50) ] * 2,
            [
                1 if bool_pos_top else 1e-3,
                np.median(list_fit_power, axis=0)[
                    getIndexClosestValue(
                        list_fit_k,
                        np.percentile(list_scale, 50)
                    )
                ]
            ],
            color=color, linestyle="-", linewidth=1.5, zorder=5
        )

class PlotSpectraAndScalesResiduals():
    def __init__(
            self,
            filepath_plot,
            spectra_obj_0, spectra_obj_1
        ):
        ## initialise figure
        factor = 1.1
        fig, axs = plt.subplots(2, 2, figsize=(2*7/factor, 8/factor), sharex=True, gridspec_kw={"height_ratios": [2, 1.1]})
        fig.subplots_adjust(hspace=0.05, wspace=0.25)
        ## plot Re 500 simulation
        self.plotSpectraObj(
            axs,
            spectra_obj = spectra_obj_0,
            list_colors = [ "green", "green" ]
        )
        ## plot Re 1500 simulation
        self.plotSpectraObj(
            axs,
            spectra_obj = spectra_obj_1,
            list_colors = [ "orange", "orange" ]
        )
        ## label axes
        axs[1,0].set_xlabel(r"$k$", fontsize=20)
        axs[1,1].set_xlabel(r"$k$", fontsize=20)
        axs[0,0].set_ylabel(r"$\widehat{\mathcal{P}}_{\mathrm{kin}}(k)$", fontsize=20)
        axs[0,1].set_ylabel(r"$\widehat{\mathcal{P}}_{\mathrm{mag}}(k)$", fontsize=20)
        axs[1,0].set_ylabel(r"$R(k)$", fontsize=20)
        axs[1,1].set_ylabel(r"$R(k)$", fontsize=20)
        ## add legend
        addLegend(
            ax = axs[0, 0],
            loc  = "upper right",
            bbox = (0.95, 1.0),
            artists = [ "-", "-" ],
            colors  = [ "green", "orange" ],
            legend_labels = [ r"Re470Pm2", r"Re1700Pm2" ],
            rspacing = 0.5,
            cspacing = 0.25,
            ncol = 1,
            fontsize = 15,
            labelcolor = "white"
        )
        ## initialise figure domain bounds
        y_max = 1e2
        y_min = 1e-14
        x_max = 300
        x_min = 1
        ## adjust top axis
        axs[0,0].set_xlim(x_min, x_max)
        axs[0,0].set_ylim(y_min, y_max)
        axs[0,0].set_xscale("log")
        axs[0,0].set_yscale("log")
        axs[1,0].set_yscale("log")
        locmin = mpl.ticker.LogLocator(
            base=10.0,
            subs=np.arange(2, 10) * 0.1,
            numticks=100
        )
        axs[0,0].yaxis.set_minor_locator(locmin)
        axs[0,0].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ## adjust bottom axis
        axs[0,1].set_xlim(x_min, x_max)
        axs[0,1].set_ylim(y_min, y_max)
        axs[0,1].set_xscale("log")
        axs[0,1].set_yscale("log")
        axs[1,1].set_yscale("log")
        locmin = mpl.ticker.LogLocator(
            base=10.0,
            subs=np.arange(2, 10) * 0.1,
            numticks=100
        )
        axs[0,0].yaxis.set_minor_locator(locmin)
        axs[0,0].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ## save plot
        fig_name = "fig_spectra_residuals_full.pdf"
        fig_filepath = createFilepath([filepath_plot, fig_name])
        plt.savefig(fig_filepath)
        print("\t> Figure saved: " + fig_name)
    def plotSpectraObj(
            self,
            axs, spectra_obj, list_colors
        ):
        ## check that a time range has been defined to collect statistics about
        sim_times = getCommonElements(spectra_obj.vel_sim_times, spectra_obj.mag_sim_times)
        bool_vel_fit = (spectra_obj.vel_fit_start_t is not None) and (spectra_obj.vel_fit_end_t is not None)
        bool_mag_fit = (spectra_obj.mag_fit_start_t is not None) and (spectra_obj.mag_fit_end_t is not None)
        if not(bool_vel_fit) or not(bool_mag_fit):
            raise Exception("Fit range has not been defined.")
        ## find indices of velocity fit time range
        vel_index_start = getIndexClosestValue(sim_times, spectra_obj.vel_fit_start_t)
        vel_index_end   = getIndexClosestValue(sim_times, spectra_obj.vel_fit_end_t)
        ## find indices of magnetic fit time range
        mag_index_start = getIndexClosestValue(sim_times, spectra_obj.mag_fit_start_t)
        mag_index_end   = getIndexClosestValue(sim_times, spectra_obj.mag_fit_end_t)
        ## load spectra
        list_vel_power = spectra_obj.vel_list_power_group_t[vel_index_start : vel_index_end]
        list_mag_power = spectra_obj.mag_list_power_group_t[mag_index_start : mag_index_end]
        ## load measured scales
        list_k_nu  = cleanMeasuredScales(spectra_obj.k_nu_group_t[vel_index_start  : vel_index_end])
        list_k_eta = cleanMeasuredScales(spectra_obj.k_eta_group_t[mag_index_start : mag_index_end])
        list_kaz, list_k_max = cleanMeasuredScales(
            list_times  = [
                sub_list[1]
                for sub_list in spectra_obj.mag_list_fit_params_group_t[mag_index_start : mag_index_end]
            ],
            list_scales = spectra_obj.k_max_group_t[mag_index_start : mag_index_end]
        )
        ## plot velocity spectra
        self.plotSpectra(
            ax_column = axs[:,0],
            color  = list_colors[0],
            list_k = spectra_obj.vel_list_k_group_t[0],
            list_power = [
                np.array(vel_power) / np.sum(vel_power)
                for vel_power in list_vel_power
            ],
            list_fit_k = spectra_obj.vel_list_fit_k_group_t[0],
            list_fit_power = [
                np.array(vel_fit_power) / np.sum(vel_power)
                for vel_fit_power, vel_power in zip(
                    spectra_obj.vel_list_fit_power_group_t[vel_index_start : vel_index_end],
                    list_vel_power
                )
            ],
            list_N = spectra_obj.vel_list_fit_k_range,
            list_error = spectra_obj.vel_list_fit_2norm_group_t[(vel_index_start + vel_index_end) // 2]
        )
        ## plot magnetic spectra
        self.plotSpectra(
            ax_column = axs[:,1],
            color  = list_colors[1],
            list_k = spectra_obj.mag_list_k_group_t[1],
            list_power = [
                np.array(mag_power) / np.sum(mag_power)
                for mag_power in list_mag_power
            ],
            list_fit_k = spectra_obj.mag_list_fit_k_group_t[0],
            list_fit_power = [
                np.array(mag_fit_power) / np.sum(mag_power)
                for mag_fit_power, mag_power in zip(
                    spectra_obj.mag_list_fit_power_group_t[mag_index_start : mag_index_end],
                    list_mag_power
                )
            ],
            list_N = spectra_obj.vel_list_fit_k_range,
            list_error = spectra_obj.mag_list_fit_2norm_group_t[(mag_index_start + mag_index_end) // 2]
        )
        ## plot measured k_nu
        self.plotScale(
            axs[0,0], list_colors[0], r"$k_\nu$",
            list_fit_k = spectra_obj.vel_list_fit_k_group_t[0],
            list_fit_power = [
                np.array(vel_fit_power) / np.sum(vel_power)
                for vel_fit_power, vel_power in zip(
                    spectra_obj.vel_list_fit_power_group_t[vel_index_start : vel_index_end],
                    list_vel_power
                )
            ],
            list_scale   = list_k_nu,
            bool_pos_top = False
        )
        ## plot measured k_eta
        self.plotScale(
            axs[0,1], list_colors[1], r"$k_\eta$",
            list_fit_k = spectra_obj.mag_list_fit_k_group_t[0],
            list_fit_power = [
                np.array(mag_fit_power) / np.sum(mag_power)
                for mag_fit_power, mag_power in zip(
                    spectra_obj.mag_list_fit_power_group_t[mag_index_start : mag_index_end],
                    list_mag_power
                )
            ],
            list_scale   = list_k_eta,
            bool_pos_top = False
        )
        ## plot measured k_max
        self.plotScale(
            axs[0,1], list_colors[1], r"$k_p$",
            list_fit_k = spectra_obj.mag_list_fit_k_group_t[0],
            list_fit_power = [
                np.array(mag_fit_power) / np.sum(mag_power)
                for mag_fit_power, mag_power in zip(
                    spectra_obj.mag_list_fit_power_group_t[mag_index_start : mag_index_end],
                    list_mag_power
                )
            ],
            list_scale   = list_k_max,
            bool_pos_top = True
        )
    def plotSpectra(
            self,
            ax_column, color,
            list_k, list_power,
            list_fit_k, list_fit_power,
            list_N, list_error
        ):
        ## plot spectra data
        ax_column[0].plot(
            list_k,
            np.median(list_power, axis=0),
            color=color, ls="-", linewidth=2
        )
        ax_column[0].plot(
            list_k,
            np.median(list_power, axis=0),
            color=color, marker="o", ms=5
        )
        ## plot spectra data error (in time average)
        ax_column[0].fill_between(
            list_k,
            1/1.2 * np.percentile(list_power, 16, axis=0),
            1.2 * np.percentile(list_power, 84, axis=0),
            facecolor=color, alpha=0.3, zorder=1
        )
        ## plot spectra fit
        ax_column[0].plot(
            list_fit_k,
            np.median(list_fit_power, axis=0),
            color="black", linestyle="-.", linewidth=2
        )
        ## plot error profile
        ax_column[1].plot(
            list_N,
            list_error,
            color=color, linestyle="-", linewidth=2
        )
        ax_column[1].plot(
            list_N,
            list_error,
            color=color, marker="o", ms=5
        )
        ## get index of minimum R
        k_index = np.argmin(list_error)
        ax_column[0].axvline(x=k_index+5, color=color, ls="--", lw=2)
        ax_column[1].axvline(x=k_index+5, color=color, ls="--", lw=2)
    def plotScale(
            self,
            ax, color, str_scale,
            list_fit_k, list_fit_power, list_scale,
            bool_pos_top = True
        ):
        ## annotate scale
        ax.text(
            0.5 * (np.percentile(list_scale, 16) + np.percentile(list_scale, 84)),
            2 if bool_pos_top else 5e-4,
            str_scale, ha="center",
            va = "bottom" if bool_pos_top else "top",
            fontsize = 18
        )
        ## indicate scale spread
        ax.annotate(
            text   = "",
            xy     = (
                np.percentile(list_scale, 16),
                1 if bool_pos_top else 1e-3
            ),
            xytext = (
                np.percentile(list_scale, 84),
                1 if bool_pos_top else 1e-3
            ),
            arrowprops=dict(
                linewidth  = 1.5,
                color      = color,
                arrowstyle = "|-|, widthA=0.25, widthB=0.25",
                shrinkA    = 0,
                shrinkB    = 0
            ), zorder = 5
        )
        ## connect scale spread to spectra curve
        ax.plot(
            [ np.percentile(list_scale, 50) ] * 2,
            [
                1 if bool_pos_top else 1e-3,
                np.median(list_fit_power, axis=0)[
                    getIndexClosestValue(
                        list_fit_k,
                        np.percentile(list_scale, 50)
                    )
                ]
            ],
            color=color, linestyle="-", linewidth=1.5, zorder=5
        )

class PlotScaleConvergence():
    def __init__(
            self,
            filepath_data,
            filepath_plot
        ):
        ## save filepath to data
        self.filepath_data = filepath_data
        ## initialise figure
        fig, axs = plt.subplots(3, 1, figsize=(6.25/1.1, 3*3.85/1.1), sharex=True)
        fig.subplots_adjust(hspace=0.05)
        ## plot Re466Pm2 data
        self.plotScales(
            axs[0],
            sim_name = "Re500",
            list_res = [ 72, 144, 288, 576 ],
            list_colors = [ "green", "green" ]
        )
        print(" ")
        ## plot Re1676Pm2 data
        self.plotScales(
            axs[0],
            sim_name = "Rm3000",
            list_res = [ 72, 144, 288, 576 ],
            list_colors = [ "orange", "orange" ]
        )
        print(" ")
        ## scale axis
        axs[0].set_xscale("log")
        axs[0].set_yscale("log")
        axs[1].set_yscale("log")
        axs[2].set_yscale("log")
        axs[0].set_ylim([0.4, 11])
        axs[1].set_ylim([0.4, 11])
        axs[2].set_ylim([0.4, 11])
        ## label plots
        axs[0].set_ylabel(r"$k_\nu$", fontsize=20)
        axs[1].set_ylabel(r"$k_\eta$", fontsize=20)
        axs[2].set_ylabel(r"$k_p$", fontsize=20)
        axs[2].set_xlabel(r"Linear grid resolution ($N_{\mathrm{res}}$)", fontsize=20)
        # axs[0].text(
        #     0.05, 0.95,
        #     r"$k_\nu$", color="black",
        #     va="top", ha="left", transform=axs[0].transAxes, fontsize=20
        # )
        # axs[1].text(
        #     0.05, 0.95,
        #     r"$k_\eta$", color="black",
        #     va="top", ha="left", transform=axs[1].transAxes, fontsize=20
        # )
        # axs[2].text(
        #     0.05, 0.95,
        #     r"$k_{\mathrm{p}}$", color="black",
        #     va="top", ha="left", transform=axs[2].transAxes, fontsize=20
        # )
        ## add legend
        addLegend(
            ax = axs[0],
            loc  = "lower right",
            bbox = (1.0, 0.0),
            artists = [ "o", "o" ],
            colors  = [ "green", "orange" ],
            legend_labels = [ r"Re470Pm2", r"Re1700Pm2" ],
            rspacing = 0.5,
            cspacing = 0.25,
            ncol = 1,
            fontsize = 15,
            labelcolor = "white"
        )
        axs[0].text(
            0.7, 0.19,
            r"Re470Pm2", color="black",
            va="bottom", ha="left", transform=axs[0].transAxes, fontsize=15, zorder=10
        )
        axs[0].text(
            0.7, 0.075,
            r"Re1700Pm2", color="black",
            va="bottom", ha="left", transform=axs[0].transAxes, fontsize=15, zorder=10
        )
        ## save plot
        fig_name = "fig_scale_convergence_full.pdf"
        fig_filepath = createFilepath([filepath_plot, fig_name])
        plt.savefig(fig_filepath)
        print("\t> Figure saved: " + fig_name)
    def plotScales(
            self,
            axs, sim_name, list_res, list_colors
        ):
        ## initialise list of scale distributions
        list_k_nu_group_res  = []
        list_k_eta_group_res = []
        list_k_max_group_res = []
        ## load and plot simulation scales
        for sim_res in list_res:
            ## load spectra object
            spectra_obj = loadPickleObject(
                createFilepath([ self.filepath_data, sim_name, str(sim_res), "Pm2" ]),
                SPECTRA_NAME
            )
            ## check that a time range has been defined to collect statistics about
            sim_times = getCommonElements(spectra_obj.vel_sim_times, spectra_obj.mag_sim_times)
            bool_vel_fit = (spectra_obj.vel_fit_start_t is not None) and (spectra_obj.vel_fit_end_t is not None)
            bool_mag_fit = (spectra_obj.mag_fit_start_t is not None) and (spectra_obj.mag_fit_end_t is not None)
            if not(bool_vel_fit) or not(bool_mag_fit):
                raise Exception("Fit range has not been defined.")
            ## find indices of velocity fit time range
            vel_index_start = getIndexClosestValue(sim_times, spectra_obj.vel_fit_start_t)
            vel_index_end   = getIndexClosestValue(sim_times, spectra_obj.vel_fit_end_t)
            ## find indices of magnetic fit time range
            mag_index_start = getIndexClosestValue(sim_times, spectra_obj.mag_fit_start_t)
            mag_index_end   = getIndexClosestValue(sim_times, spectra_obj.mag_fit_end_t)
            ## load and subset distribution of measured scales
            list_k_nu  = cleanMeasuredScales(spectra_obj.k_nu_group_t[vel_index_start  : vel_index_end])
            list_k_eta = cleanMeasuredScales(spectra_obj.k_eta_group_t[mag_index_start : mag_index_end])
            list_k_max = cleanMeasuredScales(spectra_obj.k_max_group_t[mag_index_start : mag_index_end])
            list_k_nu_group_res.append(  list_k_nu )
            list_k_eta_group_res.append( list_k_eta )
            list_k_max_group_res.append( list_k_max )
            ## plot measured scales
            plotErrorBar(axs[0], data_x=[sim_res], data_y=list_k_nu,  color=list_colors[0])
            plotErrorBar(axs[1], data_x=[sim_res], data_y=list_k_eta, color=list_colors[1])
            plotErrorBar(axs[2], data_x=[sim_res], data_y=list_k_max, color=list_colors[1])
        ## fit k_nu scales
        self.fitScales(
            ax = axs[0],
            list_scales_group = list_k_nu_group_res,
            list_res = list_res,
            color = list_colors[0]
        )
        ## fit k_eta scales
        self.fitScales(
            ax = axs[1],
            list_scales_group = list_k_eta_group_res,
            list_res = list_res,
            color = list_colors[1]
        )
        ## fit k_max scales
        self.fitScales(
            ax = axs[2],
            list_scales_group = list_k_max_group_res,
            list_res = list_res,
            color = list_colors[1]
        )
    def fitScales(
            self,
            ax, list_scales_group, list_res, color
        ):
        ## function to use for fitting
        func = ListOfModels.logistic_growth_increasing
        ## fit data
        fit_params, fit_cov = curve_fit(
            func,
            list_res,
            [
                np.median(list_scales)
                for list_scales in list_scales_group
            ],
            bounds = (
                (0.01, 1, 0.5),
                (15, 1000, 3)
            ),
            sigma  = [
                np.std(list_scales)
                for list_scales in list_scales_group
            ],
            absolute_sigma = True
        )
        ## measure fit error
        fit_std = np.sqrt(np.diag(fit_cov))[0]
        ## create plot domain
        domain_array = np.logspace(1, np.log10(1900), 300)
        ## plot converging fit
        ax.plot(
            domain_array,
            func(domain_array, *fit_params),
            color="black", ls="-", linewidth=2
        )
        ## adjust x-domain range
        ax.set_xlim([
            min(domain_array),
            max(domain_array)
        ])
        ## use integer values for y-axis
        bool_small_domain_cross = np.ceil(np.log10(ax.get_ylim()[0])) == np.floor(np.log10(ax.get_ylim()[1]))
        bool_large_domain = (np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0])) > 1
        if bool_small_domain_cross or bool_large_domain:
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.yaxis.set_minor_formatter(ScalarFormatter())


## ###############################################################
## DEFINE MAIN PROGRAM
## ###############################################################
def main():
    ## ####################
    ## INITIALISE VARIABLES
    ## ####################
    ## filepath to data
    filepath_data = "/Users/dukekriel/Documents/Studies/TurbulentDynamo/data/sub_sonic"

    ## ####################
    ## LOAD SIMULATION DATA
    ## ####################
    ## simulation parameters
    list_Re = []
    list_Rm = []
    list_Pm = []
    ## predicted scales
    list_relation_k_nu   = []
    list_relation_k_eta  = []
    ## measured (convereged) scales
    list_k_nu_converged_group  = []
    list_k_eta_converged_group = []
    list_k_max_converged_group = []
    ## kazantsev exponent
    list_alpha_vel_group = []
    list_alpha_mag_group = []
    ## list of simlation markers
    list_markers = []
    ## load data
    funcLoadData(
        filepath_data,
        list_Re, list_Rm, list_Pm,
        list_relation_k_nu, list_relation_k_eta,
        list_k_nu_converged_group, list_k_eta_converged_group, list_k_max_converged_group,
        list_alpha_vel_group, list_alpha_mag_group,
        list_markers
    )
    ## define simulation points color
    list_colors = [
        "cornflowerblue" if Re < 100
        else "orangered"
        for Re in list_Re
    ]

    ## ####################
    ## PLOT SIMULATION DATA
    ## ####################
    ## filepath to figures
    filepath_plot = "/Users/dukekriel/Documents/Studies/TurbulentDynamo/figures/sub_sonic"
    print("Saving figures in: " + filepath_plot)

    ## plot measured vs predicted scales
    funcPlotScaleRelations(
        filepath_plot,
        list_colors, list_markers,
        list_Re, list_Rm, list_Pm,
        list_relation_k_nu, list_relation_k_eta,
        list_k_nu_converged_group, list_k_eta_converged_group,
    )

    # ## plot peak scale dependance on dissipation scales
    # funcPlotScaleDependance(
    #     filepath_plot,
    #     list_colors, list_markers,
    #     list_Re, list_Rm, list_Pm,
    #     list_k_nu_converged_group, list_k_eta_converged_group, list_k_max_converged_group,
    # )

    # ## plot dependance of powerlaw exponent on Re
    # funcPlotExponent(
    #     filepath_plot,
    #     list_colors, list_markers,
    #     list_Re, list_Rm, list_Pm,
    #     list_alpha_mag_group
    # )

    # ## plot spectra + measured scales
    # PlotSpectraAndScales(
    #     filepath_plot,
    #     loadPickleObject(
    #         createFilepath([ filepath_data, "Re500",  "288", "Pm2" ]),
    #         SPECTRA_NAME,
    #         bool_hide_updates = True
    #     ),
    #     loadPickleObject(
    #         createFilepath([ filepath_data, "Rm3000",  "288", "Pm2" ]),
    #         SPECTRA_NAME,
    #         bool_hide_updates = True
    #     )
    # )
    # print(" ")
    # PlotSpectraAndScalesResiduals(
    #     filepath_plot,
    #     loadPickleObject(
    #         createFilepath([ filepath_data, "Re500",  "288", "Pm2" ]),
    #         SPECTRA_NAME,
    #         bool_hide_updates = True
    #     ),
    #     loadPickleObject(
    #         createFilepath([ filepath_data, "Rm3000",  "288", "Pm2" ]),
    #         SPECTRA_NAME,
    #         bool_hide_updates = True
    #     )
    # )

    # ## plot scale convergence
    # PlotScaleConvergence(filepath_data, filepath_plot)

    # ## ##################
    # ## MEASURE STATISTICS
    # ## ##################
    # print(" ")
    # print("Statistics...")
    # ## measure converged scales + kazantsev exponent
    # for sim_index in range(len(list_Re)):
    #     if sim_index in [4, 7, 15]: print(" ")
    #     str_k_nu  = funcPrintForm(list_k_nu_converged_group[sim_index])
    #     str_k_eta = funcPrintForm(list_k_eta_converged_group[sim_index])
    #     str_k_max = funcPrintForm(list_k_max_converged_group[sim_index])
    #     str_alpha_vel = funcPrintForm(list_alpha_vel_group[sim_index])
    #     str_alpha_mag = funcPrintForm(list_alpha_mag_group[sim_index])
    #     print("& " + str_k_nu + " & " + str_k_eta  + " & " + str_k_max)
    #     print("\t" + " & " + str_alpha_vel + " & " + str_alpha_mag + " \\\\")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
    main()
    sys.exit()


## END OF PROGRAM
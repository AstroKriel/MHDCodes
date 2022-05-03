#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit

os.environ['PYTHONPATH'].split(os.pathsep) 

## load user defined modules
from ThePlottingModule import PlotFuncs, TheMatplotlibStyler
from TheUsefulModule import WWLists, WWFnF, WWObjs
from TheLoadingModule import LoadFlashData
from TheFittingModule import UserModels


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.ioff()
plt.switch_backend("agg") # use a non-interactive plotting backend


SPECTRA_NAME = "spectra_obj_full.pkl"
## ###############################################################
## FUNCTIONS
## ###############################################################
def funcPlotTurb(axs, filepath_data):
  def fitExp(ax, data_x, data_y, index_start_fit, index_end_fit):
    ## define fit domain
    data_fit_domain = np.linspace(
      data_x[index_start_fit],
      data_x[index_end_fit],
      10**2
    )
    ## interpolate the non-uniform data
    interp_spline = make_interp_spline(
      data_x[index_start_fit : index_end_fit],
      data_y[index_start_fit : index_end_fit]
    )
    ## uniformly sample interpolated data
    data_sampled_y = interp_spline(data_fit_domain)
    ## fit exponential function to sampled data (in log-linear domain)
    fit_params_log, _ = curve_fit(
      UserModels.ListOfModels.exp_loge,
      data_fit_domain,
      np.log(data_sampled_y)
    )
    ## undo log transformation
    fit_params_linear = [
      np.exp(fit_params_log[0]),
      fit_params_log[1]
    ]
    ## initialise the plot domain
    data_plot_domain = np.linspace(0, 100, 10**3)
    ## evaluate exponential
    data_E_exp = UserModels.ListOfModels.exp_linear(
      data_plot_domain,
      *fit_params_linear
    )
    ## find where exponential enters / exists fit range
    index_E_start = WWLists.getIndexClosestValue(data_E_exp, data_y[index_start_fit])
    index_E_end = WWLists.getIndexClosestValue(data_E_exp, data_y[index_end_fit])
    ## create line data
    line_fitted = [ np.column_stack((
      data_plot_domain[index_E_start : index_E_end],
      data_E_exp[index_E_start : index_E_end]
    )) ]
    ## plot fitted line
    ax.add_collection(
      LineCollection(line_fitted, colors="red", ls="--", linewidth=3, zorder=9),
      autolim = False # ignore line when setting axis bounds
    )
  def fitSat(ax, data_x, data_y, index_start_fit, index_end_fit):
    ## define fit domain
    data_fit_domain = np.linspace(
      data_x[index_start_fit],
      data_x[index_end_fit],
      10**2
    )
    ## interpolate the non-uniform data
    interp_spline = make_interp_spline(
      data_x[index_start_fit : index_end_fit],
      data_y[index_start_fit : index_end_fit]
    )
    ## uniformly sample interpolated data
    data_sampled_y = interp_spline(data_fit_domain)
    ## measure average saturation level
    mean_y = np.mean(data_sampled_y)
    ## create line data
    line_fitted = [ np.column_stack((
      [
        data_x[index_start_fit],
        data_x[index_end_fit]
      ],
      [mean_y] * 2
    )) ]
    ## plot fitted line
    ax.add_collection(
        LineCollection(line_fitted, colors="red", ls=":", linewidth=3, zorder=9),
        autolim = False # ignore line when setting axis bounds
    )
  ## load mach data
  data_time, data_Mach = LoadFlashData.loadTurbData(
    filepath_data = filepath_data,
    var_y      = 13, # 13 (new), 8 (old)
    t_eddy     = 0.1, # TODO input: list_t_eddy,
    time_start = 0.1, # TODO input: plot_start,
    time_end   = np.inf, # TODO input: plot_end
  )
  ## load magnetic energy
  data_time, data_E_B = LoadFlashData.loadTurbData(
    filepath_data = filepath_data,
    var_y      = 11, # 11 (new), 29 (old)
    t_eddy     = 0.1, # TODO input: list_t_eddy,
    time_start = 0.1, # TODO input: plot_start,
    time_end   = np.inf, # TODO input: plot_end
  )
  ## load kinetic energy
  data_time, data_E_K = LoadFlashData.loadTurbData(
    filepath_data = filepath_data,
    var_y      = 9, # 9 (new), 6 (old)
    t_eddy     = 0.1, # TODO input: list_t_eddy,
    time_start = 0.1, # TODO input: plot_start,
    time_end   = np.inf, # TODO input: plot_end
  )
  ## calculate energy ratio: 'E_B / E_K'
  data_E_ratio = [
    (E_B / E_K) for E_B, E_K in zip(data_E_B, data_E_K)
  ]
  ## plot mach
  axs[0].plot(
    data_time, data_Mach,
    color="black", marker=".", ms=1, ls="-", lw=1
  )
  axs[0].set_xlim([0, max(data_time)])
  axs[0].set_xlabel(r"$t / t_\mathrm{turb}$", fontsize=22)
  axs[0].set_ylabel(r"$\mathcal{M}$", fontsize=22)
  ## plot energy ratio
  axs[1].plot(
    data_time, data_E_ratio,
    color="black", marker=".", ms=1, ls="-", lw=1
  )
  axs[1].set_xlabel(r"$t / t_\mathrm{turb}$", fontsize=22)
  axs[1].set_ylabel(r"$E_\mathrm{mag} / E_\mathrm{kin}$", fontsize=22)
  axs[1].set_xlim([0, max(data_time)])
  axs[1].set_yscale("log")
  axs[1].set_ylim([ 10**(-10), 10**(1) ])
  ## get index range corresponding with kinematic phase of the dynamo
  index_exp_start = WWLists.getIndexClosestValue(data_E_ratio, 10**(-9))
  index_exp_end   = WWLists.getIndexClosestValue(data_E_ratio, 10**(-3))
  ## fit mach number
  fitSat(
    axs[0],
    data_x = data_time,
    data_y = data_Mach,
    index_start_fit = index_exp_start,
    index_end_fit   = index_exp_end
  )
  ## fit exponential
  fitExp(
    axs[1],
    data_x = data_time,
    data_y = data_E_ratio,
    index_start_fit = index_exp_start,
    index_end_fit   = index_exp_end
  )
  ## fit saturation
  fitSat(
    axs[1],
    data_x = data_time,
    data_y = data_E_ratio,
    index_start_fit = WWLists.getIndexClosestValue(data_time, 0.75 * data_time[-1]),
    index_end_fit   = len(data_time)-1
  )
  return data_time[index_exp_start], data_time[index_exp_end]


def funcPlotSpectra(axs1, axs2, filepath_data, time_exp_start, time_exp_end):
  def plotData(ax, data_x, data_y):
    ax.plot(
      data_x, data_y,
      color="black", ls="-", lw=1, alpha=0.3, zorder=3
    )
    ax.plot(
      data_x, data_y,
      color="black", marker=".", ms=1, zorder=3
    )
  def plotSubsetData(ax, data_x, data_y, time_exp_start, time_exp_end):
    if (time_exp_start is not None) and (time_exp_end is not None):
      time_exp_start = WWLists.getIndexClosestValue(data_x, time_exp_start)
      time_exp_end   = WWLists.getIndexClosestValue(data_x, time_exp_end)
      ax.plot(
        data_x[time_exp_start : time_exp_end],
        data_y[time_exp_start : time_exp_end],
        color="red", marker=".", ms=2, zorder=5
      )
  ## ##########################
  ## LOAD SPECTRA OBJECT + DATA
  ## ##########################
  # print(time_exp_start, time_exp_end)
  range_k = [ 0.05, 150 ]
  range_alpha = [ -4, 4 ]
  ## load spectra object
  spectra_obj = WWObjs.loadPickleObject(filepath_data, SPECTRA_NAME, bool_hide_updates=True)
  ## load time-evolving measured parameters
  kin_sim_times = spectra_obj.kin_sim_times
  mag_sim_times = spectra_obj.mag_sim_times
  kin_num_points_fitted = spectra_obj.kin_fit_k_index_group_t
  mag_num_points_fitted = spectra_obj.mag_fit_k_index_group_t
  k_nu  = spectra_obj.k_nu_group_t
  k_eta = spectra_obj.k_eta_group_t
  k_max = spectra_obj.k_max_group_t
  kin_alpha = [
    list_fit_params[1]
    for list_fit_params in spectra_obj.kin_list_fit_params_group_t
  ]
  mag_alpha = [
    list_fit_params[1]
    for list_fit_params in spectra_obj.mag_list_fit_params_group_t
  ]
  ## #########################
  ## KINETIC SPECTRA PARMETERS
  ## #########################
  ## plot number of points fitted to kinetic spectra
  plotData(axs1[0], kin_sim_times, kin_num_points_fitted)
  plotSubsetData(
    ax = axs1[0],
    data_x = kin_sim_times,
    data_y = kin_num_points_fitted,
    time_exp_start = time_exp_start,
    time_exp_end   = time_exp_end
  )
  axs1[0].axes.xaxis.set_ticklabels([])
  axs1[0].set_yscale("log")
  PlotFuncs.FixLogAxis(axs1[0], bool_fix_y_axis=True)
  ## plot alpha_kin
  plotData(axs1[1], kin_sim_times, kin_alpha)
  plotSubsetData(
    ax = axs1[1],
    data_x = kin_sim_times,
    data_y = kin_alpha,
    time_exp_start = time_exp_start,
    time_exp_end   = time_exp_end
  )
  axs1[1].axes.xaxis.set_ticklabels([])
  axs1[1].set_ylabel(r"$\alpha_\mathrm{kin}$", fontsize=22)
  axs1[1].set_ylim(range_alpha)
  ## plot k_nu
  plotData(axs1[2], kin_sim_times, k_nu)
  plotSubsetData(
    ax = axs1[2],
    data_x = kin_sim_times,
    data_y = k_nu,
    time_exp_start = time_exp_start,
    time_exp_end   = time_exp_end
  )
  axs1[2].axes.xaxis.set_ticklabels([])
  axs1[2].set_ylabel(r"$k_\nu$", fontsize=22)
  axs1[2].set_yscale("log")
  axs1[2].set_ylim(range_k)
  ## ##########################
  ## MAGNETIC SPECTRA PARMETERS
  ## ##########################
  ## plot number of points fitted to magnetic spectra
  plotData(axs2[0], mag_sim_times, mag_num_points_fitted)
  plotSubsetData(
    ax = axs2[0],
    data_x = mag_sim_times,
    data_y = mag_num_points_fitted,
    time_exp_start = time_exp_start,
    time_exp_end   = time_exp_end
  )
  axs2[0].set_ylabel(r"$\#$ of k-modes fitted", fontsize=22, loc="bottom")
  axs2[0].set_yscale("log")
  PlotFuncs.FixLogAxis(axs2[0], bool_fix_y_axis=True)
  ## plot alpha_mag
  plotData(axs2[1], mag_sim_times, mag_alpha)
  plotSubsetData(
    ax = axs2[1],
    data_x = mag_sim_times,
    data_y = mag_alpha,
    time_exp_start = time_exp_start,
    time_exp_end   = time_exp_end
  )
  axs2[1].set_ylabel(r"$\alpha_\mathrm{mag}$", fontsize=22)
  axs2[1].set_ylim(range_alpha)
  ## plot k_eta
  plotData(axs2[2], mag_sim_times, k_eta)
  plotSubsetData(
    ax = axs2[2],
    data_x = mag_sim_times,
    data_y = k_eta,
    time_exp_start = time_exp_start,
    time_exp_end   = time_exp_end
  )
  axs2[2].set_ylabel(r"$k_\eta$", fontsize=22)
  axs2[2].set_yscale("log")
  axs2[2].set_ylim(range_k)
  ## plot k_max
  plotData(axs2[3], mag_sim_times, k_max)
  plotSubsetData(
    ax = axs2[3],
    data_x = mag_sim_times,
    data_y = k_max,
    time_exp_start = time_exp_start,
    time_exp_end   = time_exp_end
  )
  axs2[3].set_ylabel(r"$k_\mathrm{p}$", fontsize=22)
  axs2[3].set_yscale("log")
  axs2[3].set_ylim(range_k)


def funcPlotSimData(filepath_data, filepath_plot, fig_name):
  ## initialise figure
  fig = plt.figure(constrained_layout=True, figsize=(12, 8))
  ## create figure sub-axis
  gs = GridSpec(3, 4, figure=fig)
  ## top row: 'Turb.dat' data
  ax00 = fig.add_subplot(gs[0, :2])
  ax01 = fig.add_subplot(gs[0, 2:])
  ## middle row: kinetic energy spectra fits
  ax10 = fig.add_subplot(gs[1, 0])
  ax11 = fig.add_subplot(gs[1, 1])
  ax12 = fig.add_subplot(gs[1, 2])
  ## bottom row: magnetic spectra fits
  ax20 = fig.add_subplot(gs[2, 0])
  ax21 = fig.add_subplot(gs[2, 1])
  ax22 = fig.add_subplot(gs[2, 2])
  ax23 = fig.add_subplot(gs[2, 3])
  ## plot Turb.dat data
  if os.path.exists(WWFnF.createFilepath([filepath_data, "Turb.dat"])):
    time_exp_start, time_exp_end = funcPlotTurb((ax00, ax01), filepath_data)
    bool_plot_energy = True
  else:
    bool_plot_energy = False
    time_exp_start, time_exp_end = None, None
  ## plot spectra fitted data
  if os.path.exists(WWFnF.createFilepath([filepath_data, "spectra_obj_full.pkl"])):
    funcPlotSpectra(
      axs1 = [ax10, ax11, ax12],
      axs2 = [ax20, ax21, ax22, ax23],
      filepath_data   = filepath_data,
      time_exp_start = time_exp_start,
      time_exp_end   = time_exp_end
    )
    bool_plot_spectra = True
  else: bool_plot_spectra = False
  ## check if the data was plotted
  if not(bool_plot_energy) and not(bool_plot_spectra):
    print("\t> ERROR: No data in:", filepath_data)
  else:
    ## check what data was missing
    if not(bool_plot_energy):
      print("\t> ERROR: No 'Turb.dat' data in:", filepath_data)
    elif not(bool_plot_spectra):
      print("\t> ERROR: No spectra data in:", filepath_data)
    ## save the figure
    fig_filepath = WWFnF.createFilepath([filepath_plot, fig_name])
    plt.savefig(fig_filepath)
    plt.close()
    print("\t> Figure saved:", fig_name)


SONIC_REGIME = "super_sonic"
BASEPATH = "/Users/dukekriel/Documents/Studies/TurbulentDynamo/data/"
## ###############################################################
## DEFINE MAIN PROGRAM
## ###############################################################
def main():
  filepath_base = BASEPATH + SONIC_REGIME

  ## #######################
  ## LOOK AT EACH SIMULATION
  ## #######################
  for suite_folder in [
      "Re10", "Re500", "Rm3000"
    ]: # "Re10", "Re500", "Rm3000", "keta"

    for sim_res in [
        "72", "144", "288", "576"
      ]: # "18", "36", "72", "144", "288", "576"
      ## ####################################
      ## CREATE FILEPATH TO SIMULATION FOLDER
      ## ####################################
      filepath_figures = WWFnF.createFilepath([
        filepath_base, suite_folder, sim_res, "vis_folder"
      ])
      ## check that the filepath exists on MAC
      if not os.path.exists(filepath_figures):
        print("{} does not exist.".format(filepath_figures))
        continue
      str_message = "Looking at suite: {}, Nres = {}".format(suite_folder, sim_res)
      print(str_message)
      print("=" * len(str_message))

      ## ####################
      ## PLOT SIMULATION DATA
      ## ####################
      for sim_folder in [
          "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"
        ]: # "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"
        ## create filepath to the simulation folder
        filepath_data = WWFnF.createFilepath([ filepath_base, suite_folder, sim_res, sim_folder ])
        ## check that the filepath exists
        if not os.path.exists(filepath_data):
          continue
        ## plot simulation data
        fig_name = suite_folder + "_" + sim_folder + "_" + "check.png"
        funcPlotSimData(filepath_data, filepath_figures, fig_name)
    ## create an empty line after each suite
    print(" ")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
    main()
    sys.exit()


## END OF PROGRAM
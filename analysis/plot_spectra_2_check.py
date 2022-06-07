#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit

## load user defined modules
from ThePlottingModule import PlotSpectra, PlotFuncs
from TheUsefulModule import WWLists, WWFnF, WWObjs
from TheLoadingModule import LoadFlashData
from TheFittingModule import FitMHDScales, UserModels


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.ioff()
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## FUNCTIONS
## ###############################################################
def funcPlotTurb(
    axs, filepath_data,
    t_turb = 0.1 # ell_turb / (Mach * c_s)
  ):
  color_fits = "black"
  color_data = "orange"
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
    ## save time range being fitted to
    time_start = data_x[index_start_fit]
    time_end   = data_x[index_end_fit]
    ## uniformly sample interpolated data
    data_y_sampled = interp_spline(data_fit_domain)
    ## fit exponential function to sampled data (in log-linear domain)
    fit_params_log, fit_params_cov = curve_fit(
      UserModels.ListOfModels.exp_loge,
      data_fit_domain,
      np.log(data_y_sampled)
    )
    ## undo log transformation
    fit_params_linear = [
      np.exp(fit_params_log[0] + 2),
      fit_params_log[1]
    ]
    ## initialise the plot domain
    data_x_fit = np.linspace(0, 100, 10**3)
    ## evaluate exponential
    data_y_fit = UserModels.ListOfModels.exp_linear(
      data_x_fit,
      *fit_params_linear
    )
    ## find where exponential enters / exists fit range
    index_E_start = WWLists.getIndexClosestValue(data_x_fit, time_start)
    index_E_end   = WWLists.getIndexClosestValue(data_x_fit, time_end)
    ## plot fit
    gamma_val = -fit_params_log[1]
    gamma_std = max(np.sqrt(np.diag(fit_params_cov))[1], 0.01)
    str_label = r"$\Gamma =$ " + "{:.2f}".format(gamma_val) + r" $\pm$ " + "{:.2f}".format(gamma_std)
    ax.plot(
      data_x_fit[index_E_start : index_E_end],
      data_y_fit[index_E_start : index_E_end],
      label=str_label, color=color_fits, ls="--", lw=2, zorder=5
    )
  def fitConst(ax, data_x, data_y, index_start_fit, index_end_fit, label=""):
    ## define fit domain
    data_fit_domain = np.linspace(
      data_x[index_start_fit],
      data_x[index_end_fit],
      10**2
    )
    ## interpolate the non-uniform data
    print(
      len(data_x[index_start_fit : index_end_fit]),
      len(data_y[index_start_fit : index_end_fit])
    )
    interp_spline = make_interp_spline(
      data_x[index_start_fit : index_end_fit],
      data_y[index_start_fit : index_end_fit]
    )
    ## uniformly sample interpolated data
    data_y_sampled = interp_spline(data_fit_domain)
    ## measure average saturation level
    data_x_sub  = data_x[index_start_fit : index_end_fit]
    data_y_mean = np.mean(data_y_sampled)
    data_y_std  = max(np.std(data_y_sampled), 0.01)
    ## plot fit
    str_label = label + "{:.2f}".format(data_y_mean) + r" $\pm$ " + "{:.2f}".format(data_y_std)
    ax.plot(
      data_x_sub,
      [data_y_mean] * len(data_x_sub),
      label=str_label, color=color_fits, ls=":", lw=2, zorder=5
    )
    ## return mean value
    return data_y_mean
  ## load mach data
  data_time, data_Mach = LoadFlashData.loadTurbData(
    filepath_data = filepath_data,
    var_y      = 13, # 13 (new), 8 (old)
    t_turb     = t_turb,
    time_start = 0.1,
    time_end   = np.inf
  )
  ## load magnetic energy
  data_time, data_E_B = LoadFlashData.loadTurbData(
    filepath_data = filepath_data,
    var_y      = 11, # 11 (new), 29 (old)
    t_turb     = t_turb,
    time_start = 0.1,
    time_end   = np.inf
  )
  ## load kinetic energy
  data_time, data_E_K = LoadFlashData.loadTurbData(
    filepath_data = filepath_data,
    var_y      = 9, # 9 (new), 6 (old)
    t_turb     = t_turb,
    time_start = 0.1,
    time_end   = np.inf
  )
  ## calculate max of the plot domain
  max_time = max([ 100, max(data_time) ])
  ## calculate energy ratio: 'E_B / E_K'
  data_E_ratio = [
    (E_B / E_K) for E_B, E_K in zip(data_E_B, data_E_K)
  ]
  ## plot mach
  axs[0].plot(
    data_time, data_Mach,
    color=color_data, ls="-", lw=1.5, zorder=3
  )
  axs[0].set_xlabel(r"$t / t_\mathrm{turb}$")
  axs[0].set_ylabel(r"$\mathcal{M}$")
  axs[0].set_xlim([ 0, max_time ])
  ## plot energy ratio
  axs[1].plot(
    data_time, data_E_ratio,
    color=color_data, ls="-", lw=1.5, zorder=3
  )
  ## define y-axis range
  min_E_ratio         = min(data_E_ratio)
  log_min_E_ratio     = np.log10(min_E_ratio)
  new_log_min_E_ratio = np.floor(log_min_E_ratio)
  num_decades         = 1 + (-new_log_min_E_ratio)
  new_min_E_ratio     = 10**new_log_min_E_ratio
  num_y_major_ticks   = np.ceil(num_decades / 2)
  ## label figure
  axs[1].set_xlabel(r"$t / t_\mathrm{turb}$")
  axs[1].set_ylabel(r"$E_\mathrm{mag} / E_\mathrm{kin}$")
  axs[1].set_yscale("log")
  axs[1].set_xlim([ 0, max_time ])
  axs[1].set_ylim([ new_min_E_ratio, 10**(1) ])
  ## add log axis-ticks
  PlotFuncs.addLogAxisTicks(
    axs[1],
    # bool_minor_ticks    = True,
    bool_major_ticks    = True,
    max_num_major_ticks = num_y_major_ticks
  )
  ## fit saturation
  sat_ratio = fitConst(
    axs[1],
    data_x = data_time,
    data_y = data_E_ratio,
    label  = r"$\left(E_{\rm kin} / E_{\rm mag}\right)_{\rm sat} =$ ",
    index_start_fit = WWLists.getIndexClosestValue(data_time, 0.75 * data_time[-1]),
    index_end_fit   = len(data_time)-1
  )
  ## get index range corresponding with kinematic phase of the dynamo
  index_exp_start = WWLists.getIndexClosestValue(data_E_ratio, 10**(-6))
  index_exp_end   = WWLists.getIndexClosestValue(data_E_ratio, 10**(-2)) # sat_ratio/100: 1-percent of sat-ratio
  ## fit mach number
  fitConst(
    axs[0],
    data_x = data_time,
    data_y = data_Mach,
    label  = r"$\mathcal{M} =$ ",
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
  ## add legend
  legend_ax0 = axs[0].legend(frameon=False, loc="lower right", fontsize=14)
  legend_ax1 = axs[1].legend(frameon=False, loc="lower right", fontsize=14)
  axs[0].add_artist(legend_ax0)
  axs[1].add_artist(legend_ax1)
  ## return time range corresponding to kinematic phase
  return data_time[index_exp_start], data_time[index_exp_end]


def funcPlotSpectra(
    axs, ax_spectra, filepath_data,
    time_exp_start = None,
    time_exp_end   = None
  ):
  time_range = [time_exp_start, time_exp_end]
  def plotData(
      ax, data_x, data_y,
      label  = None,
      color  = "black",
      zorder = 5
    ):
    ## if a fitting range is defined
    if (time_range[0] is not None) and (time_range[1] is not None):
      index_start = WWLists.getIndexClosestValue(data_x, time_range[0])
      index_end   = WWLists.getIndexClosestValue(data_x, time_range[1])
      ## measure data statistics in the fitting range
      if "=" in label:
        data_x_sub  = data_x[index_start : index_end]
        data_y_sub  = data_y[index_start : index_end]
        data_y_mean = np.mean(data_y_sub)
        data_y_std  = max(np.std(data_y_sub), 0.01)
        label += "{:.2f}".format(data_y_mean) + r" $\pm$ " + "{:.2f}".format(data_y_std)
        ## plot fit
        ax.plot(
          data_x_sub,
          [data_y_mean] * len(data_x_sub),
          color="black", ls=":", lw=2, zorder=(zorder+2)
        )
      ## plot full dataset
      ax.plot(data_x, data_y, color=color, ls="-", lw=1, alpha=0.1, zorder=(zorder-2))
      ## plot subset of data
      ax.plot(
        data_x[index_start : index_end],
        data_y[index_start : index_end],
        label=label, color=color, ls="-", lw=1.5, zorder=zorder
      )
      ## return minimum of sub-setted measured scales
      return min(data_y[index_start : index_end])
    else:
      ## plot full dataset
      ax.plot(data_x, data_y, label=label, color=color, ls="-", lw=1.5, zorder=(zorder-2))
      ## return minimum of full range of measured scales
      return min(data_y)
  ## #################
  ## LOAD SPECTRA DATA
  ## #################
  ## load spectra-fit data as a dictionary
  spectra_fits_dict = WWObjs.loadJson2Dict(
    filepath = filepath_data,
    filename = FILENAME_SPECTRA,
    bool_hide_updates = True
  )
  ## store dictionary data in spectra-fit object
  spectra_fits_obj = FitMHDScales.SpectraFit(**spectra_fits_dict)
  ## load time-evolving measured parameters
  kin_sim_times = spectra_fits_obj.kin_sim_times
  mag_sim_times = spectra_fits_obj.mag_sim_times
  kin_num_points_fitted = spectra_fits_obj.kin_fit_k_index_group_t
  mag_num_points_fitted = spectra_fits_obj.mag_fit_k_index_group_t
  ## check that there is sufficient data points to plot
  bool_kin_spectra_fitted = len(kin_sim_times) > 0
  bool_mag_spectra_fitted = len(mag_sim_times) > 0
  bool_plot_fit_params = bool_kin_spectra_fitted or bool_mag_spectra_fitted
  if bool_plot_fit_params:
    ## load kinetic energy spectra fit paramaters
    if bool_kin_spectra_fitted:
      k_nu = spectra_fits_obj.k_nu_group_t
      kin_alpha = [
        list_fit_params[1]
        for list_fit_params in spectra_fits_obj.kin_list_fit_params_group_t
      ]
    ## load magnetic energy spectra fit paramaters
    if bool_mag_spectra_fitted:
      k_modes   = spectra_fits_obj.mag_list_k_group_t[0]
      k_eta     = spectra_fits_obj.k_eta_group_t
      k_p       = spectra_fits_obj.k_p_group_t
      k_max     = spectra_fits_obj.k_max_group_t
      mag_alpha = [
        list_fit_params[1]
        for list_fit_params in spectra_fits_obj.mag_list_fit_params_group_t
      ]
    elif bool_kin_spectra_fitted:
      k_modes = spectra_fits_obj.kin_list_k_group_t[0]
    ## define the end of the k-domain
    max_k_mode = 1.2 * max(k_modes)
  ## #####################
  ## PLOT AVERAGED SPECTRA
  ## #####################
  PlotSpectra.PlotAveSpectra(ax_spectra, spectra_fits_obj, time_range)
  ax_spectra.set_ylim([ 10**(-8), 3*10**(0) ])
  if bool_plot_fit_params:
    ## ################################
    ## PLOT NUMBER OF K-MODES FITTED TO
    ## ################################
    if bool_kin_spectra_fitted:
      plotData(axs[0], kin_sim_times, kin_num_points_fitted, label=r"kin-spectra", color="blue")
      max_sim_time = max(kin_sim_times)
    if bool_mag_spectra_fitted:
      plotData(axs[0], mag_sim_times, mag_num_points_fitted, label=r"mag-spectra", color="red")
      max_sim_time = max(mag_sim_times)
    ## label and tune figure
    axs[0].legend(frameon=False, loc="lower right", fontsize=14)
    axs[0].set_xlabel(r"$t / t_\mathrm{turb}$")
    axs[0].set_ylabel(r"max k-mode")
    axs[0].set_yscale("log")
    axs[0].set_xlim([0, max_sim_time])
    axs[0].set_ylim([1, max_k_mode])
    ## add log axis-ticks
    PlotFuncs.addLogAxisTicks(
      ax                  = axs[0],
      bool_major_ticks    = True,
      bool_minor_ticks    = True,
      max_num_major_ticks = 5
    )
    ## ####################
    ## PLOT ALPHA EXPONENTS
    ## ####################
    range_alpha = [
      min([ -4, 1.1*min(kin_alpha), 1.1*min(mag_alpha) ]),
      max([  4, 1.1*max(kin_alpha), 1.1*max(mag_alpha) ])
    ]
    if bool_kin_spectra_fitted:
      plotData(axs[1], kin_sim_times, kin_alpha, label=r"$\alpha_{\rm kin} =$ ", color="blue")
    if bool_mag_spectra_fitted:
      plotData(axs[1], mag_sim_times, mag_alpha, label=r"$\alpha_{\rm mag} =$ ", color="red")
    ## label and tune figure
    axs[1].legend(frameon=False, loc="right", fontsize=14)
    axs[1].set_xlabel(r"$t / t_\mathrm{turb}$")
    axs[1].set_ylabel(r"$\alpha$")
    axs[1].set_xlim([0, max_sim_time])
    axs[1].set_ylim(range_alpha)
    ## add log axis-ticks
    PlotFuncs.addLinearAxisTicks(
      ax               = axs[1],
      bool_minor_ticks    = True,
      bool_major_ticks    = True,
      max_num_minor_ticks = 10,
      max_num_major_ticks = 5
    )
    ## ####################
    ## PLOT MEASURED SCALES
    ## ####################
    ## fitted k modes
    if bool_kin_spectra_fitted:
      min_k_nu  = plotData(axs[2], kin_sim_times, k_nu,  color="blue",   label=r"$k_\nu =$ ")
    if bool_mag_spectra_fitted:
      min_k_eta = plotData(axs[2], mag_sim_times, k_eta, color="red",    label=r"$k_\eta =$ ")
      min_k_p   = plotData(axs[2], mag_sim_times, k_p,   color="green",  label=r"$k_{\rm p} =$ ", zorder=7)
      min_k_max = plotData(axs[2], mag_sim_times, k_max, color="purple", label=r"$k_{\rm max} =$ ")
    ## define y-range
    if bool_kin_spectra_fitted and bool_mag_spectra_fitted:
      range_k = [ 
        min([ 1, min([min_k_nu, min_k_eta, min_k_p, min_k_max]) ]),
        max_k_mode
      ]
    elif bool_mag_spectra_fitted:
      range_k = [ 
        min([ 1, min([min_k_eta, min_k_p, min_k_max]) ]),
        max_k_mode
      ]
    else:
      range_k = [ 
        min([ 1, min_k_nu ]),
        max_k_mode
      ]
    ## label and tune figure
    axs[2].legend(frameon=False, loc="upper right", fontsize=14)
    axs[2].set_xlabel(r"$t / t_\mathrm{turb}$")
    axs[2].set_ylabel(r"$k$")
    axs[2].set_yscale("log")
    axs[2].set_xlim([0, max_sim_time])
    axs[2].set_ylim(range_k)
    ## add log axis-ticks
    PlotFuncs.addLogAxisTicks(
      ax                  = axs[2],
      bool_major_ticks    = True,
      bool_minor_ticks    = True,
      max_num_major_ticks = 5
    )
  ## return simulation parameters
  return spectra_fits_obj.Re, spectra_fits_obj.Rm, spectra_fits_obj.Pm


def funcPlotSimData(filepath_sim, filepath_plot, fig_name, sim_res):
  ## initialise figure
  fig = plt.figure(constrained_layout=True, figsize=(12, 8))
  ## create figure sub-axis
  gs = GridSpec(3, 2, figure=fig)
  ## 'Turb.dat' data
  ax_mach    = fig.add_subplot(gs[0, 1])
  ax_energy  = fig.add_subplot(gs[1, 1])
  ## energy spectra
  ax_spectra = fig.add_subplot(gs[2, 1])
  ## spectra fits
  ax_kmodes  = fig.add_subplot(gs[0, 0])
  ax_alpha   = fig.add_subplot(gs[1, 0])
  ax_num_k   = fig.add_subplot(gs[2, 0])
  ## plot Turb.dat
  filepath_data_turb = filepath_sim
  if os.path.exists(WWFnF.createFilepath([filepath_data_turb, FILENAME_TURB])):
    bool_plot_energy = True
    time_exp_start, time_exp_end = funcPlotTurb([ax_mach, ax_energy], filepath_data_turb)
  else:
    bool_plot_energy = False
    time_exp_start, time_exp_end = None, None
  ## plot fitted spectra
  filepath_data_spect = filepath_sim + "/spect/"
  if os.path.exists(WWFnF.createFilepath([filepath_data_spect, FILENAME_SPECTRA])):
    bool_plot_spectra = True
    Re, Rm, Pm = funcPlotSpectra(
      axs = [ax_num_k, ax_alpha, ax_kmodes],
      ax_spectra     = ax_spectra,
      filepath_data  = filepath_data_spect,
      time_exp_start = time_exp_start,
      time_exp_end   = time_exp_end
    )
    if (Re is not None) and (Rm is not None) and (Pm is not None):
      PlotFuncs.addLegend(
        ax = ax_mach,
        list_legend_labels = [
          r"$N_{\rm res} =$ " + sim_res,
          r"Re $=$ " + "{:.0f}".format(Re),
          r"Pm $=$ " + "{:.0f}".format(Pm),
          r"Rm $=$ " + "{:.0f}".format(Rm)
        ],
        list_marker_colors = [ "w" ],
        list_artists       = [ "." ],
        loc      = "lower left",
        bbox     = (0.0, 0.0),
        ncol     = 2,
        bpad     = 0,
        tpad     = -1,
        cspacing = 0,
        fontsize = 14
      )
  else: bool_plot_spectra = False
  ## check if the data was plotted
  if not(bool_plot_energy) and not(bool_plot_spectra):
    print("\t> ERROR: No data in:")
    print("\t\t", filepath_data_turb)
    print("\t\t", filepath_data_spect)
  else:
    ## check what data was missing
    if not(bool_plot_energy):
      print("\t> ERROR: No '{}' file in:".format(FILENAME_TURB))
      print("\t\t", filepath_data_turb)
    elif not(bool_plot_spectra):
      print("\t> ERROR: No '{}' file in:".format(FILENAME_SPECTRA))
      print("\t\t", filepath_data_spect)
    ## save the figure
    fig_filepath = WWFnF.createFilepath([filepath_plot, fig_name])
    plt.savefig(fig_filepath)
    plt.close()
    print("\t> Figure saved:", fig_name)


## ###############################################################
## DEFINE MAIN PROGRAM
## ###############################################################
BASEPATH         = "/scratch/ek9/nk7952/"
SONIC_REGIME     = "super_sonic"
FILENAME_TURB    = "Turb.dat"
FILENAME_SPECTRA = "spectra_fits.json"

def main():
  ## ##############################
  ## LOOK AT EACH SIMULATION FOLDER
  ## ##############################
  ## loop over the simulation suites
  for suite_folder in [
      "Rm3000"
    ]: # "Re10", "Re500", "Rm3000", "keta"

    ## loop over the different resolution runs
    for sim_res in [
        "144"
      ]: # "18", "36", "72", "144", "288", "576"

      ## ######################################
      ## CHECK THE SUITE'S FIGURE FOLDER EXISTS
      ## ######################################
      filepath_figures = WWFnF.createFilepath([
        BASEPATH, suite_folder, sim_res, SONIC_REGIME, "vis_folder"
      ])
      if not os.path.exists(filepath_figures):
        print("{} does not exist.".format(filepath_figures))
        continue
      str_message = "Looking at suite: {}, Nres = {}".format(suite_folder, sim_res)
      print(str_message)
      print("=" * len(str_message))
      print("Saving figures in:", filepath_figures)

      ## ####################
      ## PLOT SIMULATION DATA
      ## ####################
      ## loop over the simulation folders
      for sim_folder in [
          "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"
        ]: # "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"

        ## create filepath to the simulation folder
        filepath_sim = WWFnF.createFilepath([
          BASEPATH, suite_folder, sim_res, SONIC_REGIME, sim_folder
        ])
        ## check that the filepath exists
        if not os.path.exists(filepath_sim):
          continue
        ## plot simulation data
        fig_name = suite_folder + "_" + sim_folder + "_" + "check.png"
        funcPlotSimData(filepath_sim, filepath_figures, fig_name, sim_res)
    ## create an empty line after each suite
    print(" ")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
    main()
    sys.exit()


## END OF PROGRAM
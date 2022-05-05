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
from ThePlottingModule import TheMatplotlibStyler, PlotSpectra, PlotFuncs
from TheUsefulModule import WWLists, WWFnF, WWObjs
from TheLoadingModule import LoadFlashData
from TheFittingModule import UserModels


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.ioff()
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## FUNCTIONS
## ###############################################################
def funcPlotTurb(axs, filepath_data):
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
    data_sampled_y = interp_spline(data_fit_domain)
    ## fit exponential function to sampled data (in log-linear domain)
    fit_params_log, fit_params_cov = curve_fit(
      UserModels.ListOfModels.exp_loge,
      data_fit_domain,
      np.log(data_sampled_y)
    )
    ## undo log transformation
    fit_params_linear = [
      np.exp(fit_params_log[0] + 2),
      fit_params_log[1]
    ]
    ## initialise the plot domain
    data_domain = np.linspace(0, 100, 10**3)
    ## evaluate exponential
    data_E_exp = UserModels.ListOfModels.exp_linear(
      data_domain,
      *fit_params_linear
    )
    ## find where exponential enters / exists fit range
    index_E_start = WWLists.getIndexClosestValue(data_domain, time_start)
    index_E_end   = WWLists.getIndexClosestValue(data_domain, time_end)
    ## plot fit
    gamma_val = -fit_params_log[1]
    gamma_std = max(np.sqrt(np.diag(fit_params_cov))[1], 0.01)
    str_label = r"$\Gamma =$ " + "{:.2f}".format(gamma_val) + r"$\pm$" + "{:.2f}".format(gamma_std)
    ax.plot(
      data_domain[index_E_start : index_E_end],
      data_E_exp[index_E_start : index_E_end],
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
    interp_spline = make_interp_spline(
      data_x[index_start_fit : index_end_fit],
      data_y[index_start_fit : index_end_fit]
    )
    ## uniformly sample interpolated data
    data_sampled_y = interp_spline(data_fit_domain)
    ## measure average saturation level
    sub_domain = data_x[index_start_fit : index_end_fit]
    data_mean  = np.mean(data_sampled_y)
    data_std   = max(np.std(data_sampled_y), 0.01)
    ## plot fit
    str_label = label + "{:.2f}".format(data_mean) + r"$\pm$" + "{:.2f}".format(data_std)
    ax.plot(
      sub_domain,
      [data_mean] * len(sub_domain),
      label=str_label, color=color_fits, ls=":", lw=2, zorder=5
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
    color=color_data, marker=".", ms=1, ls="-", lw=1, zorder=3
  )
  axs[0].set_xlabel(r"$t / t_\mathrm{turb}$")
  axs[0].set_ylabel(r"$\mathcal{M}$")
  axs[0].set_xlim([0, max(data_time)])
  ## plot energy ratio
  axs[1].plot(
    data_time, data_E_ratio,
    color=color_data, marker=".", ms=1, ls="-", lw=1, zorder=3
  )
  axs[1].set_xlabel(r"$t / t_\mathrm{turb}$")
  axs[1].set_ylabel(r"$E_\mathrm{mag} / E_\mathrm{kin}$")
  axs[1].set_xlim([0, max(data_time)])
  axs[1].set_yscale("log")
  axs[1].set_ylim([ 10**(-10), 10**(1) ])
  ## get index range corresponding with kinematic phase of the dynamo
  index_exp_start = WWLists.getIndexClosestValue(data_E_ratio, 10**(-9))
  index_exp_end   = WWLists.getIndexClosestValue(data_E_ratio, 10**(-3))
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
  ## fit saturation
  fitConst(
    axs[1],
    data_x = data_time,
    data_y = data_E_ratio,
    label  = r"$\left(E_{\rm kin} / E_{\rm mag}\right)_{\rm sat} =$ ",
    index_start_fit = WWLists.getIndexClosestValue(data_time, 0.75 * data_time[-1]),
    index_end_fit   = len(data_time)-1
  )
  ## add legend
  axs[0].legend(frameon=False, loc="lower right", fontsize=14)
  axs[1].legend(frameon=False, loc="lower right", fontsize=14)
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
      label = None,
      color = "black",
    ):
    plot_args = { "color":color, "ls":"-", "lw":1, "marker":".", "ms":1 }
    if (time_range[0] is not None) and (time_range[1] is not None):
      index_start = WWLists.getIndexClosestValue(data_x, time_range[0])
      index_end   = WWLists.getIndexClosestValue(data_x, time_range[1])
      ## full dataset
      ax.plot(data_x, data_y, **plot_args, alpha=0.1, zorder=3)
      ## subset of data
      ax.plot(
        data_x[index_start : index_end],
        data_y[index_start : index_end],
        label = label,
        **plot_args, zorder=5
      )
    else: ax.plot(data_x, data_y, label=label, **plot_args, zorder=3)
  ## #################
  ## LOAD SPECTRA DATA
  ## #################
  # print(time_exp_start, time_exp_end)
  range_k = [ 0.05, 150 ]
  range_alpha = [ -4, 4 ]
  ## load spectra object
  spectra_obj = WWObjs.loadPickleObject(filepath_data, "spectra_obj.pkl", bool_hide_updates=True)
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
  ## ######################
  ## PLOT SPECTRA PARMETERS
  ## ######################
  ## plot number of points fitted to
  plotData(axs[0], kin_sim_times, kin_num_points_fitted, label=r"kin-spectra", color="blue")
  plotData(axs[0], mag_sim_times, mag_num_points_fitted, label=r"mag-spectra", color="red")
  axs[0].legend(frameon=False, loc="lower right", fontsize=14)
  axs[0].set_xlabel(r"$t / t_\mathrm{turb}$")
  axs[0].set_ylabel(r"max k-mode")
  axs[0].set_yscale("log")
  axs[0].set_xlim([0, max(kin_sim_times)])
  axs[0].set_ylim([1, range_k[1]])
  ## plot alpha exponents
  plotData(axs[1], kin_sim_times, kin_alpha, label=r"$\alpha_{\rm kin}$", color="blue")
  plotData(axs[1], mag_sim_times, mag_alpha, label=r"$\alpha_{\rm alpha}$", color="red")
  axs[1].legend(frameon=False, loc="lower right", fontsize=14)
  axs[1].set_xlabel(r"$t / t_\mathrm{turb}$")
  axs[1].set_ylabel(r"$\alpha$")
  axs[1].set_xlim([0, max(kin_sim_times)])
  axs[1].set_ylim(range_alpha)
  ## plot k modes
  plotData(axs[2], kin_sim_times, k_nu,  label=r"$k_\nu$",     color="blue")
  plotData(axs[2], mag_sim_times, k_eta, label=r"$k_\eta$",    color="red")
  plotData(axs[2], mag_sim_times, k_max, label=r"$k_{\rm p}$", color="green")
  axs[2].legend(frameon=False, loc="lower right", fontsize=14)
  axs[2].set_xlabel(r"$t / t_\mathrm{turb}$")
  axs[2].set_ylabel(r"$k$")
  axs[2].set_yscale("log")
  axs[2].set_xlim([0, max(kin_sim_times)])
  axs[2].set_ylim(range_k)
  ## #####################
  ## PLOT AVERAGED SPECTRA
  ## #####################
  if (time_range[0] is not None) and (time_range[1] is not None):
    PlotSpectra.PlotAveSpectra(ax_spectra, spectra_obj, time_range)
  return spectra_obj.Re, spectra_obj.Rm, spectra_obj.Pm


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
  if os.path.exists(WWFnF.createFilepath([filepath_data_turb, "Turb.dat"])):
    bool_plot_energy = True
    time_exp_start, time_exp_end = funcPlotTurb([ax_mach, ax_energy], filepath_data_turb)
  else:
    bool_plot_energy = False
    time_exp_start, time_exp_end = None, None
  ## plot fitted spectra
  filepath_data_spect = filepath_sim + "/spect/"
  if os.path.exists(WWFnF.createFilepath([filepath_data_spect, "spectra_obj.pkl"])):
    bool_plot_spectra = True
    Re, Rm, Pm = funcPlotSpectra(
      axs = [ax_num_k, ax_alpha, ax_kmodes],
      ax_spectra     = ax_spectra,
      filepath_data  = filepath_data_spect,
      time_exp_start = time_exp_start,
      time_exp_end   = time_exp_end
    )
    PlotFuncs.plotLabelBox(
      fig, ax_mach,
      ## box placement
      box_alignment = (0.0, 0.0),
      xpos = 0.025,
      ypos = 0.025,
      ## label appearance
      alpha    = 0.25,
      fontsize = 14,
      ## list of labels to place in box
      list_fig_labels = [
        r"Re $=$ " + "{:.0f}".format(Re),
        r"Rm $=$ " + "{:.0f}".format(Rm),
        r"Pm $=$ " + "{:.0f}".format(Pm),
        r"$N_{\rm res} =$ " + sim_res
      ]
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
      print("\t> ERROR: No 'Turb.dat' data in:", filepath_data_turb)
    elif not(bool_plot_spectra):
      print("\t> ERROR: No spectra data in:", filepath_data_spect)
    ## save the figure
    fig_filepath = WWFnF.createFilepath([filepath_plot, fig_name])
    plt.savefig(fig_filepath)
    plt.close()
    print("\t> Figure saved:", fig_name)


SONIC_REGIME = "super_sonic"
BASEPATH = "/scratch/ek9/nk7952/"
## ###############################################################
## DEFINE MAIN PROGRAM
## ###############################################################
def main():
  filepath_base = BASEPATH

  ## #######################
  ## LOOK AT EACH SIMULATION
  ## #######################
  for suite_folder in [
      "Rm3000"
    ]: # "Re10", "Re500", "Rm3000", "keta"

    for sim_res in [
        "144"
      ]: # "18", "36", "72", "144", "288", "576"
      ## ####################################
      ## CREATE FILEPATH TO SIMULATION FOLDER
      ## ####################################
      filepath_figures = WWFnF.createFilepath([
        filepath_base, suite_folder, sim_res, SONIC_REGIME, "vis_folder"
      ])
      ## check that the filepath exists on MAC
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
      for sim_folder in [
          "Pm1"
        ]: # "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"
        ## create filepath to the simulation folder
        filepath_sim = WWFnF.createFilepath([
          filepath_base, suite_folder, sim_res, SONIC_REGIME, sim_folder
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
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
## FUNCTION: FIT EXPONENTIAL FUNCTION TO DATA
## ###############################################################
def funcFitExp(
    ax,
    data_x, data_y,
    index_start_fit, index_end_fit,
    color = "black"
  ):
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
  data_y_sampled = abs(interp_spline(data_fit_domain))
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
    label=str_label, color=color, ls="--", lw=2, zorder=5
  )


## ###############################################################
## FUNCTION: FIT CONSTANT FUNCTION TO DATA
## ###############################################################
def funcFitConst(
    ax,
    data_x, data_y,
    index_start_fit, index_end_fit,
    label = "",
    color = "black"
  ):
  # ## TODO: remove. for debug purposes only
  # print(
  #   len(data_x),
  #   len(data_y),
  #   index_start_fit,
  #   index_end_fit,
  #   len(data_x[index_start_fit : index_end_fit]),
  #   len(data_y[index_start_fit : index_end_fit])
  # )
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
    label=str_label, color=color, ls=":", lw=2, zorder=5
  )
  ## return mean value
  return data_y_mean


## ###############################################################
## CLASS: PLOT INTEGRATED QUANTITIES
## ###############################################################
class PlotTurbData():
  def __init__(
      self,
      axs, filepath_data
    ):
    self.axs            = axs
    self.filepath_data  = filepath_data
    self.time_exp_start = None
    self.time_exp_end   = None
    self.color_fits     = "black"
    self.color_data     = "orange"
  def getExpTimeBounds(self):
    return self.time_exp_start, self.time_exp_end
  def loadData(self):
    ## load mach data
    _, self.data_Mach = LoadFlashData.loadTurbData(
      filepath_data = self.filepath_data,
      var_y         = 13, # 13 (new), 8 (old)
      t_turb        = T_TURB,
      time_start    = 0.1,
      time_end      = np.inf
    )
    ## load magnetic energy
    _, data_E_B = LoadFlashData.loadTurbData(
      filepath_data = self.filepath_data,
      var_y         = 11, # 11 (new), 29 (old)
      t_turb        = T_TURB,
      time_start    = 0.1,
      time_end      = np.inf
    )
    ## load kinetic energy
    self.data_time, data_E_K = LoadFlashData.loadTurbData(
      filepath_data = self.filepath_data,
      var_y         = 9, # 9 (new), 6 (old)
      t_turb        = T_TURB,
      time_start    = 0.1,
      time_end      = np.inf
    )
    ## calculate plot domain range
    self.max_time = max([
      100, max(self.data_time)
    ])
    ## calculate energy ratio: 'E_B / E_K'
    self.data_E_ratio = [
      (E_B / E_K) for E_B, E_K in zip(data_E_B, data_E_K)
    ]
  def plotData(self):
    ## plot mach
    self.axs[0].plot(
      self.data_time, self.data_Mach,
      color=self.color_data, ls="-", lw=1.5, zorder=3
    )
    self.axs[0].set_xlabel(r"$t / t_\mathrm{turb}$")
    self.axs[0].set_ylabel(r"$\mathcal{M}$")
    self.axs[0].set_xlim([ 0, self.max_time ])
    ## plot energy ratio
    self.axs[1].plot(
      self.data_time, self.data_E_ratio,
      color=self.color_data, ls="-", lw=1.5, zorder=3
    )
    ## define y-axis range for the energy ratio plot
    min_E_ratio         = min(self.data_E_ratio)
    log_min_E_ratio     = np.log10(min_E_ratio)
    new_log_min_E_ratio = np.floor(log_min_E_ratio)
    num_decades         = 1 + (-new_log_min_E_ratio)
    new_min_E_ratio     = 10**new_log_min_E_ratio
    num_y_major_ticks   = np.ceil(num_decades / 2)
    ## label axis
    self.axs[1].set_xlabel(r"$t / t_\mathrm{turb}$")
    self.axs[1].set_ylabel(r"$E_\mathrm{mag} / E_\mathrm{kin}$")
    self.axs[1].set_yscale("log")
    self.axs[1].set_xlim([ 0, self.max_time ])
    self.axs[1].set_ylim([ new_min_E_ratio, 10**(1) ])
    ## add log axis-ticks
    PlotFuncs.addLogAxisTicks(
      self.axs[1],
      bool_major_ticks    = True,
      max_num_major_ticks = num_y_major_ticks
    )
  def fitData(self):
    ## fit saturation
    sat_ratio = funcFitConst(
      self.axs[1],
      data_x          = self.data_time,
      data_y          = self.data_E_ratio,
      color           = self.color_fits,
      label           = r"$\left(E_{\rm kin} / E_{\rm mag}\right)_{\rm sat} =$ ",
      index_start_fit = WWLists.getIndexClosestValue(self.data_time, (0.75 * self.data_time[-1])),
      index_end_fit   = len(self.data_time)-1,
    )
    ## get index range corresponding with kinematic phase of the dynamo
    index_exp_start = WWLists.getIndexClosestValue(self.data_E_ratio, 10**(-7))
    index_exp_end   = WWLists.getIndexClosestValue(self.data_E_ratio, sat_ratio/100) # 1-percent of sat-ratio
    ## fit mach number
    funcFitConst(
      self.axs[0],
      data_x          = self.data_time,
      data_y          = self.data_Mach,
      color           = self.color_fits,
      label           = r"$\mathcal{M} =$ ",
      index_start_fit = index_exp_start,
      index_end_fit   = index_exp_end
    )
    ## fit exponential
    funcFitExp(
      self.axs[1],
      data_x          = self.data_time,
      data_y          = self.data_E_ratio,
      color           = self.color_fits,
      index_start_fit = index_exp_start,
      index_end_fit   = index_exp_end
    )
    ## add legend
    legend_ax0 = self.axs[0].legend(frameon=False, loc="lower right", fontsize=14)
    legend_ax1 = self.axs[1].legend(frameon=False, loc="lower right", fontsize=14)
    self.axs[0].add_artist(legend_ax0)
    self.axs[1].add_artist(legend_ax1)
    ## store time range bounds corresponding with the exponential phase of the dynamo
    self.time_exp_start = self.data_time[index_exp_start]
    self.time_exp_end   = self.data_time[index_exp_end]


## ###############################################################
## CLASS: PLOT SPECTRA DATA
## ###############################################################
class PlotSpectraFitParams():
  def __init__(
      self,
      filepath_data, sim_res,
      axs_params, ax_spectra, ax_label,
      time_exp_start = None,
      time_exp_end   = None
    ):
    self.filepath_data  = filepath_data
    self.sim_res        = sim_res
    self.axs_params     = axs_params
    self.ax_spectra     = ax_spectra
    self.ax_label       = ax_label
    self.time_exp_start = time_exp_start
    self.time_exp_end   = time_exp_end
  def labelSimParams(self):
    PlotFuncs.addLegend(
      ax = self.ax_label,
      list_legend_labels = [
        r"$N_{\rm res} =$ " + self.sim_res,
        r"Re $=$ " + "{:.0f}".format(self.spectra_fits_obj.Re),
        r"Pm $=$ " + "{:.0f}".format(self.spectra_fits_obj.Rm),
        r"Rm $=$ " + "{:.0f}".format(self.spectra_fits_obj.Pm)
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
  def updateFitRange(self):
    ## update fit time-range
    WWObjs.updateObjAttr(self.spectra_fits_obj, "kin_fit_start_t", self.time_exp_start)
    WWObjs.updateObjAttr(self.spectra_fits_obj, "mag_fit_start_t", self.time_exp_start)
    WWObjs.updateObjAttr(self.spectra_fits_obj, "kin_fit_end_t",   self.time_exp_end)
    WWObjs.updateObjAttr(self.spectra_fits_obj, "mag_fit_end_t",   self.time_exp_end)
    ## save the updated spectra object
    WWObjs.saveObj2Json(
      obj      = self.spectra_fits_obj,
      filepath = self.filepath_data,
      filename = FILENAME_SPECTRA
    )
  def loadParams(self):
    ## load spectra-fit data as a dictionary
    spectra_fits_dict = WWObjs.loadJson2Dict(
      filepath          = self.filepath_data,
      filename          = FILENAME_SPECTRA,
      bool_hide_updates = True
    )
    ## store dictionary data in spectra-fit object
    self.spectra_fits_obj = FitMHDScales.SpectraFit(**spectra_fits_dict)
    ## load fitted parameters
    self.list_kin_list_sim_times = self.spectra_fits_obj.kin_list_sim_times
    self.list_mag_list_sim_times = self.spectra_fits_obj.mag_list_sim_times
    self.kin_max_k_mode_fitted   = self.spectra_fits_obj.kin_max_k_mode_fitted_group_t
    self.mag_max_k_mode_fitted   = self.spectra_fits_obj.mag_max_k_mode_fitted_group_t
    ## check that there is sufficient time-realisations
    self.bool_kin_spectra_fitted = len(self.list_kin_list_sim_times) > 0
    self.bool_mag_spectra_fitted = len(self.list_mag_list_sim_times) > 0
    self.bool_plot_fit_params = self.bool_kin_spectra_fitted or self.bool_mag_spectra_fitted
    if self.bool_plot_fit_params:
      ## load kinetic energy spectra fit paramaters
      if self.bool_kin_spectra_fitted:
        k_modes             = self.spectra_fits_obj.kin_list_k_group_t[0]
        self.list_k_nu      = self.spectra_fits_obj.k_nu_group_t
        self.list_kin_alpha = [
          list_fit_params[1] for list_fit_params
          in self.spectra_fits_obj.kin_list_fit_params_group_t
        ]
      ## load magnetic energy spectra fit paramaters
      if self.bool_mag_spectra_fitted:
        k_modes             = self.spectra_fits_obj.mag_list_k_group_t[0]
        self.list_k_eta     = self.spectra_fits_obj.k_eta_group_t
        self.list_k_p       = self.spectra_fits_obj.k_p_group_t
        self.list_k_max     = self.spectra_fits_obj.k_max_group_t
        self.list_mag_alpha = [
          list_fit_params[1] for list_fit_params
          in self.spectra_fits_obj.mag_list_fit_params_group_t
        ]
      ## define the end of the k-domain
      self.max_k_mode = 1.2 * max(k_modes)
  def plotParams(self):
    ## plot normalised and time averaged energy spectra
    PlotSpectra.PlotAveSpectra(
      ax               = self.ax_spectra,
      spectra_fits_obj = self.spectra_fits_obj,
      time_range       = [ self.time_exp_start, self.time_exp_end ]
    )
    self.ax_spectra.set_ylim([ 10**(-8), 3*10**(0) ])
    if self.bool_plot_fit_params:
      ## #############################
      ## PLOT NUMBER OF K-MODES FITTED
      ## #############################
      if self.bool_kin_spectra_fitted:
        max_sim_time = max(self.list_kin_list_sim_times)
        ## plot number of k-modes the kinetic energy spectrum was fitted to
        self.__plotParam(
          ax     = self.axs_params[0],
          data_x = self.list_kin_list_sim_times,
          data_y = self.kin_max_k_mode_fitted,
          label  = r"kin-spectra",
          color  = "blue"
        )
      if self.bool_mag_spectra_fitted:
        max_sim_time = max(self.list_mag_list_sim_times)
        ## plot number of k-modes the magnetic energy spectrum was fitted to
        self.__plotParam(
          ax     = self.axs_params[0],
          data_x = self.list_mag_list_sim_times,
          data_y = self.mag_max_k_mode_fitted,
          label  = r"mag-spectra",
          color  = "red"
        )
      ## label and tune figure
      self.axs_params[0].legend(frameon=False, loc="lower right", fontsize=14)
      self.axs_params[0].set_xlabel(r"$t / t_\mathrm{turb}$")
      self.axs_params[0].set_ylabel(r"max k-mode")
      self.axs_params[0].set_yscale("log")
      self.axs_params[0].set_xlim([0, max_sim_time])
      self.axs_params[0].set_ylim([1, self.max_k_mode])
      ## add log axis-ticks
      PlotFuncs.addLogAxisTicks(
        ax                  = self.axs_params[0],
        bool_major_ticks    = True,
        bool_minor_ticks    = True,
        max_num_major_ticks = 5
      )
      ## #############################
      ## PLOT MEASURED ALPHA EXPONENTS
      ## #############################
      range_alpha = [
        min([ -4, 1.1*min(self.list_kin_alpha), 1.1*min(self.list_mag_alpha) ]),
        max([  4, 1.1*max(self.list_kin_alpha), 1.1*max(self.list_mag_alpha) ])
      ]
      if self.bool_kin_spectra_fitted:
        ## plot alpha_kin
        self.__plotParam(
          ax     = self.axs_params[1],
          data_x = self.list_kin_list_sim_times,
          data_y = self.list_kin_alpha,
          label  = r"$\alpha_{\rm kin} =$ ",
          color  = "blue"
        )
      if self.bool_mag_spectra_fitted:
        ## plot alpha_mag
        self.__plotParam(
          ax     = self.axs_params[1],
          data_x = self.list_mag_list_sim_times,
          data_y = self.list_mag_alpha,
          label  = r"$\alpha_{\rm mag} =$ ",
          color  = "red"
        )
      ## label and tune figure
      self.axs_params[1].legend(frameon=False, loc="right", fontsize=14)
      self.axs_params[1].set_xlabel(r"$t / t_\mathrm{turb}$")
      self.axs_params[1].set_ylabel(r"$\alpha$")
      self.axs_params[1].set_xlim([0, max_sim_time])
      self.axs_params[1].set_ylim(range_alpha)
      ## add log axis-ticks
      PlotFuncs.addLinearAxisTicks(
        ax               = self.axs_params[1],
        bool_minor_ticks    = True,
        bool_major_ticks    = True,
        max_num_minor_ticks = 10,
        max_num_major_ticks = 5
      )
      ## ####################
      ## PLOT MEASURED SCALES
      ## ####################
      ## fitted k modes
      if self.bool_kin_spectra_fitted:
        ## plot k_nu
        min_k_nu  = self.__plotParam(
          ax     = self.axs_params[2],
          data_x = self.list_kin_list_sim_times,
          data_y = self.list_k_nu,
          color  = "blue",
          label  = r"$k_\nu =$ "
        )
      if self.bool_mag_spectra_fitted:
        ## plot k_eta
        min_k_eta = self.__plotParam(
          ax     = self.axs_params[2],
          data_x = self.list_mag_list_sim_times,
          data_y = self.list_k_eta,
          color  = "red",
          label  = r"$k_\eta =$ "
        )
        ## plot k_p: fitted peak scale
        min_k_p   = self.__plotParam(
          ax     = self.axs_params[2],
          data_x = self.list_mag_list_sim_times,
          data_y = self.list_k_p,
          color  = "green",
          label  = r"$k_{\rm p} =$ ", zorder=7
        )
        ## plot k_max: raw peak scale
        min_k_max = self.__plotParam(
          ax     = self.axs_params[2],
          data_x = self.list_mag_list_sim_times,
          data_y = self.list_k_max,
          color  = "purple",
          label  = r"$k_{\rm max} =$ "
        )
      ## define k-range
      if self.bool_kin_spectra_fitted and self.bool_mag_spectra_fitted:
        range_k = [ 
          min([ 1, min([min_k_nu, min_k_eta, min_k_p, min_k_max]) ]),
          self.max_k_mode
        ]
      elif self.bool_mag_spectra_fitted:
        range_k = [ 
          min([ 1, min([min_k_eta, min_k_p, min_k_max]) ]),
          self.max_k_mode
        ]
      else:
        range_k = [ 
          min([ 1, min_k_nu ]),
          self.max_k_mode
        ]
      ## label and tune figure
      self.axs_params[2].legend(frameon=False, loc="upper right", fontsize=14)
      self.axs_params[2].set_xlabel(r"$t / t_\mathrm{turb}$")
      self.axs_params[2].set_ylabel(r"$k$")
      self.axs_params[2].set_yscale("log")
      self.axs_params[2].set_xlim([0, max_sim_time])
      self.axs_params[2].set_ylim(range_k)
      ## add log axis-ticks
      PlotFuncs.addLogAxisTicks(
        ax                  = self.axs_params[2],
        bool_major_ticks    = True,
        bool_minor_ticks    = True,
        max_num_major_ticks = 5
      )
  def __plotParam(
      self,
      ax, data_x, data_y,
      label  = None,
      color  = "black",
      zorder = 5
    ):
    ## if a fitting range is defined
    if (self.time_exp_start is not None) and (self.time_exp_end is not None):
      index_start = WWLists.getIndexClosestValue(data_x, self.time_exp_start)
      index_end   = WWLists.getIndexClosestValue(data_x, self.time_exp_end)
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
      ## return smallest value in the fit-range
      return min(data_y[index_start : index_end])
    else:
      ## plot full dataset
      ax.plot(data_x, data_y, label=label, color=color, ls="-", lw=1.5, zorder=(zorder-2))
      ## return smallest value
      return min(data_y)


## ###############################################################
## FUNCTION: HANDLING PLOT CALLS
## ###############################################################
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
  ## energy spectra fits
  ax_kmodes  = fig.add_subplot(gs[0, 0])
  ax_alpha   = fig.add_subplot(gs[1, 0])
  ax_num_k   = fig.add_subplot(gs[2, 0])
  ## #####################################
  ## PLOT INTEGRATED QUANTITIES (Turb.dat)
  ## #####################################
  filepath_data_turb = WWFnF.createFilepath([ filepath_sim, FILENAME_TURB ])
  bool_plot_energy   = os.path.exists(filepath_data_turb)
  time_exp_start     = None
  time_exp_end       = None
  if bool_plot_energy:
    plot_turb_obj = PlotTurbData([ ax_mach, ax_energy ], filepath_sim)
    plot_turb_obj.loadData()
    plot_turb_obj.plotData()
    plot_turb_obj.fitData()
    time_exp_start, time_exp_end = plot_turb_obj.getExpTimeBounds()
  ## ###################
  ## PLOT FITTED SPECTRA
  ## ###################
  filepath_data_spect   = WWFnF.createFilepath([ filepath_sim, "spect" ])
  filepath_spectra_fits = WWFnF.createFilepath([ filepath_data_spect, FILENAME_SPECTRA ])
  bool_plot_spectra     = os.path.exists(filepath_spectra_fits)
  if bool_plot_spectra:
    plot_spectra_obj = PlotSpectraFitParams(
      filepath_data  = filepath_data_spect,
      sim_res        = sim_res,
      axs_params     = [ ax_num_k, ax_alpha, ax_kmodes ],
      ax_spectra     = ax_spectra,
      ax_label       = ax_mach,
      time_exp_start = time_exp_start,
      time_exp_end   = time_exp_end
    )
    plot_spectra_obj.loadParams()
    plot_spectra_obj.plotParams()
    plot_spectra_obj.updateFitRange()
    plot_spectra_obj.labelSimParams()
  ## ###########
  ## SAVE FIGURE
  ## ###########
  ## check if the data was plotted
  if not(bool_plot_energy) and not(bool_plot_spectra):
    print("\t> ERROR: No data in:")
    print("\t\t", filepath_sim)
    print("\t\t", filepath_data_spect)
  else:
    ## check what data was missing
    if not(bool_plot_energy):
      print("\t> ERROR: No '{}' file in:".format(FILENAME_TURB))
      print("\t\t", filepath_sim)
    elif not(bool_plot_spectra):
      print("\t> ERROR: No '{}' file in:".format(FILENAME_SPECTRA))
      print("\t\t", filepath_data_spect)
    ## save the figure
    fig_filepath = WWFnF.createFilepath([filepath_plot, fig_name])
    plt.savefig(fig_filepath)
    plt.close()
    print("\t> Figure saved:", fig_name)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
BASEPATH              = "/scratch/ek9/nk7952/"
SONIC_REGIME          = "super_sonic"
T_TURB                = 0.1 # ell_turb / (Mach * c_s) = (1/2) / (5 * 1) = 1/10
FILENAME_TURB         = "Turb.dat"
FILENAME_SPECTRA      = "spectra_fits.json"
FILENAME_TAG          = ""
# FILENAME_SPECTRA      = "spectra_fits_fk_fm.json"
# FILENAME_TAG          = "_fk_fm"
BOOL_UPDATE_FIT_RANGE = 1

def main():
  ## ##############################
  ## LOOK AT EACH SIMULATION FOLDER
  ## ##############################
  ## loop over the simulation suites
  for suite_folder in [
      "Re10", "Re500", "Rm3000"
    ]: # "Re10", "Re500", "Rm3000", "keta"

    ## loop over the different resolution runs
    for sim_res in [
        "18"
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
        fig_name = suite_folder + "_" + sim_folder + "_" + "check" + FILENAME_TAG + ".pdf"
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
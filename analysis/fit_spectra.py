#!/usr/bin/env python3

## TODO: https://lmfit.github.io/lmfit-py/model.html
## ###############################################################
## MODULES
## ###############################################################
import os, sys, functools
import numpy as np

## 'tmpfile' needs to be loaded before 'matplotlib'.
## This is so matplotlib stores cache in a temporary directory.
## (Useful for plotting in parallel)
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit, fsolve
from scipy import interpolate
from lmfit import Model

## load user defined modules
from ThePlottingModule import TheMatplotlibStyler, PlotSpectra, PlotFuncs
from TheUsefulModule import WWLists, WWFnF
from TheLoadingModule import LoadFlashData
from TheFittingModule import FitMHDScales, UserModels
from TheAnalysisModule import WWSpectra

## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
## use a non-interactive plotting backend
plt.ioff()
plt.switch_backend("agg")


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def fitExpFunc(
    ax, data_x, data_y, index_start_fit, index_end_fit,
    linestyle  = "-"
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
    label=str_label, color="black", ls=linestyle, lw=2, zorder=5
  )

def fitConstFunc(
    ax, data_x, data_y, index_start_fit, index_end_fit,
    str_label = "",
    linestyle = "-"
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
  ## uniformly sample interpolated data
  data_y_sampled = interp_spline(data_fit_domain)
  ## measure average saturation level
  data_x_sub  = data_x[index_start_fit : index_end_fit]
  data_y_mean = np.mean(data_y_sampled)
  data_y_std  = max(np.std(data_y_sampled), 0.01)
  ## plot fit
  str_label += "{:.2f}".format(data_y_mean) + r" $\pm$ " + "{:.2f}".format(data_y_std)
  ax.plot(
    data_x_sub,
    [ data_y_mean ] * len(data_x_sub),
    label=str_label, color="black", ls=linestyle, lw=2, zorder=5
  )
  ## return mean value
  return data_y_mean

def fitKinSpectra(ax, list_k, list_power):
  label_kin = r"$A k^{\alpha} \exp\left\{-\frac{k}{k_\nu}\right\}$"
  my_model = Model(FitMHDScales.SpectraModels.kinetic_loge)
  my_model.set_param_hint("A",     min = 10**(-10.0),  value = 10**(-5.0),  max = 10**(5.0))
  my_model.set_param_hint("alpha", min = -10.0,        value = -2.0,        max = -1.0)
  my_model.set_param_hint("k_nu",  min = 0.1,          value = 5.0,         max = 50.0)
  input_params = my_model.make_params()
  fit_results = my_model.fit(np.log(list_power), input_params, k=list_k)
  fit_params = [
    fit_results.params["A"].value,
    fit_results.params["alpha"].value,
    fit_results.params["k_nu"].value
  ]
  list_power_fit = FitMHDScales.SpectraModels.kinetic_linear(list_k, *fit_params)
  ax.plot(list_k, list_power_fit, color="black", ls="-", lw=3, label=label_kin, zorder=5)
  return fit_params

def findSpectraTail(ax, x, y):
  y_grad     = 0.05 + np.diff(y)
  tail_index = np.argmin(abs(y_grad))
  ax.plot(x[1:], y_grad, color="k", marker="o", ms=5)
  ax.axhline(y=0,             ls="--", lw=2, color="black")
  ax.axvline(x=x[tail_index], ls="--", lw=2, color="orange")
  return x[tail_index]

def fitMagSpectra(ax, list_k, list_power):
  label_mag = r"$A k^{\alpha_1} {\rm K}_0\left\{ -\left(\frac{k}{k_\eta}\right)^{\alpha_2} \right\}$"
  list_k_interp   = np.logspace(np.log10(min(list_k)), np.log10(max(list_k)), 2*len(list_power))[1:-1]
  f = interpolate.interp1d(list_k, np.log(list_power))
  list_power_loge_interp = f(list_k_interp)
  my_model = Model(FitMHDScales.SpectraModels.magnetic_loge)
  my_model.set_param_hint("A",       min = 10**(-10.0), value = 10**(-1.0), max = 10**(5.0))
  my_model.set_param_hint("alpha_1", min = 0.5,         value = 1.5,        max = 6.0)
  my_model.set_param_hint("alpha_2", min = 0.1,         value = 1.0,        max = 1.0)
  my_model.set_param_hint("k_eta",   min = 0.1,         value = 1.0,        max = 10.0)
  input_params = my_model.make_params()
  fit_results  = my_model.fit(list_power_loge_interp, input_params, k=list_k_interp)
  fit_params   = [
    fit_results.params["A"].value,
    fit_results.params["alpha_1"].value,
    fit_results.params["alpha_2"].value,
    fit_results.params["k_eta"].value
  ]
  # ax.plot(list_k_interp, np.exp(list_power_loge_interp), color="orange", ls="-", lw=2, marker="o", ms=5, zorder=12)
  list_power_fit = FitMHDScales.SpectraModels.magnetic_linear(list_k_interp, *fit_params)
  ax.plot(list_k_interp, list_power_fit, color="black", ls="-", lw=3, label=label_mag, zorder=5)
  return fit_params


## ###############################################################
## PLOT INTEGRATED QUANTITIES
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
    self.__loadData()
    self.__plotData()
    self.__fitData()

  def getExpTimeBounds(self):
    return self.time_exp_start, self.time_exp_end

  def __loadData(self):
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
    self.max_time = max([ 100, max(self.data_time) ])
    ## calculate energy ratio: 'E_B / E_K'
    self.data_E_ratio = [
      (E_B / E_K) for E_B, E_K in zip(data_E_B, data_E_K)
    ]

  def __plotData(self):
    print("Plotting energy integrated quantities...")
    ## plot mach
    self.axs[0].plot(
      self.data_time, self.data_Mach,
      color="orange", ls="-", lw=1.5, zorder=3
    )
    self.axs[0].set_ylabel(r"$\mathcal{M}$")
    self.axs[0].set_xlim([ 0, self.max_time ])
    ## plot energy ratio
    self.axs[1].plot(
      self.data_time, self.data_E_ratio,
      color="orange", ls="-", lw=1.5, zorder=3
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

  def __fitData(self):
    str_label_Esat = r"$\left(E_{\rm kin} / E_{\rm mag}\right)_{\rm sat} =$ "
    str_label_Mach = r"$\mathcal{M} =$ "
    ls_kin = "--"
    ls_sat = ":"
    ## fit saturation
    growth_percent = self.data_E_ratio[-1] / self.data_E_ratio[
      WWLists.getIndexClosestValue(self.data_time, 5)
    ]
    ## if dynamo growth occurs
    if growth_percent > 100:
      ## find saturated energy ratio
      sat_ratio = fitConstFunc(
        self.axs[1],
        data_x          = self.data_time,
        data_y          = self.data_E_ratio,
        str_label       = str_label_Esat,
        index_start_fit = WWLists.getIndexClosestValue(
          self.data_time, (0.75 * self.data_time[-1])
        ),
        index_end_fit   = len(self.data_time)-1,
        linestyle       = ls_sat
      )
      ## get index range corresponding with kinematic phase of the dynamo
      index_exp_start = WWLists.getIndexClosestValue(self.data_E_ratio, 10**(-8))
      index_exp_end   = WWLists.getIndexClosestValue(self.data_E_ratio, sat_ratio/100)
      index_start_fit = min([ index_exp_start, index_exp_end ])
      index_end_fit   = max([ index_exp_start, index_exp_end ])
      ## find growth rate of exponential
      fitExpFunc(
        self.axs[1],
        data_x          = self.data_time,
        data_y          = self.data_E_ratio,
        index_start_fit = index_start_fit,
        index_end_fit   = index_end_fit,
        linestyle       = ls_kin
      )
    else: # if no growth occurs
      ## get index range corresponding with end of the simulation
      index_start_fit = WWLists.getIndexClosestValue(self.data_time, (0.75 * self.data_time[-1]))
      index_end_fit   = len(self.data_time)-1
      ## find average energy ratio
      sat_ratio = fitConstFunc(
        self.axs[1],
        data_x          = self.data_time,
        data_y          = self.data_E_ratio,
        str_label       = str_label_Esat,
        index_start_fit = index_start_fit,
        index_end_fit   = index_end_fit,
        linestyle       = ls_sat
      )
    ## find average mach number
    fitConstFunc(
      self.axs[0],
      data_x          = self.data_time,
      data_y          = self.data_Mach,
      str_label       = str_label_Mach,
      index_start_fit = index_start_fit,
      index_end_fit   = index_end_fit,
      linestyle       = ls_kin
    )
    ## add legend
    legend_ax0 = self.axs[0].legend(frameon=False, loc="lower left", fontsize=18)
    legend_ax1 = self.axs[1].legend(frameon=False, loc="lower right", fontsize=18)
    self.axs[0].add_artist(legend_ax0)
    self.axs[1].add_artist(legend_ax1)
    ## store time range bounds corresponding with the exponential phase of the dynamo
    self.time_exp_start = self.data_time[index_start_fit]
    self.time_exp_end   = self.data_time[index_end_fit]


## ###############################################################
## PLOT NORMALISED AND TIME-AVERAGED ENERGY SPECTRA
## ###############################################################
class PlotSpectra():
  def __init__(
      self,
      fig, ax_mag_grad, axs_spect, filepath_data, time_exp_start, time_exp_end
    ):
    self.fig            = fig
    self.ax_mag_grad    = ax_mag_grad
    self.axs_spect      = axs_spect
    self.filepath_data  = filepath_data
    self.time_exp_start = time_exp_start
    self.time_exp_end   = time_exp_end
    self.__loadData()
    self.__plotData()
    self.__fitData()
    self.__labelPlot()
  
  def __loadData(self):
    print("Loading energy spectra...")
    ## extract the number of plt-files per eddy-turnover-time from 'Turb.log'
    plots_per_eddy = LoadFlashData.getPlotsPerEddy(f"{self.filepath_data}/../", bool_hide_updates=True)
    if plots_per_eddy is None:
      Exception("ERROR: # plt-files could not be read from 'Turb.log'.")
    ## load kinetic energy spectra
    list_kin_k_group_t, list_kin_power_group_t, _ = LoadFlashData.loadListSpectra(
      filepath_data     = self.filepath_data,
      str_spectra_type  = "vel",
      file_start_time   = self.time_exp_start,
      file_end_time     = self.time_exp_end,
      plots_per_eddy    = plots_per_eddy,
      bool_hide_updates = True
    )
    ## load magnetic energy spectra
    list_mag_k_group_t, list_mag_power_group_t, _ = LoadFlashData.loadListSpectra(
      filepath_data     = self.filepath_data,
      str_spectra_type  = "mag",
      file_start_time   = self.time_exp_start,
      file_end_time     = self.time_exp_end,
      plots_per_eddy    = plots_per_eddy,
      bool_hide_updates = True
    )
    ## store normalised and time-averaged energy spectra
    self.list_kin_power_ave = WWSpectra.AveNormSpectraData(list_kin_power_group_t)
    self.list_mag_power_ave = WWSpectra.AveNormSpectraData(list_mag_power_group_t)
    self.list_kin_k         = list_kin_k_group_t[0]
    self.list_mag_k         = list_mag_k_group_t[0]
  
  def __plotData(self):
    plot_args = { "marker":"o", "ms":7, "ls":"", "alpha":1.0, "zorder":3 }
    self.axs_spect[0].plot(self.list_kin_k, self.list_kin_power_ave, color="green", **plot_args)
    self.axs_spect[1].plot(self.list_mag_k, self.list_mag_power_ave, color="red",   **plot_args)

  def __fitData(self):
    print("Fitting normalised and time-avergaed energy spectra...")
    ## fit kinetic energy spectrum
    fit_params_kin     = fitKinSpectra(self.axs_spect[0], self.list_kin_k[3:80], self.list_kin_power_ave[3:80])
    self.alpha_kin     = fit_params_kin[1]
    self.k_nu          = fit_params_kin[2]
    ## fit magnetic energy spectrum
    self.k_max         = np.argmax(self.list_mag_power_ave) + 1
    if np.min(np.log10(self.list_mag_power_ave[self.k_max-1:])) < -6.0:
      k_tail = findSpectraTail(self.ax_mag_grad, self.list_mag_k[self.k_max-1:], np.log10(self.list_mag_power_ave[self.k_max-1:]))
      tail_index = int(k_tail - 1)
    else: tail_index = len(self.list_mag_power_ave)
    fit_params_mag     = fitMagSpectra(self.axs_spect[1], self.list_mag_k[:tail_index], self.list_mag_power_ave[:tail_index])
    self.alpha_mag_1   = fit_params_mag[1]
    self.alpha_mag_2   = fit_params_mag[2]
    self.k_eta         = fit_params_mag[3]
    self.k_eta_alpha_2 = self.k_eta**(1 / self.alpha_mag_2)
    k_p_guess          = FitMHDScales.SpectraModels.k_p_simple(self.alpha_mag_1, self.alpha_mag_2, self.k_eta)
    ## fit peak scale from the modified Kulsrud and Anderson 1992 model
    try:
      self.k_p = fsolve(
        functools.partial(
          FitMHDScales.SpectraModels.k_p_implicit,
          alpha_1 = self.alpha_mag_1,
          alpha_2 = self.alpha_mag_2,
          k_eta   = self.k_eta
        ),
        x0 = k_p_guess # give a guess
      )[0]
    except (RuntimeError, ValueError): self.k_p = k_p_guess

  def __labelPlot(self):
    ## measuring magnetic energy gradient
    self.ax_mag_grad.set_xlabel(r"$k$")
    self.ax_mag_grad.set_ylabel(r"${\rm d}\widehat{\mathcal{P}}_{\rm mag} / {\rm d}k$")
    self.ax_mag_grad.set_xscale("log")
    ## annotate measured scales
    self.axs_spect[0].axvline(x=self.k_nu,  ls="--", lw=2, color="green", zorder=5, label=r"$k_\nu$")
    self.axs_spect[0].axvline(x=self.k_eta, ls="--", lw=2, color="red",   zorder=5, label=r"$k_\eta$")
    self.axs_spect[0].axvline(x=self.k_p,   ls="--", lw=2, color="black", zorder=5, label=r"$k_{\rm p}$")
    self.axs_spect[1].plot(
      self.k_max, max(self.list_mag_power_ave),
      label=r"$k_{\rm max}$", color="black", marker="o", ms=10, ls="", zorder=7
    )
    ## kinetic energy labels
    label_alpha_kin     = r"$\alpha = $ "+"{:.1f}".format(self.alpha_kin)
    label_k_nu          = r"$k_\nu = $ "+"{:.1e}".format(self.k_nu)
    ## magnetic energy labels
    label_alpha_mag_1   = r"$\alpha_1 = $ "+"{:.1f}".format(self.alpha_mag_1)
    label_alpha_mag_2   = r"$\alpha_2 = $ "+"{:.1f}".format(self.alpha_mag_2)
    label_k_eta         = r"$k_\eta = $ "+"{:.1e}".format(self.k_eta)
    label_k_eta_alpha_2 = r"$k_\eta^{1 / \alpha_2} = $ "+"{:.1e}".format(self.k_eta_alpha_2)
    label_k_p           = r"$k_{\rm p} = $ "+"{:.1f}".format(self.k_p)
    label_k_max         = r"$k_{\rm max} = $ "+"{:.1f}".format(self.k_max)
    ## label energy spectra axis
    self.axs_spect[0].legend(
      frameon        = True,
      loc            = (0.015, 0.25),
      bbox_transform = self.axs_spect[0].transAxes,
      facecolor="white", edgecolor="grey", framealpha=1.0, fontsize=18
    ).set_zorder(15)
    self.axs_spect[1].legend(
      frameon        = True,
      loc            = (0.015, 0.15),
      bbox_transform = self.axs_spect[1].transAxes,
      facecolor="white", edgecolor="grey", framealpha=1.0, fontsize=18
    ).set_zorder(15)
    PlotFuncs.plotLabelBox(
      self.fig, self.axs_spect[0],
      box_alignment   = (0.0, 0.0),
      xpos            = 0.022,
      ypos            = 0.035,
      alpha           = 1.0,
      fontsize        = 18,
      list_fig_labels = [
        rf"{label_alpha_kin}, {label_k_nu}",
        rf"{label_alpha_mag_1}, {label_alpha_mag_2}, {label_k_eta}",
        rf"{label_k_eta_alpha_2}, {label_k_p}, {label_k_max}"
      ]
    )
    ## kinetic energy axis
    self.axs_spect[0].set_xlim([ 0.9, max(self.list_mag_k) ])
    self.axs_spect[0].set_xlabel(r"$k$")
    self.axs_spect[0].set_ylabel(r"$\widehat{\mathcal{P}}_{\rm kin}(k)$", color="green")
    self.axs_spect[0].tick_params(axis="y", colors="green")
    self.axs_spect[0].set_xscale("log")
    self.axs_spect[0].set_yscale("log")
    ## magnetic energy axis
    self.axs_spect[1].set_xlim([ 0.9, max(self.list_mag_k) ])
    self.axs_spect[1].set_ylabel(r"$\widehat{\mathcal{P}}_{\rm mag}(k)$", color="red")
    self.axs_spect[1].tick_params(axis="y", colors="red")
    self.axs_spect[1].spines["right"].set_edgecolor("red")
    self.axs_spect[1].spines["left"].set_edgecolor("green")
    self.axs_spect[1].set_xscale("log")
    self.axs_spect[1].set_yscale("log")


## ###############################################################
## HANDLING PLOT CALLS
## ###############################################################
def plotSimData(
    filepath_sim, filepath_plot, fig_name, Nres,
    Re=None, Rm=None, Pm=None
  ):
  print("Initialising figure...")
  fact = 0.75
  fig = plt.figure(constrained_layout=True, figsize=(fact*8*3, fact*5*3))
  gs  = GridSpec(3, 3, figure=fig)
  ax_mach      = fig.add_subplot(gs[0, 0])
  ax_energy    = fig.add_subplot(gs[1, 0])
  ax_mag_grad  = fig.add_subplot(gs[2, 0])
  ax_spect_kin = fig.add_subplot(gs[:, 1:])
  ax_spect_mag = ax_spect_kin.twinx()
  ## label simuluation
  if (Re is not None) and (Pm is not None):
    ## Re and Pm have been defined
    nu  = round(float(MACH) / (K_TURB * float(Re)), 5)
    eta = round(nu / float(Pm), 5)
    Rm  = round(float(MACH) / (K_TURB * eta))
  elif (Rm is not None) and (Pm is not None):
    ## Rm and Pm have been defined
    eta = round(float(MACH) / (K_TURB * float(Rm)), 5)
    nu  = round(eta * float(Pm), 5)
    Re  = round(float(MACH) / (K_TURB * nu))
  PlotFuncs.plotLabelBox(
    fig, ax_mach,
    box_alignment   = (1.0, 0.0),
    xpos            = 0.95,
    ypos            = 0.05,
    alpha           = 0.5,
    fontsize        = 18,
    list_fig_labels = [
      r"${\rm N}_{\rm res} = $ " + f"{Nres}",
      r"${\rm Re} = $ " + f"{Re}",
      r"${\rm Rm} = $ " + f"{Rm}",
      r"${\rm Pm} = $ " + f"{Pm}",
    ]
  )
  ## #####################################
  ## PLOT INTEGRATED QUANTITIES (Turb.dat)
  ## #####################################
  filepath_data_turb = WWFnF.createFilepath([ filepath_sim, FILENAME_TURB ])
  bool_plot_energy   = os.path.exists(filepath_data_turb)
  time_exp_start     = None
  time_exp_end       = None
  if bool_plot_energy:
    plot_turb_obj = PlotTurbData(
      axs           = [ ax_mach, ax_energy ],
      filepath_data = filepath_sim
    )
    time_exp_start, time_exp_end = plot_turb_obj.getExpTimeBounds()
  ## ###################
  ## PLOT FITTED SPECTRA
  ## ###################
  filepath_data_spect = WWFnF.createFilepath([ filepath_sim, "spect" ])
  bool_plot_spectra   = os.path.exists(filepath_data_spect)
  if bool_plot_spectra:
    PlotSpectra(
      fig            = fig,
      ax_mag_grad    = ax_mag_grad,
      axs_spect      = [ ax_spect_kin, ax_spect_mag ],
      filepath_data  = filepath_data_spect,
      time_exp_start = time_exp_start,
      time_exp_end   = time_exp_end
    )
  ## ###########
  ## SAVE FIGURE
  ## ###########
  ## check if the data was plotted
  if not(bool_plot_energy) and not(bool_plot_spectra):
    print("ERROR: No data in:")
    print("\t", filepath_sim)
    print("\t", filepath_data_spect)
  else:
    ## check what data was missing
    if not(bool_plot_energy):
      print(f"ERROR: No '{FILENAME_TURB}' file in:")
      print("\t", filepath_sim)
    elif not(bool_plot_spectra):
      print(f"ERROR: No '{...}' file in:")
      print("\t", filepath_data_spect)
    ## save the figure
    print("Saving figure...")
    fig_filepath = WWFnF.createFilepath([ filepath_plot, fig_name ])
    plt.savefig(fig_filepath)
    plt.close()
    print("Figure saved:", fig_name)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
BASEPATH      = "/scratch/ek9/nk7952/"
SONIC_REGIME  = "super_sonic"
FILENAME_TURB = "Turb.dat"
K_TURB        = 2.0
MACH          = 5.0
T_TURB        = 1 / (K_TURB * MACH) # ell_turb / (Mach * c_s)

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
        "288"
      ]: # "18", "36", "72", "144", "288", "576"

      ## ######################################
      ## CHECK THE SUITE'S FIGURE FOLDER EXISTS
      ## ######################################
      filepath_plot = WWFnF.createFilepath([
        BASEPATH, suite_folder, sim_res, SONIC_REGIME, "vis_folder"
      ])
      if not os.path.exists(filepath_plot):
        print("{} does not exist.".format(filepath_plot))
        continue
      str_message = "Looking at suite: {}, Nres = {}".format(suite_folder, sim_res)
      print(str_message)
      print("=" * len(str_message))
      print("Saving figures in:", filepath_plot)
      print(" ")

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
        if not os.path.exists(filepath_sim): continue
        ## plot simulation data
        fig_name = f"{suite_folder}_{sim_folder}_check.png"
        plotSimData(
          filepath_sim, filepath_plot, fig_name, sim_res,
          Re = float(suite_folder.replace("Re", "")) if "Re" in suite_folder else None,
          Rm = float(suite_folder.replace("Rm", "")) if "Rm" in suite_folder else None,
          Pm = float(sim_folder.replace("Pm", ""))   if "Pm" in sim_folder   else None
        )

        return

        ## create empty space
        print(" ")
      print(" ")
    print(" ")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
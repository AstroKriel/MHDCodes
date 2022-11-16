#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np

## 'tmpfile' needs to be loaded before any 'matplotlib' libraries,
## so matplotlib stores its cache in a temporary directory.
## (necessary when plotting in parallel)
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.signal import find_peaks
from lmfit import Model

## load user defined routines
from plot_turb_data import PlotTurbData

## load user defined modules
from TheUsefulModule import WWLists, WWFnF, WWObjs
from TheSimModule import SimParams
from TheLoadingModule import LoadFlashData
from ThePlottingModule import PlotFuncs
from TheFittingModule import FitMHDScales


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def createLabel_percentiles(list_vals, num_digits=3):
  perc_16 = np.percentile(list_vals, 16)
  perc_50 = np.percentile(list_vals, 50)
  perc_84 = np.percentile(list_vals, 84)
  diff_lo = perc_50 - perc_16
  diff_hi = perc_84 - perc_50
  str_val = ("{0:."+str(num_digits)+"g}").format(perc_50)
  if "." in str_val: num_decimals = len(str_val.split(".")[1])
  else: num_decimals = 2
  str_lo = ("-{0:."+str(num_decimals)+"f}").format(diff_lo)
  str_hi = ("+{0:."+str(num_decimals)+"f}").format(diff_hi)
  return r"${}_{}^{}$\;".format(
    str_val,
    "{" + str_lo + "}",
    "{" + str_hi + "}"
  )

def plotPDF(ax, list_data, color):
  list_dens, list_bin_edges = np.histogram(list_data, bins=10, density=True)
  list_dens_norm = np.append(0, list_dens / list_dens.sum())
  ax.fill_between(list_bin_edges, list_dens_norm, step="pre", alpha=0.2, color=color)
  ax.plot(list_bin_edges, list_dens_norm, drawstyle="steps", color=color)

def interpLogLogData(x, y, x_interp, kind="cubic"):
  interpolator = interpolate.interp1d(np.log10(x), np.log10(y), kind=kind)
  return np.power(10.0, interpolator(np.log10(x_interp)))

def checkFit(ax, data_x, data_y, fit_params, func):
  data_y_fit = func(data_x, *fit_params)
  ax.plot(data_x, np.log10(data_y_fit) - np.log10(data_y), color="black", ls="-", marker="o")
  ax.axhline(y=0, color="red", ls="--")
  ax.set_xscale("log")
  ax.set_xlabel(r"$k$")
  ax.set_ylabel(r"Residuals")

def fitKinSpectra(
    ax_fit, list_k, list_power,
    ax_check_fit = None,
    bool_plot    = True
  ):
  ## define model label
  label_fit = r"$A k^{\alpha} \exp\left\{-\frac{k}{k_\nu}\right\}$"
  ## define model to fit
  func_loge   = FitMHDScales.SpectraModels.kinetic_loge
  func_linear = FitMHDScales.SpectraModels.kinetic_linear
  my_model = Model(func_loge)
  my_model.set_param_hint("A",     min = 10**(-3.0),  value = 10**(1.0), max = 10**(3.0))
  my_model.set_param_hint("alpha", min = -10.0,       value = -2.0,      max = -1.0)
  my_model.set_param_hint("k_nu",  min = 10**(-1.0),  value = 5.0,       max = 10**(2.0))
  ## find k-index to stop fitting kinetic energy spectrum
  end_index_kin = WWLists.getIndexClosestValue(list_power, 10**(-6))
  ## fit kinetic energy model (in log-linear domain) to subset of data
  fit_results  = my_model.fit(
    k      = list_k[1:end_index_kin],
    data   = np.log(list_power[1:end_index_kin]),
    params = my_model.make_params()
  )
  ## extract fitted parameters
  fit_params = [
    fit_results.params["A"].value,
    fit_results.params["alpha"].value,
    fit_results.params["k_nu"].value
  ]
  ## plot fitted spectrum
  if bool_plot:
    array_k_ext = np.logspace(-1, 3, 1000)
    array_k_fit = np.logspace(
      np.log10(min(list_k[1:end_index_kin])),
      np.log10(max(list_k[1:end_index_kin])),
      int(1e3)
    )
    array_power_ext = func_linear(array_k_ext, *fit_params)
    array_power_fit = func_linear(array_k_fit, *fit_params)
    ax_fit.plot(array_k_fit, array_power_fit, color="green", ls="-", lw=2, label=label_fit, zorder=7)
    PlotFuncs.plotData_noAutoAxisScale(
      ax = ax_fit,
      x  = array_k_ext,
      y  = array_power_ext,
      color="black", ls="-", lw=6, zorder=5
    )
  ## check residuals of fit
  if ax_check_fit is not None:
    checkFit(
      ax         = ax_check_fit,
      data_x     = list_k[1:end_index_kin],
      data_y     = list_power[1:end_index_kin],
      fit_params = fit_params,
      func       = func_linear
    )
  ## return fitted parameters
  return fit_params

def fitMagSpectra(ax, list_k, list_power):
  label_fit = r"$A k^{\alpha_1} {\rm K}_0\left\{ \left(\frac{k}{k_\eta}\right)^{\alpha_2} \right\}$"
  my_model  = Model(FitMHDScales.SpectraModels.magnetic_loge)
  my_model.set_param_hint("A",       min = 1e-3, value = 1e-1, max = 1e3)
  my_model.set_param_hint("alpha_1", min = 0.1,  value = 1.5,  max = 6.0)
  my_model.set_param_hint("alpha_2", min = 0.1,  value = 1.0,  max = 1.5)
  my_model.set_param_hint("k_eta",   min = 1e-3, value = 5.0,  max = 10.0)
  input_params = my_model.make_params()
  fit_results = my_model.fit(
    k      = list_k,
    data   = np.log(list_power),
    params = input_params
  )
  fit_params = [
    fit_results.params["A"].value,
    fit_results.params["alpha_1"].value,
    fit_results.params["alpha_2"].value,
    fit_results.params["k_eta"].value
  ]
  array_k_fit     = np.logspace(0, 3, 1000)
  array_power_fit = FitMHDScales.SpectraModels.magnetic_linear(array_k_fit, *fit_params)
  PlotFuncs.plotData_noAutoAxisScale(
    ax    = ax,
    x     = array_k_fit,
    y     = array_power_fit,
    label = label_fit,
    color="black", ls="-.",
    lw=3,
    zorder=5
  )
  return fit_params

def getMagSpectraPeak(ax, list_k, list_power, bool_plot=True):
  array_k_interp = np.logspace(
    np.log10(min(list_k)),
    np.log10(max(list_k)),
    3*len(list_power)
  )[1:-1]
  array_power_interp = interpLogLogData(list_k, list_power, array_k_interp, "cubic")
  k_p   = array_k_interp[np.argmax(array_power_interp)]
  k_max = np.argmax(list_power) + 1
  if bool_plot:
    ax.plot(
      array_k_interp,
      array_power_interp,
      color="orange", ls="-"
    )
  return k_p, k_max


## ###############################################################
## OPERATOR CLASS: PLOT NORMALISED + TIME-AVERAGED ENERGY SPECTRA
## ###############################################################
class PlotSpectra():
  def __init__(
      self,
      fig, axs_spectra, axs_scales, ax_spectra_ratio, ax_check_fit,
      filepath_data, time_exp_start, time_exp_end
    ):
    ## save input arguments
    self.fig                = fig
    self.axs_spectra        = axs_spectra
    self.axs_scales         = axs_scales
    self.ax_spectra_ratio   = ax_spectra_ratio
    self.ax_check_fit       = ax_check_fit
    self.filepath_data      = filepath_data
    self.time_exp_start     = time_exp_start
    self.time_exp_end       = time_exp_end
    ## initialise quantities to measure
    self.list_mag_k         = None
    self.list_kin_power_ave = None
    self.list_mag_power_ave = None
    self.plots_per_eddy     = None
    self.list_mag_times      = None
    self.list_time_k_eq     = None
    self.alpha_kin_group_t  = None
    self.k_nu_group_t       = None
    self.k_p_group_t        = None
    self.k_eq_group_t       = None
    ## flag to check that quantities have been measured
    self.bool_fitted        = False

  def performRoutines(self):
    self.__loadData()
    self.__plotSpectra()
    self.__plotSpectraRatio()
    print("Fitting energy spectra...")
    self.__fitKinSpectra()
    self.__fitMagSpectra()
    self.bool_fitted = True
    self.__labelSpectraPlot()
    self.__labelScalesPlots()
    self.__labelSpectraRatioPlot()

  def getFittedParams(self):
    if not self.bool_fitted: self.performRoutines()
    return {
      ## normalised and time-averaged energy spectra
      "list_k"             : self.list_mag_k,
      "list_kin_power_ave" : self.list_kin_power_ave,
      "list_mag_power_ave" : self.list_mag_power_ave,
      ## measured quantities
      "plots_per_eddy"     : self.plots_per_eddy,
      "list_time_growth"   : self.list_mag_times,
      "list_time_k_eq"     : self.list_time_k_eq,
      "alpha_kin_group_t"  : self.alpha_kin_group_t,
      "k_nu_group_t"       : self.k_nu_group_t,
      "k_p_group_t"        : self.k_p_group_t,
      "k_eq_group_t"       : self.k_eq_group_t,
      "fit_params_kin_ave" : self.fit_params_kin_ave
    }

  def saveFittedParams(self, filepath_sim):
    dict_params = self.getFittedParams()
    WWObjs.saveDict2JsonFile(f"{filepath_sim}/sim_outputs.json", dict_params)

  def __loadData(self):
    print("Loading energy spectra...")
    ## extract the number of plt-files per eddy-turnover-time from 'Turb.log'
    self.plots_per_eddy = LoadFlashData.getPlotsPerEddy_fromTurbLog(
      f"{self.filepath_data}/../",
      bool_hide_updates = True
    )
    ## load kinetic energy spectra
    dict_kin_spect_data = LoadFlashData.loadAllSpectraData(
      filepath          = self.filepath_data,
      spect_field       = "vel",
      file_start_time   = self.time_exp_start,
      file_end_time     = self.time_exp_end,
      plots_per_eddy    = self.plots_per_eddy,
      bool_hide_updates = True
    )
    ## load magnetic energy spectra
    dict_mag_spect_data = LoadFlashData.loadAllSpectraData(
      filepath          = self.filepath_data,
      spect_field       = "mag",
      file_start_time   = self.time_exp_start,
      file_end_time     = self.time_exp_end,
      plots_per_eddy    = self.plots_per_eddy,
      bool_hide_updates = True
    )
    ## store time-evolving energy spectra
    self.list_kin_power_group_t = dict_kin_spect_data["list_power_group_t"]
    self.list_mag_power_group_t = dict_mag_spect_data["list_power_group_t"]
    self.list_kin_k             = dict_kin_spect_data["list_k_group_t"][0]
    self.list_mag_k             = dict_mag_spect_data["list_k_group_t"][0]
    self.list_kin_times         = dict_kin_spect_data["list_sim_times"]
    self.list_mag_times         = dict_mag_spect_data["list_sim_times"]
    ## store normalised energy spectra
    self.list_kin_power_norm_group_t = [
      np.array(list_power) / sum(list_power)
      for list_power in self.list_kin_power_group_t
    ]
    self.list_mag_power_norm_group_t = [
      np.array(list_power) / sum(list_power)
      for list_power in self.list_mag_power_group_t
    ]
    ## store normalised, and time-averaged energy spectra
    self.list_kin_power_ave = np.mean(self.list_kin_power_norm_group_t, axis=0)
    self.list_mag_power_ave = np.mean(self.list_mag_power_norm_group_t, axis=0)

  def __plotSpectra(self):
    label_kin = r"$\langle \widehat{\mathcal{P}}_{\rm kin}(k) \rangle_{\forall t/t_{\rm turb}}$"
    label_mag = r"$\langle \widehat{\mathcal{P}}_{\rm mag}(k) \rangle_{\forall t/t_{\rm turb}}$"
    args_plot_ave  = { "marker":"o", "markeredgecolor":"black", "ms":8, "ls":"", "zorder":5 }
    args_plot_time = { "ls":"-", "lw":1, "alpha":0.5, "zorder":1 }
    ## plot average normalised energy spectra
    self.axs_spectra[0].plot(
      self.list_kin_k,
      self.list_kin_power_ave,
      label=label_kin, color="green", **args_plot_ave
    )
    self.axs_spectra[1].plot(
      self.list_mag_k,
      self.list_mag_power_ave,
      label=label_mag, color="red", **args_plot_ave
    )
    ## create colormaps for time-evolving energy spectra (color as a func of time)
    cmap_kin, norm_kin = PlotFuncs.createCmap(
      cmap_name = "Greens",
      cmin      = 0.35,
      vmin      = min(self.list_kin_times),
      vmax      = max(self.list_kin_times)
    )
    cmap_mag, norm_mag = PlotFuncs.createCmap(
      cmap_name = "Reds",
      cmin      = 0.35,
      vmin      = min(self.list_mag_times),
      vmax      = max(self.list_mag_times)
    )
    ## plot each time realisation of the normalised kinetic energy spectrum
    for time_index, time_val in enumerate(self.list_kin_times):
      self.axs_spectra[0].plot(
        self.list_kin_k,
        self.list_kin_power_norm_group_t[time_index],
        color = cmap_kin(norm_kin(time_val)), **args_plot_time
      )
    ## plot each time realisation of the normalised magnetic energy spectrum
    for time_index, time_val in enumerate(self.list_mag_times):
      self.axs_spectra[1].plot(
        self.list_mag_k,
        self.list_mag_power_norm_group_t[time_index],
        color = cmap_mag(norm_mag(time_val)), **args_plot_time
      )

  def __plotSpectraRatio(self):
    ## for each time realisation
    self.k_eq_group_t   = []
    self.list_time_k_eq = []
    ## plot each time realisation
    for time_index in range(len(self.list_mag_times)):
      ## calculate energy ratio spectrum
      E_ratio_group_k = [
        mag_power / kin_power
        for kin_power, mag_power in zip(
          self.list_kin_power_group_t[time_index],
          self.list_mag_power_group_t[time_index]
        )
      ]
      ## plot ratio of spectra
      self.ax_spectra_ratio.plot(
        self.list_mag_k,
        E_ratio_group_k,
        color="black", ls="-", lw=1, alpha=0.1, zorder=3
      )
      list_index_peaks, _ = find_peaks(E_ratio_group_k)
      if len(list_index_peaks) > 0:
        index_ratio_end = min(list_index_peaks)
      else: index_ratio_end = len(E_ratio_group_k) - 1
      if BOOL_DEBUG:
        self.ax_spectra_ratio.plot(
          self.list_kin_k[index_ratio_end],
          E_ratio_group_k[index_ratio_end],
          "ro"
        )
      ## measure k_eq
      tol = 1e-1
      list_index_k_eq = [
        k_index
        for k_index, E_ratio in enumerate(E_ratio_group_k[:index_ratio_end])
        if abs(E_ratio - 1) <= tol
      ]
      if len(list_index_k_eq) > 0:
        index_k_eq = list_index_k_eq[0]
        k_eq       = self.list_mag_k[index_k_eq]
        k_eq_power = E_ratio_group_k[index_k_eq]
        self.k_eq_group_t.append(k_eq)
        self.list_time_k_eq.append(self.list_mag_times[time_index])
        self.ax_spectra_ratio.plot(k_eq, k_eq_power, "ko")
    ## plot time-evolution of measured scales
    self.axs_scales[0].plot(
      self.list_time_k_eq,
      self.k_eq_group_t,
      color="red", ls="-", label=r"$k_{\rm eq}$"
    )

  def __fitKinSpectra(self):
    self.A_kin_group_t     = []
    self.alpha_kin_group_t = []
    self.k_nu_group_t      = []
    for time_index in range(len(self.list_kin_times)):
      ## fit kinetic energy spectrum at time-realisation
      fit_params_kin = fitKinSpectra(
        ax_fit     = self.axs_spectra[0],
        list_k     = self.list_kin_k,
        list_power = self.list_kin_power_norm_group_t[time_index],
        bool_plot  = False
      )
      ## store fitted parameters
      self.A_kin_group_t.append(fit_params_kin[0])
      self.alpha_kin_group_t.append(fit_params_kin[1])
      self.k_nu_group_t.append(fit_params_kin[2])
    ## plot fitted spectrum to time-averaged spectrum
    self.fit_params_kin_ave = fitKinSpectra(
      ax_fit       = self.axs_spectra[0],
      ax_check_fit = self.ax_check_fit,
      list_k       = self.list_kin_k,
      list_power   = self.list_kin_power_ave,
      bool_plot    = True
    )
    ## plot time-evolution of measured scales
    self.axs_scales[0].plot(
      self.list_kin_times,
      self.k_nu_group_t,
      color="green", ls="-", label=r"$k_\nu$"
    )
    plotPDF(self.axs_scales[1], self.k_nu_group_t, "g")

  def __fitMagSpectra(self):
    self.k_p_group_t   = []
    self.k_max_group_t = []
    for time_index in range(len(self.list_mag_times)):
      ## extract interpolated and raw magnetic peak-scale for time-realisation
      k_p, k_max = getMagSpectraPeak(
        self.axs_spectra[1],
        self.list_mag_k,
        self.list_mag_power_norm_group_t[time_index],
        bool_plot = False
      )
      ## store measured scales
      self.k_p_group_t.append(k_p)
      self.k_max_group_t.append(k_max)
    ## plot time-evolution of measured scales
    self.axs_scales[0].plot(
      self.list_mag_times,
      self.k_p_group_t,
      color="black", ls="-", label=r"$k_{\rm p}$"
    )
    plotPDF(self.axs_scales[1], self.k_p_group_t, "k")

  def __labelSpectraPlot(self):
    ## annotate measured scales
    args_plot = { "ls":"--", "lw":2, "zorder":7 }
    self.axs_spectra[0].axvline(x=np.mean(self.k_nu_group_t), **args_plot, color="green", label=r"$k_\nu$")
    self.axs_spectra[0].axvline(x=np.mean(self.k_p_group_t),  **args_plot, color="black", label=r"$k_{\rm p}$")
    self.axs_spectra[1].plot(
      np.mean(self.k_max_group_t),
      np.mean(np.max(self.list_mag_power_norm_group_t, axis=1)),
      color="black", marker="o", ms=10, ls="", label=r"$k_{\rm max}$", zorder=7
    )
    ## create labels for measured scales
    label_A_kin     = r"$A_{\rm kin} = $ " + createLabel_percentiles(self.A_kin_group_t)
    label_alpha_kin = r"$\alpha = $ "      + createLabel_percentiles(self.alpha_kin_group_t)
    label_k_nu      = r"$k_\nu = $ "       + createLabel_percentiles(self.k_nu_group_t)
    label_k_p       = r"$k_{\rm p} = $ "   + createLabel_percentiles(self.k_p_group_t)
    label_k_max     = r"$k_{\rm max} = $ " + createLabel_percentiles(self.k_max_group_t)
    ## add legend: markers/linestyles of plotted quantities
    list_lines_ax0, list_labels_ax0 = self.axs_spectra[0].get_legend_handles_labels()
    list_lines_ax1, list_labels_ax1 = self.axs_spectra[1].get_legend_handles_labels()
    list_lines  = list_lines_ax0  + list_lines_ax1
    list_labels = list_labels_ax0 + list_labels_ax1
    self.axs_spectra[1].legend(
      list_lines,
      list_labels,
      loc="upper right", bbox_to_anchor=(0.99, 0.99),
      frameon=True, facecolor="white", edgecolor="grey", framealpha=1.0, fontsize=18
    ).set_zorder(10)
    ## add legend: measured parameter values
    PlotFuncs.addBoxOfLabels(
      fig           = self.fig,
      ax            = self.axs_spectra[0],
      box_alignment = (0.0, 0.0),
      xpos          = 0.025,
      ypos          = 0.025,
      alpha         = 1.0,
      fontsize      = 18,
      list_labels   = [
        rf"{label_A_kin}, {label_alpha_kin}, {label_k_nu}",
        rf"{label_k_p}, {label_k_max}"
      ]
    )
    ## adjust kinetic energy axis
    self.axs_spectra[0].set_xlim([ 0.9, 1.1*max(self.list_mag_k) ])
    self.axs_spectra[0].set_xlabel(r"$k$")
    self.axs_spectra[0].set_ylabel(r"$\widehat{\mathcal{P}}_{\rm kin}(k)$", color="green")
    self.axs_spectra[0].set_xscale("log")
    self.axs_spectra[0].set_yscale("log")
    ## adjust magnetic energy axis
    self.axs_spectra[1].set_xlim([ 0.9, 1.1*max(self.list_mag_k) ])
    self.axs_spectra[1].set_ylabel(r"$\widehat{\mathcal{P}}_{\rm mag}(k)$", color="red", rotation=-90, labelpad=40)
    self.axs_spectra[1].set_xscale("log")
    self.axs_spectra[1].set_yscale("log")
    ## colour left/right axis-splines
    self.axs_spectra[0].tick_params(axis="y", colors="green")
    self.axs_spectra[1].tick_params(axis="y", colors="red")
    self.axs_spectra[1].spines["left"].set_color("green")
    self.axs_spectra[1].spines["right"].set_color("red")

  def __labelScalesPlots(self):
    ## time evolution of scales
    self.axs_scales[0].legend(
      loc="upper left", bbox_to_anchor=(0.01, 0.99),
      frameon=True, facecolor="white", edgecolor="grey", framealpha=1.0, fontsize=18
    )
    self.axs_scales[0].set_xlabel(r"$t/t_{\rm turb}$")
    self.axs_scales[0].set_ylabel(r"$k$")
    self.axs_scales[0].set_yscale("log")
    ## PDF of scales 
    self.axs_scales[1].set_xlabel(r"$k$")
    self.axs_scales[1].set_ylabel(r"PDF")

  def __labelSpectraRatioPlot(self):
    self.ax_spectra_ratio.axhline(y=1, color="red", ls="--")
    self.ax_spectra_ratio.set_xlim([ 0.9, max(self.list_mag_k) ])
    self.ax_spectra_ratio.set_xlabel(r"$k$")
    self.ax_spectra_ratio.set_ylabel(r"$\mathcal{P}_{\rm mag}(k, t) / \mathcal{P}_{\rm kin}(k, t)$")
    self.ax_spectra_ratio.set_xscale("log")
    self.ax_spectra_ratio.set_yscale("log")


## ###############################################################
## HANDLING PLOT CALLS
## ###############################################################
def plotSimData(filepath_sim, filepath_vis, sim_name):
  ## GET SIMULATION PARAMETERS
  ## -------------------------
  dict_sim_inputs = SimParams.readSimInputs(filepath_sim)
  ## INITIALISE FIGURE
  ## -----------------
  print("Initialising figure...")
  fig, fig_grid = PlotFuncs.createFigure_grid(
    fig_scale        = 1.0,
    fig_aspect_ratio = (5.0, 8.0),
    num_rows         = 3,
    num_cols         = 4
  )
  ax_Mach          = fig.add_subplot(fig_grid[0,  0])
  ax_E_ratio       = fig.add_subplot(fig_grid[1,  0])
  ax_check_fit     = fig.add_subplot(fig_grid[2,  0])
  ax_spect_kin     = fig.add_subplot(fig_grid[:2, 1])
  ax_spect_mag     = ax_spect_kin.twinx()
  ax_spect_kin.set_zorder(1) # default zorder is 0 for axs[0] and axs[1]
  ax_spect_kin.set_frame_on(False) # prevents axs[0] from hiding axs[1]
  ax_spectra_ratio = fig.add_subplot(fig_grid[:2, 2])
  ax_scales_time   = fig.add_subplot(fig_grid[2,  1])
  ax_scales_pdf    = fig.add_subplot(fig_grid[2,  2])
  ## PLOT INTEGRATED QUANTITIES
  ## --------------------------
  obj_plot_turb = PlotTurbData(
    fig             = fig,
    axs             = [ ax_Mach, ax_E_ratio ],
    filepath_data   = filepath_sim,
    dict_sim_inputs = dict_sim_inputs
  )
  obj_plot_turb.performRoutines()
  obj_plot_turb.saveFittedParams(filepath_sim)
  dict_turb_params = obj_plot_turb.getFittedParams()
  ## PLOT FITTED SPECTRA
  ## -------------------
  obj_plot_spectra = PlotSpectra(
    fig              = fig,
    axs_spectra      = [ ax_spect_kin, ax_spect_mag ],
    axs_scales       = [ ax_scales_time, ax_scales_pdf ],
    ax_check_fit     = ax_check_fit,
    ax_spectra_ratio = ax_spectra_ratio,
    filepath_data    = f"{filepath_sim}/spect/",
    time_exp_start   = dict_turb_params["time_growth_start"],
    time_exp_end     = dict_turb_params["time_growth_end"]
  )
  obj_plot_spectra.performRoutines()
  obj_plot_spectra.saveFittedParams(filepath_sim)
  ## SAVE FIGURE
  ## -----------
  print("Saving figure...")
  fig_name     = f"{sim_name}_dataset.png"
  fig_filepath = f"{filepath_vis}/{fig_name}"
  plt.savefig(fig_filepath)
  plt.close()
  print("Saved figure:", fig_filepath)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## LOOK AT EACH SIMULATION FOLDER
  ## ------------------------------
  ## loop over the simulation suites
  for suite_folder in LIST_SUITE_FOLDER:

    ## loop over the simulation folders
    for sim_folder in LIST_SIM_FOLDER:

      ## CHECK THE SIMULATION EXISTS
      ## ---------------------------
      filepath_sim = WWFnF.createFilepath([
        BASEPATH, suite_folder, SONIC_REGIME, sim_folder
      ])
      if not os.path.exists(filepath_sim): continue
      str_message = f"Looking at suite: {suite_folder}, sim: {sim_folder}, regime: {SONIC_REGIME}"
      print(str_message)
      print("=" * len(str_message))
      print(" ")

      ## loop over the different resolution runs
      for sim_res in LIST_SIM_RES:

        ## CHECK THE RESOLUTION RUN EXISTS
        ## -------------------------------
        filepath_sim_res = f"{filepath_sim}/{sim_res}/"
        ## check that the filepath exists
        if not os.path.exists(filepath_sim_res): continue

        ## MAKE SURE A VISUALISATION FOLDER EXISTS
        ## ---------------------------------------
        filepath_sim_res_plot = f"{filepath_sim_res}/vis_folder"
        WWFnF.createFolder(filepath_sim_res_plot, bool_hide_updates=True)

        ## PLOT SIMULATION DATA AND SAVE MEASURED QUANTITIES
        ## -------------------------------------------------
        sim_name = f"{suite_folder}_{sim_folder}"
        plotSimData(filepath_sim_res, filepath_sim_res_plot, sim_name)

        if BOOL_DEBUG: return
        ## create empty space
        print(" ")
      print(" ")
    print(" ")


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
BOOL_DEBUG        = 0
BASEPATH          = "/scratch/ek9/nk7952/"
SONIC_REGIME      = "super_sonic"

LIST_SUITE_FOLDER = [ "Re10", "Re500", "Rm3000" ]
LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
LIST_SIM_RES      = [ "18", "36", "72", "144", "288", "576" ]

# LIST_SUITE_FOLDER = [ "Re10" ]
# LIST_SIM_FOLDER   = [ "Pm250" ]
# LIST_SIM_RES      = [ "576" ]


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
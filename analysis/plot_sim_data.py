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
from ThePlottingModule import PlotFuncs, PlotLatex
from TheFittingModule import FitMHDScales


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def getLabels_kin(fit_params_group_t):
  label_A_kin     = PlotLatex.createLabel_percentiles(WWLists.getElemFromLoL(fit_params_group_t, 0))
  label_alpha_kin = PlotLatex.createLabel_percentiles(WWLists.getElemFromLoL(fit_params_group_t, 1))
  label_k_nu      = PlotLatex.createLabel_percentiles(WWLists.getElemFromLoL(fit_params_group_t, 2))
  return r"$A_{\rm kin} = $ " + label_A_kin + r", $\alpha = $ " + label_alpha_kin + r", $k_\nu = $ " + label_k_nu

def getLabels_mag(k_p_group_t, k_max_group_t):
  label_k_p   = PlotLatex.createLabel_percentiles(k_p_group_t)
  label_k_max = PlotLatex.createLabel_percentiles(k_max_group_t)
  return r"$k_{\rm p} = $ " + label_k_p + r", $k_{\rm max} = $ " + label_k_max

def plotFitResiduals(ax, data_x, data_y, fit_params, func, color="black", label_spect=""):
  data_y_fit = func(data_x, *fit_params)
  ax.plot(
    data_x,
    np.array(data_y_fit) / np.array(data_y),
    label=label_spect, color=color, ls="-", marker="o", ms=5
  )

def fitKinSpectrum(
    ax_fit, list_k, list_power,
    ax_residuals  = None,
    color         = "black",
    label_spect   = "",
    bool_plot_fit = True
  ):
  ## define model to fit
  func_loge   = FitMHDScales.SpectraModels.kinetic_loge
  func_linear = FitMHDScales.SpectraModels.kinetic_linear
  my_model = Model(func_loge)
  my_model.set_param_hint("A",     min = 10**(-3.0),  value = 10**(1.0), max = 10**(3.0))
  my_model.set_param_hint("alpha", min = -10.0,       value = -2.0,      max = -1.0)
  my_model.set_param_hint("k_nu",  min = 10**(-1.0),  value = 5.0,       max = 10**(2.0))
  ## find k-index to stop fitting kinetic energy spectrum
  fit_index_start = 1
  fit_index_end   = WWLists.getIndexClosestValue(list_power, 10**(-6))
  ## fit kinetic energy model (in log-linear domain) to subset of data
  fit_results  = my_model.fit(
    k      = list_k[           fit_index_start : fit_index_end],
    data   = np.log(list_power[fit_index_start : fit_index_end]),
    params = my_model.make_params()
  )
  ## extract fitted parameters
  fit_params = [
    fit_results.params["A"].value,
    fit_results.params["alpha"].value,
    fit_results.params["k_nu"].value
  ]
  ## plot residuals of fit
  if ax_residuals is not None:
    plotFitResiduals(
      ax          = ax_residuals,
      data_x      = list_k[    fit_index_start : fit_index_end],
      data_y      = list_power[fit_index_start : fit_index_end],
      fit_params  = fit_params,
      func        = func_linear,
      color       = color,
      label_spect = label_spect
    )
  ## plot fitted spectrum
  if bool_plot_fit:
    array_k_fit     = np.logspace(-1, 3, 1000)
    array_power_fit = func_linear(array_k_fit, *fit_params)
    PlotFuncs.plotData_noAutoAxisScale(
      ax = ax_fit,
      x  = array_k_fit,
      y  = array_power_fit,
      color=color, ls="-", lw=6, alpha=0.5, zorder=1
    )
  ## return fitted parameters
  return fit_params

def interpLogLogData(x, y, x_interp, interp_kind="cubic"):
  interpolator = interpolate.interp1d(np.log10(x), np.log10(y), kind=interp_kind)
  return np.power(10.0, interpolator(np.log10(x_interp)))

def getMagSpectrumPeak(list_k, list_power):
  array_k_interp = np.logspace(
    start = np.log10(min(list_k)),
    stop  = np.log10(max(list_k)),
    num   = 3*len(list_power)
  )[1:-1]
  array_power_interp = interpLogLogData(
    x           = list_k,
    y           = list_power,
    x_interp    = array_k_interp,
    interp_kind = "cubic"
  )
  k_p   = array_k_interp[np.argmax(array_power_interp)]
  k_max = np.argmax(list_power) + 1
  return k_p, k_max

def plotSpectra_helper(ax, list_k, list_power_group_t, color, cmap_name, list_times):
  args_plot_ave  = { "color":color, "marker":"o", "ms":5, "zorder":5 } # "markeredgecolor":"black"
  args_plot_time = { "ls":"-", "lw":1, "alpha":0.5, "zorder":3 }
  ## plot time averaged, normalised energy spectra
  ax.plot(
    list_k,
    getSpectraAve(list_power_group_t),
    ls="", **args_plot_ave
  )
  ## create colormaps for time-evolving energy spectra (color as a func of time)
  cmap_kin, norm_kin = PlotFuncs.createCmap(
    cmap_name = cmap_name,
    vmin      = min(list_times),
    vmax      = max(list_times)
  )
  ## plot each time realisation of the normalised kinetic energy spectrum
  for time_index, time_val in enumerate(list_times):
    ax.plot(
      list_k,
      np.array(list_power_group_t[time_index]) / sum(list_power_group_t[time_index]),
      color=cmap_kin(norm_kin(time_val)), **args_plot_time
    )

def plotMeasuredScale(
    ax_time,
    list_t, scale_group_t, scale_ave,
    ax_spectrum = None,
    color       = "black",
    label       = ""
  ):
  ## plot average scale to spectrum
  if ax_spectrum is not None:
    ax_spectrum.axvline(
      x=scale_ave,
      color=color, ls="--", lw=1.5, zorder=7
    )
  ## plot time evolution of scale
  ax_time.plot(
    list_t,
    [ scale_ave ] * len(list_t),
    color=color, ls="--", lw=1.5
  )
  ax_time.plot(
    list_t,
    scale_group_t,
    color=color, ls="-", lw=1.5, label=label
  )

def fitKinSpectra_helper(
    ax_fit, ax_residuals, ax_scales,
    list_k, list_power_group_t, list_time_growth,
    color       = "black",
    label_spect = "",
    label_knu   = r"$k_\nu$"
  ):
  fit_params_group_t = []
  ## fit each time-realisation of the kinetic energy spectrum
  for time_index in range(len(list_time_growth)):
    fit_params_kin = fitKinSpectrum(
      ax_fit        = ax_fit,
      list_k        = list_k,
      list_power    = getSpectraNorm(list_power_group_t[time_index]),
      bool_plot_fit = False
    )
    ## store fitted parameters
    fit_params_group_t.append(fit_params_kin)
  ## fit time-averaged kinetic energy spectrum
  fit_params_ave = fitKinSpectrum(
    ax_fit        = ax_fit,
    ax_residuals  = ax_residuals,
    list_k        = list_k,
    list_power    = getSpectraAve(list_power_group_t),
    color         = color,
    label_spect   = label_spect,
    bool_plot_fit = True
  )
  ## plot time-evolution of measured scale
  plotMeasuredScale(
    ax_spectrum   = ax_fit,
    ax_time       = ax_scales,
    list_t        = list_time_growth,
    scale_group_t = [
      fit_params[2]
      for fit_params in fit_params_group_t
    ],
    scale_ave     = fit_params_ave[2],
    color         = color,
    label         = label_knu
  )
  return fit_params_group_t, fit_params_ave

## ###############################################################
## OPERATOR CLASS: PLOT NORMALISED + TIME-AVERAGED ENERGY SPECTRA
## ###############################################################
class PlotSpectra():
  def __init__(
      self,
      fig, dict_axs,
      filepath_spect, time_exp_start, time_exp_end
    ):
    ## save input arguments
    self.fig              = fig
    self.axs_spectra_tot  = dict_axs["axs_spectra_tot"]
    self.ax_spectra_comp  = dict_axs["ax_spectra_comp"]
    self.ax_scales_time   = dict_axs["ax_scales"]
    self.ax_residuals     = dict_axs["ax_residuals"]
    self.ax_spectra_ratio = dict_axs["ax_spectra_ratio"]
    self.filepath_spect   = filepath_spect
    self.time_exp_start   = time_exp_start
    self.time_exp_end     = time_exp_end
    ## initialise spectra labels
    self.__initialiseQuantities()
    self.dict_plot_mag_tot = {
      "color"       : "red",
      "cmap_name"   : "Reds",
      "label_spect" : getLabel_spectrum("mag", "tot"),
      "label_kp"    : r"$k_{\rm p}(t)$",
    }
    self.dict_plot_kin_tot = {
      "color"       : "darkgreen",
      "cmap_name"   : "Greens",
      "label_spect" : getLabel_spectrum("kin", "tot"),
      "label_knu"   : r"$k_{\nu, {\rm tot}}(t)$",
    }
    self.dict_plot_kin_lgt = {
      "color"       : "blue",
      "cmap_name"   : "Blues",
      "label_spect" : getLabel_spectrum("kin", "lgt"),
      "label_knu"   : r"$k_{\nu, \parallel}(t)$",
    }
    self.dict_plot_kin_trv = {
      "color"       : "darkviolet",
      "cmap_name"   : "Purples",
      "label_spect" : getLabel_spectrum("kin", "trv"),
      "label_knu"   : r"$k_{\nu, \perp}(t)$",
    }

  def performRoutines(self):
    self.__loadData()
    self.__plotSpectra()
    self.__plotSpectraRatio()
    print("Fitting energy spectra...")
    self.__fitKinSpectra()
    self.__fitMagSpectra()
    self.bool_fitted = True
    self.__labelSpectraPlot()
    self.__labelSpectraRatioPlot()
    self.__labelScalesPlot()

  def getFittedParams(self):
    list_quantities_undefined = self.__checkAnyQuantitiesNotMeasured()
    if not self.bool_fitted: self.performRoutines()
    if len(list_quantities_undefined) > 0: raise Exception("Error: failed to define quantity:", list_quantities_undefined)
    return {
      ## time-averaged energy spectra
      "list_k"                     : self.list_k,
      "list_mag_power_tot_ave"     : getSpectraAve(self.list_mag_power_tot_group_t),
      "list_kin_power_tot_ave"     : getSpectraAve(self.list_kin_power_tot_group_t),
      "list_kin_power_lgt_ave"     : getSpectraAve(self.list_kin_power_lgt_group_t),
      "list_kin_power_trv_ave"     : getSpectraAve(self.list_kin_power_trv_group_t),
      ## measured quantities
      "plots_per_eddy"             : self.plots_per_eddy,
      "list_time_growth"           : self.list_time_growth,
      "list_time_k_eq"             : self.list_time_k_eq,
      "k_p_group_t"                : self.k_p_group_t,
      "k_eq_group_t"               : self.k_eq_group_t,
      "fit_params_kin_tot_group_t" : self.fit_params_kin_tot_group_t,
      "fit_params_kin_lgt_group_t" : self.fit_params_kin_lgt_group_t,
      "fit_params_kin_trv_group_t" : self.fit_params_kin_trv_group_t,
      "fit_params_kin_tot_ave"     : self.fit_params_kin_tot_ave,
      "fit_params_kin_lgt_ave"     : self.fit_params_kin_lgt_ave,
      "fit_params_kin_trv_ave"     : self.fit_params_kin_trv_ave
    }

  def saveFittedParams(self, filepath_sim):
    dict_params = self.getFittedParams()
    WWObjs.saveDict2JsonFile(f"{filepath_sim}/sim_outputs.json", dict_params)

  def __initialiseQuantities(self):
    ## flag to check that all required quantities have been measured
    self.bool_fitted                = False
    ## initialise quantities to measure
    self.list_k                     = None
    self.list_mag_power_tot_group_t = None
    self.list_kin_power_tot_group_t = None
    self.list_kin_power_lgt_group_t = None
    self.list_kin_power_trv_group_t = None
    self.plots_per_eddy             = None
    self.list_time_growth           = None
    self.list_time_k_eq             = None
    self.k_p_group_t                = None
    self.k_eq_group_t               = None
    self.fit_params_kin_tot_group_t = None
    self.fit_params_kin_lgt_group_t = None
    self.fit_params_kin_trv_group_t = None
    self.fit_params_kin_tot_ave     = None
    self.fit_params_kin_lgt_ave     = None
    self.fit_params_kin_trv_ave     = None

  def __checkAnyQuantitiesNotMeasured(self):
    list_quantities_check = [
      self.list_k,
      self.list_mag_power_tot_group_t,
      self.list_kin_power_tot_group_t,
      self.list_kin_power_lgt_group_t,
      self.list_kin_power_trv_group_t,
      self.plots_per_eddy,
      self.list_time_growth,
      self.list_time_k_eq,
      self.k_p_group_t,
      self.k_eq_group_t,
      self.fit_params_kin_tot_group_t,
      self.fit_params_kin_lgt_group_t,
      self.fit_params_kin_trv_group_t,
      self.fit_params_kin_tot_ave,
      self.fit_params_kin_lgt_ave,
      self.fit_params_kin_trv_ave
    ]
    return [ 
      index_quantity
      for index_quantity, quantity in enumerate(list_quantities_check)
      if quantity is None
    ]

  def __loadData(self):
    print("Loading energy spectra...")
    ## extract the number of plt-files per eddy-turnover-time from 'Turb.log'
    self.plots_per_eddy = LoadFlashData.getPlotsPerEddy_fromTurbLog(
      f"{self.filepath_spect}/../",
      bool_hide_updates = True
    )
    ## load total kinetic energy spectra
    dict_kin_spect_tot_data = LoadFlashData.loadAllSpectraData(
      filepath          = self.filepath_spect,
      spect_field       = "vel",
      spect_quantity    = "tot",
      file_start_time   = self.time_exp_start,
      file_end_time     = self.time_exp_end,
      plots_per_eddy    = self.plots_per_eddy,
      bool_hide_updates = True
    )
    ## load longitudinal kinetic energy spectra
    dict_kin_spect_lgt_data = LoadFlashData.loadAllSpectraData(
      filepath          = self.filepath_spect,
      spect_field       = "vel",
      spect_quantity    = "lgt",
      file_start_time   = self.time_exp_start,
      file_end_time     = self.time_exp_end,
      plots_per_eddy    = self.plots_per_eddy,
      bool_hide_updates = True
    )
    ## load transverse kinetic energy spectra
    dict_kin_spect_trv_data = LoadFlashData.loadAllSpectraData(
      filepath          = self.filepath_spect,
      spect_field       = "vel",
      spect_quantity    = "trv",
      file_start_time   = self.time_exp_start,
      file_end_time     = self.time_exp_end,
      plots_per_eddy    = self.plots_per_eddy,
      bool_hide_updates = True
    )
    ## load total magnetic energy spectra
    dict_mag_spect_tot_data = LoadFlashData.loadAllSpectraData(
      filepath          = self.filepath_spect,
      spect_field       = "mag",
      spect_quantity    = "tot",
      file_start_time   = self.time_exp_start,
      file_end_time     = self.time_exp_end,
      plots_per_eddy    = self.plots_per_eddy,
      bool_hide_updates = True
    )
    ## store time-evolving energy spectra
    self.list_kin_power_tot_group_t = dict_kin_spect_tot_data["list_power_group_t"]
    self.list_kin_power_lgt_group_t = dict_kin_spect_lgt_data["list_power_group_t"]
    self.list_kin_power_trv_group_t = dict_kin_spect_trv_data["list_power_group_t"]
    self.list_mag_power_tot_group_t = dict_mag_spect_tot_data["list_power_group_t"]
    self.list_k                     = dict_mag_spect_tot_data["list_k_group_t"][0]
    self.list_time_growth           = dict_mag_spect_tot_data["list_sim_times"]

  def __plotSpectra(self):
    plotSpectra_helper(
      ax                 = self.axs_spectra_tot[0],
      list_k             = self.list_k,
      list_power_group_t = self.list_kin_power_tot_group_t,
      color              = self.dict_plot_kin_tot["color"],
      cmap_name          = self.dict_plot_kin_tot["cmap_name"],
      list_times         = self.list_time_growth
    )
    plotSpectra_helper(
      ax                 = self.axs_spectra_tot[1],
      list_k             = self.list_k,
      list_power_group_t = self.list_mag_power_tot_group_t,
      color              = self.dict_plot_mag_tot["color"],
      cmap_name          = self.dict_plot_mag_tot["cmap_name"],
      list_times         = self.list_time_growth
    )
    plotSpectra_helper(
      ax                 = self.ax_spectra_comp,
      list_k             = self.list_k,
      list_power_group_t = self.list_kin_power_lgt_group_t,
      color              = self.dict_plot_kin_lgt["color"],
      cmap_name          = self.dict_plot_kin_lgt["cmap_name"],
      list_times         = self.list_time_growth
    )
    plotSpectra_helper(
      ax                 = self.ax_spectra_comp,
      list_k             = self.list_k,
      list_power_group_t = self.list_kin_power_trv_group_t,
      color              = self.dict_plot_kin_trv["color"],
      cmap_name          = self.dict_plot_kin_trv["cmap_name"],
      list_times         = self.list_time_growth
    )

  def __plotSpectraRatio(self):
    ## for each time realisation
    self.k_eq_group_t   = []
    self.list_time_k_eq = []
    ## plot each time realisation
    for time_index in range(len(self.list_time_growth)):
      ## calculate energy ratio spectrum
      list_spectra_ratio = \
        np.array(self.list_mag_power_tot_group_t[time_index]) / \
        np.array(self.list_kin_power_tot_group_t[time_index])
      ## plot ratio of spectra
      self.ax_spectra_ratio.plot(
        self.list_k,
        list_spectra_ratio,
        color="black", ls="-", lw=1, alpha=0.1, zorder=3
      )
      list_index_peaks, _ = find_peaks(list_spectra_ratio)
      if len(list_index_peaks) > 0:
        index_ratio_end = min(list_index_peaks)
      else: index_ratio_end = len(list_spectra_ratio) - 1
      if BOOL_DEBUG:
        self.ax_spectra_ratio.plot(
          self.list_k[index_ratio_end],
          list_spectra_ratio[index_ratio_end],
          "ro"
        )
      ## measure k_eq
      tol = 1e-1
      list_index_k_eq = [
        k_index
        for k_index, E_ratio in enumerate(list_spectra_ratio[:index_ratio_end])
        if abs(E_ratio - 1) <= tol
      ]
      if len(list_index_k_eq) > 0:
        index_k_eq = list_index_k_eq[0]
        k_eq       = self.list_k[index_k_eq]
        k_eq_power = list_spectra_ratio[index_k_eq]
        self.k_eq_group_t.append(k_eq)
        self.list_time_k_eq.append(self.list_time_growth[time_index])
        self.ax_spectra_ratio.plot(k_eq, k_eq_power, "ko")
    ## plot time-evolution of measured scale
    if len(self.k_eq_group_t) > 0:
      plotMeasuredScale(
        ax_time       = self.ax_scales_time,
        list_t        = self.list_time_k_eq,
        scale_group_t = self.k_eq_group_t,
        scale_ave     = np.mean(self.k_eq_group_t),
        color         = "black",
        label         = r"$k_{\rm eq}$"
      )

  def __fitKinSpectra(self):
    self.fit_params_kin_tot_group_t, self.fit_params_kin_tot_ave = fitKinSpectra_helper(
      ax_fit             = self.axs_spectra_tot[0],
      ax_residuals       = self.ax_residuals,
      ax_scales          = self.ax_scales_time,
      list_k             = self.list_k,
      list_power_group_t = self.list_kin_power_tot_group_t,
      list_time_growth   = self.list_time_growth,
      color              = self.dict_plot_kin_tot["color"],
      label_spect        = self.dict_plot_kin_tot["label_spect"],
      label_knu          = self.dict_plot_kin_tot["label_knu"]
    )
    self.fit_params_kin_lgt_group_t, self.fit_params_kin_lgt_ave = fitKinSpectra_helper(
      ax_fit             = self.ax_spectra_comp,
      ax_residuals       = self.ax_residuals,
      ax_scales          = self.ax_scales_time,
      list_k             = self.list_k,
      list_power_group_t = self.list_kin_power_lgt_group_t,
      list_time_growth   = self.list_time_growth,
      color              = self.dict_plot_kin_lgt["color"],
      label_spect        = self.dict_plot_kin_lgt["label_spect"],
      label_knu          = self.dict_plot_kin_lgt["label_knu"]
    )
    self.fit_params_kin_trv_group_t, self.fit_params_kin_trv_ave = fitKinSpectra_helper(
      ax_fit             = self.ax_spectra_comp,
      ax_residuals       = self.ax_residuals,
      ax_scales          = self.ax_scales_time,
      list_k             = self.list_k,
      list_power_group_t = self.list_kin_power_trv_group_t,
      list_time_growth   = self.list_time_growth,
      color              = self.dict_plot_kin_trv["color"],
      label_spect        = self.dict_plot_kin_trv["label_spect"],
      label_knu          = self.dict_plot_kin_trv["label_knu"]
    )

  def __fitMagSpectra(self):
    self.k_p_group_t   = []
    self.k_max_group_t = []
    ## fit each time-realisation of the magnetic energy spectrum
    for time_index in range(len(self.list_time_growth)):
      k_p, k_max = getMagSpectrumPeak(
        self.list_k,
        getSpectraNorm(self.list_mag_power_tot_group_t[time_index])
      )
      ## store measured scales
      self.k_p_group_t.append(k_p)
      self.k_max_group_t.append(k_max)
    ## annotate measured raw spectrum maximum
    self.axs_spectra_tot[1].plot(
      np.mean(self.k_max_group_t),
      np.mean(np.max(getSpectraNorm_group(self.list_mag_power_tot_group_t), axis=1)),
      color="black", marker="o", ms=10, ls="", label=r"$k_{\rm max}$", zorder=7
    )
    ## plot time-evolution of measured scale
    plotMeasuredScale(
      ax_spectrum   = self.axs_spectra_tot[1],
      ax_time       = self.ax_scales_time,
      list_t        = self.list_time_growth,
      scale_group_t = self.k_p_group_t,
      scale_ave     = np.mean(self.k_p_group_t),
      color         = self.dict_plot_mag_tot["color"],
      label         = self.dict_plot_mag_tot["label_kp"]
    )

  def __adjustAxis(self, ax, bool_log_y=True):
    ax.set_xlim([ 0.9, 1.2*max(self.list_k) ])
    ax.set_xscale("log")
    if bool_log_y: ax.set_yscale("log")

  def __labelSpectraPlot(self):
    ## label fit-residuals axis
    self.__adjustAxis(self.ax_residuals, bool_log_y=False)
    self.ax_residuals.axhline(y=1, color="black", ls="--")
    addLegend_withBox(
      ax   = self.ax_residuals,
      loc  = "upper right",
      bbox = (1.0, 1.0)
    )
    self.ax_residuals.set_xlabel(r"$k$")
    self.ax_residuals.set_ylabel(addLabel_timeAve(
      getLabel_spectrum("fit") + r"$\, / \,$" + getLabel_spectrum("data")
    ))
    ## label energy spectra axis
    self.__adjustAxis(self.axs_spectra_tot[0])
    self.__adjustAxis(self.axs_spectra_tot[1])
    labelDualAxis(
      axs         = self.axs_spectra_tot,
      label_left  = self.dict_plot_kin_tot["label_spect"],
      label_right = self.dict_plot_mag_tot["label_spect"],
      color_left  = self.dict_plot_kin_tot["color"],
      color_right = self.dict_plot_mag_tot["color"]
    )
    PlotFuncs.addBoxOfLabels(
      fig           = self.fig,
      ax            = self.ax_spectra_comp,
      box_alignment = (1.0, 1.0),
      xpos          = 0.95,
      ypos          = 0.95,
      alpha         = 0.85,
      fontsize      = 20,
      list_labels   = [
        self.dict_plot_kin_lgt["label_spect"],
        self.dict_plot_kin_trv["label_spect"]
      ],
      list_colors   = [
        self.dict_plot_kin_lgt["color"],
        self.dict_plot_kin_trv["color"]
      ]
    )
    PlotFuncs.addBoxOfLabels(
      fig           = self.fig,
      ax            = self.axs_spectra_tot[0],
      box_alignment = (0.5, 0.0),
      xpos          = 0.5,
      ypos          = 0.05,
      alpha         = 0.85,
      fontsize      = 15,
      list_labels   = [
        getLabels_kin(self.fit_params_kin_tot_group_t),
        getLabels_mag(self.k_p_group_t, self.k_max_group_t)
      ],
      list_colors   = [
        self.dict_plot_kin_tot["color"],
        self.dict_plot_mag_tot["color"]
      ]
    )
    ## label components of the kinetic energy spectra axis
    self.__adjustAxis(self.ax_spectra_comp)
    self.ax_spectra_comp.set_xlabel(r"$k$")
    self.ax_spectra_comp.set_ylabel(getLabel_spectrum("kin"))
    PlotFuncs.addBoxOfLabels(
      fig           = self.fig,
      ax            = self.ax_spectra_comp,
      box_alignment = (0.5, 0.0),
      xpos          = 0.5,
      ypos          = 0.05,
      alpha         = 0.85,
      fontsize      = 15,
      list_labels   = [
        getLabels_kin(self.fit_params_kin_lgt_group_t),
        getLabels_kin(self.fit_params_kin_trv_group_t)
      ],
      list_colors   = [
        self.dict_plot_kin_lgt["color"],
        self.dict_plot_kin_trv["color"]
      ]
    )

  def __labelSpectraRatioPlot(self):
    self.ax_spectra_ratio.axhline(y=1, color="black", ls="--")
    self.__adjustAxis(self.ax_spectra_ratio)
    self.ax_spectra_ratio.set_xlabel(r"$k$")
    self.ax_spectra_ratio.set_ylabel(getLabel_spectrum("data") + r"$/$" + getLabel_spectrum("fit"))

  def __labelScalesPlot(self):
    self.ax_scales_time.set_yscale("log")
    self.ax_scales_time.set_xlabel(r"$t/t_{\rm turb}$")
    self.ax_scales_time.set_ylabel(r"$k$")
    addLegend_withBox(
      ax   = self.ax_scales_time,
      loc  = "upper left",
      bbox = (0.0, 1.0)
    )


## ###############################################################
## HANDLING PLOT CALLS
## ###############################################################
def plotSimData(filepath_sim_res, filepath_vis, sim_name):
  ## INITIALISE FIGURE
  ## -----------------
  print("Initialising figure...")
  fig, fig_grid = PlotFuncs.createFigure_grid(
    fig_scale        = 0.8,
    fig_aspect_ratio = (5.0, 8.0),
    num_rows         = 3,
    num_cols         = 3
  )
  ax_Mach          = fig.add_subplot(fig_grid[0, 0])
  ax_E_ratio       = fig.add_subplot(fig_grid[1, 0])
  ax_residuals     = fig.add_subplot(fig_grid[2, 0])
  axs_spectra_tot  = addSubplot_secondAxis(fig, fig_grid[0, 1])
  ax_spectra_comp  = fig.add_subplot(fig_grid[1, 1])
  ax_spectra_ratio = fig.add_subplot(fig_grid[:2, 2])
  ax_scales_time   = fig.add_subplot(fig_grid[2, 1:])
  ## PLOT INTEGRATED QUANTITIES
  ## --------------------------
  obj_plot_turb = PlotTurbData(
    fig              = fig,
    axs              = [ ax_Mach, ax_E_ratio ],
    filepath_sim_res = filepath_sim_res,
    dict_sim_inputs  = SimParams.readSimInputs(filepath_sim_res)
  )
  obj_plot_turb.performRoutines()
  obj_plot_turb.saveFittedParams(filepath_sim_res)
  dict_turb_params = obj_plot_turb.getFittedParams()
  ## PLOT FITTED SPECTRA
  ## -------------------
  obj_plot_spectra = PlotSpectra(
    fig              = fig,
    dict_axs         = {
      "axs_spectra_tot"  : axs_spectra_tot,
      "ax_spectra_comp"  : ax_spectra_comp,
      "ax_scales"        : ax_scales_time,
      "ax_residuals"     : ax_residuals,
      "ax_spectra_ratio" : ax_spectra_ratio,
    },
    filepath_spect   = f"{filepath_sim_res}/spect/",
    time_exp_start   = dict_turb_params["time_growth_start"],
    time_exp_end     = dict_turb_params["time_growth_end"]
  )
  obj_plot_spectra.performRoutines()
  obj_plot_spectra.saveFittedParams(filepath_sim_res)
  ## SAVE FIGURE
  ## -----------
  fig_name = f"{sim_name}_dataset.png"
  PlotFuncs.saveFigure(fig, f"{filepath_vis}/{fig_name}")


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
# LIST_SIM_FOLDER   = [ "Pm25" ]
# LIST_SIM_RES      = [ "72" ]


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
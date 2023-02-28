#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys, functools
import numpy as np
import multiprocessing as mproc
import concurrent.futures as cfut

## 'tmpfile' needs to be loaded before any 'matplotlib' libraries,
## so matplotlib stores its cache in a temporary directory.
## (necessary when plotting in parallel)
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
import matplotlib.pyplot as plt

## load user defined routines
from plot_turb_data import PlotTurbData

## load user defined modules
from TheSimModule import SimParams
from TheUsefulModule import WWFnF, WWObjs
from TheLoadingModule import LoadFlashData
from TheFittingModule import FitMHDScales, FitFuncs
from TheAnalysisModule import WWSpectra
from ThePlottingModule import PlotFuncs, PlotLatex


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def replaceNoneWNan(list_elems):
  return [ np.nan if elem is None else elem for elem in list_elems ]

def plotMeasuredScales(
    ax_spectra, ax_scales,
    list_t, scale_group_t,
    color       = "black",
    label       = ""
  ):
  if len(scale_group_t) < 5: return
  scale_group_t = replaceNoneWNan(scale_group_t)
  ax_spectra.axvline(x=np.mean(scale_group_t), color=color, ls="--", zorder=1, lw=2.0)
  ax_scales.axhline(y=np.mean(scale_group_t),  color=color, ls="--", zorder=1, lw=2.0)
  ax_scales.plot(list_t, scale_group_t,        color=color, ls="-",  zorder=1, label=label)

def plotSpectra(ax, list_k, list_power_group_t, color, cmap_name, list_times, bool_norm=False):
  args_plot_ave_spectrum   = { "color":color, "marker":"o", "ms":8, "ls":"", "zorder":5, "markeredgecolor":"black" }
  args_plot_time_evolution = { "ls":"-", "lw":1, "alpha":0.5, "zorder":3 }
  # ## plot time averaged, normalised energy spectra
  # list_power_ave = WWSpectra.aveSpectra(list_power_group_t, bool_norm=bool_norm)
  # ax.plot(list_k, list_power_ave, **args_plot_ave_spectrum)
  ## create colormaps for time-evolving spectra (color as a function of time)
  cmap, norm = PlotFuncs.createCmap(
    cmap_name = cmap_name,
    vmin      = min(list_times),
    vmax      = max(list_times)
  )
  ## plot each time realisation of the normalised kinetic energy spectrum
  for time_index, time_val in enumerate(list_times):
    list_power = list_power_group_t[time_index]
    if bool_norm: list_power = WWSpectra.normSpectra(list_power)
    ax.plot(list_k, list_power, color=cmap(norm(time_val)), **args_plot_time_evolution)

def plotReynoldsSpectrum(ax, list_times, list_k, list_power_group_t, viscosity, cmap_name, color):
  args_plot_time_evolution = { "ls":"-", "lw":1, "alpha":0.5, "zorder":3 }
  ## create colormaps for time-evolving spectra (color as a function of time)
  cmap, norm = PlotFuncs.createCmap(
    cmap_name = cmap_name,
    vmin      = min(list_times),
    vmax      = max(list_times)
  )
  ## measure dissipation scale for each time realisation
  scales_group_t = []
  for time_index, time_val in enumerate(list_times):
    ## TODO: undo normalisation
    scale = None
    array_reynolds = np.sqrt(np.cumsum(2.0 * np.array(list_power_group_t[time_index][::-1])))[::-1] / (np.array(list_k) * viscosity)
    ax.plot(list_k, array_reynolds, color=cmap(norm(time_val)), **args_plot_time_evolution)
    if np.log10(min(array_reynolds)) < 1e-1:
      list_k_interp = np.logspace(np.log10(min(list_k)), np.log10(max(list_k)), 10**4)
      list_reynolds_interp = FitFuncs.interpLogLogData(list_k, array_reynolds, list_k_interp, interp_kind="cubic")
      dis_scale_index = np.argmin(abs(list_reynolds_interp - 1.0))
      scale = list_k_interp[dis_scale_index]
    scales_group_t.append(scale)
  ax.axvline(x=np.mean(replaceNoneWNan(scales_group_t)), ls="--", lw=2.0, color=color)
  return scales_group_t


## ###############################################################
## OPERATOR CLASS: PLOT NORMALISED + TIME-AVERAGED ENERGY SPECTRA
## ###############################################################
class PlotSpectra():
  def __init__(
      self,
      fig, dict_axs, dict_sim_inputs, filepath_spect, time_exp_start, time_exp_end, plots_per_eddy,
      bool_verbose = True
    ):
    ## save input arguments
    self.fig              = fig
    self.axs_spectra      = dict_axs["axs_spectra"]
    self.axs_reynolds     = dict_axs["axs_reynolds"]
    self.ax_spectra_ratio = dict_axs["ax_spectra_ratio"]
    self.axs_scales       = dict_axs["axs_scales"]
    self.dict_sim_inputs  = dict_sim_inputs
    self.filepath_spect   = filepath_spect
    self.time_exp_start   = time_exp_start
    self.time_exp_end     = time_exp_end
    self.plots_per_eddy   = plots_per_eddy
    self.bool_verbose     = bool_verbose
    ## initialise spectra labels
    self.__initialiseQuantities()
    self.index_mag         = 0
    self.index_vel_lgt     = 1
    self.index_vel_trv     = 2
    self.color_k_eq        = "black"
    self.dict_plot_mag = {
      "color_spect" : "red",
      "color_kp"    : "darkorange",
      "color_keta"  : "red",
      "cmap_name"   : "Reds",
      "label_spect" : PlotLatex.GetLabel.spectrum("mag", "tot"),
      "label_kp"    : r"$k_{\rm p}$",
      "label_keta"  : r"$k_\eta$",
    }
    self.dict_plot_vel_trv = {
      "color_spect"     : "darkgreen",
      "color_knu"       : "darkgreen",
      "cmap_name"       : "Greens",
      "label_spect"     : PlotLatex.GetLabel.spectrum("kin", "trv"),
      "label_knu"       : r"$k_{\nu, \perp}$",
    }
    self.dict_plot_vel_lgt = {
      "color_spect"     : "royalblue",
      "color_knu"       : "royalblue",
      "cmap_name"       : "Blues",
      "label_spect"     : PlotLatex.GetLabel.spectrum("kin", "lgt"),
      "label_knu"       : r"$k_{\nu, \parallel}$",
    }

  def performRoutines(self):
    self.__loadSpectra()
    self.__plotSpectra()
    self.__plotSpectraRatio()
    if self.bool_verbose: print("Extracting scales...")
    self.__fitMagSpectra()
    self.bool_fitted = True
    self.__labelSpectra()
    self.__labelSpectraRatio()
    self.__labelScales()

  def getFittedParams(self):
    self.__checkAnyQuantitiesNotMeasured()
    if not self.bool_fitted: self.performRoutines()
    return {
      ## time-averaged energy spectra
      "list_k"                 : self.list_k,
      "list_mag_power_tot_ave" : WWSpectra.aveSpectra(self.list_mag_power_tot_group_t, bool_norm=True),
      "list_vel_power_tot_ave" : WWSpectra.aveSpectra(self.list_vel_power_tot_group_t, bool_norm=False),
      "list_vel_power_lgt_ave" : WWSpectra.aveSpectra(self.list_vel_power_lgt_group_t, bool_norm=False),
      "list_vel_power_trv_ave" : WWSpectra.aveSpectra(self.list_vel_power_trv_group_t, bool_norm=False),
      ## measured quantities
      "list_time_growth"       : self.list_time_growth,
      "list_time_k_eq"         : self.list_time_k_eq,
      "k_p_group_t"            : self.k_p_group_t,
      "k_eq_group_t"           : self.k_eq_group_t,
      "k_eta_group_t"          : self.k_eta_group_t,
      "k_nu_trv_group_t"       : self.k_nu_trv_group_t,
      "k_nu_lgt_group_t"       : self.k_nu_lgt_group_t,
    }

  def saveFittedParams(self, filepath_sim):
    dict_params = self.getFittedParams()
    WWObjs.saveDict2JsonFile(f"{filepath_sim}/sim_outputs.json", dict_params, self.bool_verbose)

  def __initialiseQuantities(self):
    ## flag to check that all required quantities have been measured
    self.bool_fitted                      = False
    ## initialise quantities to measure
    self.list_k                     = None
    self.list_mag_power_tot_group_t = None
    self.list_vel_power_tot_group_t = None # TODO: rename list_power_(field)_(sub)_group_t
    self.list_vel_power_lgt_group_t = None
    self.list_vel_power_trv_group_t = None
    self.list_time_growth           = None
    self.list_time_k_eq             = None
    self.k_p_group_t                = None
    self.k_eq_group_t               = None
    self.k_eta_group_t              = None
    self.k_nu_trv_group_t           = None
    self.k_nu_lgt_group_t           = None

  def __checkAnyQuantitiesNotMeasured(self):
    list_quantities_check = [
      self.list_k,                     # 0
      self.list_mag_power_tot_group_t, # 1
      self.list_vel_power_tot_group_t, # 2
      self.list_vel_power_lgt_group_t, # 3
      self.list_vel_power_trv_group_t, # 4
      self.list_time_growth,           # 5
      self.list_time_k_eq,             # 6
      self.k_p_group_t,                # 7
      self.k_eq_group_t,               # 8
      self.k_eta_group_t,              # 9
      self.k_nu_trv_group_t,           # 10
      self.k_nu_lgt_group_t,           # 11
    ]
    list_quantities_undefined = [ 
      index_quantity
      for index_quantity, quantity in enumerate(list_quantities_check)
      if quantity is None
    ]
    if len(list_quantities_undefined) > 0: raise Exception("Error: the following quantities were not measured:", list_quantities_undefined)

  def __loadSpectra(self):
    if self.bool_verbose: print("Loading energy spectra...")
    ## load spectra data within the growth phase of the dynamo
    ## load total magnetic energy spectra
    dict_mag_spect_tot_data = LoadFlashData.loadAllSpectraData(
      filepath        = self.filepath_spect,
      spect_field     = "mag",
      spect_quantity  = "tot",
      file_start_time = self.time_exp_start,
      file_end_time   = self.time_exp_end,
      plots_per_eddy  = self.plots_per_eddy,
      bool_verbose    = False
    )
    ## load total kinetic energy spectra
    dict_vel_spect_tot_data = LoadFlashData.loadAllSpectraData(
      filepath        = self.filepath_spect,
      spect_field     = "vel",
      spect_quantity  = "tot",
      file_start_time = self.time_exp_start,
      file_end_time   = self.time_exp_end,
      plots_per_eddy  = self.plots_per_eddy,
      bool_verbose    = False
    )
    ## load longitudinal kinetic energy spectra
    dict_vel_spect_lgt_data = LoadFlashData.loadAllSpectraData(
      filepath        = self.filepath_spect,
      spect_field     = "vel",
      spect_quantity  = "lgt",
      file_start_time = self.time_exp_start,
      file_end_time   = self.time_exp_end,
      plots_per_eddy  = self.plots_per_eddy,
      bool_verbose    = False
    )
    ## load transverse kinetic energy spectra
    dict_vel_spect_trv_data = LoadFlashData.loadAllSpectraData(
      filepath        = self.filepath_spect,
      spect_field     = "vel",
      spect_quantity  = "trv",
      file_start_time = self.time_exp_start,
      file_end_time   = self.time_exp_end,
      plots_per_eddy  = self.plots_per_eddy,
      bool_verbose    = False
    )
    ## store time-evolving energy spectra
    self.list_k                     = dict_mag_spect_tot_data["list_k_group_t"][0]
    self.list_time_growth           = dict_mag_spect_tot_data["list_sim_times"]
    self.list_mag_power_tot_group_t = dict_mag_spect_tot_data["list_power_group_t"]
    self.list_vel_power_tot_group_t = dict_vel_spect_tot_data["list_power_group_t"]
    self.list_vel_power_lgt_group_t = dict_vel_spect_lgt_data["list_power_group_t"]
    self.list_vel_power_trv_group_t = dict_vel_spect_trv_data["list_power_group_t"]

  def __plotSpectra(self):
    ## magnetic energy spectrum
    plotSpectra(
      ax                 = self.axs_spectra[self.index_mag],
      list_k             = self.list_k,
      list_power_group_t = self.list_mag_power_tot_group_t,
      color              = self.dict_plot_mag["color_spect"],
      cmap_name          = self.dict_plot_mag["cmap_name"],
      list_times         = self.list_time_growth,
      bool_norm          = True
    )
    self.k_eta_group_t = plotReynoldsSpectrum(
      ax                 = self.axs_reynolds[self.index_mag],
      list_times         = self.list_time_growth,
      list_k             = self.list_k,
      list_power_group_t = self.list_vel_power_tot_group_t,
      viscosity          = self.dict_sim_inputs["eta"],
      cmap_name          = self.dict_plot_mag["cmap_name"],
      color              = self.dict_plot_mag["color_keta"],
    )
    plotMeasuredScales(
      ax_spectra    = self.axs_spectra[self.index_mag],
      ax_scales     = self.axs_scales[0],
      list_t        = self.list_time_growth,
      scale_group_t = self.k_eta_group_t,
      color         = self.dict_plot_mag["color_keta"],
      label         = self.dict_plot_mag["label_keta"],
    )
    ## longitudinal kinetic energy spectrum
    plotSpectra(
      ax                 = self.axs_spectra[self.index_vel_lgt],
      list_k             = self.list_k,
      list_power_group_t = self.list_vel_power_lgt_group_t,
      color              = self.dict_plot_vel_lgt["color_spect"],
      cmap_name          = self.dict_plot_vel_lgt["cmap_name"],
      list_times         = self.list_time_growth,
      bool_norm          = False
    )
    self.k_nu_lgt_group_t = plotReynoldsSpectrum(
      ax                 = self.axs_reynolds[self.index_vel_lgt],
      list_times         = self.list_time_growth,
      list_k             = self.list_k,
      list_power_group_t = self.list_vel_power_lgt_group_t,
      viscosity          = self.dict_sim_inputs["nu"],
      cmap_name          = self.dict_plot_vel_lgt["cmap_name"],
      color              = self.dict_plot_vel_lgt["color_knu"],
    )
    plotMeasuredScales(
      ax_spectra    = self.axs_spectra[self.index_vel_lgt],
      ax_scales     = self.axs_scales[0],
      list_t        = self.list_time_growth,
      scale_group_t = self.k_nu_lgt_group_t,
      color         = self.dict_plot_vel_lgt["color_knu"],
      label         = self.dict_plot_vel_lgt["label_knu"],
    )
    ## transverse kinetic energy spectrum
    plotSpectra(
      ax                 = self.axs_spectra[self.index_vel_trv],
      list_k             = self.list_k,
      list_power_group_t = self.list_vel_power_trv_group_t,
      color              = self.dict_plot_vel_trv["color_spect"],
      cmap_name          = self.dict_plot_vel_trv["cmap_name"],
      list_times         = self.list_time_growth,
      bool_norm          = False
    )
    self.k_nu_trv_group_t = plotReynoldsSpectrum(
      ax                 = self.axs_reynolds[self.index_vel_trv],
      list_times         = self.list_time_growth,
      list_k             = self.list_k,
      list_power_group_t = self.list_vel_power_trv_group_t,
      viscosity          = self.dict_sim_inputs["nu"],
      cmap_name          = self.dict_plot_vel_trv["cmap_name"],
      color              = self.dict_plot_vel_trv["color_knu"],
    )
    plotMeasuredScales(
      ax_spectra    = self.axs_spectra[self.index_vel_trv],
      ax_scales     = self.axs_scales[0],
      list_t        = self.list_time_growth,
      scale_group_t = self.k_nu_trv_group_t,
      color         = self.dict_plot_vel_trv["color_knu"],
      label         = self.dict_plot_vel_trv["label_knu"],
    )

  def __plotSpectraRatio(self):
    ## load spectra data again, this time for the full duration of the simulation
    ## load total kinetic energy spectra TODO: use kinetic energy spectrum for this
    dict_vel_spect_tot_data = LoadFlashData.loadAllSpectraData(
      filepath        = self.filepath_spect,
      spect_field     = "vel",
      spect_quantity  = "tot",
      file_start_time = self.time_exp_start,
      file_end_time   = np.inf,
      plots_per_eddy  = self.plots_per_eddy,
      bool_verbose    = False
    )
    ## load total magnetic energy spectra
    dict_mag_spect_tot_data = LoadFlashData.loadAllSpectraData(
      filepath        = self.filepath_spect,
      spect_field     = "mag",
      spect_quantity  = "tot",
      file_start_time = self.time_exp_start,
      file_end_time   = np.inf,
      plots_per_eddy  = self.plots_per_eddy,
      bool_verbose    = False
    )
    ## measure + plot evolving equipartition scale
    self.k_eq_group_t, self.k_eq_power_group_t, self.list_time_k_eq = FitMHDScales.getScale_keq(
      ax_spectra             = self.ax_spectra_ratio,
      ax_scales              = self.axs_scales[1],
      list_sim_time          = dict_mag_spect_tot_data["list_sim_times"],
      list_k                 = dict_mag_spect_tot_data["list_k_group_t"][0],
      list_mag_power_group_t = dict_mag_spect_tot_data["list_power_group_t"],
      list_kin_power_group_t = dict_vel_spect_tot_data["list_power_group_t"],
      color                  = self.color_k_eq
    )

  def __fitMagSpectra(self):
    self.k_p_group_t   = []
    self.k_max_group_t = []
    ## fit each time-realisation of the magnetic energy spectrum
    for time_index in range(len(self.list_time_growth)):
      k_p, k_max = FitMHDScales.getScale_kp(
        self.list_k,
        WWSpectra.normSpectra(self.list_mag_power_tot_group_t[time_index])
      )
      ## store measured scales
      self.k_p_group_t.append(k_p)
      self.k_max_group_t.append(k_max)
    ## plot time-evolution of measured scale
    plotMeasuredScales(
      ax_spectra    = self.axs_spectra[self.index_mag],
      ax_scales     = self.axs_scales[0],
      list_t        = self.list_time_growth,
      scale_group_t = self.k_p_group_t,
      color         = self.dict_plot_mag["color_kp"],
      label         = self.dict_plot_mag["label_kp"]
    )

  # def __fitKinSpectra(self):
  #   ## define helper function
  #   def fitKinSpectra(
  #       ax_fit, ax_residuals, ax_scales,
  #       list_k, list_power_group_t, list_time_growth,
  #       bool_fix_params = False,
  #       color_fit       = "black",
  #       label_spect     = "",
  #       label_knu       = r"$k_\nu$"
  #     ):
  #     fitMethod = FitMHDScales.fitKinSpectrum
  #     ## fit each time-realisation of the kinetic energy spectrum
  #     k_nu_adj_group_t       = []
  #     fit_param_vals_group_t = []
  #     for time_index in range(len(list_time_growth)):
  #       fit_param_vals, fit_param_errs, fit_rcs = fitMethod(
  #         list_k          = list_k,
  #         list_power      = WWSpectra.normSpectra(list_power_group_t[time_index]),
  #         bool_fix_params = bool_fix_params
  #       )
  #       ## store fitted parameters
  #       k_nu_adj_group_t.append(getScaleAdj_knu(fit_param_vals))
  #       fit_param_vals_group_t.append(fit_param_vals)
  #     ## plot fit to time-averaged kinetic energy spectrum
  #     fit_param_vals_ave, fit_param_errs_ave, fit_rcs_ave = fitMethod(
  #       ax_fit          = ax_fit,
  #       ax_residuals    = ax_residuals,
  #       list_k          = list_k,
  #       list_power      = WWSpectra.aveSpectra(list_power_group_t, bool_norm=True),
  #       list_power_std  = np.std([
  #           np.log(list_power)
  #           for list_power in list_power_group_t
  #       ], axis=0),
  #       color           = color_fit,
  #       label_spect     = label_spect,
  #       bool_fix_params = bool_fix_params
  #     )
  #     ## plot measured dissipation scale
  #     plotMeasuredScales(
  #       ax_spectra    = ax_fit,
  #       ax_scales     = ax_scales,
  #       list_t        = list_time_growth,
  #       scale_group_t = k_nu_adj_group_t,
  #       scale_ave     = getScaleAdj_knu(fit_param_vals_ave),
  #       color         = color_fit,
  #       label         = label_knu
  #     )
  #     return k_nu_adj_group_t, fit_param_vals_group_t, fit_param_vals_ave
  #   ## create strings
  #   str_fixed_alpha_cas = r" fixed $\alpha_{\rm cas}$"
  #   str_fixed_alpha_dis = r" fixed $\alpha_{\rm dis}$"
  #   ## fit transverse kinetic spectrum
  #   self.k_nu_adj_trv_group_t, \
  #     self.fit_params_vel_trv_group_t, \
  #       self.fit_params_vel_trv_ave = fitKinSpectra(
  #     ax_fit              = self.axs_spectra[self.index_mag],
  #     ax_residuals        = self.ax_residuals,
  #     ax_scales           = self.axs_scales[0],
  #     list_k              = self.list_k,
  #     list_power_group_t  = self.list_vel_power_trv_group_t,
  #     list_time_growth    = self.list_time_growth,
  #     color_fit           = self.dict_plot_vel_trv["color_fit"],
  #     label_spect         = self.dict_plot_vel_trv["label_spect"] + str_fixed_alpha_cas,
  #     label_knu           = self.dict_plot_vel_trv["label_knu"]   + str_fixed_alpha_cas,
  #     bool_fix_params     = False
  #   )
  #   self.k_nu_adj_trv_fixed_group_t, \
  #     self.fit_params_vel_trv_fixed_group_t, \
  #       self.fit_params_vel_trv_fixed_ave = fitKinSpectra(
  #     ax_fit              = self.axs_spectra[self.index_mag],
  #     ax_residuals        = self.ax_residuals,
  #     ax_scales           = self.axs_scales[0],
  #     list_k              = self.list_k,
  #     list_power_group_t  = self.list_vel_power_trv_group_t,
  #     list_time_growth    = self.list_time_growth,
  #     color_fit           = self.dict_plot_vel_trv["color_fit_fixed"],
  #     label_spect         = self.dict_plot_vel_trv["label_spect"] + str_fixed_alpha_dis,
  #     label_knu           = self.dict_plot_vel_trv["label_knu"]   + str_fixed_alpha_dis,
  #     bool_fix_params     = True
  #   )

  def __adjustAxis(self, ax, bool_log_y=True):
    ax.set_xlim([ 0.9, 1.2*max(self.list_k) ])
    ax.set_xscale("log")
    if bool_log_y: ax.set_yscale("log")

  def __labelSpectra(self):
    self.axs_spectra[-1].set_xlabel(r"$k$")
    self.axs_spectra[self.index_mag].set_ylabel(self.dict_plot_mag["label_spect"])
    self.axs_spectra[self.index_vel_lgt].set_ylabel(self.dict_plot_vel_lgt["label_spect"])
    self.axs_spectra[self.index_vel_trv].set_ylabel(self.dict_plot_vel_trv["label_spect"])
    self.__adjustAxis(self.axs_spectra[self.index_mag])
    self.__adjustAxis(self.axs_spectra[self.index_vel_lgt])
    self.__adjustAxis(self.axs_spectra[self.index_vel_trv])
    self.axs_reynolds[-1].set_xlabel(r"$k$")
    self.axs_reynolds[self.index_mag].set_ylabel(r"${\rm Rm}_{\rm tot}(k)$")
    self.axs_reynolds[self.index_vel_lgt].set_ylabel(r"${\rm Re}_{\parallel}(k)$")
    self.axs_reynolds[self.index_vel_trv].set_ylabel(r"${\rm Re}_{\perp}(k)$")
    self.__adjustAxis(self.axs_reynolds[self.index_mag])
    self.__adjustAxis(self.axs_reynolds[self.index_vel_lgt])
    self.__adjustAxis(self.axs_reynolds[self.index_vel_trv])
    self.axs_reynolds[self.index_mag].axhline(y=1, ls=":", lw=2.0, color="k")
    self.axs_reynolds[self.index_vel_lgt].axhline(y=1, ls=":", lw=2.0, color="k")
    self.axs_reynolds[self.index_vel_trv].axhline(y=1, ls=":", lw=2.0, color="k")

  def __labelSpectraRatio(self):
    args_text = { "va":"bottom", "ha":"right", "transform":self.ax_spectra_ratio.transAxes, "fontsize":25 }
    self.ax_spectra_ratio.axhline(y=1, color="black", ls=":", lw=2.0)
    x = np.linspace(10**(-1), 10**(4), 10**4)
    PlotFuncs.plotData_noAutoAxisScale(
      ax     = self.ax_spectra_ratio,
      x      = x,
      y      = 10**(-5)*x**(2),
      ls     = "--",
      lw     = 2.0,
      color  = "red",
      zorder = 5
    )
    PlotFuncs.plotData_noAutoAxisScale(
      ax     = self.ax_spectra_ratio,
      x      = x,
      y      = 10**(-5)*x**(2.5),
      ls     = "--",
      lw     = 2.0,
      color  = "royalblue",
      zorder = 5
    )
    self.ax_spectra_ratio.text(0.95, 0.05, r"$\propto k^{2}$",   **args_text, color="red")
    self.ax_spectra_ratio.text(0.95, 0.10, r"$\propto k^{2.5}$", **args_text, color="royalblue")
    self.__adjustAxis(self.ax_spectra_ratio)
    self.ax_spectra_ratio.set_xlabel(r"$k$")
    self.ax_spectra_ratio.set_ylabel(
      PlotLatex.GetLabel.spectrum("mag") + r"$/$" + PlotLatex.GetLabel.spectrum("kin", "tot")
    )

  def __labelScales(self):
    self.axs_scales[0].set_yscale("log")
    self.axs_scales[0].set_ylabel(r"$k$")
    if len(self.k_eq_group_t) > 0:
      PlotFuncs.labelDualAxis_sharedY(
        axs          = self.axs_scales,
        label_bottom = r"$t_{\rm growth} \in t/t_{\rm turb}$",
        label_top    = r"$t_{\rm eq} \in t/t_{\rm turb}$",
        color_bottom = "black",
        color_top    = self.color_k_eq
      )
    else:
      self.axs_scales[0].set_xlabel(r"$t_{\rm growth} \in t/t_{\rm turb}$")
      self.axs_scales[1].set_xticks([])
    PlotFuncs.addLegend_joinedAxis(
      axs  = self.axs_scales,
      loc  = "upper right",
      bbox = (1.0, 1.0),
      alpha = 0.75,
      ncol = 2
    )
    self.axs_scales[0].set_ylim([
      0.9 * np.nanmin([
        min(self.list_k),
        min(self.k_eq_group_t) if len(self.k_eq_group_t) > 0 else np.nan
      ]),
      1.2 * np.nanmax([
        max(self.list_k),
        max(self.k_eq_group_t) if len(self.k_eq_group_t) > 0 else np.nan
      ])
    ])


## ###############################################################
## HANDLING PLOT CALLS
## ###############################################################
def plotSimData(
    filepath_sim_res,
    lock         = None,
    bool_verbose = True
  ):
  print("Looking at:", filepath_sim_res)
  dict_sim_inputs = SimParams.readSimInputs(filepath_sim_res, bool_verbose=False)
  ## INITIALISE FIGURE
  ## -----------------
  if bool_verbose: print("Initialising figure...")
  fig, fig_grid = PlotFuncs.createFigure_grid(
    fig_scale        = 0.6,
    fig_aspect_ratio = (6.0, 10.0), # height, width
    num_rows         = 4,
    num_cols         = 3
  )
  ## volume integrated qunatities
  ax_Mach         = fig.add_subplot(fig_grid[0, 0])
  ax_energy_ratio = fig.add_subplot(fig_grid[1, 0])
  ## power spectra data
  ax_spectra_ratio    = fig.add_subplot(fig_grid[2:, 0])
  ax_spectra_mag      = fig.add_subplot(fig_grid[0, 1])
  ax_spectra_vel_lgt  = fig.add_subplot(fig_grid[1, 1])
  ax_spectra_vel_trv  = fig.add_subplot(fig_grid[2, 1])
  axs_spectra         = [
    ax_spectra_mag,
    ax_spectra_vel_lgt,
    ax_spectra_vel_trv
  ]
  ## reynolds spectra
  ax_reynolds_mag     = fig.add_subplot(fig_grid[0, 2])
  ax_reynolds_vel_lgt = fig.add_subplot(fig_grid[1, 2])
  ax_reynolds_vel_trv = fig.add_subplot(fig_grid[2, 2])
  axs_reynolds        = [
    ax_reynolds_mag,
    ax_reynolds_vel_lgt,
    ax_reynolds_vel_trv
  ]
  ## measured scales
  axs_scales = PlotFuncs.addSubplot_secondAxis(
    fig         = fig,
    grid_elem   = fig_grid[3, 1:],
    shared_axis = "y"
  )
  ## PLOT INTEGRATED QUANTITIES
  ## --------------------------
  obj_plot_turb = PlotTurbData(
    fig              = fig,
    axs              = [ ax_Mach, ax_energy_ratio ],
    filepath_sim_res = filepath_sim_res,
    dict_sim_inputs  = dict_sim_inputs,
    bool_verbose     = bool_verbose
  )
  obj_plot_turb.performRoutines()
  obj_plot_turb.saveFittedParams(filepath_sim_res)
  dict_turb_params = obj_plot_turb.getFittedParams()
  ## PLOT FITTED SPECTRA
  ## -------------------
  obj_plot_spectra = PlotSpectra(
    fig             = fig,
    dict_axs        = {
      "axs_spectra"      : axs_spectra,
      "axs_reynolds"     : axs_reynolds,
      "ax_spectra_ratio" : ax_spectra_ratio,
      "axs_scales"       : axs_scales,
    },
    dict_sim_inputs = dict_sim_inputs,
    filepath_spect  = f"{filepath_sim_res}/spect/",
    time_exp_start  = dict_turb_params["time_growth_start"],
    time_exp_end    = dict_turb_params["time_growth_end"],
    plots_per_eddy  = dict_turb_params["plots_per_eddy"],
    bool_verbose    = bool_verbose
  )
  obj_plot_spectra.performRoutines()
  ## SAVE FIGURE + DATASET
  ## ---------------------
  if lock is not None: lock.acquire()
  obj_plot_spectra.saveFittedParams(filepath_sim_res)
  suite_folder = dict_sim_inputs["suite_folder"]
  sim_folder   = dict_sim_inputs["sim_folder"]
  fig_name = f"{suite_folder}_{sim_folder}_dataset.png"
  PlotFuncs.saveFigure(fig, f"{filepath_sim_res}/vis_folder/{fig_name}", bool_verbose=True)
  if lock is not None: lock.release()
  if bool_verbose: print(" ")


## ###############################################################
## CREATE LIST OF SIMULATION DIRECTORIES TO ANALYSE
## ###############################################################
def getListSimFolders():
  list_sim_filepaths = []
  ## LOOK AT EACH SIMULATION SUITE
  ## -----------------------------
  for suite_folder in LIST_SUITE_FOLDER:
    ## LOOK AT EACH SIMULATION FOLDER
    ## -----------------------------
    for sim_folder in LIST_SIM_FOLDER:
      ## CHECK THE SUITE + SIMULATION CONFIG EXISTS
      ## ------------------------------------------
      filepath_sim = WWFnF.createFilepath([
        BASEPATH, suite_folder, SONIC_REGIME, sim_folder
      ])
      if not os.path.exists(filepath_sim): continue
      ## loop over the different resolution runs
      for sim_res in LIST_SIM_RES:
        ## CHECK THE RESOLUTION RUN EXISTS
        ## -------------------------------
        filepath_sim_res = f"{filepath_sim}/{sim_res}/"
        if not os.path.exists(filepath_sim_res): continue
        list_sim_filepaths.append(filepath_sim_res)
        ## MAKE SURE A VISUALISATION FOLDER EXISTS
        ## ---------------------------------------
        WWFnF.createFolder(f"{filepath_sim_res}/vis_folder", bool_verbose=False)
  return list_sim_filepaths


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  list_sim_filepaths = getListSimFolders()
  if BOOL_MPROC:
    with cfut.ProcessPoolExecutor() as executor:
      manager = mproc.Manager()
      lock = manager.Lock()
      ## loop over all simulation folders
      futures = [
        executor.submit(
          functools.partial(plotSimData, lock=lock, bool_verbose=False),
          sim_filepath
        ) for sim_filepath in list_sim_filepaths
      ]
      ## wait to ensure that all scheduled and running tasks have completed
      cfut.wait(futures)
      ## check if any tasks failed
      for future in cfut.as_completed(futures):
        future.result()
  else: [
    plotSimData(sim_filepath, bool_verbose=True)
    for sim_filepath in list_sim_filepaths
  ]


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
BOOL_MPROC        = 0
BASEPATH          = "/scratch/ek9/nk7952/"
SONIC_REGIME      = "super_sonic"

LIST_SUITE_FOLDER = [ "Re10", "Re500", "Rm3000" ]
# LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
# LIST_SIM_RES      = [ "18", "36", "72", "144", "288", "576" ]

LIST_SUITE_FOLDER = [ "Rm3000" ]
LIST_SIM_FOLDER   = [ "Pm25", "Pm50" ]
LIST_SIM_RES      = [ "576" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
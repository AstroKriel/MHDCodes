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
from TheUsefulModule import WWLists, WWFnF, WWObjs
from TheSimModule import SimParams
from TheLoadingModule import LoadFlashData
from TheAnalysisModule import WWSpectra
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
def getScaleAdj_knu(fit_params):
  return fit_params[3]**(1 / fit_params[2])

def getLabel_kin(fit_params_group_t):
  label_A          = PlotLatex.GetLabel.percentiles(WWLists.getElemFromLoL(fit_params_group_t, 0))
  label_alpha_cas  = PlotLatex.GetLabel.percentiles(WWLists.getElemFromLoL(fit_params_group_t, 1))
  label_alpha_dis  = PlotLatex.GetLabel.percentiles(WWLists.getElemFromLoL(fit_params_group_t, 2))
  label_k_nu       = PlotLatex.GetLabel.percentiles(WWLists.getElemFromLoL(fit_params_group_t, 3))
  label_k_nu_alpha = PlotLatex.GetLabel.percentiles([
    getScaleAdj_knu(fit_params)
    for fit_params in fit_params_group_t
  ])
  return  r"$A = $ " + label_A + \
          r", $\alpha_{\rm cas} = $ " + label_alpha_cas + \
          r", $\alpha_{\rm dis} = $ " + label_alpha_dis + \
          r", $k_\nu = $ " + label_k_nu + \
          r", $k_\nu^{1 / \alpha_{\rm dis}} = $ " + label_k_nu_alpha

def getLabel_mag(k_p_group_t, k_max_group_t):
  label_k_p   = PlotLatex.GetLabel.percentiles(k_p_group_t)
  label_k_max = PlotLatex.GetLabel.percentiles(k_max_group_t)
  return r"$k_{\rm p} = $ " + label_k_p + r", $k_{\rm max} = $ " + label_k_max

def plotMeasuredScales(
    ax_spectra, ax_scales,
    list_t, scale_group_t, scale_ave,
    color       = "black",
    label       = ""
  ):
  ## plot average scale to spectrum
  ax_spectra.axvline(x=scale_ave, color=color, ls="--", lw=1.5, zorder=7)
  ## plot time evolution of scale
  ax_scales.plot(list_t, scale_group_t, color=color, ls="-", label=label)
  ax_scales.axhline(y=scale_ave, color=color, ls="--", lw=1.5, zorder=7)

def plotSpectra(ax, list_k, list_power_group_t, color, cmap_name, list_times):
  args_plot_ave  = { "color":color, "marker":"o", "ms":8, "zorder":5, "markeredgecolor":"black" }
  args_plot_time = { "ls":"-", "lw":1, "alpha":0.5, "zorder":3 }
  ## plot time averaged, normalised energy spectra
  ax.plot(
    list_k,
    WWSpectra.aveSpectra(list_power_group_t, bool_norm=True),
    ls="", **args_plot_ave
  )
  ## create colormaps for time-evolving energy spectra (color as a function of time)
  cmap_kin, norm_kin = PlotFuncs.createCmap(
    cmap_name = cmap_name,
    vmin      = min(list_times),
    vmax      = max(list_times)
  )
  ## plot each time realisation of the normalised kinetic energy spectrum
  for time_index, time_val in enumerate(list_times):
    ax.plot(
      list_k,
      WWSpectra.normSpectra(list_power_group_t[time_index]),
      color=cmap_kin(norm_kin(time_val)), **args_plot_time
    )


## ###############################################################
## OPERATOR CLASS: PLOT NORMALISED + TIME-AVERAGED ENERGY SPECTRA
## ###############################################################
class PlotSpectra():
  def __init__(
      self,
      fig, dict_axs,
      dict_sim_inputs,
      filepath_spect, time_exp_start, time_exp_end,
      bool_verbose = True
    ):
    ## save input arguments
    self.fig              = fig
    self.axs_spectra      = dict_axs["axs_spectra"]
    self.axs_scales       = dict_axs["axs_scales"]
    self.ax_residuals     = dict_axs["ax_residuals"]
    self.ax_spectra_ratio = dict_axs["ax_spectra_ratio"]
    self.dict_sim_inputs  = dict_sim_inputs
    self.filepath_spect   = filepath_spect
    self.time_exp_start   = time_exp_start
    self.time_exp_end     = time_exp_end
    self.bool_verbose     = bool_verbose
    ## initialise spectra labels
    self.__initialiseQuantities()
    self.color_k_nu = "orange"
    self.color_k_eq = "black"
    self.dict_plot_kin_trv = {
      "color"       : "darkgreen",
      "cmap_name"   : "Greens",
      "label_spect" : PlotLatex.GetLabel.spectrum("kin", "trv"),
      "label_knu"   : r"$k_{\nu, \perp}^{\alpha_{\rm dis}}$",
    }
    self.dict_plot_mag_tot = {
      "color"       : "red",
      "cmap_name"   : "Reds",
      "label_spect" : PlotLatex.GetLabel.spectrum("mag", "tot"),
      "label_kp"    : r"$k_{\rm p}$",
    }

  def performRoutines(self):
    self.__loadSpectra_kinematicPhase()
    self.__plotSpectra_kinematicPhase()
    self.__plotSpectraRatio()
    if self.bool_verbose: print("Fitting energy spectra...")
    self.__fitKinSpectra()
    self.__fitMagSpectra()
    self.bool_fitted = True
    self.__labelSpectra()
    self.__labelResiduals()
    self.__labelSpectraRatio()
    self.__labelScales()

  def getFittedParams(self):
    self.__checkAnyQuantitiesNotMeasured()
    if not self.bool_fitted: self.performRoutines()
    return {
      ## time-averaged energy spectra
      "list_k"                     : self.list_k,
      "list_mag_power_tot_ave"     : WWSpectra.aveSpectra(self.list_mag_power_tot_group_t, bool_norm=True),
      "list_kin_power_tot_ave"     : WWSpectra.aveSpectra(self.list_kin_power_tot_group_t, bool_norm=True),
      "list_kin_power_lgt_ave"     : WWSpectra.aveSpectra(self.list_kin_power_lgt_group_t, bool_norm=True),
      "list_kin_power_trv_ave"     : WWSpectra.aveSpectra(self.list_kin_power_trv_group_t, bool_norm=True),
      ## measured quantities
      "plots_per_eddy"             : self.plots_per_eddy,
      "list_time_growth"           : self.list_time_growth,
      "list_time_k_eq"             : self.list_time_k_eq,
      "k_nu_adj_trv_group_t"       : self.k_nu_adj_trv_group_t,
      "k_nu_adj_trv_group_t_fixed" : self.k_nu_adj_trv_group_t_fixed,
      "k_p_group_t"                : self.k_p_group_t,
      "k_eq_group_t"               : self.k_eq_group_t,
      "fit_params_kin_trv_group_t" : self.fit_params_kin_trv_group_t,
      "fit_params_kin_trv_ave"     : self.fit_params_kin_trv_ave,
    }

  def saveFittedParams(self, filepath_sim):
    dict_params = self.getFittedParams()
    WWObjs.saveDict2JsonFile(f"{filepath_sim}/sim_outputs.json", dict_params, self.bool_verbose)

  def __initialiseQuantities(self):
    ## flag to check that all required quantities have been measured
    self.bool_fitted                = False
    ## initialise quantities to measure
    self.list_k                     = None
    self.list_mag_power_tot_group_t = None
    self.list_kin_power_tot_group_t = None # TODO: rename list_power_(field)_(sub)_group_t
    self.list_kin_power_lgt_group_t = None
    self.list_kin_power_trv_group_t = None
    self.plots_per_eddy             = None
    self.list_time_growth           = None
    self.list_time_k_eq             = None
    self.k_nu_adj_trv_group_t       = None
    self.k_nu_adj_trv_group_t_fixed = None
    self.k_p_group_t                = None
    self.k_eq_group_t               = None
    self.fit_params_kin_trv_group_t = None
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
      self.k_nu_adj_trv_group_t,
      self.k_nu_adj_trv_group_t_fixed,
      self.k_p_group_t,
      self.k_eq_group_t,
      self.fit_params_kin_trv_group_t,
      self.fit_params_kin_trv_ave
    ]
    list_quantities_undefined = [ 
      index_quantity
      for index_quantity, quantity in enumerate(list_quantities_check)
      if quantity is None
    ]
    if len(list_quantities_undefined) > 0: raise Exception("Error: the following quantities were not measured:", list_quantities_undefined)

  def __loadSpectra_kinematicPhase(self):
    if self.bool_verbose: print("Loading energy spectra...")
    ## extract the number of plt-files per eddy-turnover-time from 'Turb.log'
    self.plots_per_eddy = LoadFlashData.getPlotsPerEddy_fromTurbLog(
      filepath     = f"{self.filepath_spect}/../",
      bool_verbose = self.bool_verbose
    )
    ## load spectra data within the growth phase of the dynamo
    ## load total kinetic energy spectra
    dict_kin_spect_tot_data = LoadFlashData.loadAllSpectraData(
      filepath        = self.filepath_spect,
      spect_field     = "vel",
      spect_quantity  = "tot",
      file_start_time = self.time_exp_start,
      file_end_time   = self.time_exp_end,
      plots_per_eddy  = self.plots_per_eddy,
      bool_verbose    = self.bool_verbose
    )
    ## load longitudinal kinetic energy spectra
    dict_kin_spect_lgt_data = LoadFlashData.loadAllSpectraData(
      filepath        = self.filepath_spect,
      spect_field     = "vel",
      spect_quantity  = "lgt",
      file_start_time = self.time_exp_start,
      file_end_time   = self.time_exp_end,
      plots_per_eddy  = self.plots_per_eddy,
      bool_verbose    = self.bool_verbose
    )
    ## load transverse kinetic energy spectra
    dict_kin_spect_trv_data = LoadFlashData.loadAllSpectraData(
      filepath        = self.filepath_spect,
      spect_field     = "vel",
      spect_quantity  = "trv",
      file_start_time = self.time_exp_start,
      file_end_time   = self.time_exp_end,
      plots_per_eddy  = self.plots_per_eddy,
      bool_verbose    = self.bool_verbose
    )
    ## load total magnetic energy spectra
    dict_mag_spect_tot_data = LoadFlashData.loadAllSpectraData(
      filepath        = self.filepath_spect,
      spect_field     = "mag",
      spect_quantity  = "tot",
      file_start_time = self.time_exp_start,
      file_end_time   = self.time_exp_end,
      plots_per_eddy  = self.plots_per_eddy,
      bool_verbose    = self.bool_verbose
    )
    ## store time-evolving energy spectra
    self.list_kin_power_tot_group_t = dict_kin_spect_tot_data["list_power_group_t"]
    self.list_kin_power_lgt_group_t = dict_kin_spect_lgt_data["list_power_group_t"]
    self.list_kin_power_trv_group_t = dict_kin_spect_trv_data["list_power_group_t"]
    self.list_mag_power_tot_group_t = dict_mag_spect_tot_data["list_power_group_t"]
    self.list_k                     = dict_mag_spect_tot_data["list_k_group_t"][0]
    self.list_time_growth           = dict_mag_spect_tot_data["list_sim_times"]

  def __plotSpectra_kinematicPhase(self):
    plotSpectra(
      ax                 = self.axs_spectra[0],
      list_k             = self.list_k,
      list_power_group_t = self.list_kin_power_trv_group_t,
      color              = self.dict_plot_kin_trv["color"],
      cmap_name          = self.dict_plot_kin_trv["cmap_name"],
      list_times         = self.list_time_growth
    )
    plotSpectra(
      ax                 = self.axs_spectra[1],
      list_k             = self.list_k,
      list_power_group_t = self.list_mag_power_tot_group_t,
      color              = self.dict_plot_mag_tot["color"],
      cmap_name          = self.dict_plot_mag_tot["cmap_name"],
      list_times         = self.list_time_growth
    )

  def __plotSpectraRatio(self):
    ## load spectra data again, this time for the full duration of the simulation
    ## load total kinetic energy spectra
    dict_kin_spect_tot_data = LoadFlashData.loadAllSpectraData(
      filepath        = self.filepath_spect,
      spect_field     = "vel",
      spect_quantity  = "tot",
      file_start_time = self.time_exp_start,
      file_end_time   = np.inf,
      plots_per_eddy  = self.plots_per_eddy,
      bool_verbose    = self.bool_verbose
    )
    ## load total magnetic energy spectra
    dict_mag_spect_tot_data = LoadFlashData.loadAllSpectraData(
      filepath        = self.filepath_spect,
      spect_field     = "mag",
      spect_quantity  = "tot",
      file_start_time = self.time_exp_start,
      file_end_time   = np.inf,
      plots_per_eddy  = self.plots_per_eddy,
      bool_verbose    = self.bool_verbose
    )
    ## measure + plot evolving equipartition scale
    self.k_eq_group_t, self.k_eq_power_group_t, self.list_time_k_eq = FitMHDScales.getScale_keq(
      ax_spectra             = self.ax_spectra_ratio,
      ax_scales              = self.axs_scales[1],
      list_sim_time          = dict_mag_spect_tot_data["list_sim_times"],
      list_k                 = dict_mag_spect_tot_data["list_k_group_t"][0],
      list_mag_power_group_t = dict_mag_spect_tot_data["list_power_group_t"],
      list_kin_power_group_t = dict_kin_spect_tot_data["list_power_group_t"],
      color                  = self.color_k_eq
    )

  def __fitKinSpectra(self):
    ## define helper function
    def fitKinSpectra(
        ax_fit, ax_residuals, ax_scales,
        list_k, list_power_group_t, list_time_growth,
        bool_fix_cascade = False,
        color_fit        = "black",
        label_spect      = "",
        label_knu        = r"$k_\nu$"
      ):
      fitMethod = FitMHDScales.fitKinSpectrum
      ## fit each time-realisation of the kinetic energy spectrum
      k_nu_adj_group_t   = []
      fit_params_group_t = []
      for time_index in range(len(list_time_growth)):
        fit_params_kin = fitMethod(
          list_k           = list_k,
          list_power       = WWSpectra.normSpectra(list_power_group_t[time_index]),
          bool_fix_cascade = bool_fix_cascade # self.dict_sim_inputs["Re"] > 100
        )
        ## store fitted parameters
        k_nu_adj_group_t.append(getScaleAdj_knu(fit_params_kin))
        fit_params_group_t.append(fit_params_kin)
      ## plot fit to time-averaged kinetic energy spectrum
      fit_params_ave = fitMethod(
        ax_fit           = ax_fit,
        ax_residuals     = ax_residuals,
        list_k           = list_k,
        list_power       = WWSpectra.aveSpectra(list_power_group_t, bool_norm=True),
        color            = color_fit,
        label_spect      = label_spect,
        bool_fix_cascade = bool_fix_cascade # self.dict_sim_inputs["Re"] > 100
      )
      ## plot measured dissipation scale
      plotMeasuredScales(
        ax_spectra    = ax_fit,
        ax_scales     = ax_scales,
        list_t        = list_time_growth,
        scale_group_t = k_nu_adj_group_t,
        scale_ave     = getScaleAdj_knu(fit_params_ave),
        color         = color_fit,
        label         = label_knu
      )
      return k_nu_adj_group_t, fit_params_group_t, fit_params_ave
    ## fit transverse kinetic spectrum
    ## power law cascade: free variable
    self.k_nu_adj_trv_group_t, \
      self.fit_params_kin_trv_group_t, \
        self.fit_params_kin_trv_ave = fitKinSpectra(
      ax_fit              = self.axs_spectra[0],
      ax_residuals        = self.ax_residuals,
      ax_scales           = self.axs_scales[0],
      list_k              = self.list_k,
      list_power_group_t  = self.list_kin_power_trv_group_t,
      list_time_growth    = self.list_time_growth,
      color_fit           = self.color_k_nu,
      label_spect         = self.dict_plot_kin_trv["label_spect"],
      label_knu           = self.dict_plot_kin_trv["label_knu"],
      bool_fix_cascade    = False
    )
    ## power law cascade: fixed -2.0
    self.k_nu_adj_trv_group_t_fixed, \
      self.fit_params_kin_trv_group_t_fixed, \
        self.fit_params_kin_trv_ave_fixed = fitKinSpectra(
      ax_fit              = self.axs_spectra[0],
      ax_residuals        = self.ax_residuals,
      ax_scales           = self.axs_scales[0],
      list_k              = self.list_k,
      list_power_group_t  = self.list_kin_power_trv_group_t,
      list_time_growth    = self.list_time_growth,
      color_fit           = "blue",
      label_spect         = self.dict_plot_kin_trv["label_spect"] + " fixed",
      label_knu           = self.dict_plot_kin_trv["label_knu"]   + " fixed",
      bool_fix_cascade    = True
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
    ## annotate measured raw spectrum maximum
    self.axs_spectra[1].plot(
      np.mean(self.k_max_group_t),
      np.mean(np.max(WWSpectra.normSpectra_grouped(self.list_mag_power_tot_group_t), axis=1)),
      color="black", marker="o", ms=8, ls="", label=r"$k_{\rm max}$", zorder=7
    )
    ## plot time-evolution of measured scale
    plotMeasuredScales(
      ax_spectra    = self.axs_spectra[1],
      ax_scales     = self.axs_scales[0],
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

  def __labelSpectra(self):
    self.__adjustAxis(self.axs_spectra[0])
    self.__adjustAxis(self.axs_spectra[1])
    PlotFuncs.labelDualAxis_sharedX(
      axs         = self.axs_spectra,
      label_left  = self.dict_plot_kin_trv["label_spect"],
      label_right = self.dict_plot_mag_tot["label_spect"],
      color_left  = self.dict_plot_kin_trv["color"],
      color_right = self.dict_plot_mag_tot["color"]
    )
    PlotFuncs.addBoxOfLabels(
      fig         = self.fig,
      ax          = self.axs_spectra[0],
      bbox        = (0.5, 0.0),
      xpos        = 0.5,
      ypos        = 1.05,
      alpha       = 0.85,
      fontsize    = 18,
      list_colors = [ "black", "black" ],
      list_labels = [
        getLabel_kin(self.fit_params_kin_trv_group_t),
        getLabel_kin(self.fit_params_kin_trv_group_t_fixed),
        getLabel_mag(self.k_p_group_t, self.k_max_group_t)
      ],
    )

  def __labelResiduals(self):
    self.__adjustAxis(self.ax_residuals, bool_log_y=False)
    self.ax_residuals.axhline(y=1, color="black", ls="--")
    PlotFuncs.addLegend_withBox(
      ax   = self.ax_residuals,
      loc  = "lower left",
      bbox = (0.0, 0.0)
    )
    self.ax_residuals.set_xlabel(r"$k$")
    self.ax_residuals.set_ylabel(
      PlotLatex.GetLabel.timeAve(
        PlotLatex.GetLabel.spectrum("fit") + r"$\, / \,$" + PlotLatex.GetLabel.spectrum("data")
    ))

  def __labelSpectraRatio(self):
    self.ax_spectra_ratio.axhline(y=1, color="black", ls="--")
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
      axs      = self.axs_scales,
      loc      = "upper right",
      bbox     = (1.0, 1.0)
    )
    self.axs_scales[0].set_ylim([
      0.9 * np.nanmin([
        min(self.list_k),
        min(self.k_nu_adj_trv_group_t),
        min(self.k_nu_adj_trv_group_t_fixed),
        min(self.k_p_group_t),
        min(self.k_eq_group_t) if len(self.k_eq_group_t) > 0 else np.nan
      ]),
      1.1 * np.nanmax([
        max(self.list_k),
        max(self.k_nu_adj_trv_group_t),
        max(self.k_nu_adj_trv_group_t_fixed),
        max(self.k_p_group_t),
        max(self.k_eq_group_t) if len(self.k_eq_group_t) > 0 else np.nan
      ])
    ])


## ###############################################################
## HANDLING PLOT CALLS
## ###############################################################
def plotSimData(
    filepath_sim_res, filepath_vis, sim_name,
    lock         = None,
    bool_verbose = True
  ):
  dict_sim_inputs = SimParams.readSimInputs(filepath_sim_res)
  ## INITIALISE FIGURE
  ## -----------------
  if bool_verbose: print("Initialising figure...")
  fig, fig_grid = PlotFuncs.createFigure_grid(
    fig_scale        = 0.4,
    fig_aspect_ratio = (10.0, 8.0),
    num_rows         = 3,
    num_cols         = 6
  )
  ## volume integrated qunatities
  ax_Mach         = fig.add_subplot(fig_grid[0, 0:2])
  ax_energy_ratio = fig.add_subplot(fig_grid[1, 0:2])
  ## spectra data
  ax_residuals = fig.add_subplot(fig_grid[2, 0:3])
  axs_spectra  = PlotFuncs.addSubplot_secondAxis(
    fig         = fig,
    grid_elem   = fig_grid[:2, 2:4],
    shared_axis = "x"
  )
  ax_spectra_ratio = fig.add_subplot(fig_grid[0:2, 4:6])
  axs_scales = PlotFuncs.addSubplot_secondAxis(
    fig         = fig,
    grid_elem   = fig.add_subplot(fig_grid[  2, 3:6]),
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
  if lock is not None: lock.acquire()
  obj_plot_turb.saveFittedParams(filepath_sim_res)
  if lock is not None: lock.release()
  dict_turb_params = obj_plot_turb.getFittedParams()
  ## PLOT FITTED SPECTRA
  ## -------------------
  obj_plot_spectra = PlotSpectra(
    fig              = fig,
    dict_axs         = {
      "axs_spectra"      : axs_spectra,
      "axs_scales"       : axs_scales,
      "ax_residuals"     : ax_residuals,
      "ax_spectra_ratio" : ax_spectra_ratio,
    },
    dict_sim_inputs  = dict_sim_inputs,
    filepath_spect   = f"{filepath_sim_res}/spect/",
    time_exp_start   = dict_turb_params["time_growth_start"],
    time_exp_end     = dict_turb_params["time_growth_end"],
    bool_verbose     = bool_verbose
  )
  obj_plot_spectra.performRoutines()
  ## SAVE FIGURE + DATASET
  ## ---------------------
  if lock is not None: lock.acquire()
  obj_plot_spectra.saveFittedParams(filepath_sim_res)
  fig_name = f"{sim_name}_dataset.png"
  PlotFuncs.saveFigure(fig, f"{filepath_vis}/{fig_name}", bool_verbose)
  if lock is not None: lock.release()


## ###############################################################
## HANDLE LOOPING OVER SIMULATION SUITES AND RESOLUTIONS
## ###############################################################
def loopOverSuitesNres(sim_folder, lock=None, bool_verbose=True):
  ## LOOK AT EACH SIMULATION SUITE
  ## -----------------------------
  ## loop over the simulation suites
  for suite_folder in LIST_SUITE_FOLDER:
    ## CHECK THE SIMULATION EXISTS IN THE SUITE
    ## ----------------------------------------
    filepath_sim = WWFnF.createFilepath([
      BASEPATH, suite_folder, SONIC_REGIME, sim_folder
    ])
    if not os.path.exists(filepath_sim): continue
    str_message = f"Looking at suite: {suite_folder}, sim: {sim_folder}, regime: {SONIC_REGIME}"
    if bool_verbose:
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
      if BOOL_MPROC: print(str_message + f", res: {sim_res}")
      ## MAKE SURE A VISUALISATION FOLDER EXISTS
      ## ---------------------------------------
      filepath_sim_res_plot = f"{filepath_sim_res}/vis_folder"
      WWFnF.createFolder(filepath_sim_res_plot, bool_verbose=False)
      ## PLOT SIMULATION DATA AND SAVE MEASURED QUANTITIES
      ## -------------------------------------------------
      sim_name = f"{suite_folder}_{sim_folder}"
      plotSimData(filepath_sim_res, filepath_sim_res_plot, sim_name, lock, bool_verbose)
      ## create trailing empty space
      if bool_verbose: print(" ")
    if bool_verbose: print(" ")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  if BOOL_MPROC:
    with cfut.ProcessPoolExecutor() as executor:
      manager = mproc.Manager()
      lock = manager.Lock()
      ## loop over all simulation folders
      futures = [
        executor.submit(
          functools.partial(loopOverSuitesNres, bool_verbose=False),
          sim_folder, lock
        ) for sim_folder in LIST_SIM_FOLDER
      ]
      ## wait to ensure that all scheduled and running tasks have completed
      cfut.wait(futures)
      ## check if any tasks failed
      for future in cfut.as_completed(futures):
        future.result()
  else: [
    loopOverSuitesNres(sim_folder, bool_verbose=True)
    for sim_folder in LIST_SIM_FOLDER
  ]


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
BOOL_MPROC        = 1
BASEPATH          = "/scratch/ek9/nk7952/"
SONIC_REGIME      = "super_sonic"

LIST_SUITE_FOLDER = [ "Re10", "Re500", "Rm3000" ]
LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
LIST_SIM_RES      = [ "18", "36", "72", "144", "288", "576" ]

# LIST_SUITE_FOLDER = [ "Rm3000" ]
# LIST_SIM_FOLDER   = [ "Pm2", "Pm25", "Pm50", "Pm125", "Pm250" ]
# LIST_SIM_RES      = [ "288" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
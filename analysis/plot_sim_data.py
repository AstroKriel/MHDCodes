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

## load user defined routines
from plot_vi_data import PlotTurbData

## load user defined modules
from TheFlashModule import SimParams, LoadFlashData, FileNames
from TheUsefulModule import WWLists, WWFnF, WWObjs
from TheFittingModule import FitMHDScales, FitFuncs
from TheAnalysisModule import WWSpectra
from ThePlottingModule import PlotFuncs, PlotLatex


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def plotScale(
    list_turb_times, scale_group_t, color_scale, label_scale,
    index_start_growth = None,
    index_end_growth   = None,
    index_start_sat    = None,
    ax_scales          = None,
    ax_spectra         = None,
    ax_reynolds        = None
  ):
  if len(scale_group_t) < 5: return
  args_plot = { "color":color_scale, "zorder":1, "lw":2.0 }
  plot_scale_group_t = WWLists.replaceNoneWNan(scale_group_t)
  ## plot time evolution of scale
  ax_scales.plot(list_turb_times, plot_scale_group_t, color=color_scale, ls="-", zorder=3, lw=1.5, label=label_scale)
  ## show average scale in growth regime
  if (index_start_growth is not None) and (index_end_growth is not None):
    scale_ave_growth = np.mean(plot_scale_group_t[index_start_growth : index_end_growth])
    if ax_spectra is not None:  ax_spectra.axvline(x=scale_ave_growth,  ls="--", **args_plot)
    if ax_reynolds is not None: ax_reynolds.axvline(x=scale_ave_growth, ls="--", **args_plot)
    if ax_scales is not None:   ax_scales.axhline(y=scale_ave_growth,   ls="--", **args_plot, alpha=0.5)
  ## show average scale in saturated regime
  if index_start_sat is not None:
    scale_ave_sat = np.mean(plot_scale_group_t[index_start_sat : ])
    if ax_spectra is not None:  ax_spectra.axvline(x=scale_ave_sat,  ls=":", **args_plot)
    if ax_reynolds is not None: ax_reynolds.axvline(x=scale_ave_sat, ls=":", **args_plot)
    if ax_scales is not None:   ax_scales.axhline(y=scale_ave_sat,   ls=":", **args_plot, alpha=0.5)

def plotSpectra(ax, list_k, list_power_group_t, cmap_name, bool_norm=False):
  ## create colormap
  cmap, norm = PlotFuncs.createCmap(
    cmap_name,
    vmin = 0,
    vmax = len(list_power_group_t)
  )
  ## plot each spectra realisation
  for index, list_power in enumerate(list_power_group_t):
    if bool_norm: list_power = WWSpectra.normSpectra(list_power)
    ax.plot(list_k, list_power, color=cmap(norm(index)), ls="-", lw=1.0, alpha=0.5, zorder=1)

def plotReynoldsSpectrum(ax, list_k, list_power_group_t, viscosity, cmap_name, bool_norm):
  scales_group_t = []
  ## helper function
  def reynoldsSpectrum(list_power):
    return np.sqrt(
      np.cumsum( np.array(list_power[::-1]) )
    )[::-1] / (viscosity * np.array(list_k))
  ## create colormap
  cmap, norm = PlotFuncs.createCmap(
    cmap_name,
    vmin = 0,
    vmax = len(list_power_group_t)
  )
  ## plot each reynolds spectrum realisation
  for index, list_power in enumerate(list_power_group_t):
    ## plot reynolds spectrum
    if bool_norm: list_power = WWSpectra.normSpectra(list_power)
    array_reynolds = reynoldsSpectrum(list_power)
    ax.plot(list_k, array_reynolds, color=cmap(norm(index)), ls="-", lw=1.0, alpha=0.5, zorder=1)
    ## measure proxy for "physical dissipation" scale if it exists
    if np.log10(min(array_reynolds)) < 1e-1:
      list_k_interp        = np.logspace(np.log10(min(list_k)), np.log10(max(list_k)), 10**4)
      list_reynolds_interp = FitFuncs.interpLogLogData(list_k, array_reynolds, list_k_interp, interp_kind="cubic")
      dis_scale_index      = np.argmin(abs(list_reynolds_interp - 1.0))
      scale                = list_k_interp[dis_scale_index]
    else: scale = None
    scales_group_t.append(scale)
  return scales_group_t


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class PlotSpectra():
  def __init__(
      self,
      fig, dict_axs, filepath_spect, dict_sim_inputs, outputs_per_t_turb, time_bounds_growth, time_start_sat,
      bool_verbose = True
    ):
    ## save input arguments
    self.fig                = fig
    self.axs_spectra        = dict_axs["axs_spectra"]
    self.axs_reynolds       = dict_axs["axs_reynolds"]
    self.ax_spectra_ratio   = dict_axs["ax_spectra_ratio"]
    self.ax_scales          = dict_axs["ax_scales"]
    self.filepath_spect     = filepath_spect
    self.dict_sim_inputs    = dict_sim_inputs
    self.outputs_per_t_turb = outputs_per_t_turb
    self.time_bounds_growth = time_bounds_growth
    self.time_start_sat     = time_start_sat
    self.bool_verbose       = bool_verbose
    ## initialise spectra labels
    self.index_mag     = 0
    self.index_vel_lgt = 1
    self.index_vel_trv = 2
    self.color_k_eq    = "black"
    self.dict_plot_mag = {
      "color_spect" : "red",
      "color_k_p"   : "darkorange",
      "color_k_eta" : "red",
      "cmap_name"   : "Reds",
      "label_spect" : PlotLatex.GetLabel.spectrum("mag", "tot"),
      "label_k_p"   : r"$k_{\rm p}$",
      "label_k_eta" : r"$k_\eta$",
    }
    self.dict_plot_vel_trv = {
      "color_spect" : "darkgreen",
      "color_k_nu"  : "darkgreen",
      "cmap_name"   : "Greens",
      "label_spect" : PlotLatex.GetLabel.spectrum("vel", "trv"),
      "label_k_nu"  : r"$k_{\nu, \perp}$",
    }
    self.dict_plot_vel_lgt = {
      "color_spect" : "royalblue",
      "color_k_nu"  : "royalblue",
      "cmap_name"   : "Blues",
      "label_spect" : PlotLatex.GetLabel.spectrum("vel", "lgt"),
      "label_k_nu"  : r"$k_{\nu, \parallel}$",
    }

  def performRoutines(self):
    if self.bool_verbose: print("Loading power spectra...")
    self._loadData()
    if self.bool_verbose: print("Plotting power spectra...")
    self._plotSpectra()
    self._plotSpectraRatio()
    self._fitMagSpectra()
    self.bool_fitted = True
    self._labelSpectra()
    self._labelSpectraRatio()
    self._labelScales()

  def getFittedParams(self):
    if not self.bool_fitted: self.performRoutines()
    return {
      ## time-averaged energy spectra
      "list_k"              : self.list_k,
      "list_power_mag_tot"  : self.list_power_mag_tot_group_t[self.index_start_growth : self.index_end_growth],
      "list_power_vel_tot"  : self.list_power_vel_tot_group_t[self.index_start_growth : self.index_end_growth],
      "list_power_vel_lgt"  : self.list_power_vel_lgt_group_t[self.index_start_growth : self.index_end_growth],
      "list_power_vel_trv"  : self.list_power_vel_trv_group_t[self.index_start_growth : self.index_end_growth],
      ## measured quantities
      "index_bounds_growth" : [ self.index_start_growth, self.index_end_growth ],
      "index_start_sat"     : self.index_start_sat,
      "list_time_growth"    : self.list_time_growth,
      "list_time_eq"        : self.list_time_eq,
      "list_time_sat"       : self.list_time_sat,
      "k_nu_trv_group_t"    : self.k_nu_trv_group_t,
      "k_nu_lgt_group_t"    : self.k_nu_lgt_group_t,
      "k_eta_group_t"       : self.k_eta_group_t,
      "k_p_group_t"         : self.k_p_group_t,
      "k_eq_group_t"        : self.k_eq_group_t,
    }

  def saveFittedParams(self, filepath_sim):
    dict_params = self.getFittedParams()
    WWObjs.saveDict2JsonFile(f"{filepath_sim}/{FileNames.FILENAME_SIM_OUTPUTS}", dict_params, self.bool_verbose)

  def _loadData(self):
    ## load total magnetic energy spectra
    dict_mag_spect_tot_data = LoadFlashData.loadAllSpectra(
      filepath           = self.filepath_spect,
      spect_field        = "mag",
      spect_comp         = "tot",
      file_start_time    = self.time_bounds_growth[0],
      outputs_per_t_turb = self.outputs_per_t_turb,
      bool_verbose       = False
    )
    ## load total kinetic energy spectra
    dict_vel_spect_tot_data = LoadFlashData.loadAllSpectra(
      filepath           = self.filepath_spect,
      spect_field        = "vel",
      spect_comp         = "tot",
      file_start_time    = self.time_bounds_growth[0],
      outputs_per_t_turb = self.outputs_per_t_turb,
      bool_verbose       = False
    )
    ## load longitudinal kinetic energy spectra
    dict_vel_spect_lgt_data = LoadFlashData.loadAllSpectra(
      filepath           = self.filepath_spect,
      spect_field        = "vel",
      spect_comp         = "lgt",
      file_start_time    = self.time_bounds_growth[0],
      outputs_per_t_turb = self.outputs_per_t_turb,
      bool_verbose       = False
    )
    ## load transverse kinetic energy spectra
    dict_vel_spect_trv_data = LoadFlashData.loadAllSpectra(
      filepath           = self.filepath_spect,
      spect_field        = "vel",
      spect_comp         = "trv",
      file_start_time    = self.time_bounds_growth[0],
      outputs_per_t_turb = self.outputs_per_t_turb,
      bool_verbose       = False
    )
    ## store time realisations in growth regime
    self.list_turb_times = dict_mag_spect_tot_data["list_turb_times"]
    if None not in self.time_bounds_growth:
      self.index_start_growth = WWLists.getIndexClosestValue(self.list_turb_times, self.time_bounds_growth[0])
      self.index_end_growth   = WWLists.getIndexClosestValue(self.list_turb_times, self.time_bounds_growth[1])
      self.list_time_growth   = self.list_turb_times[self.index_start_growth : self.index_end_growth]
    else:
      self.index_start_growth = None
      self.index_end_growth   = None
    ## store time realisations in saturated regime
    self.index_start_sat = WWLists.getIndexClosestValue(self.list_turb_times, self.time_start_sat)
    self.list_time_sat   = self.list_turb_times[self.index_start_sat : ]
    ## store time-evolving energy spectra
    self.list_k                     = dict_mag_spect_tot_data["list_k_group_t"][0]
    self.list_power_mag_tot_group_t = dict_mag_spect_tot_data["list_power_group_t"]
    self.list_power_vel_tot_group_t = dict_vel_spect_tot_data["list_power_group_t"]
    self.list_power_vel_lgt_group_t = dict_vel_spect_lgt_data["list_power_group_t"]
    self.list_power_vel_trv_group_t = dict_vel_spect_trv_data["list_power_group_t"]

  def _plotSpectra(self):
    ## define helper function
    def __plot(
        ax_spectra, ax_reynolds, list_power_group_t, viscosity, cmap_name, color_scale, label_scale,
        bool_plot_growth = True,
        bool_plot_sat    = True,
        bool_norm        = True
      ):
      plotSpectra(
        ax                 = ax_spectra,
        list_k             = self.list_k,
        list_power_group_t = list_power_group_t,
        cmap_name          = cmap_name,
        bool_norm          = bool_norm
      )
      scale_group_t = plotReynoldsSpectrum(
        ax                 = ax_reynolds,
        list_k             = self.list_k,
        list_power_group_t = list_power_group_t,
        viscosity          = viscosity,
        cmap_name          = cmap_name,
        bool_norm          = bool_norm
      )
      plotScale(
        ax_scales          = self.ax_scales,
        ax_spectra         = ax_spectra,
        ax_reynolds        = ax_reynolds,
        list_turb_times    = self.list_turb_times,
        scale_group_t      = scale_group_t,
        index_start_growth = self.index_start_growth if bool_plot_growth else None,
        index_end_growth   = self.index_end_growth   if bool_plot_growth else None,
        index_start_sat    = self.index_start_sat    if bool_plot_sat    else None,
        color_scale        = color_scale,
        label_scale        = label_scale
      )
      return scale_group_t
    ## magnetic energy spectrum
    self.k_eta_group_t = __plot(
      ax_spectra         = self.axs_spectra[self.index_mag],
      ax_reynolds        = self.axs_reynolds[self.index_mag],
      list_power_group_t = self.list_power_mag_tot_group_t,
      viscosity          = self.dict_sim_inputs["eta"],
      cmap_name          = self.dict_plot_mag["cmap_name"],
      color_scale        = self.dict_plot_mag["color_k_eta"],
      label_scale        = self.dict_plot_mag["label_k_eta"],
      bool_plot_growth   = True,
      bool_plot_sat      = True,
      bool_norm          = True
    )
    ## longitudinal kinetic energy spectrum
    self.k_nu_lgt_group_t = __plot(
      ax_spectra         = self.axs_spectra[self.index_vel_lgt],
      ax_reynolds        = self.axs_reynolds[self.index_vel_lgt],
      list_power_group_t = self.list_power_vel_lgt_group_t,
      viscosity          = self.dict_sim_inputs["nu"],
      cmap_name          = self.dict_plot_vel_lgt["cmap_name"],
      color_scale        = self.dict_plot_vel_lgt["color_k_nu"],
      label_scale        = self.dict_plot_vel_lgt["label_k_nu"],
      bool_plot_growth   = True,
      bool_plot_sat      = True,
      bool_norm          = False
    )
    ## transverse kinetic energy spectrum
    self.k_nu_trv_group_t = __plot(
      ax_spectra         = self.axs_spectra[self.index_vel_trv],
      ax_reynolds        = self.axs_reynolds[self.index_vel_trv],
      list_power_group_t = self.list_power_vel_trv_group_t,
      viscosity          = self.dict_sim_inputs["nu"],
      cmap_name          = self.dict_plot_vel_trv["cmap_name"],
      color_scale        = self.dict_plot_vel_trv["color_k_nu"],
      label_scale        = self.dict_plot_vel_trv["label_k_nu"],
      bool_plot_growth   = True,
      bool_plot_sat      = True,
      bool_norm          = False
    )

  def _plotSpectraRatio(self):
    ## measure + plot evolving equipartition scale
    self.k_eq_group_t, _, self.list_time_eq = FitMHDScales.getScale_keq(
      ax_spectra             = self.ax_spectra_ratio,
      ax_scales              = self.ax_scales,
      list_times             = self.list_turb_times,
      list_k                 = self.list_k,
      list_power_mag_group_t = self.list_power_mag_tot_group_t,
      list_power_kin_group_t = self.list_power_vel_tot_group_t,
      color                  = self.color_k_eq
    )

  def _fitMagSpectra(self):
    self.k_p_group_t   = []
    self.k_max_group_t = []
    ## fit each time-realisation of the magnetic energy spectrum
    for time_index in range(len(self.list_turb_times)):
      k_p, k_max = FitMHDScales.getScale_kp(
        self.list_k,
        WWSpectra.normSpectra(self.list_power_mag_tot_group_t[time_index])
      )
      ## store measured scales
      self.k_p_group_t.append(k_p)
      self.k_max_group_t.append(k_max)
    ## plot time-evolution of peak scale
    plotScale(
      ax_scales          = self.ax_scales,
      ax_spectra         = self.axs_spectra[self.index_mag],
      list_turb_times     = self.list_turb_times,
      scale_group_t      = self.k_p_group_t,
      index_start_growth = self.index_start_growth,
      index_end_growth   = self.index_end_growth,
      index_start_sat    = self.index_start_sat,
      color_scale        = self.dict_plot_mag["color_k_p"],
      label_scale        = self.dict_plot_mag["label_k_p"]
    )

  def _adjustAxis(self, ax, bool_log_y=True):
    ax.set_xlim([ 0.9, 1.2*max(self.list_k) ])
    ax.set_xscale("log")
    if bool_log_y: ax.set_yscale("log")

  def _labelSpectra(self):
    def getArtists(list_scales, label, color):
      list_colors = []
      list_labels = []
      if (self.index_start_growth is not None) and (self.index_end_growth is not None):
        ## append things for growth stats
        scale_growth = PlotLatex.GetLabel.modes(list_scales[self.index_start_growth : self.index_end_growth])
        list_colors.append(color)
        list_labels.append("{" + label + r"}$_{,{\rm growth}}$ = " + scale_growth)
      ## append things for saturated stats
      scale_sat = PlotLatex.GetLabel.modes(list_scales[self.index_start_sat : ])
      list_colors.append(color)
      list_labels.append("{" + label + r"}$_{,{\rm sat}}$ = " + scale_sat)
      return {
        "list_colors" : list_colors,
        "list_labels" : list_labels
      }
    ## label spectra axis
    self.axs_spectra[-1].set_xlabel(r"$k$")
    self.axs_spectra[self.index_mag].set_ylabel(self.dict_plot_mag["label_spect"])
    self.axs_spectra[self.index_vel_lgt].set_ylabel(self.dict_plot_vel_lgt["label_spect"])
    self.axs_spectra[self.index_vel_trv].set_ylabel(self.dict_plot_vel_trv["label_spect"])
    self._adjustAxis(self.axs_spectra[self.index_mag])
    self._adjustAxis(self.axs_spectra[self.index_vel_lgt])
    self._adjustAxis(self.axs_spectra[self.index_vel_trv])
    ## label reynolds axis
    self.axs_reynolds[-1].set_xlabel(r"$k$")
    self.axs_reynolds[self.index_mag].set_ylabel(r"${\rm Rm}_{\rm tot}(k)$")
    self.axs_reynolds[self.index_vel_lgt].set_ylabel(r"${\rm Re}_{\parallel}(k)$")
    self.axs_reynolds[self.index_vel_trv].set_ylabel(r"${\rm Re}_{\perp}(k)$")
    self._adjustAxis(self.axs_reynolds[self.index_mag])
    self._adjustAxis(self.axs_reynolds[self.index_vel_lgt])
    self._adjustAxis(self.axs_reynolds[self.index_vel_trv])
    ## annotate spectra plots
    dict_legend_args = {
      "list_artists"       : [ "--", ":" ],
      "loc"                : "lower left",
      "bbox"               : (0.0, 0.0),
      "bool_frame"         : True,
      "fontsize"           : 18
    }
    dict_artists_k_p = getArtists(
      list_scales = self.k_p_group_t,
      label       = self.dict_plot_mag["label_k_p"],
      color       = self.dict_plot_mag["color_k_p"]
    )
    dict_artists_k_eta = getArtists(
      list_scales = self.k_eta_group_t,
      label       = self.dict_plot_mag["label_k_eta"],
      color       = self.dict_plot_mag["color_k_eta"]
    )
    dict_artists_k_nu_lgt = getArtists(
      list_scales = self.k_nu_lgt_group_t,
      label       = self.dict_plot_vel_lgt["label_k_nu"],
      color       = self.dict_plot_vel_lgt["color_k_nu"]
    )
    dict_artists_k_nu_trv = getArtists(
      list_scales = self.k_nu_trv_group_t,
      label       = self.dict_plot_vel_trv["label_k_nu"],
      color       = self.dict_plot_vel_trv["color_k_nu"]
    )
    PlotFuncs.addLegend_fromArtists(
      ax                 = self.axs_spectra[self.index_mag],
      list_legend_labels = dict_artists_k_p["list_labels"] + dict_artists_k_eta["list_labels"],
      list_marker_colors = dict_artists_k_p["list_colors"] + dict_artists_k_eta["list_colors"],
      **dict_legend_args
    )
    PlotFuncs.addLegend_fromArtists(
      ax                 = self.axs_spectra[self.index_vel_lgt],
      list_legend_labels = dict_artists_k_nu_lgt["list_labels"],
      list_marker_colors = dict_artists_k_nu_lgt["list_colors"],
      **dict_legend_args
    )
    PlotFuncs.addLegend_fromArtists(
      ax                 = self.axs_spectra[self.index_vel_trv],
      list_legend_labels = dict_artists_k_nu_trv["list_labels"],
      list_marker_colors = dict_artists_k_nu_trv["list_colors"],
      **dict_legend_args
    )
    ## annotate reynolds plots
    args_hline = { "y":1, "ls":"-", "lw":2.0, "color":"k", "zorder":3 }
    self.axs_reynolds[self.index_mag].axhline(**args_hline)
    self.axs_reynolds[self.index_vel_lgt].axhline(**args_hline)
    self.axs_reynolds[self.index_vel_trv].axhline(**args_hline)
    

  def _labelSpectraRatio(self):
    args_text = { "va":"bottom", "ha":"right", "transform":self.ax_spectra_ratio.transAxes, "fontsize":25 }
    self.ax_spectra_ratio.axhline(y=1, color="black", ls=":", lw=2.0)
    x = np.linspace(10**(-1), 10**(4), 10**4)
    PlotFuncs.plotData_noAutoAxisScale(
      ax     = self.ax_spectra_ratio,
      x      = x,
      y      = 10**(-5) * x**(2),
      ls     = "--",
      lw     = 2.0,
      color  = "red",
      zorder = 5
    )
    PlotFuncs.plotData_noAutoAxisScale(
      ax     = self.ax_spectra_ratio,
      x      = x,
      y      = 10**(-5) * x**(2.5),
      ls     = "--",
      lw     = 2.0,
      color  = "royalblue",
      zorder = 5
    )
    self.ax_spectra_ratio.text(0.95, 0.05, r"$\propto k^{2}$",   **args_text, color="red")
    self.ax_spectra_ratio.text(0.95, 0.10, r"$\propto k^{2.5}$", **args_text, color="royalblue")
    self._adjustAxis(self.ax_spectra_ratio)
    self.ax_spectra_ratio.set_xlabel(r"$k$")
    self.ax_spectra_ratio.set_ylabel(
      PlotLatex.GetLabel.spectrum("mag") + r"$/$" + PlotLatex.GetLabel.spectrum("kin", "tot")
    )

  def _labelScales(self):
    self.ax_scales.set_yscale("log")
    self.ax_scales.set_ylabel(r"$k$")
    self.ax_scales.set_xlabel(r"$t/t_{\rm turb}$")
    obj_legened = PlotFuncs.addLegend(
      ax       = self.ax_scales,
      loc      = "upper right",
      bbox     = (1.0, 1.0),
      ncol     = 2,
      fontsize = 20,
      alpha    = 0.75
    )
    for line in obj_legened.get_lines():
      line.set_linewidth(2.0)
    self.ax_scales.set_ylim([
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
## OPPERATOR HANDLING PLOT CALLS
## ###############################################################
def plotSimData(
    filepath_sim_res,
    lock            = None,
    bool_check_only = False,
    bool_verbose    = True
  ):
  print("Looking at:", filepath_sim_res)
  ## get simulation parameters
  dict_sim_inputs = SimParams.readSimInputs(filepath_sim_res, bool_verbose=False)
  ## make sure a visualisation folder exists
  filepath_vis = f"{filepath_sim_res}/vis_folder/"
  WWFnF.createFolder(filepath_vis, bool_verbose=False)
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
  ax_scales = fig.add_subplot(fig_grid[3, 1:])
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
  if not(bool_check_only): obj_plot_turb.saveFittedParams(filepath_sim_res)
  dict_turb_params = obj_plot_turb.getFittedParams()
  ## PLOT SPECTRA + MEASURED SCALES
  ## ------------------------------
  obj_plot_spectra = PlotSpectra(
    fig             = fig,
    dict_axs        = {
      "axs_spectra"      : axs_spectra,
      "axs_reynolds"     : axs_reynolds,
      "ax_spectra_ratio" : ax_spectra_ratio,
      "ax_scales"        : ax_scales,
    },
    filepath_spect     = f"{filepath_sim_res}/spect/",
    dict_sim_inputs    = dict_sim_inputs,
    outputs_per_t_turb = dict_turb_params["outputs_per_t_turb"],
    time_bounds_growth = dict_turb_params["time_bounds_growth"],
    time_start_sat     = dict_turb_params["time_start_sat"],
    bool_verbose       = bool_verbose
  )
  obj_plot_spectra.performRoutines()
  ## SAVE FIGURE + DATASET
  ## ---------------------
  if lock is not None: lock.acquire()
  if not(bool_check_only): obj_plot_spectra.saveFittedParams(filepath_sim_res)
  sim_name = SimParams.getSimName(dict_sim_inputs)
  fig_name = f"{sim_name}_dataset.png"
  PlotFuncs.saveFigure(fig, f"{filepath_vis}/{fig_name}", bool_verbose=True)
  if lock is not None: lock.release()
  if bool_verbose: print(" ")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  SimParams.callFuncForAllSimulations(
    func               = plotSimData,
    bool_mproc         = BOOL_MPROC,
    bool_check_only    = BOOL_CHECK_ONLY,
    basepath           = BASEPATH,
    list_suite_folders = LIST_SUITE_FOLDERS,
    list_sonic_regimes = LIST_SONIC_REGIMES,
    list_sim_folders   = LIST_SIM_FOLDERS,
    list_sim_res       = LIST_SIM_RES
  )


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
BOOL_MPROC         = 1
BOOL_CHECK_ONLY    = 1
BASEPATH           = "/scratch/ek9/nk7952/"

## PLASMA PARAMETER SET
# LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm3000" ]
LIST_SUITE_FOLDERS = [ "Rm3000" ]
LIST_SONIC_REGIMES = [ "Mach5" ]
# LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
LIST_SIM_FOLDERS   = [ "Pm5", "Pm50", "Pm250" ]
# LIST_SIM_RES       = [ "18", "36", "72", "144", "288", "576" ]
LIST_SIM_RES       = [ "288" ]

# ## MACH NUMBER SET
# LIST_SUITE_FOLDERS = [ "Re300" ]
# LIST_SONIC_REGIMES = [ "Mach0.3", "Mach1", "Mach10" ]
# LIST_SIM_FOLDERS   = [ "Pm4" ]
# LIST_SIM_RES       = [ "288" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
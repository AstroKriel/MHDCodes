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
from TheFlashModule import LoadData, SimParams, FileNames
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
def reynoldsSpectrum(list_k, list_power, diss_rate):
    list_power_reverse = np.array(list_power[::-1])
    list_sqt_sum_power = np.sqrt(np.cumsum(list_power_reverse))[::-1]
    return list_sqt_sum_power / (diss_rate * np.array(list_k))

def plotReynoldsSpectrum(ax, list_k, list_power_group_t, diss_rate, cmap_name, bool_norm=False):
  cmap, norm = PlotFuncs.createCmap(cmap_name, vmin=0, vmax=len(list_power_group_t))
  scales_group_t = []
  for time_index, list_power in enumerate(list_power_group_t):
    if bool_norm: list_power = WWSpectra.normSpectra(list_power)
    array_reynolds = reynoldsSpectrum(list_k, list_power, diss_rate)
    ax.plot(list_k, array_reynolds, color=cmap(norm(time_index)), ls="-", lw=1.0, alpha=0.5, zorder=1)
    if np.log10(min(array_reynolds)) < 1e-1:
      list_k_interp = np.logspace(np.log10(min(list_k)), np.log10(max(list_k)), 10**4)
      list_reynolds_interp = FitFuncs.interpLogLogData(list_k, array_reynolds, list_k_interp, interp_kind="cubic")
      diss_scale_index = np.argmin(abs(list_reynolds_interp - 1.0))
      diss_scale = list_k_interp[diss_scale_index]
    else: diss_scale = None
    scales_group_t.append(diss_scale)
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
    self.color_k_eq = "black"
    self.dict_plot_cur_tot = {
      "ax_spectra"  : self.axs_spectra[0],
      "ax_reynolds" : None,
      "diss_rate"   : self.dict_sim_inputs["eta"],
      "color_spect" : "purple",
      "color_k_eta" : "purple",
      "cmap_name"   : "Purples",
      "label_spect" : PlotLatex.GetLabel.spectrum("cur"),
      "label_k_eta" : r"$k_{\eta,{\rm cur}}$",
    }
    self.dict_plot_mag_tot = {
      "ax_spectra"     : self.axs_spectra[1],
      "ax_reynolds"    : self.axs_reynolds[1],
      "diss_rate"      : self.dict_sim_inputs["eta"],
      "color_spect"    : "red",
      "color_k_p"      : "darkorange",
      "color_k_eta"    : "red",
      "cmap_name"      : "Reds",
      "label_spect"    : PlotLatex.GetLabel.spectrum("mag"),
      "label_reynolds" : r"${\rm Rm}(k)$",
      "label_k_p"      : r"$k_{\rm p}$",
      "label_k_eta"    : r"$k_{\eta,{\rm mag}}$",
    }
    self.dict_plot_kin_tot = {
      "ax_spectra"     : self.axs_spectra[2],
      "ax_reynolds"    : self.axs_reynolds[2],
      "diss_rate"      : self.dict_sim_inputs["nu"],
      "color_spect"    : "black",
      "color_k_nu"     : "black",
      "cmap_name"      : "Greys",
      "label_spect"    : PlotLatex.GetLabel.spectrum("kin"),
      "label_reynolds" : r"${\rm Re}(k)$",
      "label_k_nu"     : r"$k_\nu$",
    }
    self.dict_plot_kin_lgt = {
      "ax_spectra"     : self.axs_spectra[3],
      "ax_reynolds"    : self.axs_reynolds[3],
      "diss_rate"      : self.dict_sim_inputs["nu"],
      "color_spect"    : "royalblue",
      "color_k_nu"     : "royalblue",
      "cmap_name"      : "Blues",
      "label_spect"    : PlotLatex.GetLabel.spectrum("kin", "lgt"),
      "label_reynolds" : r"${\rm Re}_\parallel(k)$",
      "label_k_nu"     : r"$k_{\nu, \parallel}$",
    }
    self.dict_plot_kin_trv = {
      "ax_spectra"     : self.axs_spectra[4],
      "ax_reynolds"    : self.axs_reynolds[4],
      "diss_rate"      : self.dict_sim_inputs["nu"],
      "color_spect"    : "darkgreen",
      "color_k_nu"     : "darkgreen",
      "cmap_name"      : "Greens",
      "label_spect"    : PlotLatex.GetLabel.spectrum("kin", "trv"),
      "label_reynolds" : r"${\rm Re}_\perp(k)$",
      "label_k_nu"     : r"$k_{\nu, \perp}$",
    }

  def performRoutines(self):
    if self.bool_verbose: print("Loading power spectra...")
    self._loadData()
    if self.bool_verbose: print("Plotting power spectra...")
    self._plotSpectra()
    self._plotSpectraRatio()
    self._fitMagScales()
    self._fitKinScales()
    self.bool_fitted = True
    self._labelSpectra()
    self._labelSpectraRatio()
    self._labelScales()

  def getFittedParams(self):
    if not self.bool_fitted: self.performRoutines()
    return {
      ## time-averaged energy spectra
      "list_k"                     : self.list_k,
      "list_power_mag_tot_group_t" : self.list_power_mag_tot_group_t,
      "list_power_cur_tot_group_t" : self.list_power_cur_tot_group_t,
      "list_power_kin_tot_group_t" : self.list_power_kin_tot_group_t,
      "list_power_kin_lgt_group_t" : self.list_power_kin_lgt_group_t,
      "list_power_kin_trv_group_t" : self.list_power_kin_trv_group_t,
      ## measured quantities
      "index_bounds_growth" : [ self.index_start_growth, self.index_end_growth ],
      "index_start_sat"     : self.index_start_sat,
      "list_time_growth"    : self.list_time_growth,
      "list_time_eq"        : self.list_time_eq,
      "list_time_sat"       : self.list_time_sat,
      "k_nu_tot_group_t"    : self.k_nu_tot_group_t,
      "k_nu_lgt_group_t"    : self.k_nu_lgt_group_t,
      "k_nu_trv_group_t"    : self.k_nu_trv_group_t,
      "k_eta_mag_group_t"   : self.k_eta_mag_group_t,
      "k_eta_cur_group_t"   : self.k_eta_cur_group_t,
      "k_p_group_t"         : self.k_p_group_t,
      "k_max_group_t"       : self.k_max_group_t,
      "k_eq_group_t"        : self.k_eq_group_t,
    }

  def saveFittedParams(self, filepath_sim):
    dict_params = self.getFittedParams()
    WWObjs.saveDict2JsonFile(f"{filepath_sim}/{FileNames.FILENAME_SIM_OUTPUTS}", dict_params, self.bool_verbose)

  def __plotScale(
      self,
      scale_group_t, color_scale, label_scale,
      ax_spectra  = None,
      ax_reynolds = None
    ):
    args_plot = { "color":color_scale, "zorder":1, "lw":2.0 }
    scale_group_t = WWLists.replaceNoneWNan(scale_group_t)
    self.ax_scales.plot(self.list_turb_times, scale_group_t, color=color_scale, ls="-", zorder=3, lw=1.5, label=label_scale)
    ## growth regime
    if self.index_start_growth is not None:
      scale_group_t_growth = scale_group_t[self.index_start_growth : self.index_end_growth]
      if WWLists.countElemsFromList(scale_group_t_growth) > 5:
        scale_ave_growth = np.nanmean(scale_group_t_growth)
        if ax_spectra  is not None: ax_spectra.axvline(x=scale_ave_growth,  ls="--", **args_plot)
        if ax_reynolds is not None: ax_reynolds.axvline(x=scale_ave_growth, ls="--", **args_plot)
    ## saturated regime
    if self.index_start_sat is not None:
      scale_group_t_sat = scale_group_t[self.index_start_sat : ]
      if WWLists.countElemsFromList(scale_group_t_sat) > 5:
        scale_ave_sat = np.nanmean(scale_group_t_sat)
        if ax_spectra  is not None: ax_spectra.axvline(x=scale_ave_sat,  ls=":", **args_plot)
        if ax_reynolds is not None: ax_reynolds.axvline(x=scale_ave_sat, ls=":", **args_plot)

  def __measureReynoldsScale(
      self,
      dict_plot, list_power_group_t, color_scale, label_scale
    ):
    scale_group_t = plotReynoldsSpectrum(
      ax                 = dict_plot["ax_reynolds"],
      list_k             = self.list_k,
      list_power_group_t = list_power_group_t,
      diss_rate          = dict_plot["diss_rate"],
      cmap_name          = dict_plot["cmap_name"]
    )
    self.__plotScale(
      ax_spectra    = dict_plot["ax_spectra"],
      ax_reynolds   = dict_plot["ax_reynolds"],
      scale_group_t = scale_group_t,
      color_scale   = color_scale,
      label_scale   = label_scale
    )
    return scale_group_t

  def __adjustAxis(self, ax):
    ax.set_xlim([ 0.9, 1.1*max(self.list_k) ])
    ax.set_xscale("log")
    ax.set_yscale("log")

  def _loadData(self):
    if self.time_bounds_growth[0] is not None:
      file_start_time = self.time_bounds_growth[0]
    else: file_start_time = 2
    ## load total magnetic energy spectra
    dict_mag_tot_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "mag",
      spect_comp         = "tot",
      file_start_time    = file_start_time,
      outputs_per_t_turb = self.outputs_per_t_turb,
      bool_verbose       = False
    )
    ## load total current power spectra
    dict_cur_tot_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "cur",
      spect_comp         = "tot",
      file_start_time    = file_start_time,
      outputs_per_t_turb = self.outputs_per_t_turb,
      bool_verbose       = False
    )
    ## load total kinetic energy spectra
    dict_kin_tot_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "kin",
      spect_comp         = "tot",
      file_start_time    = file_start_time,
      outputs_per_t_turb = self.outputs_per_t_turb,
      bool_verbose       = False
    )
    ## load longitudinal kinetic energy spectra
    dict_kin_lgt_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "kin",
      spect_comp         = "lgt",
      file_start_time    = file_start_time,
      outputs_per_t_turb = self.outputs_per_t_turb,
      bool_verbose       = False
    )
    ## load transverse kinetic energy spectra
    dict_kin_trv_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "kin",
      spect_comp         = "trv",
      file_start_time    = file_start_time,
      outputs_per_t_turb = self.outputs_per_t_turb,
      bool_verbose       = False
    )
    ## store time realisations in growth regime
    self.list_turb_times    = dict_mag_tot_data["list_turb_times"]
    self.index_start_growth = WWLists.getIndexClosestValue(self.list_turb_times, self.time_bounds_growth[0])
    self.index_end_growth   = WWLists.getIndexClosestValue(self.list_turb_times, self.time_bounds_growth[1])
    self.list_time_growth   = self.list_turb_times[self.index_start_growth : self.index_end_growth]
    ## store time realisations in saturated regime
    self.index_start_sat = WWLists.getIndexClosestValue(self.list_turb_times, self.time_start_sat)
    self.list_time_sat   = self.list_turb_times[self.index_start_sat : ]
    ## store time-evolving energy spectra
    self.list_k                     = dict_mag_tot_data["list_k_group_t"][0]
    self.list_power_mag_tot_group_t = dict_mag_tot_data["list_power_group_t"]
    self.list_power_cur_tot_group_t = dict_cur_tot_data["list_power_group_t"]
    self.list_power_kin_tot_group_t = dict_kin_tot_data["list_power_group_t"]
    self.list_power_kin_lgt_group_t = dict_kin_lgt_data["list_power_group_t"]
    self.list_power_kin_trv_group_t = dict_kin_trv_data["list_power_group_t"]

  def _plotSpectra(self):
    ## helper function
    def __plotSpectra(dict_plot, list_power_group_t, bool_norm=False):
      cmap, norm = PlotFuncs.createCmap(dict_plot["cmap_name"], vmin=0, vmax=len(list_power_group_t))
      for index, list_power in enumerate(list_power_group_t):
        if bool_norm: list_power = WWSpectra.normSpectra(list_power)
        dict_plot["ax_spectra"].plot(self.list_k, list_power, color=cmap(norm(index)), ls="-", lw=1.0, alpha=0.5, zorder=1)
    ## plot spectra
    __plotSpectra(self.dict_plot_mag_tot, self.list_power_mag_tot_group_t, bool_norm=True)
    __plotSpectra(self.dict_plot_cur_tot, self.list_power_cur_tot_group_t, bool_norm=True)
    __plotSpectra(self.dict_plot_kin_tot, self.list_power_kin_tot_group_t)
    __plotSpectra(self.dict_plot_kin_lgt, self.list_power_kin_lgt_group_t)
    __plotSpectra(self.dict_plot_kin_trv, self.list_power_kin_trv_group_t)

  def _plotSpectraRatio(self):
    self.k_eq_group_t, _, self.list_time_eq = FitMHDScales.getEquipartitionScale(
      ax_spectra             = self.ax_spectra_ratio,
      ax_scales              = self.ax_scales,
      list_times             = self.list_turb_times,
      list_k                 = self.list_k,
      list_power_mag_group_t = self.list_power_mag_tot_group_t,
      list_power_kin_group_t = self.list_power_kin_tot_group_t,
      color                  = self.color_k_eq
    )

  def _fitMagScales(self):
    self.k_p_group_t       = []
    self.k_max_group_t     = []
    self.k_eta_cur_group_t = []
    self.k_eta_mag_group_t = self.__measureReynoldsScale(
      dict_plot          = self.dict_plot_mag_tot,
      list_power_group_t = self.list_power_kin_tot_group_t,
      color_scale        = self.dict_plot_mag_tot["color_k_eta"],
      label_scale        = self.dict_plot_mag_tot["label_k_eta"]
    )
    ## fit each time-realisation of the magnetic energy spectrum
    for time_index in range(len(self.list_turb_times)):
      k_eta_cur, _ = FitMHDScales.getSpectrumPeakScale(
        self.list_k,
        WWSpectra.normSpectra(self.list_power_cur_tot_group_t[time_index])
      )
      k_p, k_max = FitMHDScales.getSpectrumPeakScale(
        self.list_k,
        WWSpectra.normSpectra(self.list_power_mag_tot_group_t[time_index])
      )
      self.k_eta_cur_group_t.append(k_eta_cur)
      self.k_p_group_t.append(k_p)
      self.k_max_group_t.append(k_max)
    ## resistive scale from current density
    self.__plotScale(
      ax_spectra    = self.dict_plot_cur_tot["ax_spectra"],
      scale_group_t = self.k_eta_cur_group_t,
      color_scale   = self.dict_plot_cur_tot["color_k_eta"],
      label_scale   = self.dict_plot_cur_tot["label_k_eta"]
    )
    ## magnetic peak scale
    self.__plotScale(
      ax_spectra    = self.dict_plot_mag_tot["ax_spectra"],
      scale_group_t = self.k_p_group_t,
      color_scale   = self.dict_plot_mag_tot["color_k_p"],
      label_scale   = self.dict_plot_mag_tot["label_k_p"]
    )

  def _fitKinScales(self):
    ## total kinetic energy spectrum
    self.k_nu_tot_group_t = self.__measureReynoldsScale(
      dict_plot          = self.dict_plot_kin_tot,
      list_power_group_t = self.list_power_kin_tot_group_t,
      color_scale        = self.dict_plot_kin_tot["color_k_nu"],
      label_scale        = self.dict_plot_kin_tot["label_k_nu"]
    )
    ## longitudinal kinetic energy spectrum
    self.k_nu_lgt_group_t = self.__measureReynoldsScale(
      dict_plot          = self.dict_plot_kin_lgt,
      list_power_group_t = self.list_power_kin_lgt_group_t,
      color_scale        = self.dict_plot_kin_lgt["color_k_nu"],
      label_scale        = self.dict_plot_kin_lgt["label_k_nu"]
    )
    ## transverse kinetic energy spectrum
    self.k_nu_trv_group_t = self.__measureReynoldsScale(
      dict_plot          = self.dict_plot_kin_trv,
      list_power_group_t = self.list_power_kin_trv_group_t,
      color_scale        = self.dict_plot_kin_trv["color_k_nu"],
      label_scale        = self.dict_plot_kin_trv["label_k_nu"]
    )

  def _labelSpectra(self):
    ## helper function
    def __labelAxis(dict_plot):
      if dict_plot["ax_spectra"] is not None:
        dict_plot["ax_spectra"].set_ylabel(dict_plot["label_spect"])
        self.__adjustAxis(dict_plot["ax_spectra"])
      if dict_plot["ax_reynolds"] is not None:
        dict_plot["ax_reynolds"].axhline(y=1, ls="-", lw=2, color="black", zorder=3)
        dict_plot["ax_reynolds"].set_ylabel(dict_plot["label_reynolds"])
        self.__adjustAxis(dict_plot["ax_reynolds"])
    ## helper function
    def __getArtists(scales_group_t, label, color):
      list_colors  = []
      list_markers = []
      list_labels  = []
      ## growth phase
      if self.index_start_growth is not None:
        scales_group_t_growth = scales_group_t[self.index_start_growth : self.index_end_growth]
        if WWLists.countElemsFromList(scales_group_t_growth) > 5:
          scale_growth = PlotLatex.GetLabel.modes(scales_group_t_growth)
          list_colors.append(color)
          list_markers.append("--")
          list_labels.append("{" + label + r"}$_{,{\rm growth}}$ = " + scale_growth)
      ## saturated phase
      if self.index_start_sat is not None:
        scales_group_t_sat = scales_group_t[self.index_start_sat : ]
        if WWLists.countElemsFromList(scales_group_t_sat) > 5:
          scale_sat = PlotLatex.GetLabel.modes(scales_group_t_sat)
          list_colors.append(color)
          list_markers.append(":")
          list_labels.append("{" + label + r"}$_{,{\rm sat}}$ = " + scale_sat)
      return {
        "list_colors"  : list_colors,
        "list_markers" : list_markers,
        "list_labels"  : list_labels
      }
    ## label axis
    self.axs_spectra[-1].set_xlabel(r"$k$")
    self.axs_reynolds[-1].set_xlabel(r"$k$")
    __labelAxis(self.dict_plot_mag_tot)
    __labelAxis(self.dict_plot_cur_tot)
    __labelAxis(self.dict_plot_kin_tot)
    __labelAxis(self.dict_plot_kin_lgt)
    __labelAxis(self.dict_plot_kin_trv)
    ## annotate spectra plots
    dict_legend_args = {
      "loc"        : "lower left",
      "bbox"       : (0.0, 0.0),
      "bool_frame" : True,
      "fontsize"   : 18
    }
    dict_artists_k_p = __getArtists(
      scales_group_t = self.k_p_group_t,
      label          = self.dict_plot_mag_tot["label_k_p"],
      color          = self.dict_plot_mag_tot["color_k_p"]
    )
    dict_artists_k_eta_mag = __getArtists(
      scales_group_t = self.k_eta_mag_group_t,
      label          = self.dict_plot_mag_tot["label_k_eta"],
      color          = self.dict_plot_mag_tot["color_k_eta"]
    )
    dict_artists_k_eta_cur = __getArtists(
      scales_group_t = self.k_eta_cur_group_t,
      label          = self.dict_plot_cur_tot["label_k_eta"],
      color          = self.dict_plot_cur_tot["color_k_eta"]
    )
    dict_artists_k_nu_tot = __getArtists(
      scales_group_t = self.k_nu_tot_group_t,
      label          = self.dict_plot_kin_tot["label_k_nu"],
      color          = self.dict_plot_kin_tot["color_k_nu"]
    )
    dict_artists_k_nu_lgt = __getArtists(
      scales_group_t = self.k_nu_lgt_group_t,
      label          = self.dict_plot_kin_lgt["label_k_nu"],
      color          = self.dict_plot_kin_lgt["color_k_nu"]
    )
    dict_artists_k_nu_trv = __getArtists(
      scales_group_t = self.k_nu_trv_group_t,
      label          = self.dict_plot_kin_trv["label_k_nu"],
      color          = self.dict_plot_kin_trv["color_k_nu"]
    )
    PlotFuncs.addLegend_fromArtists(
      ax                 = self.dict_plot_mag_tot["ax_spectra"],
      list_legend_labels = dict_artists_k_p["list_labels"]  + dict_artists_k_eta_mag["list_labels"],
      list_marker_colors = dict_artists_k_p["list_colors"]  + dict_artists_k_eta_mag["list_colors"],
      list_artists       = dict_artists_k_p["list_markers"] + dict_artists_k_eta_mag["list_markers"],
      **dict_legend_args
    )
    PlotFuncs.addLegend_fromArtists(
      ax                 = self.dict_plot_cur_tot["ax_spectra"],
      list_legend_labels = dict_artists_k_eta_cur["list_labels"],
      list_marker_colors = dict_artists_k_eta_cur["list_colors"],
      list_artists       = dict_artists_k_eta_cur["list_markers"],
      **dict_legend_args
    )
    PlotFuncs.addLegend_fromArtists(
      ax                 = self.dict_plot_kin_tot["ax_spectra"],
      list_legend_labels = dict_artists_k_nu_tot["list_labels"],
      list_marker_colors = dict_artists_k_nu_tot["list_colors"],
      list_artists       = dict_artists_k_nu_tot["list_markers"],
      **dict_legend_args
    )
    PlotFuncs.addLegend_fromArtists(
      ax                 = self.dict_plot_kin_lgt["ax_spectra"],
      list_legend_labels = dict_artists_k_nu_lgt["list_labels"],
      list_marker_colors = dict_artists_k_nu_lgt["list_colors"],
      list_artists       = dict_artists_k_nu_lgt["list_markers"],
      **dict_legend_args
    )
    PlotFuncs.addLegend_fromArtists(
      ax                 = self.dict_plot_kin_trv["ax_spectra"],
      list_legend_labels = dict_artists_k_nu_trv["list_labels"],
      list_marker_colors = dict_artists_k_nu_trv["list_colors"],
      list_artists       = dict_artists_k_nu_trv["list_markers"],
      **dict_legend_args
    )

  def _labelSpectraRatio(self):
    args_text = { "va":"bottom", "ha":"right", "transform":self.ax_spectra_ratio.transAxes, "fontsize":25 }
    self.ax_spectra_ratio.axhline(y=1, color="black", ls="-", lw=2.0)
    x = np.linspace(10**(-1), 10**(4), 10**4)
    PlotFuncs.plotData_noAutoAxisScale(
      ax     = self.ax_spectra_ratio,
      x      = x,
      y      = 10**(-5) * x**(1),
      ls     = "--",
      lw     = 2.0,
      color  = "blue",
      zorder = 5
    )
    PlotFuncs.plotData_noAutoAxisScale(
      ax     = self.ax_spectra_ratio,
      x      = x,
      y      = 10**(-5) * x**(2),
      ls     = "--",
      lw     = 2.0,
      color  = "red",
      zorder = 5
    )
    self.ax_spectra_ratio.text(0.95, 0.10, r"$\propto k$",   color="blue", **args_text)
    self.ax_spectra_ratio.text(0.95, 0.05, r"$\propto k^2$", color="red",  **args_text)
    self.__adjustAxis(self.ax_spectra_ratio)
    self.ax_spectra_ratio.set_xlabel(r"$k$")
    self.ax_spectra_ratio.set_ylabel(
      self.dict_plot_mag_tot["label_spect"] + r"$/$" + self.dict_plot_kin_tot["label_spect"]
    )

  def _labelScales(self):
    self.ax_scales.set_yscale("log")
    self.ax_scales.set_ylabel(r"$k$")
    self.ax_scales.set_xlabel(r"$t/t_{\rm turb}$")
    PlotFuncs.addLegend(
      ax       = self.ax_scales,
      loc      = "upper right",
      bbox     = (1.0, 1.0),
      ncol     = 1,
      lw       = 2.0,
      fontsize = 20,
      alpha    = 0.75
    )
    ## scale axis limits
    self.ax_scales.set_ylim([
      0.9 * min(self.list_k),
      1.1 * max(self.list_k)
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
    num_rows         = 6,
    num_cols         = 3
  )
  ## volume integrated qunatities
  ax_Mach         = fig.add_subplot(fig_grid[0, 0])
  ax_energy_ratio = fig.add_subplot(fig_grid[1, 0])
  ## power spectra data
  ax_spectra_ratio    = fig.add_subplot(fig_grid[2:5, 0])
  axs_spectra         = [
    fig.add_subplot(fig_grid[0, 1]),
    fig.add_subplot(fig_grid[1, 1]),
    fig.add_subplot(fig_grid[2, 1]),
    fig.add_subplot(fig_grid[3, 1]),
    fig.add_subplot(fig_grid[4, 1])
  ]
  ## reynolds spectra
  axs_reynolds        = [
    fig.add_subplot(fig_grid[0, 2]),
    fig.add_subplot(fig_grid[1, 2]),
    fig.add_subplot(fig_grid[2, 2]),
    fig.add_subplot(fig_grid[3, 2]),
    fig.add_subplot(fig_grid[4, 2])
  ]
  ## measured scales
  ax_scales = fig.add_subplot(fig_grid[5, :])
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
BOOL_MPROC         = 0
BOOL_CHECK_ONLY    = 1
BASEPATH           = "/scratch/ek9/nk7952/"

# ## PLASMA PARAMETER SET
# LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm3000" ]
# LIST_SONIC_REGIMES = [ "Mach5" ]
# LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
# LIST_SIM_RES       = [ "18", "36", "72", "144", "288", "576" ]

LIST_SUITE_FOLDERS = [ "Rm3000" ]
LIST_SONIC_REGIMES = [ "Mach5" ]
LIST_SIM_FOLDERS   = [ "Pm10" ]
LIST_SIM_RES       = [ "288" ]

# ## MACH NUMBER SET
# LIST_SUITE_FOLDERS = [ "Re300" ]
# LIST_SONIC_REGIMES = [ "Mach0.3", "Mach1", "Mach5", "Mach10" ]
# LIST_SIM_FOLDERS   = [ "Pm4" ]
# LIST_SIM_RES       = [ "18", "36", "72", "144", "288" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
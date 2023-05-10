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
      "label_reynolds" : None,
      "label_k_eta" : r"$k_{\eta, {\rm cur}}$",
    }
    self.dict_plot_rho_tot = {
      "ax_spectra"     : self.axs_reynolds[0],
      "ax_reynolds"    : None,
      "diss_rate"      : None,
      "color_spect"    : "darkorange",
      "color_k_p"      : "darkorange",
      "cmap_name"      : "Oranges",
      "label_spect"    : PlotLatex.GetLabel.spectrum("rho"),
      "label_reynolds" : None,
      "label_k_p"      : r"$k_\rho$",
    }
    self.dict_plot_mag_tot = {
      "ax_spectra"     : self.axs_spectra[1],
      "ax_reynolds"    : self.axs_reynolds[1],
      "diss_rate"      : self.dict_sim_inputs["eta"],
      "color_spect"    : "red",
      "color_k_p"      : "red",
      "color_k_eta"    : "red",
      "cmap_name"      : "Reds",
      "label_spect"    : PlotLatex.GetLabel.spectrum("mag"),
      "label_reynolds" : r"${\rm Rm}(k)$",
      "label_k_p"      : r"$k_{\rm p}$",
      "label_k_eta"    : r"$k_{\eta, {\rm mag}}$",
    }
    self.dict_plot_kin_tot = {
      "ax_spectra"     : self.axs_spectra[2],
      "ax_reynolds"    : self.axs_reynolds[2],
      "diss_rate"      : self.dict_sim_inputs["nu"],
      "color_spect"    : "black",
      "color_k_nu"     : "black",
      "cmap_name"      : "Greys",
      "label_spect"    : PlotLatex.GetLabel.spectrum("kin"),
      "label_reynolds" : r"${\rm Re}_{\rm kin}(k)$",
      "label_k_nu"     : r"$k_{\nu, {\rm kin}}$",
    }
    self.dict_plot_vel_tot = {
      "ax_spectra"     : self.axs_spectra[3],
      "ax_reynolds"    : self.axs_reynolds[3],
      "diss_rate"      : self.dict_sim_inputs["nu"],
      "color_spect"    : "magenta",
      "color_k_nu"     : "magenta",
      "cmap_name"      : "Greys",
      "label_spect"    : PlotLatex.GetLabel.spectrum("vel", "tot"),
      "label_reynolds" : r"${\rm Re}_{\rm vel}(k)$",
      "label_k_nu"     : r"$k_{\nu, {\rm vel}}$",
    }
    self.dict_plot_vel_lgt = {
      "ax_spectra"     : self.axs_spectra[4],
      "ax_reynolds"    : self.axs_reynolds[4],
      "diss_rate"      : self.dict_sim_inputs["nu"],
      "color_spect"    : "royalblue",
      "color_k_nu"     : "royalblue",
      "cmap_name"      : "Blues",
      "label_spect"    : PlotLatex.GetLabel.spectrum("vel", "lgt"),
      "label_reynolds" : r"${\rm Re}_{{\rm vel}, \parallel}(k)$",
      "label_k_nu"     : r"$k_{\nu, {\rm vel}, \parallel}$",
    }
    self.dict_plot_vel_trv = {
      "ax_spectra"     : self.axs_spectra[5],
      "ax_reynolds"    : self.axs_reynolds[5],
      "diss_rate"      : self.dict_sim_inputs["nu"],
      "color_spect"    : "darkgreen",
      "color_k_nu"     : "darkgreen",
      "cmap_name"      : "Greens",
      "label_spect"    : PlotLatex.GetLabel.spectrum("vel", "trv"),
      "label_reynolds" : r"${\rm Re}_{{\rm vel}, \perp}(k)$",
      "label_k_nu"     : r"$k_{\nu, {\rm vel}, \perp}$",
    }

  def performRoutines(self):
    if self.bool_verbose: print("Loading spectra data...")
    self._loadData()
    if self.bool_verbose: print("Plotting spectra...")
    self._plotSpectra()
    self._plotSpectraRatio()
    self._fitMagScales()
    self._fitRhoScales()
    self._fitKinScales()
    self.bool_fitted = True
    self._labelSpectra()
    self._labelSpectraRatio()
    self._labelScales()

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
    else: file_start_time = 5
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
    ## load total current power spectra
    dict_rho_tot_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "rho",
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
    ## load total velocity power spectra
    dict_vel_tot_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "vel",
      spect_comp         = "tot",
      file_start_time    = file_start_time,
      outputs_per_t_turb = self.outputs_per_t_turb,
      bool_verbose       = False
    )
    ## load longitudinal velocity power spectra
    dict_vel_lgt_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "vel",
      spect_comp         = "lgt",
      file_start_time    = file_start_time,
      outputs_per_t_turb = self.outputs_per_t_turb,
      bool_verbose       = False
    )
    ## load transverse velocity power spectra
    dict_vel_trv_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "vel",
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
    ## store time-evolving spectra
    self.list_k                     = dict_mag_tot_data["list_k_group_t"][0]
    self.list_power_mag_tot_group_t = dict_mag_tot_data["list_power_group_t"]
    self.list_power_cur_tot_group_t = dict_cur_tot_data["list_power_group_t"]
    self.list_power_rho_tot_group_t = dict_rho_tot_data["list_power_group_t"]
    self.list_power_kin_tot_group_t = dict_kin_tot_data["list_power_group_t"]
    self.list_power_vel_tot_group_t = dict_vel_tot_data["list_power_group_t"]
    self.list_power_vel_lgt_group_t = dict_vel_lgt_data["list_power_group_t"]
    self.list_power_vel_trv_group_t = dict_vel_trv_data["list_power_group_t"]

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
    __plotSpectra(self.dict_plot_rho_tot, self.list_power_rho_tot_group_t)
    __plotSpectra(self.dict_plot_kin_tot, self.list_power_kin_tot_group_t)
    __plotSpectra(self.dict_plot_vel_tot, self.list_power_vel_tot_group_t)
    __plotSpectra(self.dict_plot_vel_lgt, self.list_power_vel_lgt_group_t)
    __plotSpectra(self.dict_plot_vel_trv, self.list_power_vel_trv_group_t)

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
    self.k_p_mag_group_t   = []
    self.k_max_mag_group_t = []
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
      self.k_p_mag_group_t.append(k_p)
      self.k_max_mag_group_t.append(k_max)
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
      scale_group_t = self.k_p_mag_group_t,
      color_scale   = self.dict_plot_mag_tot["color_k_p"],
      label_scale   = self.dict_plot_mag_tot["label_k_p"]
    )

  def _fitRhoScales(self):
    self.k_p_rho_group_t   = []
    self.k_max_rho_group_t = []
    ## fit each time-realisation of the magnetic energy spectrum
    for time_index in range(len(self.list_turb_times)):
      k_p, k_max = FitMHDScales.getSpectrumPeakScale(
        self.list_k,
        WWSpectra.normSpectra(self.list_power_rho_tot_group_t[time_index])
      )
      self.k_p_rho_group_t.append(k_p)
      self.k_max_rho_group_t.append(k_max)
    ## density peak scale
    self.__plotScale(
      ax_spectra    = self.dict_plot_rho_tot["ax_spectra"],
      scale_group_t = self.k_p_rho_group_t,
      color_scale   = self.dict_plot_rho_tot["color_k_p"],
      label_scale   = self.dict_plot_rho_tot["label_k_p"]
    )

  def _fitKinScales(self):
    ## total kinetic energy spectrum
    self.k_nu_kin_group_t = self.__measureReynoldsScale(
      dict_plot          = self.dict_plot_kin_tot,
      list_power_group_t = self.list_power_kin_tot_group_t,
      color_scale        = self.dict_plot_kin_tot["color_k_nu"],
      label_scale        = self.dict_plot_kin_tot["label_k_nu"]
    )
    ## total velocity power spectrum
    self.k_nu_vel_tot_group_t = self.__measureReynoldsScale(
      dict_plot          = self.dict_plot_vel_tot,
      list_power_group_t = self.list_power_vel_tot_group_t,
      color_scale        = self.dict_plot_vel_tot["color_k_nu"],
      label_scale        = self.dict_plot_vel_tot["label_k_nu"]
    )
    ## longitudinal velocity power spectrum
    self.k_nu_vel_lgt_group_t = self.__measureReynoldsScale(
      dict_plot          = self.dict_plot_vel_lgt,
      list_power_group_t = self.list_power_vel_lgt_group_t,
      color_scale        = self.dict_plot_vel_lgt["color_k_nu"],
      label_scale        = self.dict_plot_vel_lgt["label_k_nu"]
    )
    ## transverse velocity power spectrum
    self.k_nu_vel_trv_group_t = self.__measureReynoldsScale(
      dict_plot          = self.dict_plot_vel_trv,
      list_power_group_t = self.list_power_vel_trv_group_t,
      color_scale        = self.dict_plot_vel_trv["color_k_nu"],
      label_scale        = self.dict_plot_vel_trv["label_k_nu"]
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
    __labelAxis(self.dict_plot_rho_tot)
    __labelAxis(self.dict_plot_kin_tot)
    __labelAxis(self.dict_plot_vel_tot)
    __labelAxis(self.dict_plot_vel_lgt)
    __labelAxis(self.dict_plot_vel_trv)
    ## annotate spectra plots
    dict_legend_args = {
      "loc"        : "lower left",
      "bbox"       : (0.0, 0.0),
      "bool_frame" : True,
      "fontsize"   : 18
    }
    dict_artists_k_p_mag = __getArtists(
      scales_group_t = self.k_p_mag_group_t,
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
    dict_artists_k_p_rho = __getArtists(
      scales_group_t = self.k_p_rho_group_t,
      label          = self.dict_plot_rho_tot["label_k_p"],
      color          = self.dict_plot_rho_tot["color_k_p"]
    )
    dict_artists_k_nu_kin = __getArtists(
      scales_group_t = self.k_nu_kin_group_t,
      label          = self.dict_plot_kin_tot["label_k_nu"],
      color          = self.dict_plot_kin_tot["color_k_nu"]
    )
    dict_artists_k_nu_vel_tot = __getArtists(
      scales_group_t = self.k_nu_vel_tot_group_t,
      label          = self.dict_plot_vel_tot["label_k_nu"],
      color          = self.dict_plot_vel_tot["color_k_nu"]
    )
    dict_artists_k_nu_vel_lgt = __getArtists(
      scales_group_t = self.k_nu_vel_lgt_group_t,
      label          = self.dict_plot_vel_lgt["label_k_nu"],
      color          = self.dict_plot_vel_lgt["color_k_nu"]
    )
    dict_artists_k_nu_vel_trv = __getArtists(
      scales_group_t = self.k_nu_vel_trv_group_t,
      label          = self.dict_plot_vel_trv["label_k_nu"],
      color          = self.dict_plot_vel_trv["color_k_nu"]
    )
    PlotFuncs.addLegend_fromArtists(
      ax                 = self.dict_plot_mag_tot["ax_spectra"],
      list_legend_labels = dict_artists_k_p_mag["list_labels"]  + dict_artists_k_eta_mag["list_labels"],
      list_marker_colors = dict_artists_k_p_mag["list_colors"]  + dict_artists_k_eta_mag["list_colors"],
      list_artists       = dict_artists_k_p_mag["list_markers"] + dict_artists_k_eta_mag["list_markers"],
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
      ax                 = self.dict_plot_rho_tot["ax_spectra"],
      list_legend_labels = dict_artists_k_p_rho["list_labels"],
      list_marker_colors = dict_artists_k_p_rho["list_colors"],
      list_artists       = dict_artists_k_p_rho["list_markers"],
      **dict_legend_args
    )
    PlotFuncs.addLegend_fromArtists(
      ax                 = self.dict_plot_kin_tot["ax_spectra"],
      list_legend_labels = dict_artists_k_nu_kin["list_labels"],
      list_marker_colors = dict_artists_k_nu_kin["list_colors"],
      list_artists       = dict_artists_k_nu_kin["list_markers"],
      **dict_legend_args
    )
    PlotFuncs.addLegend_fromArtists(
      ax                 = self.dict_plot_vel_tot["ax_spectra"],
      list_legend_labels = dict_artists_k_nu_vel_tot["list_labels"],
      list_marker_colors = dict_artists_k_nu_vel_tot["list_colors"],
      list_artists       = dict_artists_k_nu_vel_tot["list_markers"],
      **dict_legend_args
    )
    PlotFuncs.addLegend_fromArtists(
      ax                 = self.dict_plot_vel_lgt["ax_spectra"],
      list_legend_labels = dict_artists_k_nu_vel_lgt["list_labels"],
      list_marker_colors = dict_artists_k_nu_vel_lgt["list_colors"],
      list_artists       = dict_artists_k_nu_vel_lgt["list_markers"],
      **dict_legend_args
    )
    PlotFuncs.addLegend_fromArtists(
      ax                 = self.dict_plot_vel_trv["ax_spectra"],
      list_legend_labels = dict_artists_k_nu_vel_trv["list_labels"],
      list_marker_colors = dict_artists_k_nu_vel_trv["list_colors"],
      list_artists       = dict_artists_k_nu_vel_trv["list_markers"],
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
      loc      = "right",
      bbox     = (1.0, 0.5),
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
def plotSimData():
  ## initialise figure
  print("Initialising figure...")
  figscale = 1.2
  fig, axs = plt.subplots(nrows=2, figsize=(6*figscale, 2*4*figscale), sharex=True)
  fig.subplots_adjust(hspace=0.075)
  ## plot subsonic data
  obj_plot_turb = PlotTurbData(
    fig               = fig,
    axs               = axs,
    filepath_sim_res  = f"{PATH_SCRATCH}/Rm3000/Mach0.3/Pm5/288/",
    color             = "#66c2a5",
    time_start_growth = 5,
    time_end_growth   = 15,
    time_start_sat    = 33,
  )
  obj_plot_turb.performRoutines()
  ## plot supersonic data
  obj_plot_turb = PlotTurbData(
    fig               = fig,
    axs               = axs,
    filepath_sim_res  = f"{PATH_SCRATCH}/Rm3000/Mach5/Pm5/288/",
    color             = "#fc8d62",
    time_start_growth = 5,
    time_end_growth   = 35,
    time_start_sat    = 62,
  )
  obj_plot_turb.performRoutines()
  ## label figure
  axs[1].set_xlabel(r"$t / t_\mathrm{turb}$")
  axs[0].set_ylabel(r"$\mathcal{M}$")
  axs[1].set_ylabel(r"$E_\mathrm{mag} / E_\mathrm{kin}$")
  axs[0].set_yscale("log")
  axs[1].set_yscale("log")
  axs[0].set_xlim([ -2, 102 ])
  axs[0].set_ylim([ 10**(-2), 10**(1) ])
  axs[1].set_ylim([ 10**(-11), 10**(1) ])
  ## add log axis-ticks
  PlotFuncs.addAxisTicks_log10(
    axs[1],
    bool_major_ticks = True,
    num_major_ticks  = 7
  )
  ## add legends
  PlotFuncs.addLegend_fromArtists(
    axs[1],
    list_artists       = [
      "--",
      ":"
    ],
    list_legend_labels = [
      r"kinematic phase",
      r"saturated phase",
    ],
    list_marker_colors = [ "k" ],
    label_color        = "black",
    loc                = "lower right",
    bbox               = (1.0, 0.0),
    fontsize           = 18
  )
  ## save figure
  print("Saving figure...")
  fig_name     = f"time_evolution.pdf"
  filepath_fig = f"{PATH_PLOT}/{fig_name}"
  fig.savefig(filepath_fig, dpi=100)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")
## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  SimParams.callFuncForAllSimulations(
    func               = plotSimData,
    bool_mproc         = BOOL_MPROC,
    bool_check_only    = BOOL_CHECK_ONLY,
    basepath           = PATH_SCRATCH,
    list_suite_folders = LIST_SUITE_FOLDERS,
    list_sonic_regimes = LIST_SONIC_REGIMES,
    list_sim_folders   = LIST_SIM_FOLDERS,
    list_sim_res       = LIST_SIM_RES
  )


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BOOL_MPROC      = 1
BOOL_CHECK_ONLY = 0
PATH_SCRATCH    = "/scratch/ek9/nk7952/"
# PATH_SCRATCH    = "/scratch/jh2/nk7952/"

## PLASMA PARAMETER SET
LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm3000" ]
LIST_SONIC_REGIMES = [ "Mach5" ]
LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
LIST_SIM_RES       = [ "18", "36", "72", "144", "288", "576" ]

# ## MACH NUMBER SET
# LIST_SUITE_FOLDERS = [ "Rm3000" ]
# LIST_SONIC_REGIMES = [ "Mach0.3", "Mach1", "Mach10" ]
# LIST_SIM_FOLDERS   = [ "Pm1", "Pm5", "Pm10", "Pm125" ]
# LIST_SIM_RES       = [ "18", "36", "72", "144", "288" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
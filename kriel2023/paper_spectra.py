#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np
import matplotlib.pyplot as plt

## load user defined modules
from TheFlashModule import LoadData, SimParams
from TheUsefulModule import WWFnF
from TheAnalysisModule import WWSpectra
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def plotPowerLawPassingThroughPoint(
    ax, x_domain, slope, coord,
    ls = "-",
    lw = 2.0
  ):
  x1, y1 = coord
  a0 = y1 / x1**(slope)
  x = np.logspace(np.log10(x_domain[0]), np.log10(x_domain[1]), 100)
  y = a0 * x**(slope)
  PlotFuncs.plotData_noAutoAxisScale(ax, x, y, ls=ls, lw=lw, zorder=10)

def getSimLabel(mach_regime, Re):
  return "$" + mach_regime.replace("Mach", "\mathcal{M}") + "\\text{Re}" + f"{Re:.0f}" + "\\text{Pm}5$"

def getSimPath(scratch_path, sim_suite, mach_regime, sim_res):
  return f"{scratch_path}/{sim_suite}/{mach_regime}/Pm5/{int(sim_res):d}/"

def initFigure(ncols=1, nrows=1):
  return plt.subplots(
    ncols   = ncols,
    nrows   = nrows,
    figsize = (7*ncols, 4*nrows)
  )

def addText(ax, pos, text, fontsize=24):
  ax.text(
    pos[0], pos[1],
    text,
    va        = "bottom",
    ha        = "left",
    transform = ax.transAxes,
    color     = "black",
    fontsize  = fontsize,
    zorder    = 10
  )

def reynoldsSpectrum(list_k, list_power, diss_rate):
    list_power_reverse = np.array(list_power[::-1])
    list_sqt_sum_power = np.sqrt(np.cumsum(list_power_reverse))[::-1]
    return list_sqt_sum_power / (diss_rate * np.array(list_k))

def plotReynoldsSpectrum(
    ax, list_k, list_power_group_t, diss_rate,
    color     = "black",
    bool_norm = False,
    label     = None,
    zorder    = 3
  ):
  array_reynolds_group_t = []
  for list_power in list_power_group_t:
    if bool_norm: list_power = WWSpectra.normSpectra(list_power)
    array_reynolds_group_t.append(reynoldsSpectrum(list_k, list_power, diss_rate))
  array_reynolds_ave = np.mean(array_reynolds_group_t, axis=0)
  ax.plot(
    list_k,
    array_reynolds_ave,
    color  = color,
    label  = label,
    zorder = zorder,
    ls     = "-",
    lw     = 2.0
  )
  array_reynolds_group_k = [
    [
      array_reynolds[k_index]
      for array_reynolds in array_reynolds_group_t
    ]
    for k_index in range(len(list_k))
  ]
  for k_index in range(len(list_k)):
    PlotFuncs.plotErrorBar_1D(
      ax      = ax,
      x       = list_k[k_index],
      array_y = array_reynolds_group_k[k_index],
      color   = color,
      marker  = "",
      capsize = 2.0,
      alpha   = 0.25,
      zorder  = zorder
    )

def plotSpectrum(
    ax, list_k, list_power_group_t,
    comp_factor    = None,
    bool_norm      = False,
    color          = "black",
    label          = None,
    zorder         = 3
  ):
  array_power_group_t = []
  for t_index in range(len(list_power_group_t)):
    array_power = np.array(list_power_group_t[t_index])
    if bool_norm: array_power = WWSpectra.normSpectra(array_power)
    if comp_factor is not None: array_power *= np.array(list_k)**(comp_factor)
    array_power_group_t.append(array_power)
  ## plot time-averaged spectrum
  array_power_ave = np.mean(array_power_group_t, axis=0)
  ax.plot(
    list_k,
    array_power_ave,
    color  = color,
    label  = label,
    zorder = zorder,
    ls     = "-",
    lw     = 2.0
  )
  ## rearrange data: [spectrum(t0), spectrum(t1), ...] -> [k=1 data, k=2 data, ...]
  array_power_group_k = [
    [
      array_power[k_index]
      for array_power in array_power_group_t
    ]
    for k_index in range(len(list_k))
  ]
  ## plot spectrum variation
  for k_index in range(len(list_k)):
    PlotFuncs.plotErrorBar_1D(
      ax      = ax,
      x       = list_k[k_index],
      array_y = array_power_group_k[k_index],
      color   = color,
      marker  = "",
      capsize = 2.0,
      alpha   = 0.25,
      zorder  = zorder
    )

def plotSpectrumRatio(ax, list_k, list_power_1_group_t, list_power_2_group_t, color="black", label=None):
  array_power_ratio_group_t = []
  for list_power_1, list_power_2 in zip(
      list_power_1_group_t,
      list_power_2_group_t
    ):
    array_power_ratio = np.array(list_power_1) / np.array(list_power_2)
    array_power_ratio_group_t.append(array_power_ratio)
  array_power_ratio_ave = np.mean(array_power_ratio_group_t, axis=0)
  ax.plot(list_k, array_power_ratio_ave, color=color, label=label, ls="-", lw=2.0, zorder=5)
  array_power_ratio_group_k = [
    [
      array_power_ratio[k_index]
      for array_power_ratio in array_power_ratio_group_t
    ]
    for k_index in range(len(list_k))
  ]
  for k_index in range(len(list_k)):
    PlotFuncs.plotErrorBar_1D(
      ax      = ax,
      x       = list_k[k_index],
      array_y = array_power_ratio_group_k[k_index],
      color   = color,
      marker  = "",
      capsize = 0.0,
      zorder  = 3
    )


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class PlotSpectra():
  def __init__(
      self,
      filepath_sim_res
    ):
    self.filepath_spect   = f"{filepath_sim_res}/spect/"
    self.dict_sim_inputs  = SimParams.readSimInputs(filepath_sim_res, False)
    self.dict_sim_outputs = SimParams.readSimOutputs(filepath_sim_res, False)
    self.desired_Mach     = self.dict_sim_inputs["desired_Mach"]
    self.time_bounds      = self.dict_sim_outputs["time_bounds_growth"]

  def _plotScales(self, ax, kscale_group_t, color, label=None):
    index_start = self.dict_sim_outputs["index_bounds_growth"][0]
    index_end   = self.dict_sim_outputs["index_bounds_growth"][1]
    kscale_group_t = kscale_group_t[index_start : index_end]
    kscale_ave = np.mean(kscale_group_t)
    kscale_std = np.std(kscale_group_t)
    ax.axvline(x=kscale_ave, color=color, ls="--", lw=1.5, zorder=2)
    ax.axvspan(kscale_ave-kscale_std, kscale_ave+kscale_std, color=color, alpha=0.25, zorder=1)

  def plotKinSpectrumRatios(self, ax, color="black"):
    dict_lgt_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "vel",
      spect_comp         = "lgt",
      file_start_time    = self.time_bounds[0],
      file_end_time      = self.time_bounds[1],
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    dict_trv_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "vel",
      spect_comp         = "trv",
      file_start_time    = self.time_bounds[0],
      file_end_time      = self.time_bounds[1],
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    list_k = dict_lgt_data["list_k_group_t"][0]
    plotSpectrumRatio(
      ax                   = ax,
      list_k               = list_k,
      list_power_1_group_t = dict_lgt_data["list_power_group_t"],
      list_power_2_group_t = dict_trv_data["list_power_group_t"],
      color                = color,
      label                = r"$\frac{\mathcal{P}_{\rm vel, \parallel}(k)}{\mathcal{P}_{\rm vel, \perp}(k)}$"
    )

  def plotKinTotSpectrum(self, ax, label="tot", color="#003f5c", comp_factor=None, zorder=3, bool_plot_scale=False):
    dict_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "vel",
      spect_comp         = "tot",
      file_start_time    = self.time_bounds[0],
      file_end_time      = self.time_bounds[1],
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    plotSpectrum(
      ax                 = ax,
      list_k             = dict_data["list_k_group_t"][0],
      list_power_group_t = dict_data["list_power_group_t"],
      bool_norm          = True,
      comp_factor        = comp_factor,
      zorder             = zorder,
      color              = color,
      label              = label
    )
    if bool_plot_scale: self._plotScales(ax, self.dict_sim_outputs["k_nu_vel_tot_group_t"], color)

  def plotKinLgtSpectrum(self, ax, label="lgt", color="#bc5090", comp_factor=None, zorder=3):
    dict_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "vel",
      spect_comp         = "lgt",
      file_start_time    = self.time_bounds[0],
      file_end_time      = self.time_bounds[1],
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    plotSpectrum(
      ax                 = ax,
      list_k             = dict_data["list_k_group_t"][0],
      list_power_group_t = dict_data["list_power_group_t"],
      comp_factor        = comp_factor,
      zorder             = zorder,
      color              = color,
      label              = label
    )

  def plotKinTrvSpectrum(self, ax, label="trv", color="#ffa600", comp_factor=None, zorder=3):
    dict_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "vel",
      spect_comp         = "trv",
      file_start_time    = self.time_bounds[0],
      file_end_time      = self.time_bounds[1],
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    plotSpectrum(
      ax                 = ax,
      list_k             = dict_data["list_k_group_t"][0],
      list_power_group_t = dict_data["list_power_group_t"],
      comp_factor        = comp_factor,
      zorder             = zorder,
      color              = color,
      label              = label
    )

  def plotKinReynoldsSpectrum(self, ax, color="black", bool_plot_scale=False, zorder=5):
    dict_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "vel",
      spect_comp         = "tot",
      file_start_time    = self.time_bounds[0],
      file_end_time      = self.time_bounds[1],
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    plotReynoldsSpectrum(
      ax                 = ax,
      list_k             = dict_data["list_k_group_t"][0],
      list_power_group_t = dict_data["list_power_group_t"],
      diss_rate          = self.dict_sim_inputs["nu"],
      color              = color,
      label              = r"$" + self.dict_sim_inputs["sim_res"] + r"^3$",
      zorder             = zorder
    )
    if bool_plot_scale: self._plotScales(ax, self.dict_sim_outputs["k_nu_vel_tot_group_t"], color)

  def plotMagSpectrum(self, ax, color="black", label="mag", zorder=3):
    dict_mag_tot_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "mag",
      spect_comp         = "tot",
      file_start_time    = self.time_bounds[0],
      file_end_time      = self.time_bounds[1],
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    plotSpectrum(
      ax                 = ax,
      list_k             = dict_mag_tot_data["list_k_group_t"][0],
      list_power_group_t = dict_mag_tot_data["list_power_group_t"],
      bool_norm          = True,
      color              = color,
      label              = label,
      zorder             = zorder
    )
    self._plotScales(ax, self.dict_sim_outputs["k_p_mag_group_t"], "red")

  def plotCurSpectrum(self, ax, color="black", label="cur", zorder=3):
    dict_cur_tot_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "cur",
      spect_comp         = "tot",
      file_start_time    = self.time_bounds[0],
      file_end_time      = self.time_bounds[1],
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    list_k = dict_cur_tot_data["list_k_group_t"][0]
    list_power_group_t = dict_cur_tot_data["list_power_group_t"]
    plotSpectrum(
      ax                 = ax,
      list_k             = list_k,
      list_power_group_t = list_power_group_t,
      bool_norm          = True,
      color              = color,
      label              = label,
      zorder             = zorder
    )
    self._plotScales(ax, self.dict_sim_outputs["k_eta_cur_group_t"], "green")


## ###############################################################
## PLOT KINETIC ENERGY SPECTRA
## ###############################################################
def plotKinSpectra():
  ## initialise figure
  print("Initialising figure...")
  fig, ax = initFigure()
  list_filepath_sim_res = []
  for mach_regime in [ "Mach0.3", "Mach5" ]:
    for scratch_path in LIST_SCRATCH_PATHS:
      for sim_res in LIST_SIM_RES:
        filepath_sim_res = getSimPath(scratch_path, "Rm3000", mach_regime, sim_res)
        if not os.path.exists(filepath_sim_res): continue
        list_filepath_sim_res.append(filepath_sim_res)
    if len(list_filepath_sim_res) == 0: raise Exception(f"Error: there are no {mach_regime} runs")
    filepath_highest_sim_res = list_filepath_sim_res[-1]
    print("Looking at:", filepath_highest_sim_res)
    obj_plot = PlotSpectra(filepath_highest_sim_res)
    label = getSimLabel(mach_regime, obj_plot.dict_sim_inputs["Re"])
    if obj_plot.desired_Mach < 1:
      obj_plot.plotKinTotSpectrum(ax, label=label, color=COLOR_SUBSONIC, bool_plot_scale=True, zorder=7)
    else: obj_plot.plotKinTotSpectrum(ax, label=label, color=COLOR_SUPERSONIC, bool_plot_scale=True, zorder=5)
  ## label main axis
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlim([ 0.9, 200 ])
  ax.set_ylim([ 1e-7, 1 ])
  ax.set_xlabel(r"$k L_{\rm box} / 2\pi$", fontsize=22)
  ax.set_ylabel(r"$\widehat{\mathcal{P}}_{\rm kin}(k)$", fontsize=22)
  ax.legend(loc="lower left", fontsize=18)
  plotPowerLawPassingThroughPoint(
    ax,
    slope    = -5/2,
    x_domain = (2, 40),
    coord    = (5, 1e-2),
    ls       = "--",
    lw       = 1.75
  )
  ax.text(
    0.225, 0.75,
    r"$k^{-5/2}$",
    va        = "top",
    ha        = "right",
    transform = ax.transAxes,
    color     = "black",
    fontsize  = 20,
    zorder    = 10
  )
  plotPowerLawPassingThroughPoint(
    ax,
    slope    = -5/3,
    x_domain = (2.5, 10),
    coord    = (5, 1.25e-1),
    ls       = ":",
    lw       = 1.75
  )
  ax.text(
    0.375, 0.85,
    r"$k^{-5/3}$",
    va        = "bottom",
    ha        = "left",
    transform = ax.transAxes,
    color     = "black",
    fontsize  = 20,
    zorder    = 10
  )
  ax.text(
    0.75, 0.95,
    r"$k_\nu$",
    va        = "top",
    ha        = "left",
    transform = ax.transAxes,
    color     = "black",
    fontsize  = 20,
    zorder    = 10
  )
  ## save figure
  print("Saving figure...")
  fig_name     = f"spectra_kin_spectra.pdf"
  filepath_fig = f"{PATH_PLOT}/{fig_name}"
  fig.savefig(filepath_fig, dpi=100)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## PLOT KINETIC ENERGY SPECTRA
## ###############################################################
def plotKinReynoldsSpectra():
  ## initialise figure
  print("Initialising figure...")
  fig, ax = initFigure()
  list_filepath_sim_res = []
  for mach_regime in [ "Mach0.3", "Mach5" ]:
    for scratch_path in LIST_SCRATCH_PATHS:
      for sim_res in LIST_SIM_RES:
        filepath_sim_res = getSimPath(scratch_path, "Rm3000", mach_regime, sim_res)
        if not os.path.exists(filepath_sim_res): continue
        list_filepath_sim_res.append(filepath_sim_res)
    if len(list_filepath_sim_res) == 0: raise Exception(f"Error: there are no {mach_regime} runs")
    filepath_highest_sim_res = list_filepath_sim_res[-1]
    print("Looking at:", filepath_highest_sim_res)
    obj_plot = PlotSpectra(filepath_highest_sim_res)
    if obj_plot.desired_Mach < 1:
      obj_plot.plotKinReynoldsSpectrum(ax, color=COLOR_SUBSONIC, bool_plot_scale=True, zorder=7)
    else: obj_plot.plotKinReynoldsSpectrum(ax, color=COLOR_SUPERSONIC, bool_plot_scale=True, zorder=5)
  ## label main axis
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlim([ 0.9, 200 ])
  ax.set_ylim([ 1e-2, 1e4 ])
  ax.set_xlabel(r"$k L_{\rm box} / 2\pi$", fontsize=22)
  ax.set_ylabel(r"$\mathrm{Re}(k)$", fontsize=22)
  ax.axhline(y=1, color="black", ls=":", lw=2, zorder=15)
  ax.text(
    0.75, 0.95,
    r"$k_\nu$",
    va        = "top",
    ha        = "left",
    transform = ax.transAxes,
    color     = "black",
    fontsize  = 20,
    zorder    = 10
  )
  ## save figure
  print("Saving figure...")
  fig_name     = f"spectra_kin_reynolds_spectra.pdf"
  filepath_fig = f"{PATH_PLOT}/{fig_name}"
  fig.savefig(filepath_fig, dpi=100)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## PLOT RATIO OF KINETIC ENERGY SPECTRA COMPONENTS
## ###############################################################
def plotKinCompRatioNres(mach_regime):
  ## initialise figure
  print("Initialising figure...")
  fig, ax = initFigure()
  ## define colormap
  cmap, norm = PlotFuncs.createCmap(
    cmap_name = "cmr.cosmic_r",
    cmin      = 0.1,
    vmin      = 0.0,
    vmax      = len(LIST_SIM_RES)
  )
  ## plot hydro-Reynolds spectra
  for scratch_path in LIST_SCRATCH_PATHS:
    for sim_index in range(len(LIST_SIM_RES)):
      filepath_sim_res = getSimPath(scratch_path, "Re2000", mach_regime, LIST_SIM_RES[sim_index])
      if not os.path.exists(filepath_sim_res): continue
      print("Looking at:", filepath_sim_res)
      obj_plot = PlotSpectra(filepath_sim_res)
      obj_plot.plotKinSpectrumRatios(ax, cmap(norm(sim_index)))
  ## label figure
  ax.set_xlabel(r"$k L_\mathrm{box} / 2\pi$", fontsize=22)
  ax.set_ylabel(r"$\mathcal{P}_{{\rm vel}, \parallel}(k) / \mathcal{P}_{{\rm vel}, \perp}(k)$", fontsize=22)
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_ylim(top=1.2*10**(1))
  ax.axhline(y=1, color="black", ls=":", lw=2, zorder=1)
  ax.text(
    0.05, 0.925,
    getSimLabel(mach_regime, obj_plot.dict_sim_inputs["Re"]),
    va        = "top",
    ha        = "left",
    transform = ax.transAxes,
    color     = "black",
    fontsize  = 20,
    zorder    = 10
  )
  ## save figure
  print("Saving figure...")
  fig_name     = f"spectra_{mach_regime}_kin_component_ratios_Nres.pdf"
  filepath_fig = f"{PATH_PLOT}/{fig_name}"
  fig.savefig(filepath_fig, dpi=100)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## PLOT REYNOLDS SPECTRA (RESOLUTION STUDY)
## ###############################################################
def plotKinReynoldsSpectraNres(mach_regime):
  ## initialise figure
  print("Initialising figure...")
  fig, ax = initFigure()
  ## define colormap
  cmap, norm = PlotFuncs.createCmap(
    cmap_name = "cmr.cosmic_r",
    cmin      = 0.1,
    vmin      = 0.0,
    vmax      = len(LIST_SIM_RES)
  )
  ## plot hydro-Reynolds spectra
  for scratch_path in LIST_SCRATCH_PATHS:
    for sim_index in range(len(LIST_SIM_RES)):
      filepath_sim_res = getSimPath(scratch_path, "Re2000", mach_regime, LIST_SIM_RES[sim_index])
      if not os.path.exists(filepath_sim_res): continue
      print("Looking at:", filepath_sim_res)
      obj_plot = PlotSpectra(filepath_sim_res)
      obj_plot.plotKinReynoldsSpectrum(ax, cmap(norm(sim_index)), bool_plot_scale=False)
  ## label figure
  ax.set_xlabel(r"$k L_\mathrm{box} / 2\pi$", fontsize=22)
  ax.set_ylabel(r"${\rm Re}(k)$", fontsize=22)
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlim([ 0.9, 7*10**(2) ])
  ax.set_ylim([ 10**(-3), 10**(4) ])
  ax.axhline(y=1, color="black", ls=":", lw=2, zorder=1)
  ax.text(
    0.935, 0.925,
    getSimLabel(mach_regime, obj_plot.dict_sim_inputs["Re"]),
    va        = "top",
    ha        = "right",
    transform = ax.transAxes,
    color     = "black",
    fontsize  = 20,
    zorder    = 10
  )
  ax.legend(
    loc            = "lower left",
    ncol           = 2,
    bbox_to_anchor = (0.0, -0.02),
    columnspacing  = 1,
    labelspacing   = 0.35,
    handletextpad  = 0.5,
    fontsize       = 16
  )
  ## save figure
  print("Saving figure...")
  fig_name     = f"spectra_{mach_regime}_reyonolds_Nres.pdf"
  filepath_fig = f"{PATH_PLOT}/{fig_name}"
  fig.savefig(filepath_fig, dpi=100)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## PLOT SUBRSONIC MAGNETIC ENERGY + CURRENT DENSITY SPECTRA
## ###############################################################
def plotMagCurSpectra_subsonic():
  mach_regime = "Mach0.3"
  ## initialise figure
  print("Initialising figure...")
  fig, ax = initFigure()
  list_filepath_sim_res = []
  for scratch_path in LIST_SCRATCH_PATHS:
    for sim_res in LIST_SIM_RES:
      filepath_sim_res = getSimPath(scratch_path, "Rm3000", mach_regime, sim_res)
      if not os.path.exists(filepath_sim_res): continue
      list_filepath_sim_res.append(filepath_sim_res)
  if len(list_filepath_sim_res) == 0: raise Exception(f"Error: there are no {mach_regime} runs")
  filepath_highest_sim_res = list_filepath_sim_res[-1]
  print("Looking at:", filepath_highest_sim_res)
  obj_plot = PlotSpectra(filepath_highest_sim_res)
  obj_plot.plotMagSpectrum(ax, color="red", label=r"$\widehat{\mathcal{P}}_{\rm mag}(k)$")
  obj_plot.plotCurSpectrum(ax, color="green", label=r"$\widehat{\mathcal{P}}_{\rm cur}(k)$")
  ## label axis
  ax.set_xlabel(r"$k L_\mathrm{box} / 2\pi$", fontsize=22)
  ax.set_ylabel(r"$\widehat{\mathcal{P}}(k)$", fontsize=22)
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlim([ 0.9, 400 ])
  ax.set_ylim([ 0.9*10**(-5), 1.1*10**(-1) ])
  ax.text(
    0.95, 0.935,
    getSimLabel(mach_regime, obj_plot.dict_sim_inputs["Re"]),
    va        = "top",
    ha        = "right",
    transform = ax.transAxes,
    color     = "black",
    fontsize  = 20,
    zorder    = 10
  )
  plotPowerLawPassingThroughPoint(
    ax       = ax,
    slope    = 3/2,
    x_domain = ( 1.5, 6 ),
    coord    = ( 1, 1e-3 ),
    ls       = ":"
  )
  ax.text(
    0.15, 0.6,
    "$k^{3/2}$",
    va        = "top",
    ha        = "left",
    transform = ax.transAxes,
    color     = "black",
    fontsize  = 20,
    zorder    = 10
  )
  ## save figure
  print("Saving figure...")
  fig_name     = f"spectra_{mach_regime}_mag_cur_power.pdf"
  filepath_fig = f"{PATH_PLOT}/{fig_name}"
  fig.savefig(filepath_fig, dpi=100)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## PLOT SUPERSONIC MAGNETIC ENERGY + CURRENT DENSITY SPECTRA
## ###############################################################
def plotMagCurSpectra_supersonic():
  mach_regime = "Mach5"
  ## initialise figure
  print("Initialising figure...")
  fig, ax = initFigure()
  list_filepath_sim_res = []
  for scratch_path in LIST_SCRATCH_PATHS:
    for sim_res in LIST_SIM_RES:
      filepath_sim_res = getSimPath(scratch_path, "Rm3000", mach_regime, sim_res)
      if not os.path.exists(filepath_sim_res): continue
      list_filepath_sim_res.append(filepath_sim_res)
  if len(list_filepath_sim_res) == 0: raise Exception(f"Error: there are no {mach_regime} runs")
  filepath_highest_sim_res = list_filepath_sim_res[-1]
  print("Looking at:", filepath_highest_sim_res)
  obj_plot = PlotSpectra(filepath_highest_sim_res)
  obj_plot.plotMagSpectrum(ax, color="red", label=r"$\widehat{\mathcal{P}}_{\rm mag}(k)$")
  obj_plot.plotCurSpectrum(ax, color="green", label=r"$\widehat{\mathcal{P}}_{\rm cur}(k)$")
  ## label axis
  ax.set_xlabel(r"$k L_\mathrm{box} / 2\pi$", fontsize=22)
  ax.set_ylabel(r"$\widehat{\mathcal{P}}(k)$", fontsize=22)
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlim([ 0.9, 400 ])
  ax.set_ylim([ 0.9*10**(-5), 1.1*10**(-1) ])
  ax.legend(
    loc           = "lower right",
    fontsize      = 20,
    labelspacing  = 0.35,
    handletextpad = 0.5,
    frameon       = True,
    framealpha    = 0.75
  )
  ax.text(
    0.95, 0.935,
    getSimLabel(mach_regime, obj_plot.dict_sim_inputs["Re"]),
    va        = "top",
    ha        = "right",
    transform = ax.transAxes,
    color     = "black",
    fontsize  = 20,
    zorder    = 10
  )
  plotPowerLawPassingThroughPoint(
    ax       = ax,
    slope    = 3/2,
    x_domain = ( 1, 3 ),
    coord    = ( 1, 3e-3 ),
    ls       = ":"
  )
  ax.text(
    0.05, 0.6,
    "$k^{3/2}$",
    va        = "top",
    ha        = "left",
    transform = ax.transAxes,
    color     = "black",
    fontsize  = 20,
    zorder    = 10
  )
  ## save figure
  print("Saving figure...")
  fig_name     = f"spectra_{mach_regime}_mag_cur_power.pdf"
  filepath_fig = f"{PATH_PLOT}/{fig_name}"
  fig.savefig(filepath_fig, dpi=100)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  WWFnF.createFolder(PATH_PLOT, bool_verbose=False)
  ## main study
  plotKinSpectra()
  plotKinReynoldsSpectra()
  # plotMagCurSpectra_subsonic()
  # plotMagCurSpectra_supersonic()
  # ## resolution study
  # plotKinCompRatioNres("Mach0.3")
  # plotKinCompRatioNres("Mach5")
  # plotKinReynoldsSpectraNres("Mach0.3")
  # plotKinReynoldsSpectraNres("Mach5")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
COLOR_SUBSONIC   = "#B80EF6"
COLOR_SUPERSONIC = "#F4A123"
PATH_PLOT = "/home/586/nk7952/MHDCodes/kriel2023/spectra/"
LIST_SIM_RES = [ 18, 36, 72, 144, 288, 576, 1152 ]
LIST_SCRATCH_PATHS = [
  "/scratch/ek9/nk7952/",
  "/scratch/jh2/nk7952/"
]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
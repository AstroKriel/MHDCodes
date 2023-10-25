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

def plotSpectraRatios(ax, list_k, list_power_1_group_t, list_power_2_group_t, color, zorder):
  for list_power_1, list_power_2 in zip(
      list_power_1_group_t,
      list_power_2_group_t
    ):
    array_power_ratio = np.array(list_power_1) / np.array(list_power_2)
    ax.plot(list_k, array_power_ratio, color=color, ls="-", lw=1.0, zorder=zorder, alpha=0.25)


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

  def plotEnergySpectraRatios(self, ax, color="black"):
    index_start = self.dict_sim_outputs["index_bounds_growth"][0]
    index_end   = self.dict_sim_outputs["index_bounds_growth"][1]
    k_nu = np.mean(self.dict_sim_outputs["k_nu_kin_group_t"][index_start : index_end])
    k_eta = np.mean(self.dict_sim_outputs["k_eta_cur_group_t"][index_start : index_end])
    ax.axvline(x=k_nu, ls="-", lw=2, color="black")
    ax.axvline(x=k_eta, ls="-", lw=2, color="black")
    ## initial phase
    print("initial phase")
    dict_mag_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "mag",
      file_end_time      = 3,
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    dict_kin_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "kin",
      file_end_time      = 3,
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    list_k = dict_mag_data["list_k_group_t"][0]
    plotSpectraRatios(
      ax                   = ax,
      list_k               = list_k,
      list_power_1_group_t = dict_mag_data["list_power_group_t"],
      list_power_2_group_t = dict_kin_data["list_power_group_t"],
      color  = "black",
      zorder = 3
    )
    ## kinematic phase
    print("kinematic phase")
    dict_mag_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "mag",
      file_start_time    = 3,
      file_end_time      = 10,
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    dict_kin_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "kin",
      file_start_time    = 3,
      file_end_time      = 10,
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    list_k = dict_mag_data["list_k_group_t"][0]
    plotSpectraRatios(
      ax                   = ax,
      list_k               = list_k,
      list_power_1_group_t = dict_mag_data["list_power_group_t"],
      list_power_2_group_t = dict_kin_data["list_power_group_t"],
      color  = "dodgerblue",
      zorder = 5
    )
    ## nonlinear phase
    print("nonlinear phase")
    dict_mag_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "mag",
      file_start_time    = 10,
      file_end_time      = 25,
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    dict_kin_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "kin",
      file_start_time    = 10,
      file_end_time      = 25,
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    list_k = dict_mag_data["list_k_group_t"][0]
    plotSpectraRatios(
      ax                   = ax,
      list_k               = list_k,
      list_power_1_group_t = dict_mag_data["list_power_group_t"],
      list_power_2_group_t = dict_kin_data["list_power_group_t"],
      color  = "limegreen",
      zorder = 3
    )
    ## saturated phase
    print("saturated phase")
    dict_mag_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "mag",
      file_start_time    = 25,
      read_every         = 10,
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    dict_kin_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "kin",
      file_start_time    = 25,
      read_every         = 10,
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    list_k = dict_mag_data["list_k_group_t"][0]
    plotSpectraRatios(
      ax                   = ax,
      list_k               = list_k,
      list_power_1_group_t = dict_mag_data["list_power_group_t"],
      list_power_2_group_t = dict_kin_data["list_power_group_t"],
      color  = "red",
      zorder = 3
    )

  def plotKinSpectrumRatios(self, ax, color="black"):
    dict_lgt_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "kin",
      spect_comp         = "lgt",
      file_start_time    = self.time_bounds[0],
      file_end_time      = self.time_bounds[1],
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    dict_trv_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "kin",
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
      spect_field        = "kin",
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
      spect_field        = "kin",
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
      spect_field        = "kin",
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
      spect_field        = "kin",
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
## PLOT ENERGY SPECTRA RATIO
## ###############################################################
def plotSpectraRatio(filepath_sim_res):
  ## initialise figure
  print("Initialising figure...")
  fig, ax = initFigure()
  ## label main axis
  ax.set_xscale("log")
  ax.set_yscale("log")
  # ax.set_xlim([ 0.9, 200 ])
  # ax.set_ylim([ 1e-7, 1 ])
  ax.set_xlabel(r"$k L_{\rm box} / 2\pi$", fontsize=22)
  ax.set_ylabel(r"$\mathcal{P}_{\rm mag}(k) / \mathcal{P}_{\rm kin}(k)$", fontsize=22)
  ## plot spectra
  obj_plot = PlotSpectra(filepath_sim_res)
  if obj_plot.desired_Mach < 0.75:
    obj_plot.plotEnergySpectraRatios(ax, color=COLOR_SUBSONIC)
  else: obj_plot.plotEnergySpectraRatios(ax, color=COLOR_SUPERSONIC)
  ax.axhline(y=1, color="red", ls="--", lw=1.5, zorder=15)
  ## save figure
  print("Saving figure...")
  if obj_plot.desired_Mach < 0.75:
    sim_name = "subsonic"
  else: sim_name = "supersonic"
  fig_name = f"spectra_{sim_name}_spectra_ratio.png"
  filepath_fig = f"{PATH_PLOT}/{fig_name}"
  fig.savefig(filepath_fig, dpi=200)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## PLOT KINETIC ENERGY SPECTRA
## ###############################################################
def plotKinSpectra(filepath_sim_res):
  ## initialise figure
  print("Initialising figure...")
  fig, ax = initFigure()
  print("Looking at:", filepath_sim_res)
  obj_plot = PlotSpectra(filepath_sim_res)
  if obj_plot.desired_Mach < 1:
    obj_plot.plotKinTotSpectrum(ax, color=COLOR_SUBSONIC, bool_plot_scale=True)
  else: obj_plot.plotKinTotSpectrum(ax, color=COLOR_SUPERSONIC, bool_plot_scale=True)
  ## label main axis
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlabel(r"$k L_{\rm box} / 2\pi$", fontsize=22)
  ax.set_ylabel(r"$\mathcal{P}_{\rm kin}(k)$", fontsize=22)
  if obj_plot.desired_Mach < 0.75:
    domain = (2, 10)
  else: domain = (2, 30)
  plotPowerLawPassingThroughPoint(
    ax,
    slope    = -5/3,
    x_domain = domain,
    coord    = (2, 5e-2),
    ls       = "-",
    lw       = 1.5
  )
  plotPowerLawPassingThroughPoint(
    ax,
    slope    = -2,
    x_domain = domain,
    coord    = (2, 5e-2),
    ls       = "--",
    lw       = 1.5
  )
  ## save figure
  ax.set_xlim([0.8, 200])
  print("Saving figure...")
  if obj_plot.desired_Mach < 0.75:
    sim_name = "subsonic"
    ax.set_ylim([10**(-9), 10**(0)])
  else:
    sim_name = "supersonic"
    ax.set_ylim([10**(-7), 10**(0)])
  fig_name = f"spectra_{sim_name}_spectra_kin.png"
  filepath_fig = f"{PATH_PLOT}/{fig_name}"
  fig.savefig(filepath_fig, dpi=200)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  WWFnF.createFolder(PATH_PLOT, bool_verbose=False)
  for filepath in [
      "/scratch/ek9/nk7952/Rm3000/Mach0.3/Pm5/288",
      "/scratch/ek9/nk7952/Rm3000/Mach5/Pm5/288"
    ]:
    plotSpectraRatio(filepath)
    # plotKinSpectra(filepath)


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
COLOR_SUBSONIC   = "#B80EF6"
COLOR_SUPERSONIC = "#F4A123"
PATH_PLOT = "/home/586/nk7952/MHDCodes/ii6/spectra/"
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
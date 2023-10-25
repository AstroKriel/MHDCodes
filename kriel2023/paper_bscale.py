#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys
import cmasher as cmr
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["axes.axisbelow"] = False


## load user defined modules
from TheFlashModule import LoadData, SimParams
from TheUsefulModule import WWFnF
from TheFittingModule import FitMHDScales
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def plotLinePassingThroughPoint(
    ax, x_domain, slope, coord,
    ls = "-",
    lw = 2.0
  ):
  x1, y1 = coord
  a0 = y1 - slope * x1
  x = np.linspace(x_domain[0], x_domain[1], 100)
  y = a0 + slope * x
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

def addText(
    ax, pos, text,
    rotation = 0,
    fontsize = 20,
    va       = "bottom",
    ha       = "left",
  ):
  ax.text(
    pos[0], pos[1],
    text,
    va            = va,
    ha            = ha,
    transform     = ax.transAxes,
    rotation      = rotation,
    fontsize      = fontsize,
    rotation_mode = "anchor",
    color         = "black",
    zorder        = 10
  )

def measureScales(list_k, list_power_group_t):
  if not(len(list_k) == len(list_power_group_t[0])): return
  array_k = np.array(list_k)
  kp_group_t = []
  kcor_group_t = []
  for t_index in range(len(list_power_group_t)):
    array_power = np.array(list_power_group_t[t_index])
    kp, _ = FitMHDScales.getSpectrumPeakScale(list_k, array_power)
    t1 = np.sum([ p/k for k,p in zip(array_k, array_power) ])
    t2 = np.sum(array_power)
    kcor = t2 / t1
    kp_group_t.append(kp)
    kcor_group_t.append(kcor)
  return kp_group_t, kcor_group_t


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class CompareScales():
  def __init__(self, filepath_sim_res):
    self.filepath_spect   = f"{filepath_sim_res}/spect/"
    self.dict_sim_inputs  = SimParams.readSimInputs(filepath_sim_res, False)
    self.dict_sim_outputs = SimParams.readSimOutputs(filepath_sim_res, False)
    self.desired_Mach     = self.dict_sim_inputs["desired_Mach"]
    if self.desired_Mach < 2:
      self.time_start_exp   = 5.5
      self.time_end_exp     = 16
      self.time_start_sat   = 26
    else:
      self.time_start_exp   = 5.5
      self.time_end_exp     = 38
      self.time_start_sat   = 53

  def plotScales(self, ax, fig):
    dict_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "mag",
      file_start_time    = 1,
      file_end_time      = 100,
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    list_time = dict_data["list_turb_times"]
    kp_group_t, kcor_group_t = measureScales(
      dict_data["list_k_group_t"][0],
      dict_data["list_power_group_t"]
    )
    y_vals = np.array(kcor_group_t) / np.array(kp_group_t)
    if self.desired_Mach < 2:
      y_vals[y_vals > 2] = np.nan
    obj_plot = ax.hexbin(
      list_time,
      y_vals,
      cmap = cmr.get_sub_cmap("viridis", 0.0, 1.0),
      gridsize=(40,15), extent=(0, 2, 0.5, 4.8), xscale="log",
      mincnt=1, bins="log", zorder=3
    )
    if self.desired_Mach < 2:
      ax_cbar = fig.add_axes([ 0.125, 0.9, 0.775, 0.06 ])
      fig.colorbar(mappable=obj_plot, cax=ax_cbar, orientation="horizontal")
      ax_cbar.set_title(r"$\mathrm{counts}$", fontsize=20, pad=10)
      ax_cbar.xaxis.set_ticks_position("top")
      dict_label = { "ha":"right", "va":"center", "rotation":90 }
      addText(ax, (0.04, 0.795), r"${\rm transient\, phase}$", **dict_label)
      addText(ax, (0.04+np.log10(self.time_start_exp)/2, 0.95), r"${\rm exponential\, growth}$", **dict_label)
      addText(ax, (0.035+np.log10(self.time_end_exp)/2, 0.95), r"${\rm linear\, growth}$", **dict_label)
      addText(ax, (0.025+np.log10(self.time_start_sat)/2, 0.95), r"${\rm saturated}$", **dict_label)
    ax.axvspan(
      0.9,
      self.time_start_exp,
      color="grey", alpha=0.25, zorder=1, linewidth=0.1
    )
    ax.axvspan(
      self.time_start_exp,
      self.time_end_exp,
      color="forestgreen", alpha=0.25, zorder=1, linewidth=0.1
    )
    ax.axvspan(
      self.time_end_exp,
      self.time_start_sat,
      color="dodgerblue", alpha=0.35, zorder=1, linewidth=0.1
    )
    ax.axvspan(
      self.time_start_sat,
      120,
      color="crimson", alpha=0.35, zorder=1, linewidth=0.1
    )


## ###############################################################
## PLOT B-FIELD SCALES
## ###############################################################
def plotScaleComparison(mach_regime):
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
  obj_plot = CompareScales(filepath_highest_sim_res)
  obj_plot.plotScales(ax, fig)
  ## label axis
  ax.set_xscale("log")
  if obj_plot.desired_Mach < 2:
    ax.set_xticklabels([ ])
    addText(
      ax, (0.025, 0.95),
      r"${\rm }$",
      va = "bottom",
      ha = "left"
    )
  else: ax.set_xlabel(r"$t / t_{\rm turb}$", fontsize=22)
  ax.set_ylabel(r"$k_{\rm cor} / k_{\rm p}$", fontsize=22)
  ax.set_xlim([ 0.9, 120 ])
  ax.set_ylim([ 0.5, 4.8 ])
  ax.axhline(y=1, ls=":", lw=2, color="black", alpha=0.5)
  ax.axhline(y=2, ls=":", lw=2, color="black", alpha=0.5)
  ax.axhline(y=3, ls=":", lw=2, color="black", alpha=0.5)
  ax.axhline(y=4, ls=":", lw=2, color="black", alpha=0.5)
  addText(
    ax, (0.0235, 0.935),
    getSimLabel(mach_regime, obj_plot.dict_sim_inputs["Re"]),
    va = "top",
    ha = "left"
  )
  ## save figure
  print("Saving figure...")
  fig_name     = f"mag_scales_{mach_regime}.pdf"
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
  plotScaleComparison("Mach0.3")
  plotScaleComparison("Mach5")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
COLOR_SUBSONIC   = "#B80EF6"
COLOR_SUPERSONIC = "#F4A123"
PATH_PLOT = "/home/586/nk7952/MHDCodes/kriel2023/spectra/"
LIST_SIM_RES = [ 18, 36, 72, 144 ] # , 288, 576, 1152
LIST_SCRATCH_PATHS = [
  "/scratch/ek9/nk7952/",
  # "/scratch/jh2/nk7952/"
]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
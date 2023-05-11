#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np
import matplotlib.pyplot as plt

## load user defined modules
from TheFlashModule import LoadData, SimParams
from TheAnalysisModule import WWSpectra
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def addText(ax, pos, text):
  ax.text(
    pos[0], pos[1],
    text,
    va        = "bottom",
    ha        = "left",
    transform = ax.transAxes,
    color     = "black",
    fontsize  = 24,
    zorder    = 10
  )

def reynoldsSpectrum(list_k, list_power, diss_rate):
    list_power_reverse = np.array(list_power[::-1])
    list_sqt_sum_power = np.sqrt(np.cumsum(list_power_reverse))[::-1]
    return list_sqt_sum_power / (diss_rate * np.array(list_k))

def plotReynoldsSpectrum(ax, list_k, list_power_group_t, diss_rate, color="black", bool_norm=False):
  array_reynolds_group_t = []
  for list_power in list_power_group_t:
    if bool_norm: list_power = WWSpectra.normSpectra(list_power)
    array_reynolds_group_t.append(reynoldsSpectrum(list_k, list_power, diss_rate))
  array_reynolds_ave = np.mean(array_reynolds_group_t, axis=0)
  ax.plot(list_k, array_reynolds_ave, color=color, ls="-", lw=2.0, zorder=5)
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
      capsize = 0.0,
      zorder  = 3
    )

def plotSpectrum(
    ax, list_k, list_power_group_t,
    label       = None,
    color       = "black",
    comp_factor = 1.0,
    bool_norm   = False,
  ):
  array_power_comp_group_t = []
  for list_power in list_power_group_t:
    if bool_norm: list_power = WWSpectra.normSpectra(list_power)
    array_power_comp  = np.array(list_power)*np.array(list_k)**(comp_factor)
    array_power_comp_group_t.append(array_power_comp)
  array_power_comp_ave = np.mean(array_power_comp_group_t, axis=0)
  ax.plot(list_k, array_power_comp_ave, color=color, label=label, ls="-", lw=2.0, zorder=5)
  array_power_comp_group_k = [
    [
      array_power_comp[k_index]
      for array_power_comp in array_power_comp_group_t
    ]
    for k_index in range(len(list_k))
  ]
  for k_index in range(len(list_k)):
    PlotFuncs.plotErrorBar_1D(
      ax      = ax,
      x       = list_k[k_index],
      array_y = array_power_comp_group_k[k_index],
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

  def plotKinSpectrum(self, axs, color="black"):
    dict_kin_tot_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "kin",
      spect_comp         = "tot",
      file_start_time    = self.dict_sim_outputs["time_bounds_growth"][0],
      file_end_time      = self.dict_sim_outputs["time_bounds_growth"][1],
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    if self.dict_sim_outputs["rms_Mach_growth"] > 1:
      comp_factor = 3/2
    else: comp_factor = 5/3
    plotSpectrum(
      ax                 = axs[0],
      list_k             = dict_kin_tot_data["list_k_group_t"][0],
      list_power_group_t = dict_kin_tot_data["list_power_group_t"],
      comp_factor        = comp_factor,
      color              = color,
      label              = r"$"+self.dict_sim_inputs["sim_res"]+r"^3$"
    )
    plotReynoldsSpectrum(
      ax                 = axs[1],
      list_k             = dict_kin_tot_data["list_k_group_t"][0],
      list_power_group_t = dict_kin_tot_data["list_power_group_t"],
      diss_rate          = self.dict_sim_inputs["nu"],
      color              = color
    )
    return comp_factor

  def plotMagSpectra(self, ax, color="black"):
    dict_mag_tot_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "mag",
      spect_comp         = "tot",
      file_start_time    = self.dict_sim_outputs["time_bounds_growth"][0],
      file_end_time      = self.dict_sim_outputs["time_bounds_growth"][1],
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    plotSpectrum(
      ax                 = ax,
      list_k             = dict_mag_tot_data["list_k_group_t"][0],
      list_power_group_t = dict_mag_tot_data["list_power_group_t"],
      bool_norm          = True,
      comp_factor        = -3/2,
      color              = color,
      label              = r"$"+self.dict_sim_inputs["sim_res"]+r"^3$"
    )

  def plotCurSpectra(self, ax, color="black"):
    dict_cur_tot_data = LoadData.loadAllSpectra(
      directory          = self.filepath_spect,
      spect_field        = "cur",
      spect_comp         = "tot",
      file_start_time    = self.dict_sim_outputs["time_bounds_growth"][0],
      file_end_time      = self.dict_sim_outputs["time_bounds_growth"][1],
      outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"],
      bool_verbose       = False
    )
    plotSpectrum(
      ax                 = ax,
      list_k             = dict_cur_tot_data["list_k_group_t"][0],
      list_power_group_t = dict_cur_tot_data["list_power_group_t"],
      bool_norm          = True,
      comp_factor        = -3,
      color              = color,
      label              = r"$"+self.dict_sim_inputs["sim_res"]+r"^3$"
    )


## ###############################################################
## PLOT KINETIC ENERGY SPECTRA
## ###############################################################
def plotKinSpectra(mach_regime):
  ## initialise figure
  print("Initialising figure...")
  figscale = 1.2
  fig, axs = plt.subplots(
    nrows   = 2,
    figsize = (10*figscale, 2*6*figscale),
    sharex  = True
  )
  fig.subplots_adjust(hspace=0.05)
  ## define colormap
  list_sim_res = [ 18, 36, 72, 144, 288, 576, 1152 ]
  cmap, norm = PlotFuncs.createCmap(
    cmap_name = "cmr.cosmic_r",
    cmin      = 0.1,
    vmin      = 0.0,
    vmax      = len(list_sim_res)
  )
  ## plot kinetic energy spectra
  for scratch_path in [
      "/scratch/ek9/nk7952/",
      # "/scratch/jh2/nk7952/"
    ]:
    for sim_index in range(len(list_sim_res)):
      filepath_sim_res = f"{scratch_path}/Re2000/{mach_regime}/Pm5/{list_sim_res[sim_index]:d}/"
      if not os.path.exists(filepath_sim_res): continue
      print("Looking at:", filepath_sim_res)
      obj_plot = PlotSpectra(filepath_sim_res)
      comp_factor = obj_plot.plotKinSpectrum(axs, cmap(norm(sim_index)))
  ## label figure
  if "0.3" in mach_regime: addText(axs[0], (0.05, 0.35), r"$\mathcal{M} = 0.3$")
  elif "5" in mach_regime: addText(axs[0], (0.05, 0.35), r"$\mathcal{M} = 5$")
  else: raise Exception(f"Error: mach_regime={mach_regime} is unexpected.")
  addText(axs[0], (0.05, 0.25), r"{\rm Re} = 2,000")
  addText(axs[0], (0.05, 0.15), r"{\rm Rm} = 10,000")
  addText(axs[0], (0.05, 0.05), r"{\rm Pm} = 5")
  axs[0].legend(loc="upper right", fontsize=24)
  axs[1].set_xlabel(r"$k$")
  if   comp_factor == 5/3: axs[0].set_ylabel(r"$k^{5/3} \, \mathcal{P}_{\rm kin}(k, t)$")
  elif comp_factor == 3/2: axs[0].set_ylabel(r"$k^{3/2} \, \mathcal{P}_{\rm kin}(k, t)$")
  else: raise Exception(f"Error: comp_factor={comp_factor} is unexpected.")
  axs[1].set_ylabel(r"$\int_k^\infty \big[ \mathcal{P}_{\rm kin}(k^\prime, t) \big]^{1/2} {\rm d}k^\prime / \nu k$")
  axs[1].axhline(y=1, ls=":", color="black", lw=2.0)
  for ax in axs:
    ax.set_xscale("log")
    ax.set_yscale("log")
  ## save figure
  print("Saving figure...")
  fig_name     = f"spectra_kin_{mach_regime}.pdf"
  filepath_fig = f"{PATH_PLOT}/{fig_name}"
  fig.savefig(filepath_fig, dpi=100)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## PLOT MAGNETIC ENERGY SPECTRA
## ###############################################################
def plotMagSpectra(mach_regime):
  ## initialise figure
  print("Initialising figure...")
  figscale = 1.2
  fig, ax = plt.subplots(figsize=(10*figscale, 6*figscale))
  ## define colormap
  list_sim_res = [ 18, 36, 72, 144, 288, 576, 1152 ]
  cmap, norm = PlotFuncs.createCmap(
    cmap_name = "cmr.sunburst_r",
    cmin      = 0.1,
    vmin      = 0.0,
    vmax      = len(list_sim_res)
  )
  ## plot magnetic energy spectra
  for scratch_path in [
      "/scratch/ek9/nk7952/",
      # "/scratch/jh2/nk7952/"
    ]:
    for sim_index in range(len(list_sim_res)):
      filepath_sim_res = f"{scratch_path}/Re2000/{mach_regime}/Pm5/{list_sim_res[sim_index]:d}/"
      if not os.path.exists(filepath_sim_res): continue
      print("Looking at:", filepath_sim_res)
      obj_plot = PlotSpectra(filepath_sim_res)
      obj_plot.plotMagSpectra(ax, cmap(norm(sim_index)))
  ## label figure
  if "0.3" in mach_regime: addText(ax, (0.05, 0.35), r"$\mathcal{M} = 0.3$")
  elif "5" in mach_regime: addText(ax, (0.05, 0.35), r"$\mathcal{M} = 5$")
  else: raise Exception(f"Error: mach_regime={mach_regime} is unexpected.")
  addText(ax, (0.05, 0.25), r"{\rm Re} = 2,000")
  addText(ax, (0.05, 0.15), r"{\rm Rm} = 10,000")
  addText(ax, (0.05, 0.05), r"{\rm Pm} = 5")
  ax.legend(loc="upper right", fontsize=24)
  ax.set_xlabel(r"$k$")
  ax.set_ylabel(r"$\mathcal{P}_{\rm mag}(k, t)$")
  ax.set_xscale("log")
  ax.set_yscale("log")
  ## save figure
  print("Saving figure...")
  fig_name     = f"spectra_mag_{mach_regime}.pdf"
  filepath_fig = f"{PATH_PLOT}/{fig_name}"
  fig.savefig(filepath_fig, dpi=100)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## PLOT CURRENT DENSITY SPECTRA
## ###############################################################
def plotCurSpectra(mach_regime):
  ## initialise figure
  print("Initialising figure...")
  figscale = 1.2
  fig, ax = plt.subplots(figsize=(10*figscale, 6*figscale))
  ## define colormap
  list_sim_res = [ 18, 36, 72, 144, 288, 576, 1152 ]
  cmap, norm = PlotFuncs.createCmap(
    cmap_name = "cmr.nuclear_r",
    cmin      = 0.1,
    vmin      = 0.0,
    vmax      = len(list_sim_res)
  )
  ## plot current density spectra
  for scratch_path in [
      "/scratch/ek9/nk7952/",
      # "/scratch/jh2/nk7952/"
    ]:
    for sim_index in range(len(list_sim_res)):
      filepath_sim_res = f"{scratch_path}/Re2000/{mach_regime}/Pm5/{list_sim_res[sim_index]:d}/"
      if not os.path.exists(filepath_sim_res): continue
      print("Looking at:", filepath_sim_res)
      obj_plot = PlotSpectra(filepath_sim_res)
      obj_plot.plotCurSpectra(ax, cmap(norm(sim_index)))
  ## label figure
  if "0.3" in mach_regime: addText(ax, (0.05, 0.35), r"$\mathcal{M} = 0.3$")
  elif "5" in mach_regime: addText(ax, (0.05, 0.35), r"$\mathcal{M} = 5$")
  else: raise Exception(f"Error: mach_regime={mach_regime} is unexpected.")
  addText(ax, (0.05, 0.25), r"{\rm Re} = 2,000")
  addText(ax, (0.05, 0.15), r"{\rm Rm} = 10,000")
  addText(ax, (0.05, 0.05), r"{\rm Pm} = 5")
  ax.legend(loc="upper right", fontsize=24)
  ax.set_xlabel(r"$k$")
  ax.set_ylabel(r"$\mathcal{P}_{\rm cur}(k, t)$")
  ax.set_xscale("log")
  ax.set_yscale("log")
  ## save figure
  print("Saving figure...")
  fig_name     = f"spectra_cur_{mach_regime}.pdf"
  filepath_fig = f"{PATH_PLOT}/{fig_name}"
  fig.savefig(filepath_fig, dpi=100)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  # plotKinSpectra("Mach0.3")
  # plotKinSpectra("Mach5")
  plotMagSpectra("Mach0.3")
  plotMagSpectra("Mach5")
  plotCurSpectra("Mach0.3")
  plotCurSpectra("Mach5")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
PATH_PLOT = "/home/586/nk7952/MHDCodes/kriel2023/"


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
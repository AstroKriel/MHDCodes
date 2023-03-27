#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate

## load user defined modules
from TheFlashModule import SimParams, LoadFlashData, FileNames
from TheUsefulModule import WWFnF, WWLists, WWObjs
from ThePlottingModule import PlotFuncs, PlotLatex

## ##############################################################
## PREPARE WORKSPACE
## ##############################################################
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## MAIN PROGRAM
## ###############################################################
class PlotCurrentSpectra():
  def __init__(self, filepath_sim_res):
    ## save inputs
    self.filepath_sim_res = filepath_sim_res
    ## get simulation parameters
    self.dict_sim_inputs = SimParams.readSimInputs(filepath_sim_res, bool_verbose=False)
    self.sim_name = SimParams.getSimName(self.dict_sim_inputs)
    ## make sure a visualisation folder exists
    self.filepath_vis = f"{self.filepath_sim_res}/vis_folder/"
    WWFnF.createFolder(self.filepath_vis, bool_verbose=False)

  def performRoutines(self):
    self.fig, self.ax = plt.subplots()
    self._loadCurrentSpectra()
    self._plotCurrentSpectra()
    self._labelAxis()
    fig_name = f"{self.sim_name}_current_spectra.png"
    PlotFuncs.saveFigure(self.fig, f"{self.filepath_vis}/{fig_name}")

  def _loadCurrentSpectra(self):
    ## load simulation parameters
    dict_sim_inputs    = SimParams.readSimInputs(self.filepath_sim_res)
    dict_sim_outputs   = SimParams.readSimOutputs(self.filepath_sim_res)
    outputs_per_t_turb = dict_sim_outputs["outputs_per_t_turb"]
    time_growth_end    = dict_sim_outputs["time_growth_end"]
    time_growth_start  = dict_sim_outputs["time_growth_start"]
    self.sim_res       = dict_sim_inputs["sim_res"]
    self.Re            = dict_sim_inputs["Re"]
    self.Rm            = dict_sim_inputs["Rm"]
    self.Pm            = dict_sim_inputs["Pm"]
    ## load energy spectra
    print("Loading kinetic energy spectra...")
    dict_current_spect = LoadFlashData.loadAllSpectra(
      filepath           = f"{self.filepath_sim_res}/plt/",
      spect_field        = "cur",
      spect_comp         = "tot",
      file_start_time    = time_growth_start,
      file_end_time      = time_growth_end,
      outputs_per_t_turb = outputs_per_t_turb
    )
    ## extract data
    self.list_k                     = dict_current_spect["list_k_group_t"][0]
    self.list_power_cur_tot_group_t = dict_current_spect["list_power_group_t"]
    self.list_turb_times            = dict_current_spect["list_turb_times"]

  def __plotAveKinComponents(self, ax):
    ax.plot(
      self.list_k,
      np.mean(self.list_power_kin_lgt_group_t, axis=0),
      "k--", lw=2, label=r"$\mathcal{P}_{\rm kin, \parallel}(k, t)$", zorder=3
    )
    ax.plot(
      self.list_k,
      np.mean(self.list_power_kin_trv_group_t, axis=0),
      "k-",  lw=2, label=r"$\mathcal{P}_{\rm kin, \perp}(k, t)$", zorder=3
    )

  def __plotData(self):
    label_kin_tot = r"$\mathcal{P}_{\rm kin, tot}(k, t)$"
    label_mag_tot = r"$\mathcal{P}_{\rm mag, tot}(k, t)$"
    label_kin_lgt = r"$\mathcal{P}_{\rm kin, \parallel}(k, t)$"
    label_kin_trv = r"$\mathcal{P}_{\rm kin, \perp}(k, t)$"
    ## TOTAL KINETIC ENERGY
    ## --------------------
    ## plot total spectra
    self.axs[0,0].set_ylabel(label_kin_tot)
    self.__plotAveKinComponents(self.axs[0,0])
    plotSpectra_timeEvolve(
      fig                = self.fig,
      ax                 = self.axs[0,0],
      list_turb_times     = self.list_kin_turb_times,
      list_k             = self.list_k,
      list_power_group_t = self.list_power_kin_tot_group_t,
      cmap_name          = "Greens",
      bool_add_colorbar  = True
    )


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  





## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
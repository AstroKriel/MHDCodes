#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys, functools
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mproc
import concurrent.futures as cfut

## load user defined modules
from TheSimModule import SimParams
from TheUsefulModule import WWFnF, WWObjs
from TheFittingModule import FitFuncs
from TheLoadingModule import LoadFlashData, FileNames
from TheAnalysisModule import WWSpectra
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def createSubAxis(fig, fig_grid, row_index):
  return [
    fig.add_subplot(fig_grid[row_index, 0]),
    fig.add_subplot(fig_grid[row_index, 1]),
    fig.add_subplot(fig_grid[row_index, 2])
  ]

def getSpectrumCompensated(list_k, list_power, alpha_comp):
  return np.array(list_power) * np.array(list_k)**(alpha_comp)

def getReynoldsSpectrum(list_k, list_power, viscosity):
  array_reynolds = np.sqrt(np.cumsum(2 * np.array(list_power[::-1])))[::-1] / (np.array(list_k) * viscosity)
  k_dissipation  = None
  if np.log10(min(array_reynolds)) < 1e-1:
    list_k_interp = np.logspace(np.log10(min(list_k)), np.log10(max(list_k)), 10**4)
    list_reynolds_interp = FitFuncs.interpLogLogData(list_k, array_reynolds, list_k_interp, interp_kind="cubic")
    dis_scale_index = np.argmin(abs(list_reynolds_interp - 1.0))
    k_dissipation   = list_k_interp[dis_scale_index]
  return array_reynolds, k_dissipation

def plotSpectra_res(axs, filepath_sim_res, sim_res, bool_verbose=True):
  ## look up table for plot styles
  dict_plot_style = {
    "18"  : "r--",
    "36"  : "g--",
    "72"  : "b--",
    "144" : "r-",
    "288" : "g-",
    "576" : "b-"
  }
  plot_style = dict_plot_style[sim_res]
  ## helper functions
  def plotSpectrum(ax_row, list_k, list_power, alpha_comp):
    list_power_comp = getSpectrumCompensated(list_k, list_power, alpha_comp)
    axs[ax_row][0].plot(list_k, list_power,      plot_style, label=sim_res)
    axs[ax_row][1].plot(list_k, list_power_comp, plot_style, label=sim_res)
  def plotReynoldsSpectrum(ax_row, list_k, list_power, viscosity):
    array_reynolds, k_dis = getReynoldsSpectrum(list_k, list_power, viscosity)
    axs[ax_row][2].plot(list_k, array_reynolds, plot_style, label=sim_res)
    if k_dis is not None: axs[ax_row][2].plot(k_dis, 1.0, "ko")
    axs[ax_row][2].axhline(y=1, ls=":", c="k")
    return k_dis
  ## load relevant data
  if bool_verbose: print(f"Reading in data:", filepath_sim_res)
  dict_sim_inputs  = SimParams.readSimInputs(filepath_sim_res,  bool_verbose=False)
  dict_sim_outputs = SimParams.readSimOutputs(filepath_sim_res, bool_verbose=False)
  ## read in time-averaged spectra
  filepath_spect     = f"{filepath_sim_res}/spect"
  list_k             = dict_sim_outputs["list_k"]
  plots_per_eddy     = dict_sim_outputs["plots_per_eddy"]
  time_growth_start  = dict_sim_outputs["time_growth_start"]
  time_growth_end    = dict_sim_outputs["time_growth_end"]
  dict_mag_spect_tot_data = LoadFlashData.loadAllSpectra(
    filepath        = filepath_spect,
    spect_field     = "mag",
    spect_quantity  = "tot",
    file_start_time = time_growth_start,
    file_end_time   = time_growth_end,
    plots_per_eddy  = plots_per_eddy,
    bool_verbose    = False
  )
  dict_kin_spect_tot_data = LoadFlashData.loadAllSpectra(
    filepath        = filepath_spect,
    spect_field     = "vel",
    spect_quantity  = "tot",
    file_start_time = time_growth_start,
    file_end_time   = time_growth_end,
    plots_per_eddy  = plots_per_eddy,
    bool_verbose    = False
  )
  dict_kin_spect_trv_data = LoadFlashData.loadAllSpectra(
    filepath        = filepath_spect,
    spect_field     = "vel",
    spect_quantity  = "trv",
    file_start_time = time_growth_start,
    file_end_time   = time_growth_end,
    plots_per_eddy  = plots_per_eddy,
    bool_verbose    = False
  )
  dict_kin_spect_lgt_data = LoadFlashData.loadAllSpectra(
    filepath        = filepath_spect,
    spect_field     = "vel",
    spect_quantity  = "lgt",
    file_start_time = time_growth_start,
    file_end_time   = time_growth_end,
    plots_per_eddy  = plots_per_eddy,
    bool_verbose    = False
  )
  ## plot total magnetic energy spectra
  plotSpectrum(
    ax_row     = 0,
    list_k     = list_k,
    list_power = WWSpectra.aveSpectra(dict_mag_spect_tot_data["list_power_group_t"], bool_norm=True),
    alpha_comp = -3/2
  )
  k_eta = plotReynoldsSpectrum(
    ax_row     = 0,
    list_k     = list_k,
    list_power = WWSpectra.aveSpectra(dict_kin_spect_tot_data["list_power_group_t"], bool_norm=True),
    viscosity  = dict_sim_inputs["eta"]
  )
  list_mag_power = WWSpectra.aveSpectra(dict_mag_spect_tot_data["list_power_group_t"], bool_norm=True)
  k_p_index = np.argmax(list_mag_power)
  k_p = list_k[k_p_index]
  axs[0][0].plot(k_p, list_mag_power[k_p_index], "ko")
  ## plot total kinetic energy spectra
  plotSpectrum(
    ax_row     = 1,
    list_k     = list_k,
    list_power = WWSpectra.aveSpectra(dict_kin_spect_tot_data["list_power_group_t"], bool_norm=False),
    alpha_comp = 2.0
  )
  k_nu_tot = plotReynoldsSpectrum(
    ax_row     = 1,
    list_k     = list_k,
    list_power = WWSpectra.aveSpectra(dict_kin_spect_tot_data["list_power_group_t"], bool_norm=True),
    viscosity  = dict_sim_inputs["nu"]
  )
  ## plot longitudinal kinetic energy spectra
  plotSpectrum(
    ax_row     = 2,
    list_k     = list_k,
    list_power = WWSpectra.aveSpectra(dict_kin_spect_lgt_data["list_power_group_t"], bool_norm=False),
    alpha_comp = 2.0
  )
  k_nu_lgt = plotReynoldsSpectrum(
    ax_row     = 2,
    list_k     = list_k,
    list_power = WWSpectra.aveSpectra(dict_kin_spect_lgt_data["list_power_group_t"], bool_norm=True),
    viscosity  = dict_sim_inputs["nu"]
  )
  ## plot transverse kinetic energy spectra
  plotSpectrum(
    ax_row     = 3,
    list_k     = list_k,
    list_power = WWSpectra.aveSpectra(dict_kin_spect_trv_data["list_power_group_t"], bool_norm=False),
    alpha_comp = 2.0
  )
  k_nu_trv = plotReynoldsSpectrum(
    ax_row     = 3,
    list_k     = list_k,
    list_power = WWSpectra.aveSpectra(dict_kin_spect_trv_data["list_power_group_t"], bool_norm=True),
    viscosity  = dict_sim_inputs["nu"]
  )
  ## return all scales measured at resolution
  return k_p, k_eta, k_nu_tot, k_nu_lgt, k_nu_trv


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def plotSpectra(filepath_sim, bool_verbose=True):
  fig, fig_grid = PlotFuncs.createFigure_grid(num_rows=4, num_cols=3)
  axs = [
    createSubAxis(fig, fig_grid, row_index=0),
    createSubAxis(fig, fig_grid, row_index=1),
    createSubAxis(fig, fig_grid, row_index=2),
    createSubAxis(fig, fig_grid, row_index=3)
  ]
  dict_scales = {}
  for sim_res in LIST_SIM_RES:
    filepath_sim_res = f"{filepath_sim}/{sim_res}/"
    if not os.path.exists(f"{filepath_sim_res}/spect/"): continue
    k_p, k_eta, k_nu_tot, k_nu_lgt, k_nu_trv = plotSpectra_res(axs, filepath_sim_res, sim_res, bool_verbose)
    dict_scales.update({
      f"{sim_res} k_p"      : k_p,
      f"{sim_res} k_eta"    : k_eta,
      f"{sim_res} k_nu_tot" : k_nu_tot,
      f"{sim_res} k_nu_lgt" : k_nu_lgt,
      f"{sim_res} k_nu_trv" : k_nu_trv
    })
  ## save measured scales
  if bool_verbose: print("Saving measured scales...")
  WWObjs.saveDict2JsonFile(f"{filepath_sim}/{FileNames.FILENAME_SIM_SCALES}", dict_scales, bool_verbose=True)
  ## adjust figure axis
  for row_index in range(len(axs)):
    for col_index in range(len(axs[0])):
      axs[row_index][col_index].legend(loc="upper right")
      axs[row_index][col_index].set_xscale("log")
      axs[row_index][col_index].set_yscale("log")
  ## label axis
  label_mag_tot = r"$\widehat{\mathcal{P}}_{\rm mag, tot}(k)$"
  label_kin_tot = r"$\widehat{\mathcal{P}}_{\rm kin, tot}(k)$"
  label_kin_lgt = r"$\widehat{\mathcal{P}}_{\rm kin, \parallel}(k)$"
  label_kin_trv = r"$\widehat{\mathcal{P}}_{\rm kin, \perp}(k)$"
  axs[0][0].set_ylabel(label_mag_tot)
  axs[1][0].set_ylabel(label_kin_tot)
  axs[2][0].set_ylabel(label_kin_lgt)
  axs[3][0].set_ylabel(label_kin_trv)
  axs[0][1].set_ylabel(r"$k^{-3/2} \,$" + label_mag_tot)
  axs[1][1].set_ylabel(r"$k^{2} \,$"    + label_kin_tot)
  axs[2][1].set_ylabel(r"$k^{2} \,$"    + label_kin_lgt)
  axs[3][1].set_ylabel(r"$k^{2} \,$"    + label_kin_trv)
  axs[0][2].set_ylabel(r"${\rm Rm}_{\rm tot}(k)$")
  axs[1][2].set_ylabel(r"${\rm Re}_{\rm tot}(k)$")
  axs[2][2].set_ylabel(r"${\rm Re}_{\parallel}(k)$")
  axs[3][2].set_ylabel(r"${\rm Re}_{\perp}(k)$")
  for col_index in range(len(axs[0])):
    axs[-1][col_index].set_xlabel(r"$k$")
  SimParams.addLabel_simInputs(
    fig      = fig,
    ax       = axs[0][0],
    filepath = f"{filepath_sim}/288/",
  )
  ## save figure
  dict_sim_inputs = SimParams.readSimInputs(f"{filepath_sim}/288/", bool_verbose=False)
  fig_name = "{}_{}_nres_spectra.png".format(
    dict_sim_inputs["suite_folder"],
    dict_sim_inputs["sim_folder"]
  )
  PlotFuncs.saveFigure(fig, f"{filepath_sim}/vis_folder/{fig_name}")
  if bool_verbose: print(" ")


## ###############################################################
## CREATE LIST OF SIMULATION DIRECTORIES TO ANALYSE
## ###############################################################
def getListOfSimFolders():
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
      ## CHECK THE NRES=288 RUN EXISTS
      ## -----------------------------
      if not os.path.exists(f"{filepath_sim}/288/"): continue
      list_sim_filepaths.append(filepath_sim)
      ## MAKE SURE A VISUALISATION FOLDER EXISTS
      ## ---------------------------------------
      WWFnF.createFolder(f"{filepath_sim}/vis_folder", bool_verbose=False)
  return list_sim_filepaths


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  list_sim_filepaths = getListOfSimFolders()
  if BOOL_MPROC:
    with cfut.ProcessPoolExecutor() as executor:
      ## loop over all simulation folders
      futures = [
        executor.submit(
          functools.partial(plotSpectra, bool_verbose=False),
          sim_filepath
        ) for sim_filepath in list_sim_filepaths
      ]
      ## wait to ensure that all scheduled and running tasks have completed
      cfut.wait(futures)
  else: [
    plotSpectra(sim_filepath, bool_verbose=True)
    for sim_filepath in list_sim_filepaths
  ]


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BOOL_MPROC        = 1
BOOL_DEBUG        = 0
BASEPATH          = "/scratch/ek9/nk7952/"
SONIC_REGIME      = "super_sonic"

LIST_SUITE_FOLDER = [ "Re10", "Re500", "Rm3000" ]
LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
# LIST_SIM_RES      = [ "18", "36", "72", "144", "288", "576" ]

# LIST_SUITE_FOLDER = [ "Rm3000" ]
# LIST_SIM_FOLDER   = [ "Pm1", "Pm10", "Pm25" ]
LIST_SIM_RES      = [ "36", "72", "144", "288", "576" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
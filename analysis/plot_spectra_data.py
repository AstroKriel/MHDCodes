#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np
import matplotlib.pyplot as plt

## load user defined modules
from TheUsefulModule import WWFnF
from TheSimModule import SimParams
from TheLoadingModule import LoadFlashData
from ThePlottingModule import PlotFuncs


## ##############################################################
## PREPARE WORKSPACE
## ##############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


def plotSpectra_timeEvolve(
    fig, ax, list_sim_times, list_k, list_power_group_t, cmap_name,
    bool_add_colorbar = True
  ):
  cmap, norm = PlotFuncs.createCmap(
    cmap_name = cmap_name,
    vmin      = min(list_sim_times),
    vmax      = max(list_sim_times)
  )
  for time_index, time_val in enumerate(list_sim_times):
    ax.plot(
      list_k,
      list_power_group_t[time_index],
      color=cmap(norm(time_val)), ls="-", alpha=0.2, zorder=1
    )
  if bool_add_colorbar:
    PlotFuncs.addColorbar_fromCmap(
      fig, ax, cmap, norm,
      label = r"$t = t_{\rm sim} / t_{\rm turb}$"
    )

def plotSpectra_ratio(ax, list_sim_times, list_k, list_power_1_group_t, list_power_2_group_t, cmap_name):
  cmap, norm = PlotFuncs.createCmap(
    cmap_name = cmap_name,
    vmin      = min(list_sim_times),
    vmax      = max(list_sim_times)
  )
  for time_index, time_val in enumerate(list_sim_times):
    ax.plot(
      list_k,
      np.array(list_power_1_group_t[time_index]) / np.array(list_power_2_group_t[time_index]),
      color=cmap(norm(time_val)), ls="-", alpha=0.2, zorder=1
    )

class PlotSpectra():
  def __init__(self,
      fig, axs, filepath_sim_res
    ):
    self.fig              = fig
    self.axs              = axs
    self.filepath_sim_res = filepath_sim_res
  
  def performRoutines(self):
    self.__loadData()
    self.__plotData()
    self.__labelAxis()

  def __loadData(self):
    ## load simulation parameters
    dict_sim_inputs   = SimParams.readSimInputs(self.filepath_sim_res)
    dict_sim_outputs  = SimParams.readSimOutputs(self.filepath_sim_res)
    plots_per_eddy    = dict_sim_outputs["plots_per_eddy"]
    time_growth_end   = dict_sim_outputs["time_growth_end"]
    time_growth_start = dict_sim_outputs["time_growth_start"]
    ## annotate simulation parameters
    PlotFuncs.addBoxOfLabels(
      self.fig, self.axs[0,0],
      box_alignment = (0.0, 0.0),
      xpos          = 0.05,
      ypos          = 0.05,
      alpha         = 0.5,
      fontsize      = 16,
      list_labels   = [
        r"${\rm N}_{\rm res} = $ " + "{:d}".format(int(dict_sim_inputs["sim_res"])),
        r"${\rm Re} = $ "          + "{:d}".format(int(dict_sim_inputs["Re"])),
        r"${\rm Rm} = $ "          + "{:d}".format(int(dict_sim_inputs["Rm"])),
        r"${\rm Pm} = $ "          + "{:d}".format(int(dict_sim_inputs["Pm"])),
      ]
    )
    ## load energy spectra
    print("Loading kinetic energy spectra...")
    dict_kin_spect_tot = LoadFlashData.loadAllSpectraData(
      filepath        = f"{self.filepath_sim_res}/spect/",
      spect_field     = "vel",
      spect_quantity  = "tot",
      file_start_time = time_growth_start,
      file_end_time   = time_growth_end,
      plots_per_eddy  = plots_per_eddy
    )
    dict_kin_spect_lgt = LoadFlashData.loadAllSpectraData(
      filepath        = f"{self.filepath_sim_res}/spect/",
      spect_field     = "vel",
      spect_quantity  = "lgt",
      file_start_time = time_growth_start,
      file_end_time   = time_growth_end,
      plots_per_eddy  = plots_per_eddy
    )
    dict_kin_spect_trv = LoadFlashData.loadAllSpectraData(
      filepath        = f"{self.filepath_sim_res}/spect/",
      spect_field     = "vel",
      spect_quantity  = "trv",
      file_start_time = time_growth_start,
      file_end_time   = time_growth_end,
      plots_per_eddy  = plots_per_eddy
    )
    print("Loading magnetic energy spectra...")
    dict_mag_spect_tot = LoadFlashData.loadAllSpectraData(
      filepath        = f"{self.filepath_sim_res}/spect/",
      spect_field     = "mag",
      spect_quantity  = "tot",
      file_start_time = time_growth_start,
      file_end_time   = time_growth_end,
      plots_per_eddy  = plots_per_eddy
    )
    ## extract data
    self.list_k                     = dict_mag_spect_tot["list_k_group_t"][0]
    self.list_kin_power_tot_group_t = dict_kin_spect_tot["list_power_group_t"]
    self.list_kin_power_lgt_group_t = dict_kin_spect_lgt["list_power_group_t"]
    self.list_kin_power_trv_group_t = dict_kin_spect_trv["list_power_group_t"]
    self.list_mag_power_tot_group_t = dict_mag_spect_tot["list_power_group_t"]
    self.list_kin_sim_times         = dict_kin_spect_tot["list_sim_times"]
    self.list_mag_sim_times         = dict_mag_spect_tot["list_sim_times"]

  def __plotAveData(self, ax):
    ax.plot(
      self.list_k,
      np.mean(self.list_kin_power_lgt_group_t, axis=0),
      "k--", lw=2, label=r"$\mathcal{P}_{\rm kin, \parallel}(k, t)$", zorder=3
    )
    ax.plot(
      self.list_k,
      np.mean(self.list_kin_power_trv_group_t, axis=0),
      "k-",  lw=2, label=r"$\mathcal{P}_{\rm kin, \perp}(k, t)$", zorder=3
    )
    ax.legend(loc="upper right", fontsize=16)

  def __plotData(self):
    ## TOTAL KINETIC ENERGY
    ## --------------------
    self.axs[0,0].set_ylabel(r"$\mathcal{P}_{\rm kin, tot}(k, t)$")
    self.__plotAveData(self.axs[0,0])
    plotSpectra_timeEvolve(
      fig                = self.fig,
      ax                 = self.axs[0,0],
      list_sim_times     = self.list_kin_sim_times,
      list_k             = self.list_k,
      list_power_group_t = self.list_kin_power_tot_group_t,
      cmap_name          = "Greens",
      bool_add_colorbar  = True
    )
    ## PARALLEL KINETIC ENERGY
    ## -----------------------
    self.axs[1,0].set_ylabel(r"$\mathcal{P}_{\rm kin, \parallel}(k, t)$")
    self.__plotAveData(self.axs[1,0])
    plotSpectra_timeEvolve(
      fig                = self.fig,
      ax                 = self.axs[1,0],
      list_sim_times     = self.list_kin_sim_times,
      list_k             = self.list_k,
      list_power_group_t = self.list_kin_power_lgt_group_t,
      cmap_name          = "Greens",
      bool_add_colorbar  = False
    )
    self.axs[2,0].set_ylabel(r"$\mathcal{P}_{\rm kin, tot}(k, t) / \mathcal{P}_{\rm kin, \parallel}(k, t)$")
    plotSpectra_ratio(
      ax                   = self.axs[2,0],
      list_sim_times       = self.list_kin_sim_times,
      list_k               = self.list_k,
      list_power_1_group_t = self.list_kin_power_tot_group_t,
      list_power_2_group_t = self.list_kin_power_lgt_group_t,
      cmap_name = "Greens"
    )
    ## PERPENDICULAR KINETIC ENERGY
    ## ----------------------------
    self.axs[1,1].set_ylabel(r"$\mathcal{P}_{\rm kin, \perp}(k, t)$")
    self.__plotAveData(self.axs[1,1])
    plotSpectra_timeEvolve(
      fig                = self.fig,
      ax                 = self.axs[1,1],
      list_sim_times     = self.list_kin_sim_times,
      list_k             = self.list_k,
      list_power_group_t = self.list_kin_power_trv_group_t,
      cmap_name          = "Greens",
      bool_add_colorbar  = False
    )
    self.axs[2,1].set_ylabel(r"$\mathcal{P}_{\rm kin, tot}(k, t) / \mathcal{P}_{\rm kin, \perp}(k, t)$")
    plotSpectra_ratio(
      ax                   = self.axs[2,1],
      list_sim_times       = self.list_kin_sim_times,
      list_k               = self.list_k,
      list_power_1_group_t = self.list_kin_power_tot_group_t,
      list_power_2_group_t = self.list_kin_power_trv_group_t,
      cmap_name = "Greens"
    )
    ## TOTAL MAGNETIC ENERGY
    ## ---------------------
    self.axs[0,1].set_ylabel(r"$\mathcal{P}_{\rm mag, tot}(k, t)$")
    plotSpectra_timeEvolve(
      fig                = self.fig,
      ax                 = self.axs[0,1],
      list_sim_times     = self.list_mag_sim_times,
      list_k             = self.list_k,
      list_power_group_t = self.list_mag_power_tot_group_t,
      cmap_name          = "Reds",
      bool_add_colorbar  = True
    )

  def __labelAxis(self):
    num_axs = self.axs.size
    for index_ax, ax in enumerate(self.axs.flatten()):
      if index_ax >= num_axs-2: ax.set_xlabel(r"$k$")
      ax.set_xscale("log")
      ax.set_yscale("log")
      PlotFuncs.addAxisTicks_log10(
        ax,
        bool_major_ticks = True,
        num_major_ticks  = 10
      )


## ###############################################################
## HANDLING PLOT CALLS
## ###############################################################
def plotSpectraData(filepath_sim_res, filepath_sim_res_plot, sim_name):
  fig, axs = plt.subplots(
    nrows              = 3,
    ncols              = 2, 
    figsize            = (5*2, 4*3),
    sharex             = True,
    constrained_layout = True
  )
  obj_plot_spectra = PlotSpectra(fig, axs, filepath_sim_res)
  obj_plot_spectra.performRoutines()
  fig_name = f"{sim_name}_spectra.png"
  PlotFuncs.saveFigure(fig, f"{filepath_sim_res_plot}/{fig_name}")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## LOOK AT EACH SIMULATION
  ## -----------------------
  ## loop over each simulation suite
  for suite_folder in LIST_SUITE_FOLDER:

    ## loop over each simulation folder
    for sim_folder in LIST_SIM_FOLDER:

      ## CHECK THE SIMULATION EXISTS
      ## ---------------------------
      filepath_sim = WWFnF.createFilepath([
        BASEPATH, suite_folder, SONIC_REGIME, sim_folder
      ])
      if not os.path.exists(filepath_sim): continue
      str_message = f"Looking at suite: {suite_folder}, sim: {sim_folder}, regime: {SONIC_REGIME}"
      print(str_message)
      print("=" * len(str_message))
      print(" ")

      ## loop over each resolution
      for sim_res in LIST_SIM_RES:

        ## CHECK THE RESOLUTION RUN EXISTS
        ## -------------------------------
        filepath_sim_res = f"{filepath_sim}/{sim_res}/"
        ## check that the filepath exists
        if not os.path.exists(filepath_sim_res): continue

        ## MAKE SURE A VISUALISATION FOLDER EXISTS
        ## ---------------------------------------
        filepath_sim_res_plot = f"{filepath_sim_res}/vis_folder/"
        WWFnF.createFolder(filepath_sim_res_plot, bool_hide_updates=True)

        ## PLOT SIMULATION DATA
        ## --------------------
        sim_name = f"{suite_folder}_{sim_folder}"
        plotSpectraData(filepath_sim_res, filepath_sim_res_plot, sim_name)

        if BOOL_DEBUG: return
        ## create empty space
        print(" ")
      print(" ")
    print(" ")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BOOL_DEBUG        = 0
BASEPATH          = "/scratch/ek9/nk7952/"
SONIC_REGIME      = "super_sonic"

LIST_SUITE_FOLDER = [ "Re10", "Re500", "Rm3000" ]
LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
LIST_SIM_RES      = [ "18", "36", "72", "144", "288", "576" ]

# LIST_SUITE_FOLDER = [ "Rm3000" ]
# LIST_SIM_FOLDER   = [ "Pm5" ]
# LIST_SIM_RES      = [ "288" ]


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
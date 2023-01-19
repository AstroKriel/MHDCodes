#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate

## load user defined modules
from TheUsefulModule import WWFnF, WWLists, WWObjs
from TheSimModule import SimParams
from TheLoadingModule import LoadFlashData
from ThePlottingModule import PlotFuncs, PlotLatex

## ##############################################################
## PREPARE WORKSPACE
## ##############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ##############################################################
## HELPER FUNCTIONS
## ##############################################################
def plotSpectra_timeEvolve(
    fig, ax, list_sim_times, list_k, list_power_group_t, cmap_name,
    bool_add_colorbar = True
  ):
  cmap, norm = PlotFuncs.createCmap(
    cmap_name = cmap_name,
    vmin      = min(list_sim_times),
    vmax      = max(list_sim_times)
  )
  ## plot each time realisation
  for time_index, time_val in enumerate(list_sim_times):
    ax.plot(
      list_k,
      list_power_group_t[time_index],
      color=cmap(norm(time_val)), ls="-", alpha=0.2, zorder=1
    )
  ## add colour bar
  if bool_add_colorbar:
    PlotFuncs.addColorbar_fromCmap(
      fig, ax, cmap, norm,
      label = r"$t = t_{\rm sim} / t_{\rm turb}$"
    )

def getSpectraRatio_grouped(list_power_1_group_t, list_power_2_group_t):
  return [
    list_power_1 / list_power_2
    for list_power_1, list_power_2 in zip(
      list_power_1_group_t,
      list_power_2_group_t
    )
  ]

def plotSpectra_ratio(
    ax, list_sim_times, list_k, list_power_1_group_t, list_power_2_group_t, cmap_name,
  ):
  cmap, norm = PlotFuncs.createCmap(
    cmap_name = cmap_name,
    vmin      = min(list_sim_times),
    vmax      = max(list_sim_times)
  )
  ## compute ratio of spectra at each time realisation
  list_power_ratio = getSpectraRatio_grouped(list_power_1_group_t, list_power_2_group_t)
  ## plot time-average
  ax.plot(
    list_k,
    np.mean(list_power_ratio, axis=0),
    color="black", ls="-", lw=2, alpha=1.0, zorder=5,
    label=r"time-average"
  )
  ## plot each time realisation
  for time_index, time_val in enumerate(list_sim_times):
    ax.plot(
      list_k,
      list_power_ratio[time_index],
      color=cmap(norm(time_val)), ls="-", alpha=0.2, zorder=1
    )

def interpLogLogData(x, y, x_interp, interp_kind="cubic"):
  interpolator = interpolate.interp1d(np.log10(x), np.log10(y), kind=interp_kind)
  return np.power(10.0, interpolator(np.log10(x_interp)))

def measureScales_peak(ax, list_k, list_power_group_t):
  array_k_interp = np.logspace(
    start = np.log10(min(list_k)),
    stop  = np.log10(max(list_k)),
    num   = 3*len(list_k)
  )[1:-1]
  distribution_scales = []
  for list_power in list_power_group_t:
    array_power_interp = interpLogLogData(
      x           = list_k,
      y           = list_power,
      x_interp    = array_k_interp,
      interp_kind = "cubic"
    )
    distribution_scales.append(
      array_k_interp[ np.argmax(array_power_interp) ]
    )
  # distribution_scales = list_k[ np.argmax(list_power_group_t, axis=1) ]
  scale_p16 = np.percentile(distribution_scales, 16)
  scale_p50 = np.percentile(distribution_scales, 50)
  scale_p84 = np.percentile(distribution_scales, 84)
  ax.axvline(x=scale_p16, color="black", ls="--")
  ax.axvline(x=scale_p50, color="black", ls=":")
  ax.axvline(x=scale_p84, color="black", ls="--")
  return distribution_scales

def measureScales_trv(ax, list_k, list_power_group_t, target_value):
  array_k_interp = np.logspace(
    start = np.log10(min(list_k)),
    stop  = np.log10(max(list_k)),
    num   = 3*len(list_k)
  )[1:-1]
  distribution_scales = []
  for list_power in list_power_group_t:
    array_power_interp = interpLogLogData(
      x           = list_k,
      y           = list_power,
      x_interp    = array_k_interp,
      interp_kind = "cubic"
    )
    distribution_scales.append(
      array_k_interp[ WWLists.getIndexClosestValue(array_power_interp, target_value) ]
    )
  # distribution_indices = [
  #   WWLists.getIndexClosestValue(list_power, target_value)
  #   for list_power in list_power_group_t
  # ]
  # distribution_scales = [
  #   list_k[peak_index]
  #   for peak_index in distribution_indices
  # ]
  scale_p16 = np.percentile(distribution_scales, 16)
  scale_p50 = np.percentile(distribution_scales, 50)
  scale_p84 = np.percentile(distribution_scales, 84)
  ax.axhline(y=target_value, color="black", ls="--")
  ax.axvline(x=scale_p16,    color="black", ls="--")
  ax.axvline(x=scale_p50,    color="black", ls=":")
  ax.axvline(x=scale_p84,    color="black", ls="--")
  return distribution_scales


## ###############################################################
## MAIN PROGRAM
## ###############################################################
class PlotSpectra():
  def __init__(
      self,
      filepath_sim_res, filepath_sim_res_plot, sim_name
    ):
    ## save inputs
    self.filepath_sim_res      = filepath_sim_res
    self.filepath_sim_res_plot = filepath_sim_res_plot
    self.sim_name              = sim_name
    ## initialise output data
    self.k_p_tot_group_t     = None
    self.k_nu_lgt_group_t    = None
    self.k_nu_trv_group_t_p5 = None
    self.k_nu_trv_group_t_1  = None
    self.k_nu_trv_group_t_2  = None

  def performRoutines(self):
    self.ax_nrows = 4
    self.ax_ncols = 2
    self.fig, self.axs = plt.subplots(
      nrows              = self.ax_nrows,
      ncols              = self.ax_ncols,
      figsize            = (6*self.ax_ncols, 4*self.ax_nrows),
      sharex             = True,
      constrained_layout = True
    )
    self.__loadData()
    self.__plotData()
    self.__saveScales()
    self.__labelAxis()
    fig_name = f"{self.sim_name}_spectra.png"
    PlotFuncs.saveFigure(self.fig, f"{self.filepath_sim_res_plot}/{fig_name}")

  def __saveScales(self):
    # if (self.k_nu_lgt_group_t is None) or\
    #    (self.k_nu_trv_group_t is None) or\
    #    (self.k_p_tot_group_t  is None):
    #   raise Exception("Error: scales have not been computed yet/correctly.")
    dict_scales = {
      "k_p_tot_group_t"     : self.k_p_tot_group_t,
      "k_nu_lgt_group_t"    : self.k_nu_lgt_group_t,
      "k_nu_trv_group_t_p5" : self.k_nu_trv_group_t_p5,
      "k_nu_trv_group_t_1"  : self.k_nu_trv_group_t_1,
      "k_nu_trv_group_t_2"  : self.k_nu_trv_group_t_2,
    }
    WWObjs.saveDict2JsonFile(f"{self.filepath_sim_res}/sim_outputs.json", dict_scales)

  def __loadData(self):
    ## load simulation parameters
    dict_sim_inputs   = SimParams.readSimInputs(self.filepath_sim_res)
    dict_sim_outputs  = SimParams.readSimOutputs(self.filepath_sim_res)
    plots_per_eddy    = dict_sim_outputs["plots_per_eddy"]
    time_growth_end   = dict_sim_outputs["time_growth_end"]
    time_growth_start = dict_sim_outputs["time_growth_start"]
    self.sim_res      = dict_sim_inputs["sim_res"]
    self.Re           = dict_sim_inputs["Re"]
    self.Rm           = dict_sim_inputs["Rm"]
    self.Pm           = dict_sim_inputs["Pm"]
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

  def __plotAveKinComponents(self, ax):
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
      list_sim_times     = self.list_kin_sim_times,
      list_k             = self.list_k,
      list_power_group_t = self.list_kin_power_tot_group_t,
      cmap_name          = "Greens",
      bool_add_colorbar  = True
    )
    ## TOTAL MAGNETIC ENERGY
    ## ---------------------
    ## plot total spectra
    self.axs[0,1].set_ylabel(label_mag_tot)
    plotSpectra_timeEvolve(
      fig                = self.fig,
      ax                 = self.axs[0,1],
      list_sim_times     = self.list_mag_sim_times,
      list_k             = self.list_k,
      list_power_group_t = self.list_mag_power_tot_group_t,
      cmap_name          = "Reds",
      bool_add_colorbar  = True
    )
    self.k_p_tot_group_t = measureScales_peak(
      ax                 = self.axs[0,1],
      list_k             = self.list_k,
      list_power_group_t = self.list_mag_power_tot_group_t
    )
    ## LONGITUDINAL KINETIC ENERGY
    ## ---------------------------
    ## plot spectra component only
    self.axs[1,0].set_ylabel(label_kin_lgt)
    self.__plotAveKinComponents(self.axs[1,0])
    plotSpectra_timeEvolve(
      fig                = self.fig,
      ax                 = self.axs[1,0],
      list_sim_times     = self.list_kin_sim_times,
      list_k             = self.list_k,
      list_power_group_t = self.list_kin_power_lgt_group_t,
      cmap_name          = "Greens",
      bool_add_colorbar  = False
    )
    ## plot spectra component vs total
    self.axs[2,0].set_ylabel(PlotLatex.addLabel_frac(label_kin_lgt, label_kin_tot))
    plotSpectra_ratio(
      ax                   = self.axs[2,0],
      list_sim_times       = self.list_kin_sim_times,
      list_k               = self.list_k,
      list_power_1_group_t = self.list_kin_power_lgt_group_t,
      list_power_2_group_t = self.list_kin_power_tot_group_t,
      cmap_name            = "Greens"
    )
    measureScales_peak(
      ax                 = self.axs[2,0],
      list_k             = self.list_k,
      list_power_group_t = getSpectraRatio_grouped(
        self.list_kin_power_lgt_group_t,
        self.list_kin_power_tot_group_t
      )
    )
    ## plot spectra component compared
    self.axs[3,0].set_ylabel(PlotLatex.addLabel_frac(label_kin_lgt, label_kin_trv))
    plotSpectra_ratio(
      ax                   = self.axs[3,0],
      list_sim_times       = self.list_kin_sim_times,
      list_k               = self.list_k,
      list_power_1_group_t = self.list_kin_power_lgt_group_t,
      list_power_2_group_t = self.list_kin_power_trv_group_t,
      cmap_name            = "Greens"
    )
    self.k_nu_lgt_group_t = measureScales_peak(
      ax                 = self.axs[3,0],
      list_k             = self.list_k,
      list_power_group_t = getSpectraRatio_grouped(
        self.list_kin_power_lgt_group_t,
        self.list_kin_power_trv_group_t
      )
    )
    index_end_k_nu_trv = int(np.mean(self.k_nu_lgt_group_t))
    # TRANSVERSE KINETIC ENERGY
    # -------------------------
    ## plot spectra component only
    self.axs[1,1].set_ylabel(label_kin_trv)
    self.__plotAveKinComponents(self.axs[1,1])
    plotSpectra_timeEvolve(
      fig                = self.fig,
      ax                 = self.axs[1,1],
      list_sim_times     = self.list_kin_sim_times,
      list_k             = self.list_k,
      list_power_group_t = self.list_kin_power_trv_group_t,
      cmap_name          = "Greens",
      bool_add_colorbar  = False
    )
    ## plot spectra component vs total
    self.axs[2,1].set_ylabel(PlotLatex.addLabel_frac(label_kin_trv, label_kin_tot))
    plotSpectra_ratio(
      ax                   = self.axs[2,1],
      list_sim_times       = self.list_kin_sim_times,
      list_k               = self.list_k,
      list_power_1_group_t = self.list_kin_power_trv_group_t,
      list_power_2_group_t = self.list_kin_power_tot_group_t,
      cmap_name            = "Greens"
    )
    ## plot spectra component compared
    self.axs[3,1].set_ylabel(PlotLatex.addLabel_frac(label_kin_trv, label_kin_lgt))
    plotSpectra_ratio(
      ax                   = self.axs[3,1],
      list_sim_times       = self.list_kin_sim_times,
      list_k               = self.list_k,
      list_power_1_group_t = self.list_kin_power_trv_group_t,
      list_power_2_group_t = self.list_kin_power_lgt_group_t,
      cmap_name            = "Greens"
    )
    self.k_nu_trv_group_t_p5 = measureScales_trv(
      ax                 = self.axs[3,1],
      target_value       = 0.5,
      list_k             = self.list_k[:index_end_k_nu_trv],
      list_power_group_t = getSpectraRatio_grouped(
        [ list_power[:index_end_k_nu_trv] for list_power in self.list_kin_power_trv_group_t ],
        [ list_power[:index_end_k_nu_trv] for list_power in self.list_kin_power_lgt_group_t ]
      )
    )
    self.k_nu_trv_group_t_1 = measureScales_trv(
      ax                 = self.axs[3,1],
      target_value       = 1.0,
      list_k             = self.list_k[:index_end_k_nu_trv],
      list_power_group_t = getSpectraRatio_grouped(
        [ list_power[:index_end_k_nu_trv] for list_power in self.list_kin_power_trv_group_t ],
        [ list_power[:index_end_k_nu_trv] for list_power in self.list_kin_power_lgt_group_t ]
      )
    )
    self.k_nu_trv_group_t_2 = measureScales_trv(
      ax                 = self.axs[3,1],
      target_value       = 2.0,
      list_k             = self.list_k[:index_end_k_nu_trv],
      list_power_group_t = getSpectraRatio_grouped(
        [ list_power[:index_end_k_nu_trv] for list_power in self.list_kin_power_trv_group_t ],
        [ list_power[:index_end_k_nu_trv] for list_power in self.list_kin_power_lgt_group_t ]
      )
    )

  def __labelAxis(self):
    ## annotate simulation parameters
    PlotFuncs.addBoxOfLabels(
      self.fig, self.axs[0,0],
      bbox        = (0.0, 0.0),
      xpos        = 0.05,
      ypos        = 0.05,
      alpha       = 0.5,
      fontsize    = 16,
      list_labels = [
        r"${\rm N}_{\rm res} = $ " + "{:d}".format(int(self.sim_res)),
        r"${\rm Re} = $ "          + "{:d}".format(int(self.Re)),
        r"${\rm Rm} = $ "          + "{:d}".format(int(self.Rm)),
        r"${\rm Pm} = $ "          + "{:d}".format(int(self.Pm)),
      ]
    )
    for index_row in range(self.ax_nrows):
      for index_col in range(self.ax_ncols):
        ax = self.axs[index_row, index_col]
        if index_row == self.ax_nrows:            ax.set_xlabel(r"$k$")
        if (index_row == 0) and (index_col == 0): ax.legend(loc="upper right",  fontsize=16)
        if index_row == 1:                        ax.legend(loc="lower center", fontsize=16)
        if (index_row >= 2) and (index_col == 0): ax.legend(loc="upper left",   fontsize=16)
        if (index_row >= 2) and (index_col == 1): ax.legend(loc="lower left",   fontsize=16)
        ax.set_xscale("log")
        ax.set_yscale("log")
        PlotFuncs.addAxisTicks_log10(
          ax,
          bool_major_ticks = True,
          num_major_ticks  = 10
        )


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
        WWFnF.createFolder(filepath_sim_res_plot, bool_verbose=True)

        ## PLOT SIMULATION DATA
        ## --------------------
        sim_name = f"{suite_folder}_{sim_folder}"
        obj_plot = PlotSpectra(filepath_sim_res, filepath_sim_res_plot, sim_name)
        obj_plot.performRoutines()

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
# LIST_SIM_FOLDER   = [ "Pm2" ]
# LIST_SIM_RES      = [ "288" ]

# LIST_SIM_RES      = [ "18", "36", "72" ]
# LIST_SIM_RES      = [ "144", "288", "576" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
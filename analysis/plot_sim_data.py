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
from plot_turb_data import PlotTurbData

## load user defined modules
from TheUsefulModule import WWLists, WWFnF, WWObjs
from TheSimModule import SimParams
from TheLoadingModule import LoadFlashData
from TheAnalysisModule import WWSpectra
from ThePlottingModule import PlotFuncs, PlotLatex
from TheFittingModule import FitMHDScales


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def getLabel_kin(fit_params_group_t):
  label_A_kin     = PlotLatex.GetLabel.percentiles(WWLists.getElemFromLoL(fit_params_group_t, 0))
  label_alpha_kin = PlotLatex.GetLabel.percentiles(WWLists.getElemFromLoL(fit_params_group_t, 1))
  label_k_nu      = PlotLatex.GetLabel.percentiles(WWLists.getElemFromLoL(fit_params_group_t, 2))
  return r"$A_{\rm kin} = $ " + label_A_kin + r", $\alpha = $ " + label_alpha_kin + r", $k_\nu = $ " + label_k_nu

def getLabel_mag(k_p_group_t, k_max_group_t):
  label_k_p   = PlotLatex.GetLabel.percentiles(k_p_group_t)
  label_k_max = PlotLatex.GetLabel.percentiles(k_max_group_t)
  return r"$k_{\rm p} = $ " + label_k_p + r", $k_{\rm max} = $ " + label_k_max

def plotMeasuredScale(
    ax_time, ax_spectrum,
    list_t, scale_group_t, scale_ave,
    color       = "black",
    label       = ""
  ):
  ## plot average scale to spectrum
  ax_spectrum.axvline(
    x=scale_ave,
    color=color, ls="--", lw=1.5, zorder=7
  )
  ## plot time evolution of scale
  ax_time.plot(
    list_t,
    [ scale_ave ] * len(list_t),
    color=color, ls="--", lw=1.5
  )
  ax_time.plot(
    list_t,
    scale_group_t,
    color=color, ls="-", lw=1.5, label=label
  )

def plotSpectra(ax, list_k, list_power_group_t, color, cmap_name, list_times):
  args_plot_ave  = { "color":color, "marker":"o", "ms":8, "zorder":5, "markeredgecolor":"black" }
  args_plot_time = { "ls":"-", "lw":1, "alpha":0.5, "zorder":3 }
  ## plot time averaged, normalised energy spectra
  ax.plot(
    list_k,
    WWSpectra.aveSpectra(list_power_group_t, bool_norm=True),
    ls="", **args_plot_ave
  )
  ## create colormaps for time-evolving energy spectra (color as a func of time)
  cmap_kin, norm_kin = PlotFuncs.createCmap(
    cmap_name = cmap_name,
    vmin      = min(list_times),
    vmax      = max(list_times)
  )
  ## plot each time realisation of the normalised kinetic energy spectrum
  for time_index, time_val in enumerate(list_times):
    ax.plot(
      list_k,
      WWSpectra.normSpectra(list_power_group_t[time_index]),
      color=cmap_kin(norm_kin(time_val)), **args_plot_time
    )


## ###############################################################
## OPERATOR CLASS: PLOT NORMALISED + TIME-AVERAGED ENERGY SPECTRA
## ###############################################################
class PlotSpectra():
  def __init__(
      self,
      fig, dict_axs,
      filepath_spect, time_exp_start, time_exp_end
    ):
    ## save input arguments
    self.fig              = fig
    self.axs_spectra      = dict_axs["axs_spectra"]
    self.ax_scales        = dict_axs["ax_scales"]
    self.ax_residuals     = dict_axs["ax_residuals"]
    self.ax_spectra_ratio = dict_axs["ax_spectra_ratio"]
    self.filepath_spect   = filepath_spect
    self.time_exp_start   = time_exp_start
    self.time_exp_end     = time_exp_end
    ## initialise spectra labels
    self.__initialiseQuantities()
    self.dict_plot_kin_trv = {
      "color"       : "darkgreen",
      "cmap_name"   : "Greens",
      "label_spect" : PlotLatex.GetLabel.spectrum("kin", "trv"),
      "label_knu"   : r"$k_{\nu, \perp}(t)$",
    }
    self.dict_plot_mag_tot = {
      "color"       : "red",
      "cmap_name"   : "Reds",
      "label_spect" : PlotLatex.GetLabel.spectrum("mag", "tot"),
      "label_kp"    : r"$k_{\rm p}(t)$",
    }

  def performRoutines(self):
    self.__loadData()
    self.__plotSpectra()
    # self.__plotSpectraRatio()
    print("Fitting energy spectra...")
    self.__fitKinSpectra()
    self.__fitMagSpectra()
    self.bool_fitted = True
    self.__labelSpectra()
    self.__labelResiduals()
    # self.__labelSpectraRatio()
    self.__labelScales()

  def getFittedParams(self):
    # list_quantities_undefined = self.__checkAnyQuantitiesNotMeasured()
    # if not self.bool_fitted: self.performRoutines()
    # if len(list_quantities_undefined) > 0: raise Exception("Error: failed to define quantity:", list_quantities_undefined)
    return {
      ## time-averaged energy spectra
      "list_k"                     : self.list_k,
      "list_mag_power_tot_ave"     : WWSpectra.aveSpectra(self.list_mag_power_tot_group_t, bool_norm=True),
      "list_kin_power_tot_ave"     : WWSpectra.aveSpectra(self.list_kin_power_tot_group_t, bool_norm=True),
      "list_kin_power_lgt_ave"     : WWSpectra.aveSpectra(self.list_kin_power_lgt_group_t, bool_norm=True),
      "list_kin_power_trv_ave"     : WWSpectra.aveSpectra(self.list_kin_power_trv_group_t, bool_norm=True),
      ## measured quantities
      "plots_per_eddy"             : self.plots_per_eddy,
      "list_time_growth"           : self.list_time_growth,
      "list_time_k_eq"             : self.list_time_k_eq,
      "k_p_group_t"                : self.k_p_group_t,
      "k_eq_group_t"               : self.k_eq_group_t,
      "fit_params_kin_trv_group_t" : self.fit_params_kin_trv_group_t,
      "fit_params_kin_trv_ave"     : self.fit_params_kin_trv_ave
    }

  def saveFittedParams(self, filepath_sim):
    dict_params = self.getFittedParams()
    WWObjs.saveDict2JsonFile(f"{filepath_sim}/sim_outputs.json", dict_params)

  def __initialiseQuantities(self):
    ## flag to check that all required quantities have been measured
    self.bool_fitted                = False
    ## initialise quantities to measure
    self.list_k                     = None
    self.list_mag_power_tot_group_t = None
    self.list_kin_power_tot_group_t = None # TODO: rename list_power_(field)_(sub)_group_t
    self.list_kin_power_lgt_group_t = None
    self.list_kin_power_trv_group_t = None
    self.plots_per_eddy             = None
    self.list_time_growth           = None
    self.list_time_k_eq             = None
    self.k_p_group_t                = None
    self.k_eq_group_t               = None
    self.fit_params_kin_trv_group_t = None
    self.fit_params_kin_trv_ave     = None

  def __checkAnyQuantitiesNotMeasured(self):
    list_quantities_check = [
      self.list_k,
      self.list_mag_power_tot_group_t,
      self.list_kin_power_tot_group_t,
      self.list_kin_power_lgt_group_t,
      self.list_kin_power_trv_group_t,
      self.plots_per_eddy,
      self.list_time_growth,
      self.list_time_k_eq,
      self.k_p_group_t,
      self.k_eq_group_t,
      self.fit_params_kin_trv_group_t,
      self.fit_params_kin_trv_ave
    ]
    return [ 
      index_quantity
      for index_quantity, quantity in enumerate(list_quantities_check)
      if quantity is None
    ]

  def __loadData(self):
    print("Loading energy spectra...")
    ## extract the number of plt-files per eddy-turnover-time from 'Turb.log'
    self.plots_per_eddy = LoadFlashData.getPlotsPerEddy_fromTurbLog(
      f"{self.filepath_spect}/../",
      bool_hide_updates = True
    )
    ## load total kinetic energy spectra
    dict_kin_spect_tot_data = LoadFlashData.loadAllSpectraData(
      filepath          = self.filepath_spect,
      spect_field       = "vel",
      spect_quantity    = "tot",
      file_start_time   = self.time_exp_start,
      file_end_time     = self.time_exp_end,
      plots_per_eddy    = self.plots_per_eddy,
      bool_hide_updates = True
    )
    ## load longitudinal kinetic energy spectra
    dict_kin_spect_lgt_data = LoadFlashData.loadAllSpectraData(
      filepath          = self.filepath_spect,
      spect_field       = "vel",
      spect_quantity    = "lgt",
      file_start_time   = self.time_exp_start,
      file_end_time     = self.time_exp_end,
      plots_per_eddy    = self.plots_per_eddy,
      bool_hide_updates = True
    )
    ## load transverse kinetic energy spectra
    dict_kin_spect_trv_data = LoadFlashData.loadAllSpectraData(
      filepath          = self.filepath_spect,
      spect_field       = "vel",
      spect_quantity    = "trv",
      file_start_time   = self.time_exp_start,
      file_end_time     = self.time_exp_end,
      plots_per_eddy    = self.plots_per_eddy,
      bool_hide_updates = True
    )
    ## load total magnetic energy spectra
    dict_mag_spect_tot_data = LoadFlashData.loadAllSpectraData(
      filepath          = self.filepath_spect,
      spect_field       = "mag",
      spect_quantity    = "tot",
      file_start_time   = self.time_exp_start,
      file_end_time     = self.time_exp_end,
      plots_per_eddy    = self.plots_per_eddy,
      bool_hide_updates = True
    )
    ## store time-evolving energy spectra
    self.list_kin_power_tot_group_t = dict_kin_spect_tot_data["list_power_group_t"]
    self.list_kin_power_lgt_group_t = dict_kin_spect_lgt_data["list_power_group_t"]
    self.list_kin_power_trv_group_t = dict_kin_spect_trv_data["list_power_group_t"]
    self.list_mag_power_tot_group_t = dict_mag_spect_tot_data["list_power_group_t"]
    self.list_k                     = dict_mag_spect_tot_data["list_k_group_t"][0]
    self.list_time_growth           = dict_mag_spect_tot_data["list_sim_times"]

  def __plotSpectra(self):
    plotSpectra(
      ax                 = self.axs_spectra[0],
      list_k             = self.list_k,
      list_power_group_t = self.list_kin_power_trv_group_t,
      color              = self.dict_plot_kin_trv["color"],
      cmap_name          = self.dict_plot_kin_trv["cmap_name"],
      list_times         = self.list_time_growth
    )
    plotSpectra(
      ax                 = self.axs_spectra[1],
      list_k             = self.list_k,
      list_power_group_t = self.list_mag_power_tot_group_t,
      color              = self.dict_plot_mag_tot["color"],
      cmap_name          = self.dict_plot_mag_tot["cmap_name"],
      list_times         = self.list_time_growth
    )

  # def __plotSpectraRatio(self):
  #   bla

  def __fitKinSpectra(self):
    ## define helper function
    def fitKinSpectra(
        ax_fit, ax_residuals, ax_scales,
        list_k, list_power_group_t, list_time_growth,
        color       = "black",
        label_spect = "",
        label_knu   = r"$k_\nu$"
      ):
      fit_params_group_t = []
      ## fit each time-realisation of the kinetic energy spectrum
      for time_index in range(len(list_time_growth)):
        fit_params_kin = FitMHDScales.fitKinSpectrum(
          list_k        = list_k,
          list_power    = WWSpectra.normSpectra(list_power_group_t[time_index])
        )
        ## store fitted parameters
        fit_params_group_t.append(fit_params_kin)
      ## fit time-averaged kinetic energy spectrum
      fit_params_ave = FitMHDScales.fitKinSpectrum(
        ax_fit        = ax_fit,
        ax_residuals  = ax_residuals,
        list_k        = list_k,
        list_power    = WWSpectra.aveSpectra(list_power_group_t, bool_norm=True),
        color         = "royalblue",
        label_spect   = label_spect
      )
      ## plot time-evolution of measured scale
      plotMeasuredScale(
        ax_spectrum   = ax_fit,
        ax_time       = ax_scales,
        list_t        = list_time_growth,
        scale_group_t = WWLists.getElemFromLoL(fit_params_group_t, 2),
        scale_ave     = fit_params_ave[2],
        color         = color,
        label         = label_knu
      )
      return fit_params_group_t, fit_params_ave
    ## fit transverse kinetic spectrum
    self.fit_params_kin_trv_group_t, self.fit_params_kin_trv_ave = fitKinSpectra(
      ax_fit             = self.axs_spectra[0],
      ax_residuals       = self.ax_residuals,
      ax_scales          = self.ax_scales,
      list_k             = self.list_k,
      list_power_group_t = self.list_kin_power_trv_group_t,
      list_time_growth   = self.list_time_growth,
      color              = self.dict_plot_kin_trv["color"],
      label_spect        = self.dict_plot_kin_trv["label_spect"],
      label_knu          = self.dict_plot_kin_trv["label_knu"]
    )

  def __fitMagSpectra(self):
    self.k_p_group_t   = []
    self.k_max_group_t = []
    ## fit each time-realisation of the magnetic energy spectrum
    for time_index in range(len(self.list_time_growth)):
      k_p, k_max = FitMHDScales.getMagSpectrumPeak(
        self.list_k,
        WWSpectra.normSpectra(self.list_mag_power_tot_group_t[time_index])
      )
      ## store measured scales
      self.k_p_group_t.append(k_p)
      self.k_max_group_t.append(k_max)
    ## annotate measured raw spectrum maximum
    self.axs_spectra[1].plot(
      np.mean(self.k_max_group_t),
      np.mean(np.max(WWSpectra.normSpectra_grouped(self.list_mag_power_tot_group_t), axis=1)),
      color="black", marker="o", ms=8, ls="", label=r"$k_{\rm max}$", zorder=7
    )
    ## plot time-evolution of measured scale
    plotMeasuredScale(
      ax_spectrum   = self.axs_spectra[1],
      ax_time       = self.ax_scales,
      list_t        = self.list_time_growth,
      scale_group_t = self.k_p_group_t,
      scale_ave     = np.mean(self.k_p_group_t),
      color         = self.dict_plot_mag_tot["color"],
      label         = self.dict_plot_mag_tot["label_kp"]
    )

  def __adjustAxis(self, ax, bool_log_y=True):
    ax.set_xlim([ 0.9, 1.2*max(self.list_k) ])
    ax.set_xscale("log")
    if bool_log_y: ax.set_yscale("log")

  def __labelSpectra(self):
    self.__adjustAxis(self.axs_spectra[0])
    self.__adjustAxis(self.axs_spectra[1])
    PlotFuncs.labelDualAxis(
      axs         = self.axs_spectra,
      label_left  = self.dict_plot_kin_trv["label_spect"],
      label_right = self.dict_plot_mag_tot["label_spect"],
      color_left  = self.dict_plot_kin_trv["color"],
      color_right = self.dict_plot_mag_tot["color"]
    )
    PlotFuncs.addBoxOfLabels(
      fig           = self.fig,
      ax            = self.axs_spectra[0],
      box_alignment = (0.5, 0.0),
      xpos          = 0.5,
      ypos          = 0.05,
      alpha         = 0.85,
      fontsize      = 18,
      list_labels   = [
        getLabel_kin(self.fit_params_kin_trv_group_t),
        getLabel_mag(self.k_p_group_t, self.k_max_group_t)
      ],
      list_colors   = [
        self.dict_plot_kin_trv["color"],
        self.dict_plot_mag_tot["color"]
      ]
    )

  def __labelResiduals(self):
    self.__adjustAxis(self.ax_residuals, bool_log_y=False)
    self.ax_residuals.axhline(y=1, color="black", ls="--")
    PlotFuncs.addLegend_withBox(
      ax   = self.ax_residuals,
      loc  = "lower left",
      bbox = (0.0, 0.0)
    )
    self.ax_residuals.set_xlabel(r"$k$")
    self.ax_residuals.set_ylabel(
      PlotLatex.GetLabel.timeAve(
        PlotLatex.GetLabel.spectrum("fit") + r"$\, / \,$" + PlotLatex.GetLabel.spectrum("data")
    ))

  def __labelSpectraRatio(self):
    self.ax_spectra_ratio.axhline(y=1, color="black", ls="--")
    self.__adjustAxis(self.ax_spectra_ratio)
    self.ax_spectra_ratio.set_xlabel(r"$k$")
    self.ax_spectra_ratio.set_ylabel(
      PlotLatex.GetLabel.spectrum("mag") + r"$/$" + PlotLatex.GetLabel.spectrum("kin", "tot")
    )

  def __labelScales(self):
    self.ax_scales.set_yscale("log")
    self.ax_scales.set_xlabel(r"$t/t_{\rm turb}$")
    self.ax_scales.set_ylabel(r"$k$")
    PlotFuncs.addLegend_withBox(
      ax   = self.ax_scales,
      loc  = "lower left",
      bbox = (0.0, 1.0)
    )


## ###############################################################
## HANDLING PLOT CALLS
## ###############################################################
def plotSimData(filepath_sim_res, filepath_vis, sim_name):
  ## INITIALISE FIGURE
  ## -----------------
  print("Initialising figure...")
  fig, fig_grid = PlotFuncs.createFigure_grid(
    fig_scale        = 0.45,
    fig_aspect_ratio = (10.0, 8.0),
    num_rows         = 3,
    num_cols         = 6
  )
  ## volume integrated qunatities
  ax_Mach          = fig.add_subplot(fig_grid[0, 0:2])
  ax_energy_ratio  = fig.add_subplot(fig_grid[1, 0:2])
  ## spectra data
  ax_residuals     = fig.add_subplot(fig_grid[2, 0:3])
  axs_spectra      = PlotFuncs.addSubplot_secondAxis(fig, fig_grid[:2, 2:4])
  ax_spectra_ratio = fig.add_subplot(fig_grid[0:2, 4:6])
  ax_scales        = fig.add_subplot(fig_grid[  2, 3:6])
  ## PLOT INTEGRATED QUANTITIES
  ## --------------------------
  obj_plot_turb = PlotTurbData(
    fig              = fig,
    axs              = [ ax_Mach, ax_energy_ratio ],
    filepath_sim_res = filepath_sim_res,
    dict_sim_inputs  = SimParams.readSimInputs(filepath_sim_res)
  )
  obj_plot_turb.performRoutines()
  obj_plot_turb.saveFittedParams(filepath_sim_res)
  dict_turb_params = obj_plot_turb.getFittedParams()
  ## PLOT FITTED SPECTRA
  ## -------------------
  obj_plot_spectra = PlotSpectra(
    fig              = fig,
    dict_axs         = {
      "axs_spectra"      : axs_spectra,
      "ax_scales"        : ax_scales,
      "ax_residuals"     : ax_residuals,
      "ax_spectra_ratio" : ax_spectra_ratio,
    },
    filepath_spect   = f"{filepath_sim_res}/spect/",
    time_exp_start   = dict_turb_params["time_growth_start"],
    time_exp_end     = dict_turb_params["time_growth_end"]
  )
  obj_plot_spectra.performRoutines()
  obj_plot_spectra.saveFittedParams(filepath_sim_res)
  ## SAVE FIGURE
  ## -----------
  fig_name = f"{sim_name}_dataset.png"
  PlotFuncs.saveFigure(fig, f"{filepath_vis}/{fig_name}")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## LOOK AT EACH SIMULATION FOLDER
  ## ------------------------------
  ## loop over the simulation suites
  for suite_folder in LIST_SUITE_FOLDER:

    ## loop over the simulation folders
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

      ## loop over the different resolution runs
      for sim_res in LIST_SIM_RES:

        ## CHECK THE RESOLUTION RUN EXISTS
        ## -------------------------------
        filepath_sim_res = f"{filepath_sim}/{sim_res}/"
        ## check that the filepath exists
        if not os.path.exists(filepath_sim_res): continue

        ## MAKE SURE A VISUALISATION FOLDER EXISTS
        ## ---------------------------------------
        filepath_sim_res_plot = f"{filepath_sim_res}/vis_folder"
        WWFnF.createFolder(filepath_sim_res_plot, bool_hide_updates=True)

        ## PLOT SIMULATION DATA AND SAVE MEASURED QUANTITIES
        ## -------------------------------------------------
        sim_name = f"{suite_folder}_{sim_folder}"
        plotSimData(filepath_sim_res, filepath_sim_res_plot, sim_name)

        if BOOL_DEBUG: return
        ## create empty space
        print(" ")
      print(" ")
    print(" ")


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
BOOL_DEBUG        = 1
BASEPATH          = "/scratch/ek9/nk7952/"
SONIC_REGIME      = "super_sonic"

LIST_SUITE_FOLDER = [ "Re10", "Re500", "Rm3000" ]
LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
LIST_SIM_RES      = [ "18", "36", "72", "144", "288", "576" ]

LIST_SUITE_FOLDER = [ "Rm3000" ]
LIST_SIM_FOLDER   = [ "Pm25" ]
LIST_SIM_RES      = [ "288" ]


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
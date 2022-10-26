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
from matplotlib.gridspec import GridSpec

from scipy import interpolate
from lmfit import Model

## load user routines
from plot_turb import PlotTurbData

## load user defined modules
from ThePlottingModule import PlotFuncs
from TheUsefulModule import WWLists, WWFnF, WWObjs
from TheLoadingModule import LoadFlashData
from TheFittingModule import FitMHDScales
from TheJobModule import SimParams

## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## DATASET
## ###############################################################
class DataSet():
  def __init__(
      self,
      Nres,
      Re, Rm, Pm,
      rms_mach, Gamma, E_sat_ratio,
      list_alpha_kin, list_k_nu, list_k_p, list_k_eq
    ):
    self.Nres           = Nres
    self.Re             = Re
    self.Rm             = Rm
    self.Pm             = Pm
    self.rms_mach       = rms_mach
    self.Gamma          = Gamma
    self.E_sat_ratio    = E_sat_ratio
    self.list_alpha_kin = list_alpha_kin
    self.list_k_nu      = list_k_nu
    self.list_k_p       = list_k_p
    self.list_k_eq      = list_k_eq

  def saveDataset(self):
    a = 10


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def getPlasmaNumberFromSimName(sim_name, plasma_number):
  return float(sim_name.replace(plasma_number, "")) if plasma_number in sim_name else None

def interpLogLogData(x, y, x_interp, kind="cubic"):
  interpolator = interpolate.interp1d(np.log10(x), np.log10(y), kind=kind)
  return np.power(10.0, interpolator(np.log10(x_interp)))

def fitKinSpectra(ax, list_k_data, list_power_data, bool_plot=True):
  label_fit = r"$A k^{\alpha} \exp\left\{-\frac{k}{k_\nu}\right\}$"
  my_model  = Model(FitMHDScales.SpectraModels.kinetic_loge)
  my_model.set_param_hint("A",     min = 10**(-3.0),  value = 10**(-1.0),  max = 10**(3.0))
  my_model.set_param_hint("alpha", min = -10.0,       value = -2.0,        max = -1.0)
  my_model.set_param_hint("k_nu",  min = 0.1,         value = 5.0,         max = 50.0)
  input_params = my_model.make_params()
  fit_results  = my_model.fit(
    k      = list_k_data,
    data   = np.log(list_power_data),
    params = input_params
  )
  fit_params = [
    fit_results.params["A"].value,
    fit_results.params["alpha"].value,
    fit_results.params["k_nu"].value
  ]
  array_k_fit     = np.logspace(np.log10(min(list_k_data)), np.log10(max(list_k_data)), 1000)
  array_power_fit = FitMHDScales.SpectraModels.kinetic_linear(array_k_fit, *fit_params)
  if bool_plot: ax.plot(array_k_fit, array_power_fit, label=label_fit, color="black", ls="-", lw=3, zorder=5)
  return fit_params

def fitMagSpectra(ax, list_k_data, list_power_data):
  label_fit = r"$A k^{\alpha_1} {\rm K}_0\left\{ \left(\frac{k}{k_\eta}\right)^{\alpha_2} \right\}$"
  my_model  = Model(FitMHDScales.SpectraModels.magnetic_loge)
  my_model.set_param_hint("A",       min = 1e-3, value = 1e-1, max = 1e3)
  my_model.set_param_hint("alpha_1", min = 0.1,  value = 1.5,  max = 6.0)
  my_model.set_param_hint("alpha_2", min = 0.1,  value = 1.0,  max = 1.5)
  my_model.set_param_hint("k_eta",   min = 1e-3, value = 5.0,  max = 10.0)
  input_params = my_model.make_params()
  fit_results = my_model.fit(
    k      = list_k_data,
    data   = np.log(list_power_data),
    params = input_params
  )
  fit_params = [
    fit_results.params["A"].value,
    fit_results.params["alpha_1"].value,
    fit_results.params["alpha_2"].value,
    fit_results.params["k_eta"].value
  ]
  array_k_fit     = np.logspace(0, 2, 1000)
  array_power_fit = FitMHDScales.SpectraModels.magnetic_linear(array_k_fit, *fit_params)
  ax.plot(array_k_fit, array_power_fit, label=label_fit, color="black", ls="-.", lw=3, zorder=5)
  return fit_params

def getMagSpectraPeak(ax, list_k_data, list_power_data, bool_plot=True):
  array_k_interp = np.logspace(np.log10(min(list_k_data)), np.log10(max(list_k_data)), 3*len(list_power_data))[1:-1]
  array_power_interp = interpLogLogData(list_k_data, list_power_data, array_k_interp, "cubic")
  k_p   = array_k_interp[np.argmax(array_power_interp)]
  k_max = np.argmax(list_power_data) + 1
  if bool_plot: ax.plot(array_k_interp, array_power_interp, ls="-", c="orange")
  return k_p, k_max

def plotPDF(ax, list_data, color):
  list_dens, list_bin_edges = np.histogram(list_data, bins=10, density=True)
  list_dens_norm = np.append(0, list_dens / list_dens.sum())
  ax.fill_between(list_bin_edges, list_dens_norm, step="pre", alpha=0.2, color=color)
  ax.plot(list_bin_edges, list_dens_norm, drawstyle="steps", color=color)


## ###############################################################
## PLOT NORMALISED + TIME-AVERAGED ENERGY SPECTRA
## ###############################################################
class PlotSpectra():
  def __init__(
      self,
      fig, axs_spect_data, axs_scales, ax_spect_ratio,
      filepath_data, time_exp_start, time_exp_end
    ):
    self.fig            = fig
    self.axs_spect_data = axs_spect_data
    self.axs_scales     = axs_scales
    self.ax_spect_ratio = ax_spect_ratio
    self.filepath_data  = filepath_data
    self.time_exp_start = time_exp_start
    self.time_exp_end   = time_exp_end
    print("Loading energy spectra...")
    self.__loadData()
    self.__plotEnergySpectra()
    self.__plotEnergyRatio()
    print("Fitting energy spectra...")
    self.__fitKinSpectra()
    self.__fitMagSpectra()
    self.__labelEnergySpectraPlot()
    self.__labelScalesPlots()
    self.__labelEnergyRatioPlot()

  def getKinScales(self):
    return self.list_alpha_kin, self.list_k_nu

  def getMagScales(self):
    return self.list_k_p, self.list_k_eq

  def __loadData(self):
    ## extract the number of plt-files per eddy-turnover-time from 'Turb.log'
    plots_per_eddy = LoadFlashData.getPlotsPerTturbFromFlashParamFile(f"{self.filepath_data}/../", bool_hide_updates=True)
    if plots_per_eddy is None:
      Exception("ERROR: # plt-files could not be read from 'Turb.log'.")
    ## load kinetic energy spectra
    list_kin_k_group_t, list_kin_power_group_t, self.list_kin_time = LoadFlashData.loadListOfSpectraDataInDirectory(
      filepath_data     = self.filepath_data,
      str_spectra_type  = "vel",
      file_start_time   = self.time_exp_start,
      file_end_time     = self.time_exp_end,
      plots_per_eddy    = plots_per_eddy,
      bool_hide_updates = True
    )
    ## load magnetic energy spectra
    list_mag_k_group_t, list_mag_power_group_t, self.list_mag_time = LoadFlashData.loadListOfSpectraDataInDirectory(
      filepath_data     = self.filepath_data,
      str_spectra_type  = "mag",
      file_start_time   = self.time_exp_start,
      file_end_time     = self.time_exp_end,
      plots_per_eddy    = plots_per_eddy,
      bool_hide_updates = True
    )
    ## store energy spectra
    self.list_kin_power_group_t = list_kin_power_group_t
    self.list_mag_power_group_t = list_mag_power_group_t
    self.list_kin_k             = list_kin_k_group_t[0]
    self.list_mag_k             = list_mag_k_group_t[0]
    ## store normalised and time-averaged energy spectra
    self.list_kin_power_norm_group_t = [ np.array(list_power) / sum(list_power) for list_power in list_kin_power_group_t ]
    self.list_mag_power_norm_group_t = [ np.array(list_power) / sum(list_power) for list_power in list_mag_power_group_t ]

  def __plotEnergySpectra(self):
    plot_args = { "marker":"o", "color":"k", "ms":7, "ls":"", "alpha":1.0, "zorder":3 }
    label_kin = r"$\widehat{\mathcal{P}}_{\rm kin}(k)$ data"
    label_mag = r"$\widehat{\mathcal{P}}_{\rm mag}(k)$ data"
    ## plot average normalised energy spectra
    self.axs_spect_data[0].plot(
      self.list_kin_k,
      np.mean(self.list_kin_power_norm_group_t, axis=0),
      label=label_kin, markerfacecolor="green", **plot_args
    )
    self.axs_spect_data[1].plot(
      self.list_mag_k,
      np.mean(self.list_mag_power_norm_group_t, axis=0),
      label=label_mag, markerfacecolor="red", **plot_args
    )
    ## plot each time realisation of the normalised kinetic energy spectrum
    for time_index in range(len(self.list_kin_power_norm_group_t)):
      self.axs_spect_data[0].plot(
        self.list_kin_k,
        self.list_kin_power_norm_group_t[time_index],
        color="green", ls="-", lw=1, alpha=0.1, zorder=1
      )
    ## plot each time realisation of the normalised magnetic energy spectrum
    for time_index in range(len(self.list_mag_power_norm_group_t)):
      self.axs_spect_data[1].plot(
        self.list_mag_k,
        self.list_mag_power_norm_group_t[time_index],
        color="red", ls="-", lw=1, alpha=0.1, zorder=1
      )

  def __plotEnergyRatio(self):
    ## for each time realisation
    self.list_k_eq = []
    self.list_k_eq_time = []
    for time_index in range(len(self.list_mag_power_norm_group_t)):
      ## measure energy ratio spectrum
      list_E_ratio = [
        list_mag_power / list_kin_power
        for list_kin_power, list_mag_power in zip(
          self.list_kin_power_group_t[time_index],
          self.list_mag_power_group_t[time_index]
        )
      ]
      # ## find the vally in the energy ratio spectrum
      # list_E_ratio_inv       = [ 1/ratio for ratio in list_E_ratio ]
      # list_E_ratio_minima, _ = signal.find_peaks(list_E_ratio_inv)
      # if len(list_E_ratio_minima) > 0:
      #   minima_prominance = signal.peak_prominences(list_E_ratio_inv, list_E_ratio_minima)[0]
      #   minima_index      = np.argmax(minima_prominance)
      #   k_end_index       = list_E_ratio_minima[minima_index]
      # else: k_end_index   = len(self.list_mag_k)-1
      # ## interpolate spectrum
      # array_k_interp       = np.linspace(1, self.list_mag_k[k_end_index-1], 10**5)
      # array_E_ratio_interp = interpLogLogData(self.list_mag_k[:k_end_index], list_E_ratio[:k_end_index], array_k_interp, "cubic")
      self.ax_spect_ratio.plot(self.list_mag_k, list_E_ratio, color="black", ls="-", lw=1, alpha=0.1, zorder=3)
      ## measure k_eq
      tol = 1e-1
      if any(
          abs(E_ratio_i - 1) <= tol
          for E_ratio_i in list_E_ratio
        ):
        list_tmp = [
          k if abs(E_ratio_i - 1) <= tol
          else np.nan
          for E_ratio_i, k in zip(list_E_ratio, self.list_mag_k)
        ]
        if BOOL_DEBUG: print([tmp for tmp in list_tmp if tmp is not np.nan])
        k_eq_index = np.nanargmin(list_tmp)
        k_eq       = self.list_mag_k[k_eq_index]
        k_eq_power = list_E_ratio[k_eq_index]
        self.list_k_eq.append(k_eq)
        self.list_k_eq_time.append(self.list_mag_time[time_index])
        if BOOL_DEBUG: self.ax_spect_ratio.plot(k_eq, k_eq_power, "ko")
    self.axs_scales[0].plot(self.list_k_eq_time, self.list_k_eq, "r-", label=r"$k_{\rm eq}$")

  def __fitKinSpectra(self):
    self.list_A_kin     = []
    self.list_alpha_kin = []
    self.list_k_nu      = []
    for time_index in range(len(self.list_kin_power_norm_group_t)):
      end_index_kin = WWLists.getIndexClosestValue(self.list_kin_power_norm_group_t[time_index], 10**(-7))
      params_kin    = fitKinSpectra(
        self.axs_spect_data[0],
        self.list_kin_k[1:end_index_kin],
        self.list_kin_power_norm_group_t[time_index][1:end_index_kin],
        bool_plot = False
      )
      self.list_A_kin.append(params_kin[0])
      self.list_alpha_kin.append(params_kin[1])
      self.list_k_nu.append(params_kin[2])
    self.axs_scales[0].plot(self.list_kin_time, self.list_k_nu, "g-", label=r"$k_\nu$")
    plotPDF(self.axs_scales[1], self.list_k_nu, "g")

  def __fitMagSpectra(self):
    self.list_k_p   = []
    self.list_k_max = []
    for time_index in range(len(self.list_mag_power_norm_group_t)):
      k_p, k_max = getMagSpectraPeak(
        self.axs_spect_data[1],
        self.list_mag_k,
        self.list_mag_power_norm_group_t[time_index],
        bool_plot = False
      )
      self.list_k_p.append(k_p)
      self.list_k_max.append(k_max)
    self.axs_scales[0].plot(self.list_mag_time, self.list_k_p, "k-", label=r"$k_{\rm p}$")
    plotPDF(self.axs_scales[1], self.list_k_p, "k")

  def __labelEnergySpectraPlot(self):
    ## annotate measured scales
    plot_args = { "ls":"--", "lw":2, "zorder":7 }
    self.axs_spect_data[0].axvline(x=np.mean(self.list_k_nu), **plot_args, color="green", label=r"$k_\nu$")
    self.axs_spect_data[1].axvline(x=np.mean(self.list_k_p),  **plot_args, color="black", label=r"$k_{\rm p}$")
    self.axs_spect_data[1].plot(
      np.mean(self.list_k_max),
      np.mean(np.max(self.list_mag_power_norm_group_t, axis=1)),
      label=r"$k_{\rm max}$", color="black", marker="o", ms=10, ls="", zorder=7
    )
    ## create labels
    label_A_kin     = r"$A_{\rm kin} = $ "+"{:.1e}".format(np.mean(self.list_A_kin))
    label_alpha_kin = r"$\alpha = $ "+"{:.1f}".format(np.mean(self.list_alpha_kin))
    label_k_nu      = r"$k_\nu = $ "+"{:.1e}".format(np.mean(self.list_k_nu))
    label_k_p       = r"$k_{\rm p} = $ "+"{:.1f}".format(np.mean(self.list_k_p))
    label_k_max     = r"$k_{\rm max} = $ "+"{:.1f}".format(np.mean(self.list_k_max))
    ## add legends
    list_lines_ax0, list_labels_ax0 = self.axs_spect_data[0].get_legend_handles_labels()
    list_lines_ax1, list_labels_ax1 = self.axs_spect_data[1].get_legend_handles_labels()
    list_lines  = list_lines_ax0  + list_lines_ax1
    list_labels = list_labels_ax0 + list_labels_ax1
    self.axs_spect_data[1].legend(
      list_lines,
      list_labels,
      loc="upper right", bbox_to_anchor=(0.99, 0.99),
      frameon=True, facecolor="white", edgecolor="grey", framealpha=1.0, fontsize=18
    ).set_zorder(10)
    PlotFuncs.plotBoxOfLabels(
      self.fig, self.axs_spect_data[0],
      box_alignment   = (0.0, 0.0),
      xpos            = 0.025,
      ypos            = 0.025,
      alpha           = 1.0,
      fontsize        = 18,
      list_fig_labels = [
        rf"{label_A_kin}, {label_alpha_kin}, {label_k_nu}",
        rf"{label_k_p}, {label_k_max}"
      ]
    )
    ## adjust kinetic energy axis
    self.axs_spect_data[0].set_xlim([ 0.9, max(self.list_mag_k) ])
    self.axs_spect_data[0].set_xlabel(r"$k$")
    self.axs_spect_data[0].set_ylabel(r"$\widehat{\mathcal{P}}_{\rm kin}(k)$", color="green")
    self.axs_spect_data[0].tick_params(axis="y", colors="green")
    self.axs_spect_data[0].set_xscale("log")
    self.axs_spect_data[0].set_yscale("log")
    ## adjust magnetic energy axis
    self.axs_spect_data[1].set_xlim([ 0.9, max(self.list_mag_k) ])
    self.axs_spect_data[1].set_ylabel(r"$\widehat{\mathcal{P}}_{\rm mag}(k)$", color="red")
    self.axs_spect_data[1].tick_params(axis="y", colors="red")
    self.axs_spect_data[1].spines["left"].set_edgecolor("green")
    self.axs_spect_data[1].spines["right"].set_edgecolor("red")
    self.axs_spect_data[1].set_xscale("log")
    self.axs_spect_data[1].set_yscale("log")

  def __labelScalesPlots(self):
    ## time evolution of scales
    self.axs_scales[0].legend(
      loc="upper left", bbox_to_anchor=(0.01, 0.99),
      frameon=True, facecolor="white", edgecolor="grey", framealpha=1.0, fontsize=18
    )
    self.axs_scales[0].set_xlabel(r"$t/t_{\rm turb}$")
    self.axs_scales[0].set_ylabel(r"$k$")
    self.axs_scales[0].set_yscale("log")
    ## PDF of scales
    self.axs_scales[1].set_xlabel(r"$k$")
    self.axs_scales[1].set_ylabel(r"PDF")

  def __labelEnergyRatioPlot(self):
    self.ax_spect_ratio.axhline(y=1, color="red", ls="--")
    self.ax_spect_ratio.set_xlim([ 0.9, max(self.list_mag_k) ])
    self.ax_spect_ratio.set_xlabel(r"$k$")
    self.ax_spect_ratio.set_ylabel(r"$\mathcal{P}_{\rm mag}(k) / \mathcal{P}_{\rm kin}(k)$")
    self.ax_spect_ratio.set_xscale("log")
    self.ax_spect_ratio.set_yscale("log")


## ###############################################################
## HANDLING PLOT CALLS
## ###############################################################
def plotSimData(
    filepath_sim, filepath_plot, sim_name, Nres,
    Re=None, Rm=None, Pm=None
  ):
  ## INITIALISE FIGURE
  ## -----------------
  print("Initialising figure...")
  fig, fig_grid = PlotFuncs.initFigureGrid(
    fig_scale        = 1.0,
    fig_aspect_ratio = (5.0, 8.0),
    num_rows         = 3,
    num_cols         = 4
  )
  ax_mach        = fig.add_subplot(fig_grid[0,  0])
  ax_energy      = fig.add_subplot(fig_grid[1,  0])
  ax_spect_kin   = fig.add_subplot(fig_grid[:2, 1])
  ax_spect_mag   = ax_spect_kin.twinx()
  ax_spect_ratio = fig.add_subplot(fig_grid[:2, 2])
  ax_scales_time = fig.add_subplot(fig_grid[2,  1])
  ax_scales_pdf  = fig.add_subplot(fig_grid[2,  2])
  ## ANNOTATE SIMULATION PARAMETERS
  ## ------------------------------
  ## annotate plasma parameters
  Re, Rm, Pm, _, _ = SimParams.getPlasmaParams(RMS_MACH, K_TURB, Re, Rm, Pm)
  PlotFuncs.plotBoxOfLabels(
    fig, ax_mach,
    box_alignment   = (1.0, 0.0),
    xpos            = 0.95,
    ypos            = 0.05,
    alpha           = 0.5,
    fontsize        = 18,
    list_fig_labels = [
      r"${\rm N}_{\rm res} = $ " + f"{int(Nres)}",
      r"${\rm Re} = $ " + f"{int(Re)}",
      r"${\rm Rm} = $ " + f"{int(Rm)}",
      r"${\rm Pm} = $ " + f"{int(Pm)}",
    ]
  )
  ## PLOT INTEGRATED QUANTITIES (Turb.dat)
  ## -------------------------------------
  plot_turb_obj = PlotTurbData(
    axs           = [ ax_mach, ax_energy ],
    filepath_data = filepath_sim
  )
  time_exp_start, time_exp_end = plot_turb_obj.getExpTimeBounds()
  rms_mach    = plot_turb_obj.getMach()
  Gamma       = plot_turb_obj.getGamma()
  E_sat_ratio = plot_turb_obj.getEsatRatio()
  ## PLOT FITTED SPECTRA
  ## -------------------
  plot_spectra_obj = PlotSpectra(
    fig            = fig,
    axs_spect_data = [ ax_spect_kin, ax_spect_mag ],
    axs_scales     = [ ax_scales_time, ax_scales_pdf ],
    ax_spect_ratio = ax_spect_ratio,
    filepath_data  = f"{filepath_sim}/spect",
    time_exp_start = time_exp_start,
    time_exp_end   = time_exp_end
  )
  list_alpha_kin, list_k_nu = plot_spectra_obj.getKinScales()
  list_k_p, list_k_eq = plot_spectra_obj.getMagScales()
  ## SAVE FIGURE
  ## -----------
  print("Saving figure...")
  fig_name = f"{sim_name}_dataset.png"
  fig_filepath = WWFnF.createFilepath([ filepath_plot, fig_name ])
  plt.savefig(fig_filepath)
  plt.close()
  print("Figure saved:", fig_name)
  ## SAVE DATASET
  ## ------------
  dataset_name = f"{sim_name}_dataset.json"
  dataset_obj  = DataSet(
    Nres,
    Pm, Re, Rm,
    rms_mach, Gamma, E_sat_ratio,
    list_alpha_kin, list_k_nu, list_k_p, list_k_eq
  )
  WWObjs.saveObj2Json(
    obj      = dataset_obj,
    filepath = filepath_sim,
    filename = dataset_name
  )


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## LOOK AT EACH SIMULATION FOLDER
  ## ------------------------------
  ## loop over the simulation suites
  for suite_folder in LIST_SUITE_FOLDER:

    ## loop over the different resolution runs
    for sim_res in LIST_SIM_RES:

      ## CHECK THE SUITE'S FIGURE FOLDER EXISTS
      ## --------------------------------------
      filepath_plot = WWFnF.createFilepath([
        BASEPATH, suite_folder, sim_res, SONIC_REGIME, "vis_folder"
      ])
      if not os.path.exists(filepath_plot):
        print(f"{filepath_plot} does not exist.")
        continue
      str_message = "Looking at suite: {}, Nres = {}".format(suite_folder, sim_res)
      print(str_message)
      print("=" * len(str_message))
      print("Saving figures in:", filepath_plot)
      print(" ")

      ## loop over the simulation folders
      for sim_folder in LIST_SIM_FOLDER:

        ## CHECK THE SIMULATION FOLDER EXISTS
        ## ----------------------------------
        ## create filepath to the simulation folder
        filepath_sim = WWFnF.createFilepath([
          BASEPATH, suite_folder, sim_res, SONIC_REGIME, sim_folder
        ])
        ## check that the filepath exists
        if not os.path.exists(filepath_sim): continue

        ## PLOT SIMULATION DATA AND MEASURE QUANTITIES
        ## -------------------------------------------
        sim_name = f"{suite_folder}_{sim_folder}"
        plotSimData(
          filepath_sim, filepath_plot, sim_name, sim_res,
          Re = getPlasmaNumberFromSimName(suite_folder, "Re"),
          Rm = getPlasmaNumberFromSimName(suite_folder, "Rm"),
          Pm = getPlasmaNumberFromSimName(sim_folder, "Pm")
        )

        if BOOL_DEBUG: return
        ## create empty space
        print(" ")
      print(" ")
    print(" ")


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
BOOL_DEBUG        = 0
BASEPATH          = "/scratch/ek9/nk7952/"
SONIC_REGIME      = "super_sonic"
FILENAME_TURB     = "Turb.dat"
K_TURB            = 2.0
RMS_MACH          = 5.0
T_TURB            = 1 / (K_TURB * RMS_MACH) # ell_turb / (rms_mach * c_s)
LIST_SUITE_FOLDER = [ "Re10", "Re500", "Rm3000" ]
LIST_SIM_RES      = [ "144", "288", "576" ]
LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
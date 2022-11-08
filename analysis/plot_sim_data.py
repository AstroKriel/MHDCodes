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

from scipy import interpolate
from lmfit import Model

## load user defined routines
from plot_turb_data import PlotTurbData

## load user defined modules
from TheUsefulModule import WWLists, WWFnF, WWObjs
from TheJobModule import SimInputParams
from TheLoadingModule import LoadFlashData
from ThePlottingModule import PlotFuncs
from TheFittingModule import FitMHDScales


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def plotPDF(ax, list_data, color):
  list_dens, list_bin_edges = np.histogram(list_data, bins=10, density=True)
  list_dens_norm = np.append(0, list_dens / list_dens.sum())
  ax.fill_between(list_bin_edges, list_dens_norm, step="pre", alpha=0.2, color=color)
  ax.plot(list_bin_edges, list_dens_norm, drawstyle="steps", color=color)

def interpLogLogData(x, y, x_interp, kind="cubic"):
  interpolator = interpolate.interp1d(np.log10(x), np.log10(y), kind=kind)
  return np.power(10.0, interpolator(np.log10(x_interp)))

def fitKinSpectra(ax, list_k_data, list_power_data, bool_plot=True):
  label_fit = r"$A k^{\alpha} \exp\left\{-\frac{k}{k_\nu}\right\}$"
  my_model  = Model(FitMHDScales.SpectraModels.kinetic_loge)
  my_model.set_param_hint("A",     min = 10**(-3.0),  value = 10**(-1.0),  max = 10**(3.0))
  my_model.set_param_hint("alpha", min = -10.0,       value = -2.0,        max = -1.0)
  my_model.set_param_hint("k_nu",  min = 10**(-1.0),  value = 5.0,         max = 10**(3.0))
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
  if bool_plot:
    ax.plot(
      array_k_fit,
      array_power_fit,
      label=label_fit, color="black", ls="-", lw=3, zorder=5
    )
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
  ax.plot(
    array_k_fit,
    array_power_fit,
    label=label_fit, color="black", ls="-.", lw=3, zorder=5
  )
  return fit_params

def getMagSpectraPeak(ax, list_k_data, list_power_data, bool_plot=True):
  array_k_interp = np.logspace(
    np.log10(min(list_k_data)),
    np.log10(max(list_k_data)),
    3*len(list_power_data)
  )[1:-1]
  array_power_interp = interpLogLogData(list_k_data, list_power_data, array_k_interp, "cubic")
  k_p   = array_k_interp[np.argmax(array_power_interp)]
  k_max = np.argmax(list_power_data) + 1
  if bool_plot:
    ax.plot(
      array_k_interp,
      array_power_interp,
      color="orange", ls="-"
    )
  return k_p, k_max


## ###############################################################
## OPERATOR CLASS: PLOT NORMALISED + TIME-AVERAGED ENERGY SPECTRA
## ###############################################################
class PlotSpectra():
  def __init__(
      self,
      fig, axs_spectra, axs_scales, ax_spectra_ratio,
      filepath_data, time_exp_start, time_exp_end
    ):
    ## save input arguments
    self.fig                = fig
    self.axs_spectra        = axs_spectra
    self.axs_scales         = axs_scales
    self.ax_spectra_ratio   = ax_spectra_ratio
    self.filepath_data      = filepath_data
    self.time_exp_start     = time_exp_start
    self.time_exp_end       = time_exp_end
    ## initialise quantities to measure
    self.list_mag_k         = None
    self.list_kin_power_ave = None
    self.list_mag_power_ave = None
    self.plots_per_eddy     = None
    self.list_mag_time      = None
    self.list_time_k_eq     = None
    self.alpha_kin_group_t  = None
    self.k_nu_group_t       = None
    self.k_p_group_t        = None
    self.k_eq_group_t       = None
    ## flag to check that quantities have been measured
    self.bool_fitted        = False

  def performRoutines(self):
    self.__loadData()
    self.__plotSpectra()
    self.__plotSpectraRatio()
    print("Fitting energy spectra...")
    self.__fitKinSpectra()
    self.__fitMagSpectra()
    self.bool_fitted = True
    self.__labelSpectraPlot()
    self.__labelScalesPlots()
    self.__labelSpectraRatioPlot()

  def getFittedParams(self):
    if not self.bool_fitted: self.performRoutines()
    return {
      ## normalised and time-averaged energy spectra
      "list_k"             : self.list_mag_k,
      "list_kin_power_ave" : self.list_kin_power_ave,
      "list_mag_power_ave" : self.list_mag_power_ave,
      ## measured quantities
      "plots_per_eddy"     : self.plots_per_eddy,
      "list_time_growth"   : self.list_mag_time,
      "list_time_k_eq"     : self.list_time_k_eq,
      "alpha_kin_group_t"  : self.alpha_kin_group_t,
      "k_nu_group_t"       : self.k_nu_group_t,
      "k_p_group_t"        : self.k_p_group_t,
      "k_eq_group_t"       : self.k_eq_group_t,
    }

  def saveFittedParams(self, filepath_sim):
    dict_params = self.getFittedParams()
    WWObjs.saveDict2JsonFile(f"{filepath_sim}/sim_outputs.json", dict_params)

  def __loadData(self):
    print("Loading energy spectra...")
    ## extract the number of plt-files per eddy-turnover-time from 'Turb.log'
    self.plots_per_eddy = LoadFlashData.getPlotsPerEddy_fromTurbLog(
      f"{self.filepath_data}/../",
      bool_hide_updates = True
    )
    if self.plots_per_eddy is None:
      raise Exception("ERROR: failed to read number of plt-files per turn-over-time from 'Turb.log'!")
    ## load kinetic energy spectra
    list_kin_k_group_t, list_kin_power_group_t, self.list_kin_time = LoadFlashData.loadAllSpectraData(
      filepath          = self.filepath_data,
      str_spectra_type  = "vel",
      file_start_time   = self.time_exp_start,
      file_end_time     = self.time_exp_end,
      plots_per_eddy    = self.plots_per_eddy,
      bool_hide_updates = True
    )
    ## load magnetic energy spectra
    list_mag_k_group_t, list_mag_power_group_t, self.list_mag_time = LoadFlashData.loadAllSpectraData(
      filepath          = self.filepath_data,
      str_spectra_type  = "mag",
      file_start_time   = self.time_exp_start,
      file_end_time     = self.time_exp_end,
      plots_per_eddy    = self.plots_per_eddy,
      bool_hide_updates = True
    )
    ## store time-evolving energy spectra
    self.list_kin_power_group_t = list_kin_power_group_t
    self.list_mag_power_group_t = list_mag_power_group_t
    self.list_kin_k             = list_kin_k_group_t[0]
    self.list_mag_k             = list_mag_k_group_t[0]
    ## store normalised energy spectra
    self.list_kin_power_norm_group_t = [
      np.array(list_power) / sum(list_power)
      for list_power in list_kin_power_group_t
    ]
    self.list_mag_power_norm_group_t = [
      np.array(list_power) / sum(list_power)
      for list_power in list_mag_power_group_t
    ]
    ## store normalised, and time-averaged energy spectra
    self.list_kin_power_ave = np.mean(self.list_kin_power_norm_group_t, axis=0)
    self.list_mag_power_ave = np.mean(self.list_mag_power_norm_group_t, axis=0)

  def __plotSpectra(self):
    plot_args = { "marker":"o", "color":"k", "ms":7, "ls":"", "alpha":1.0, "zorder":3 }
    label_kin = r"$\widehat{\mathcal{P}}_{\rm kin}(k)$ data"
    label_mag = r"$\widehat{\mathcal{P}}_{\rm mag}(k)$ data"
    ## plot average normalised energy spectra
    self.axs_spectra[0].plot(
      self.list_kin_k,
      self.list_kin_power_ave,
      label=label_kin, markerfacecolor="green", **plot_args
    )
    self.axs_spectra[1].plot(
      self.list_mag_k,
      self.list_mag_power_ave,
      label=label_mag, markerfacecolor="red", **plot_args
    )
    ## plot each time realisation of the normalised kinetic energy spectrum
    for time_index in range(len(self.list_kin_time)):
      self.axs_spectra[0].plot(
        self.list_kin_k,
        self.list_kin_power_norm_group_t[time_index],
        color="green", ls="-", lw=1, alpha=0.1, zorder=1
      )
    ## plot each time realisation of the normalised magnetic energy spectrum
    for time_index in range(len(self.list_mag_time)):
      self.axs_spectra[1].plot(
        self.list_mag_k,
        self.list_mag_power_norm_group_t[time_index],
        color="red", ls="-", lw=1, alpha=0.1, zorder=1
      )

  def __plotSpectraRatio(self):
    ## for each time realisation
    self.k_eq_group_t   = []
    self.list_time_k_eq = []
    for time_index in range(len(self.list_mag_time)):
      ## calculate energy ratio spectrum
      list_spectra_ratio = [
        list_mag_power / list_kin_power
        for list_kin_power, list_mag_power in zip(
          self.list_kin_power_group_t[time_index],
          self.list_mag_power_group_t[time_index]
        )
      ]
      ## plot ratio of spectra
      self.ax_spectra_ratio.plot(
        self.list_mag_k,
        list_spectra_ratio,
        color="black", ls="-", lw=1, alpha=0.1, zorder=3
      )
      ## measure k_eq
      tol = 1e-1
      if any(
          abs(E_ratio_i - 1) <= tol
          for E_ratio_i in list_spectra_ratio
        ):
        list_tmp = [
          k if abs(E_ratio_i - 1) <= tol
          else np.nan
          for E_ratio_i, k in zip(list_spectra_ratio, self.list_mag_k)
        ]
        if BOOL_DEBUG: print([tmp for tmp in list_tmp if tmp is not np.nan])
        k_eq_index = np.nanargmin(list_tmp)
        k_eq       = self.list_mag_k[k_eq_index]
        k_eq_power = list_spectra_ratio[k_eq_index]
        self.k_eq_group_t.append(k_eq)
        self.list_time_k_eq.append(self.list_mag_time[time_index])
        if BOOL_DEBUG: self.ax_spectra_ratio.plot(k_eq, k_eq_power, "ko")
    self.axs_scales[0].plot(
      self.list_time_k_eq,
      self.k_eq_group_t,
      color="red", ls="-", label=r"$k_{\rm eq}$"
    )

  def __fitKinSpectra(self):
    self.A_kin_group_t     = []
    self.alpha_kin_group_t = []
    self.k_nu_group_t      = []
    for time_index in range(len(self.list_kin_time)):
      ## find k-index to stop fitting kinetic energy spectrum
      end_index_kin = WWLists.getIndexClosestValue(
        self.list_kin_power_norm_group_t[time_index],
        10**(-7)
      )
      ## fit kinetic energy spectrum at time-realisation
      params_kin = fitKinSpectra(
        self.axs_spectra[0],
        self.list_kin_k[1:end_index_kin],
        self.list_kin_power_norm_group_t[time_index][1:end_index_kin],
        bool_plot = False
      )
      ## store fitted parameters
      self.A_kin_group_t.append(params_kin[0])
      self.alpha_kin_group_t.append(params_kin[1])
      self.k_nu_group_t.append(params_kin[2])
    ## plot scales
    self.axs_scales[0].plot(
      self.list_kin_time,
      self.k_nu_group_t,
      color="green", ls="-", label=r"$k_\nu$"
    )
    plotPDF(self.axs_scales[1], self.k_nu_group_t, "g")

  def __fitMagSpectra(self):
    self.k_p_group_t   = []
    self.k_max_group_t = []
    for time_index in range(len(self.list_mag_time)):
      ## extract interpolated and raw magnetic peak-scale for time-realisation
      k_p, k_max = getMagSpectraPeak(
        self.axs_spectra[1],
        self.list_mag_k,
        self.list_mag_power_norm_group_t[time_index],
        bool_plot = False
      )
      ## store measured scales
      self.k_p_group_t.append(k_p)
      self.k_max_group_t.append(k_max)
    ## plot scales
    self.axs_scales[0].plot(
      self.list_mag_time,
      self.k_p_group_t,
      color="black", ls="-", label=r"$k_{\rm p}$"
    )
    plotPDF(self.axs_scales[1], self.k_p_group_t, "k")

  def __labelSpectraPlot(self):
    ## annotate measured scales
    plot_args = { "ls":"--", "lw":2, "zorder":7 }
    self.axs_spectra[0].axvline(x=np.mean(self.k_nu_group_t), **plot_args, color="green", label=r"$k_\nu$")
    self.axs_spectra[1].axvline(x=np.mean(self.k_p_group_t),  **plot_args, color="black", label=r"$k_{\rm p}$")
    self.axs_spectra[1].plot(
      np.mean(self.k_max_group_t),
      np.mean(np.max(self.list_mag_power_norm_group_t, axis=1)),
      color="black", marker="o", ms=10, ls="", label=r"$k_{\rm max}$", zorder=7
    )
    ## create labels for measured scales
    label_A_kin     = r"$A_{\rm kin} = $ " +"{:.1e}".format(np.mean(self.A_kin_group_t))
    label_alpha_kin = r"$\alpha = $ "      +"{:.1f}".format(np.mean(self.alpha_kin_group_t))
    label_k_nu      = r"$k_\nu = $ "       +"{:.1e}".format(np.mean(self.k_nu_group_t))
    label_k_p       = r"$k_{\rm p} = $ "   +"{:.1f}".format(np.mean(self.k_p_group_t))
    label_k_max     = r"$k_{\rm max} = $ " +"{:.1f}".format(np.mean(self.k_max_group_t))
    ## add legend: markers/linestyles of plotted quantities
    list_lines_ax0, list_labels_ax0 = self.axs_spectra[0].get_legend_handles_labels()
    list_lines_ax1, list_labels_ax1 = self.axs_spectra[1].get_legend_handles_labels()
    list_lines  = list_lines_ax0  + list_lines_ax1
    list_labels = list_labels_ax0 + list_labels_ax1
    self.axs_spectra[1].legend(
      list_lines,
      list_labels,
      loc="upper right", bbox_to_anchor=(0.99, 0.99),
      frameon=True, facecolor="white", edgecolor="grey", framealpha=1.0, fontsize=18
    ).set_zorder(10)
    ## add legend: measured parameter values
    PlotFuncs.addBoxOfLabels(
      self.fig, self.axs_spectra[0],
      box_alignment = (0.0, 0.0),
      xpos          = 0.025,
      ypos          = 0.025,
      alpha         = 1.0,
      fontsize      = 18,
      list_labels   = [
        rf"{label_A_kin}, {label_alpha_kin}, {label_k_nu}",
        rf"{label_k_p}, {label_k_max}"
      ]
    )
    ## adjust kinetic energy axis
    self.axs_spectra[0].set_xlim([ 0.9, 1.1*max(self.list_mag_k) ])
    self.axs_spectra[0].set_xlabel(r"$k$")
    self.axs_spectra[0].set_ylabel(r"$\widehat{\mathcal{P}}_{\rm kin}(k)$", color="green")
    self.axs_spectra[0].tick_params(axis="y", colors="green")
    self.axs_spectra[0].set_xscale("log")
    self.axs_spectra[0].set_yscale("log")
    ## adjust magnetic energy axis
    self.axs_spectra[1].set_xlim([ 0.9, 1.1*max(self.list_mag_k) ])
    self.axs_spectra[1].set_ylabel(r"$\widehat{\mathcal{P}}_{\rm mag}(k)$", color="red")
    self.axs_spectra[1].tick_params(axis="y", colors="red")
    self.axs_spectra[1].spines["left"].set_edgecolor("green")
    self.axs_spectra[1].spines["right"].set_edgecolor("red")
    self.axs_spectra[1].set_xscale("log")
    self.axs_spectra[1].set_yscale("log")

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

  def __labelSpectraRatioPlot(self):
    self.ax_spectra_ratio.axhline(y=1, color="red", ls="--")
    self.ax_spectra_ratio.set_xlim([ 0.9, max(self.list_mag_k) ])
    self.ax_spectra_ratio.set_xlabel(r"$k$")
    self.ax_spectra_ratio.set_ylabel(r"$\mathcal{P}_{\rm mag}(k) / \mathcal{P}_{\rm kin}(k)$")
    self.ax_spectra_ratio.set_xscale("log")
    self.ax_spectra_ratio.set_yscale("log")


## ###############################################################
## HANDLING PLOT CALLS
## ###############################################################
def plotSimData(filepath_sim, filepath_vis, sim_name):
  ## GET SIMULATION PARAMETERS
  ## -------------------------
  obj_sim_params  = SimInputParams.readSimInputParams(filepath_sim)
  dict_sim_params = obj_sim_params.getSimParams()
  ## INITIALISE FIGURE
  ## -----------------
  print("Initialising figure...")
  fig, fig_grid = PlotFuncs.createFigure_grid(
    fig_scale        = 1.0,
    fig_aspect_ratio = (5.0, 8.0),
    num_rows         = 3,
    num_cols         = 4
  )
  ax_Mach          = fig.add_subplot(fig_grid[0,  0])
  ax_E_ratio       = fig.add_subplot(fig_grid[1,  0])
  ax_spect_kin     = fig.add_subplot(fig_grid[:2, 1])
  ax_spect_mag     = ax_spect_kin.twinx()
  ax_spectra_ratio = fig.add_subplot(fig_grid[:2, 2])
  ax_scales_time   = fig.add_subplot(fig_grid[2,  1])
  ax_scales_pdf    = fig.add_subplot(fig_grid[2,  2])
  ## PLOT INTEGRATED QUANTITIES
  ## --------------------------
  obj_plot_turb = PlotTurbData(
    fig             = fig,
    axs             = [ ax_Mach, ax_E_ratio ],
    filepath_data   = filepath_sim,
    dict_sim_params = dict_sim_params
  )
  obj_plot_turb.performRoutines()
  obj_plot_turb.saveFittedParams(filepath_sim)
  dict_turb_params = obj_plot_turb.getFittedParams()
  ## PLOT FITTED SPECTRA
  ## -------------------
  obj_plot_spectra = PlotSpectra(
    fig              = fig,
    axs_spectra      = [ ax_spect_kin, ax_spect_mag ],
    axs_scales       = [ ax_scales_time, ax_scales_pdf ],
    ax_spectra_ratio = ax_spectra_ratio,
    filepath_data    = f"{filepath_sim}/spect/",
    time_exp_start   = dict_turb_params["time_growth_start"],
    time_exp_end     = dict_turb_params["time_growth_end"]
  )
  obj_plot_spectra.performRoutines()
  obj_plot_spectra.saveFittedParams(filepath_sim)
  ## SAVE FIGURE
  ## -----------
  print("Saving figure...")
  fig_name     = f"{sim_name}_dataset.png"
  fig_filepath = f"{filepath_vis}/{fig_name}"
  plt.savefig(fig_filepath)
  plt.close()
  print("Saved figure:", fig_filepath)


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
BOOL_DEBUG        = 0
BASEPATH          = "/scratch/ek9/nk7952/"
SONIC_REGIME      = "super_sonic"

# LIST_SUITE_FOLDER = [ "Re10", "Re500", "Rm3000" ]
# LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
LIST_SIM_RES      = [ "18", "36", "72", "144", "288", "576" ]

LIST_SUITE_FOLDER = [ "Rm3000" ]
LIST_SIM_FOLDER   = [ "Pm1" ]


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
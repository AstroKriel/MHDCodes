#!/usr/bin/env python3

## TODO: https://lmfit.github.io/lmfit-py/model.html
## ###############################################################
## MODULES
## ###############################################################
import os, sys, functools
import numpy as np

# ## 'tmpfile' needs to be loaded before 'matplotlib'.
# ## This is so matplotlib stores cache in a temporary directory.
# ## (Useful for plotting in parallel)
# import tempfile
# os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit, fsolve
from lmfit import Model

## load user defined modules
from ThePlottingModule import PlotSpectra, PlotFuncs
from TheUsefulModule import WWLists, WWFnF
from TheLoadingModule import LoadFlashData
from TheFittingModule import FitMHDScales, UserModels
from TheAnalysisModule import WWSpectra

## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
## use a non-interactive plotting backend
plt.ioff()
plt.switch_backend("agg")


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def fitExpFunc(
    ax, data_x, data_y, index_start_fit, index_end_fit,
    color      = "black",
    linestyle  = "-"
  ):
  ## define fit domain
  data_fit_domain = np.linspace(
    data_x[index_start_fit],
    data_x[index_end_fit],
    10**2
  )
  ## interpolate the non-uniform data
  interp_spline = make_interp_spline(
    data_x[index_start_fit : index_end_fit],
    data_y[index_start_fit : index_end_fit]
  )
  ## save time range being fitted to
  time_start = data_x[index_start_fit]
  time_end   = data_x[index_end_fit]
  ## uniformly sample interpolated data
  data_y_sampled = abs(interp_spline(data_fit_domain))
  ## fit exponential function to sampled data (in log-linear domain)
  fit_params_log, fit_params_cov = curve_fit(
    UserModels.ListOfModels.exp_loge,
    data_fit_domain,
    np.log(data_y_sampled)
  )
  ## undo log transformation
  fit_params_linear = [
    np.exp(fit_params_log[0] + 2),
    fit_params_log[1]
  ]
  ## initialise the plot domain
  data_x_fit = np.linspace(0, 100, 10**3)
  ## evaluate exponential
  data_y_fit = UserModels.ListOfModels.exp_linear(
    data_x_fit,
    *fit_params_linear
  )
  ## find where exponential enters / exists fit range
  index_E_start = WWLists.getIndexClosestValue(data_x_fit, time_start)
  index_E_end   = WWLists.getIndexClosestValue(data_x_fit, time_end)
  ## plot fit
  gamma_val = -fit_params_log[1]
  gamma_std = max(np.sqrt(np.diag(fit_params_cov))[1], 0.01)
  str_label = r"$\Gamma =$ " + "{:.2f}".format(gamma_val) + r" $\pm$ " + "{:.2f}".format(gamma_std)
  ax.plot(
    data_x_fit[index_E_start : index_E_end],
    data_y_fit[index_E_start : index_E_end],
    label=str_label, c=color, ls=linestyle, lw=2, zorder=5
  )

def fitConstFunc(
    ax, data_x, data_y, index_start_fit, index_end_fit,
    str_label = "",
    color     = "black",
    linestyle = "-"
  ):
  ## define fit domain
  data_fit_domain = np.linspace(
    data_x[index_start_fit],
    data_x[index_end_fit],
    10**2
  )
  ## interpolate the non-uniform data
  interp_spline = make_interp_spline(
    data_x[index_start_fit : index_end_fit],
    data_y[index_start_fit : index_end_fit]
  )
  ## uniformly sample interpolated data
  data_y_sampled = interp_spline(data_fit_domain)
  ## measure average saturation level
  data_x_sub  = data_x[index_start_fit : index_end_fit]
  data_y_mean = np.mean(data_y_sampled)
  data_y_std  = max(np.std(data_y_sampled), 0.01)
  ## plot fit
  str_label += "{:.2f}".format(data_y_mean) + r" $\pm$ " + "{:.2f}".format(data_y_std)
  ax.plot(
    data_x_sub,
    [ data_y_mean ] * len(data_x_sub),
    label=str_label, c=color, ls=linestyle, lw=2, zorder=5
  )
  ## return mean value
  return data_y_mean

def fitKinSpectra(ax, list_k, list_power):
  list_power_loge = np.log(list_power)
  my_model = Model(FitMHDScales.SpectraModels.kinetic_loge)
  my_model.set_param_hint("A_loge", value = -5.0,  min = -10.0,  max =  5.0)
  my_model.set_param_hint("alpha",  value = -2.0,  min = -10.0,  max =  3.0)
  my_model.set_param_hint("ell_nu", value =  1/10, min =  1/100, max =  1/0.01)
  input_params = my_model.make_params()
  fit_results = my_model.fit(list_power_loge, input_params, k=list_k)
  fit_params = [
    np.exp(fit_results.params["A_loge"].value),
    fit_results.params["alpha"].value,
    fit_results.params["ell_nu"].value
  ]
  list_power_fit = FitMHDScales.SpectraModels.kinetic_linear(list_k, *fit_params)
  ax.plot(list_k, list_power_fit, c="k", ls="-", lw=2, zorder=5)
  return fit_params

def fitMagSpectra(ax, list_k, list_power):
  list_power_loge = np.log(list_power)
  my_model = Model(FitMHDScales.SpectraModels.magnetic_loge)
  my_model.set_param_hint("A_loge",  value = -1.0,   min = -10.0,  max = 5.0)
  my_model.set_param_hint("alpha_1", value =  5.0,   min =  0.01,  max = 10.0)
  my_model.set_param_hint("alpha_2", value =  0.5,   min =  0.01,  max = 2.0)
  my_model.set_param_hint("ell_eta", value =  1/0.2, min =  1/100, max = 1/0.01)
  input_params = my_model.make_params()
  fit_results = my_model.fit(list_power_loge, input_params, k=list_k)
  fit_params = [
    np.exp(fit_results.params["A_loge"].value),
    fit_results.params["alpha_1"].value,
    fit_results.params["alpha_2"].value,
    fit_results.params["ell_eta"].value
  ]
  list_power_fit = FitMHDScales.SpectraModels.magnetic_linear(list_k, *fit_params)
  ax.plot(list_k, list_power_fit, c="k", ls="-", lw=2, zorder=5)
  return fit_params


## ###############################################################
## PLOT INTEGRATED QUANTITIES
## ###############################################################
class PlotTurbData():
  def __init__(
      self,
      axs, filepath_data
    ):
    self.axs            = axs
    self.filepath_data  = filepath_data
    self.time_exp_start = None
    self.time_exp_end   = None
    self.color_fits     = "black"
    self.color_data     = "orange"
    self.__loadData()
    self.__plotData()
    self.__fitData()

  def getExpTimeBounds(self):
    return self.time_exp_start, self.time_exp_end

  def __loadData(self):
    ## load mach data
    _, self.data_Mach = LoadFlashData.loadTurbData(
      filepath_data = self.filepath_data,
      var_y         = 13, # 13 (new), 8 (old)
      t_turb        = T_TURB,
      time_start    = 0.1,
      time_end      = np.inf
    )
    ## load magnetic energy
    _, data_E_B = LoadFlashData.loadTurbData(
      filepath_data = self.filepath_data,
      var_y         = 11, # 11 (new), 29 (old)
      t_turb        = T_TURB,
      time_start    = 0.1,
      time_end      = np.inf
    )
    ## load kinetic energy
    self.data_time, data_E_K = LoadFlashData.loadTurbData(
      filepath_data = self.filepath_data,
      var_y         = 9, # 9 (new), 6 (old)
      t_turb        = T_TURB,
      time_start    = 0.1,
      time_end      = np.inf
    )
    ## calculate plot domain range
    self.max_time = max([ 100, max(self.data_time) ])
    ## calculate energy ratio: 'E_B / E_K'
    self.data_E_ratio = [
      (E_B / E_K) for E_B, E_K in zip(data_E_B, data_E_K)
    ]

  def __plotData(self):
    print("Plotting energy integrated quantities...")
    ## plot mach
    self.axs[0].plot(
      self.data_time, self.data_Mach,
      c=self.color_data, ls="-", lw=1.5, zorder=3
    )
    self.axs[0].set_xlabel(r"$t / t_\mathrm{turb}$")
    self.axs[0].set_ylabel(r"$\mathcal{M}$")
    self.axs[0].set_xlim([ 0, self.max_time ])
    ## plot energy ratio
    self.axs[1].plot(
      self.data_time, self.data_E_ratio,
      c=self.color_data, ls="-", lw=1.5, zorder=3
    )
    ## define y-axis range for the energy ratio plot
    min_E_ratio         = min(self.data_E_ratio)
    log_min_E_ratio     = np.log10(min_E_ratio)
    new_log_min_E_ratio = np.floor(log_min_E_ratio)
    num_decades         = 1 + (-new_log_min_E_ratio)
    new_min_E_ratio     = 10**new_log_min_E_ratio
    num_y_major_ticks   = np.ceil(num_decades / 2)
    ## label axis
    self.axs[1].set_xlabel(r"$t / t_\mathrm{turb}$")
    self.axs[1].set_ylabel(r"$E_\mathrm{mag} / E_\mathrm{kin}$")
    self.axs[1].set_yscale("log")
    self.axs[1].set_xlim([ 0, self.max_time ])
    self.axs[1].set_ylim([ new_min_E_ratio, 10**(1) ])
    ## add log axis-ticks
    PlotFuncs.addLogAxisTicks(
      self.axs[1],
      bool_major_ticks    = True,
      max_num_major_ticks = num_y_major_ticks
    )

  def __fitData(self):
    str_label_Esat = r"$\left(E_{\rm kin} / E_{\rm mag}\right)_{\rm sat} =$ "
    str_label_Mach = r"$\mathcal{M} =$ "
    ls_kin = "--"
    ls_sat = ":"
    ## fit saturation
    growth_percent = self.data_E_ratio[-1] / self.data_E_ratio[WWLists.getIndexClosestValue(self.data_time, 5)]
    ## if dynamo growth occurs
    if growth_percent > 100:
      ## find saturated energy ratio
      sat_ratio = fitConstFunc(
        self.axs[1],
        data_x          = self.data_time,
        data_y          = self.data_E_ratio,
        color           = self.color_fits,
        str_label       = str_label_Esat,
        index_start_fit = WWLists.getIndexClosestValue(self.data_time, (0.75 * self.data_time[-1])),
        index_end_fit   = len(self.data_time)-1,
        linestyle       = ls_sat
      )
      ## get index range corresponding with kinematic phase of the dynamo
      index_exp_start = WWLists.getIndexClosestValue(self.data_E_ratio, 10**(-7))
      index_exp_end   = WWLists.getIndexClosestValue(self.data_E_ratio, sat_ratio/100) # 1-percent of sat-ratio
      index_start_fit = min([ index_exp_start, index_exp_end ])
      index_end_fit   = max([ index_exp_start, index_exp_end ])
      ## find growth rate of exponential
      fitExpFunc(
        self.axs[1],
        data_x          = self.data_time,
        data_y          = self.data_E_ratio,
        color           = self.color_fits,
        index_start_fit = index_start_fit,
        index_end_fit   = index_end_fit,
        linestyle       = ls_kin
      )
    else: # if no growth occurs
      ## get index range corresponding with end of the simulation
      index_start_fit = WWLists.getIndexClosestValue(self.data_time, (0.75 * self.data_time[-1]))
      index_end_fit   = len(self.data_time)-1
      ## find average energy ratio
      sat_ratio = fitConstFunc(
        self.axs[1],
        data_x          = self.data_time,
        data_y          = self.data_E_ratio,
        color           = self.color_fits,
        str_label       = str_label_Esat,
        index_start_fit = index_start_fit,
        index_end_fit   = index_end_fit,
        linestyle       = ls_sat
      )
    ## find average mach number
    fitConstFunc(
      self.axs[0],
      data_x          = self.data_time,
      data_y          = self.data_Mach,
      color           = self.color_fits,
      str_label       = str_label_Mach,
      index_start_fit = index_start_fit,
      index_end_fit   = index_end_fit,
      linestyle       = ls_kin
    )
    ## add legend
    legend_ax0 = self.axs[0].legend(frameon=False, loc="lower left", fontsize=18)
    legend_ax1 = self.axs[1].legend(frameon=False, loc="lower right", fontsize=18)
    self.axs[0].add_artist(legend_ax0)
    self.axs[1].add_artist(legend_ax1)
    ## store time range bounds corresponding with the exponential phase of the dynamo
    self.time_exp_start = self.data_time[index_start_fit]
    self.time_exp_end   = self.data_time[index_end_fit]


## ###############################################################
## PLOT NORMALISED AND TIME-AVERAGED ENERGY SPECTRA
## ###############################################################
class PlotSpectra():
  def __init__(
      self,
      fig, ax, filepath_data, time_exp_start, time_exp_end
    ):
    self.fig            = fig
    self.ax             = ax
    self.filepath_data  = filepath_data
    self.time_exp_start = time_exp_start
    self.time_exp_end   = time_exp_end
    self.__loadData()
    self.__plotData()
    self.__fitData()
    self.__labelPlot()
  
  def __loadData(self):
    print("Loading energy spectra...")
    ## extract the number of plt-files per eddy-turnover-time from 'Turb.log'
    plots_per_eddy = LoadFlashData.getPlotsPerEddy(f"{self.filepath_data}/../", bool_hide_updates=True)
    if plots_per_eddy is None:
      Exception("ERROR: # plt-files could not be read from 'Turb.log'.")
    ## load kinetic energy spectra
    list_kin_k_group_t, list_kin_power_group_t, _ = LoadFlashData.loadListSpectra(
      filepath_data     = self.filepath_data,
      str_spectra_type  = "vel",
      file_start_time   = self.time_exp_start,
      file_end_time     = self.time_exp_end,
      plots_per_eddy    = plots_per_eddy,
      bool_hide_updates = True
    )
    ## load magnetic energy spectra
    list_mag_k_group_t, list_mag_power_group_t, _ = LoadFlashData.loadListSpectra(
      filepath_data     = self.filepath_data,
      str_spectra_type  = "mag",
      file_start_time   = self.time_exp_start,
      file_end_time     = self.time_exp_end,
      plots_per_eddy    = plots_per_eddy,
      bool_hide_updates = True
    )
    ## store normalised and time-averaged energy spectra
    self.list_kin_power_ave = WWSpectra.AveNormSpectraData(list_kin_power_group_t)
    self.list_mag_power_ave = WWSpectra.AveNormSpectraData(list_mag_power_group_t)
    self.list_kin_k         = list_kin_k_group_t[0]
    self.list_mag_k         = list_mag_k_group_t[0]
  
  def __plotData(self):
    plot_args = { "marker":"o", "ms":5, "alpha":0.2, "zorder":3 }
    self.ax.plot(
      self.list_kin_k, self.list_kin_power_ave,
      label=r"$\widehat{\mathcal{P}}_{\rm kin}(k)$", c="blue", **plot_args
    )
    self.ax.plot(
      self.list_mag_k, self.list_mag_power_ave,
      label=r"$\widehat{\mathcal{P}}_{\rm mag}(k)$", c="red", **plot_args
    )

  def __fitData(self):
    print("Fitting normalised and time-avergaed energy spectra...")
    ## fit kinetic energy spectrum
    fit_params_kin     = fitKinSpectra(self.ax, self.list_kin_k[2:], self.list_kin_power_ave[2:])
    self.alpha_kin     = fit_params_kin[1]
    ell_nu             = fit_params_kin[2]
    self.k_nu          = 1 / ell_nu
    ## fit magnetic energy spectrum
    fit_params_mag     = fitMagSpectra(self.ax, self.list_mag_k[:50], self.list_mag_power_ave[:50])
    self.alpha_mag_1   = fit_params_mag[1]
    self.alpha_mag_2   = fit_params_mag[2]
    ell_eta            = fit_params_mag[3]
    self.k_eta         = 1 / ell_eta
    self.k_eta_alpha_2 = self.k_eta**(1/self.alpha_mag_2)
    self.k_max         = np.argmax(self.list_mag_power_ave) + 1
    k_p_guess          = FitMHDScales.SpectraModels.k_p_simple(self.alpha_mag_1, self.alpha_mag_2, ell_eta)
    ## fit peak scale from the modified Kulsrud and Anderson 1992 model
    try:
      self.k_p = fsolve(
        functools.partial(
          FitMHDScales.SpectraModels.k_p_implicit,
          alpha_1 = self.alpha_mag_1,
          alpha_2 = self.alpha_mag_2,
          ell_eta = ell_eta
        ),
        x0 = k_p_guess # give a guess
      )[0]
    except (RuntimeError, ValueError): self.k_p = k_p_guess

  def __labelPlot(self):
    ## annotate measured scales
    self.ax.axvline(x=self.k_nu,          ls="--", color="blue",  label=r"$k_\nu$")
    self.ax.axvline(x=self.k_eta,         ls="--", color="red",   label=r"$k_\eta$")
    self.ax.axvline(x=self.k_eta_alpha_2, ls="--", color="black", label=r"$k_\eta^{\alpha_{{\rm mag}, 2}}$")
    self.ax.axvline(x=self.k_p,           ls="--", color="green", label=r"$k_{\rm p}$")
    self.ax.plot(self.k_max, max(self.list_mag_power_ave), label=r"$k_{\rm max}$", c="k", marker="o", ms=7, ls="", zorder=7)
    ## create parameter labels
    label_kin_spectra   = r"$\mathcal{P}_{\rm kin}(k) = A_{\rm kin} k^{\alpha_{\rm kin}} \exp\left\{-\frac{k}{k_\nu}\right\}$"
    label_alpha_kin     = r"$\alpha_{\rm kin} = $ "+"{:.1f}".format(self.alpha_kin)
    label_k_nu          = r"$k_\nu = $ "+"{:.1f}".format(self.k_nu)
    label_mag_spectra   = r"$\mathcal{P}_{\rm mag}(k) = A_{\rm mag} k^{\alpha_{{\rm mag}, 1}} \exp\left\{ -\left(\frac{k}{k_\eta}\right)^{\alpha_{{\rm mag}, 2}} \right\}$"
    label_alpha_mag_1   = r"$\alpha_{{\rm mag}, 1} = $ "+"{:.1f}".format(self.alpha_mag_1)
    label_alpha_mag_2   = r"$\alpha_{{\rm mag}, 2} = $ "+"{:.1f}".format(self.alpha_mag_2)
    label_k_eta         = r"$k_\eta = $ "+"{:.2f}".format(self.k_eta)
    label_k_eta_alpha_2 = r"$k_\eta^{1/\alpha_{{\rm mag}, 2}} = $ "+"{:.2f}".format(self.k_eta_alpha_2)
    label_k_p           = r"$k_{\rm p} = $ "+"{:.1f}".format(self.k_p)
    label_k_max         = r"$k_{\rm max} = $ "+"{:.1f}".format(self.k_max)
    ## annotate measured parameters
    PlotFuncs.plotLabelBox(
      self.fig, self.ax,
      box_alignment   = (0.5, 0.0),
      xpos            = 0.5,
      ypos            = 0.025,
      alpha           = 0.5,
      fontsize        = 18,
      list_fig_labels = [
        label_kin_spectra,
        rf"{label_alpha_kin}, {label_k_nu}",
        label_mag_spectra,
        rf"{label_alpha_mag_1}, {label_alpha_mag_2}, {label_k_eta}",
        rf"{label_k_eta_alpha_2}, {label_k_p}, {label_k_max}"
      ]
    )
    ## tune plot
    self.ax.legend(frameon=True, loc="center left", fontsize=18)
    self.ax.set_xlabel(r"$k$")
    self.ax.set_ylabel(r"$\widehat{\mathcal{P}}(k)$")
    self.ax.set_xscale("log")
    self.ax.set_yscale("log")
    PlotFuncs.addLogAxisTicks(self.ax, bool_major_ticks=True, max_num_major_ticks=6)


## ###############################################################
## HANDLING PLOT CALLS
## ###############################################################
def plotSimData(filepath_sim, filepath_plot, fig_name):
  print("Initialising figure...")
  fig = plt.figure(constrained_layout=True, figsize=(12, 8))
  ## create figure sub-axis
  gs = GridSpec(2, 2, figure=fig)
  ## 'Turb.dat' data
  ax_mach    = fig.add_subplot(gs[0, 0])
  ax_energy  = fig.add_subplot(gs[1, 0])
  ## energy spectra
  ax_spectra = fig.add_subplot(gs[:, 1])
  ## #####################################
  ## PLOT INTEGRATED QUANTITIES (Turb.dat)
  ## #####################################
  filepath_data_turb = WWFnF.createFilepath([ filepath_sim, FILENAME_TURB ])
  bool_plot_energy   = os.path.exists(filepath_data_turb)
  time_exp_start     = None
  time_exp_end       = None
  if bool_plot_energy:
    plot_turb_obj = PlotTurbData(
      axs           = [ ax_mach, ax_energy ],
      filepath_data = filepath_sim
    )
    time_exp_start, time_exp_end = plot_turb_obj.getExpTimeBounds()
  ## ###################
  ## PLOT FITTED SPECTRA
  ## ###################
  filepath_data_spect = WWFnF.createFilepath([ filepath_sim, "spect" ])
  bool_plot_spectra   = os.path.exists(filepath_data_spect)
  if bool_plot_spectra:
    PlotSpectra(
      fig            = fig,
      ax             = ax_spectra,
      filepath_data  = filepath_data_spect,
      time_exp_start = time_exp_start,
      time_exp_end   = time_exp_end
    )
  ## ###########
  ## SAVE FIGURE
  ## ###########
  ## check if the data was plotted
  if not(bool_plot_energy) and not(bool_plot_spectra):
    print("ERROR: No data in:")
    print("\t", filepath_sim)
    print("\t", filepath_data_spect)
  else:
    ## check what data was missing
    if not(bool_plot_energy):
      print(f"ERROR: No '{FILENAME_TURB}' file in:")
      print("\t", filepath_sim)
    elif not(bool_plot_spectra):
      print(f"ERROR: No '{...}' file in:")
      print("\t", filepath_data_spect)
    ## save the figure
    print("Saving figure...")
    fig_filepath = WWFnF.createFilepath([ filepath_plot, fig_name ])
    plt.savefig(fig_filepath)
    plt.close()
    print("Figure saved:", fig_name)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
BASEPATH      = "/scratch/ek9/nk7952/"
SONIC_REGIME  = "super_sonic"
FILENAME_TURB = "Turb.dat"
T_TURB        = 0.1 # = ell_turb / (Mach * c_s) = (1/2) / (5 * 1) = 1/10

def main():
  suite_folder = "Re10"
  sim_folder   = "Pm25"
  sim_res      = "288"

  filepath_base = WWFnF.createFilepath([ BASEPATH, suite_folder, sim_res, SONIC_REGIME ])
  filepath_sim  = WWFnF.createFilepath([ filepath_base, sim_folder ])
  filepath_plot = WWFnF.createFilepath([ filepath_base, "vis_folder" ])

  plotSimData(filepath_sim, filepath_plot, "test.png")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
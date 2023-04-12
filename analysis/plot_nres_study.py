#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys, copy
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

## load user defined modules
from TheFlashModule import SimParams, FileNames
from TheUsefulModule import WWFnF, WWObjs
from TheFittingModule import UserModels
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def createLabel_logisticModel(stats):
  return r"${} \pm {}$\;".format(
    str(round(stats[0], 2)),
    str(round(stats[1], 2))
  )

def createLabel_powerLaw(stats):
  return r"${} k^{}$\;".format(
    str(round(stats[0], 2)),
    "{" + str(round(stats[1], 2)) + "}"
  )

def check_mean_within_10_percent(list_of_lists):
  list_means = [
    np.mean(sub_list)
    for sub_list in list_of_lists
  ]  # calculate the mean of each sublist
  list_log_means = np.log10(list_means) # take the logarithm of the means
  mean_log_means = np.mean(list_log_means)
  list_abs_diffs = np.abs(mean_log_means - list_log_means) # calculate the absolute differences using broadcasting
  return np.all(list_abs_diffs < 0.1) # check if all absolute differences are less than or equal to 0.1

def fitLogisticModel(
    list_res, scales_group_t_res,
    ax     = None,
    color  = "black",
    ls     = ":",
    bounds = ( (1.0, 0.1, 0.0), (1e3, 1e4, 5.0) ) # amplitude, turnover scale, turnover rate
  ):
  ## check if measured scales increase or decrease with resolution
  if np.nanmedian(scales_group_t_res[1]) <= np.nanmedian(scales_group_t_res[-1]):
    func = UserModels.ListOfModels.logistic_growth_increasing
  else: func = UserModels.ListOfModels.logistic_growth_decreasing
  fit_params, fit_cov = curve_fit(
    f     = func,
    xdata = list_res,
    ydata = [ np.nanpercentile(scales_group_t, 50) for scales_group_t in scales_group_t_res ],
    sigma = [ np.nanstd(scales_group_t)+1e-3       for scales_group_t in scales_group_t_res ],
    absolute_sigma=True, bounds=bounds, maxfev=10**5
  )
  fit_std = np.nanmean([
    np.nanstd(sub_list)
    for sub_list in scales_group_t_res
  ])
  # fit_std = np.sqrt(np.diag(fit_cov))[0] # confidence in fit
  data_x  = np.logspace(np.log10(1), np.log10(10**4), 100)
  data_y  = func(data_x, *fit_params)
  if ax is not None: ax.plot(data_x, data_y, color=color, ls=ls, lw=1.5)
  return data_y[-1], fit_std


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class PlotConvergence():
  def __init__(self, filepath_sim_288):
    self.filepath_sim = f"{filepath_sim_288}/../"
    self.filepath_vis = f"{self.filepath_sim}/vis_folder/"
    WWFnF.createFolder(self.filepath_vis, bool_verbose=False)
    dict_sim_inputs = SimParams.readSimInputs(filepath_sim_288, bool_verbose=False)
    self.sim_name   = SimParams.getSimName(dict_sim_inputs)

  def saveData(self):
    dict_stats_nres = {
      "k_p_stats_nres"       : self.k_p_stats_nres,
      "k_eta_mag_stats_nres" : self.k_eta_mag_stats_nres,
      "k_eta_cur_stats_nres" : self.k_eta_cur_stats_nres,
      "k_nu_lgt_stats_nres"  : self.k_nu_lgt_stats_nres,
      "k_nu_trv_stats_nres"  : self.k_nu_trv_stats_nres
    }
    WWObjs.saveDict2JsonFile(
      filepath_file = f"{self.filepath_sim}/{FileNames.FILENAME_SIM_SCALES}",
      input_dict    = dict_stats_nres
    )

  def readData(self):
    self.list_sim_res                    = []
    self.k_p_group_t_group_sim_res       = []
    self.k_eta_mag_group_t_group_sim_res = []
    self.k_eta_cur_group_t_group_sim_res = []
    self.k_nu_tot_group_t_group_sim_res  = []
    self.k_nu_lgt_group_t_group_sim_res  = []
    self.k_nu_trv_group_t_group_sim_res  = []
    for sim_res in LIST_SIM_RES:
      filepath_sim_res = f"{self.filepath_sim}/{sim_res}/"
      if not(os.path.isdir(filepath_sim_res)): continue
      dict_sim_outputs = SimParams.readSimOutputs(filepath_sim_res)
      index_bounds_growth = dict_sim_outputs["index_bounds_growth"]
      k_p_group_t       = dict_sim_outputs["k_p_group_t"][index_bounds_growth[0] : index_bounds_growth[1]]
      k_eta_mag_group_t = dict_sim_outputs["k_eta_mag_group_t"][index_bounds_growth[0] : index_bounds_growth[1]]
      k_eta_cur_group_t = dict_sim_outputs["k_eta_cur_group_t"][index_bounds_growth[0] : index_bounds_growth[1]]
      k_nu_lgt_group_t  = dict_sim_outputs["k_nu_lgt_group_t"][index_bounds_growth[0] : index_bounds_growth[1]]
      k_nu_trv_group_t  = dict_sim_outputs["k_nu_trv_group_t"][index_bounds_growth[0] : index_bounds_growth[1]]
      self.list_sim_res.append(int(sim_res))
      self.k_p_group_t_group_sim_res.append(k_p_group_t)
      self.k_eta_mag_group_t_group_sim_res.append(k_eta_mag_group_t)
      self.k_eta_cur_group_t_group_sim_res.append(k_eta_cur_group_t)
      self.k_nu_lgt_group_t_group_sim_res.append(k_nu_lgt_group_t)
      self.k_nu_trv_group_t_group_sim_res.append(k_nu_trv_group_t)

  def plotData(self):
    self.fig, fig_grid = PlotFuncs.createFigure_grid(
      fig_scale        = 1.0,
      fig_aspect_ratio = (5.0, 8.0),
      num_rows         = 2,
      num_cols         = 3
    )
    self.ax_k_p           = self.fig.add_subplot(fig_grid[0, 0])
    self.ax_k_eta_mag     = self.fig.add_subplot(fig_grid[0, 1])
    self.ax_k_eta_cur     = self.fig.add_subplot(fig_grid[1, 1])
    self.ax_k_nu_lgt      = self.fig.add_subplot(fig_grid[0, 2])
    self.ax_k_nu_trv      = self.fig.add_subplot(fig_grid[1, 2])
    self.list_axs = [
      self.ax_k_p,
      self.ax_k_eta_mag,
      self.ax_k_eta_cur,
      self.ax_k_nu_lgt,
      self.ax_k_nu_trv
    ]
    self.__plotScales()
    self.__fitLogisticModel()
    self.__labelAxis()
    filepath_fig = f"{self.filepath_vis}/{self.sim_name}_nres_scales.png"
    PlotFuncs.saveFigure(self.fig, filepath_fig)

  def __plotScales(self):
    for res_index, sim_res in enumerate(self.list_sim_res):
      PlotFuncs.plotErrorBar_1D(
        ax      = self.ax_k_p,
        x       = sim_res,
        array_y = self.k_p_group_t_group_sim_res[res_index],
        color   = "black"
      )
      PlotFuncs.plotErrorBar_1D(
        ax      = self.ax_k_eta_mag,
        x       = sim_res,
        array_y = self.k_eta_mag_group_t_group_sim_res[res_index],
        color   = "black"
      )
      PlotFuncs.plotErrorBar_1D(
        ax      = self.ax_k_eta_cur,
        x       = sim_res,
        array_y = self.k_eta_cur_group_t_group_sim_res[res_index],
        color   = "black"
      )
      PlotFuncs.plotErrorBar_1D(
        ax      = self.ax_k_nu_lgt,
        x       = sim_res,
        array_y = self.k_nu_lgt_group_t_group_sim_res[res_index],
        color   = "black"
      )
      PlotFuncs.plotErrorBar_1D(
        ax      = self.ax_k_nu_trv,
        x       = sim_res,
        array_y = self.k_nu_trv_group_t_group_sim_res[res_index],
        color   = "black"
      )

  def __fitLogisticModel(self):
    ## define helper function
    def fitData(ax, scales_group_t_res, bool_extend=False):
      list_sim_res = copy.deepcopy(self.list_sim_res)
      scales_group_t_res = copy.deepcopy(scales_group_t_res)
      if bool_extend or True:
        list_sim_res.append(2*list_sim_res[-1])
        scales_group_t_res.append(scales_group_t_res[-1])
      ## check that there is at least five data points to fit to for each resolution run
      list_bool_fit = [
        len(scales_group_t) - 5 > sum([
          1
          for scale in scales_group_t
          if scale is None
        ])
        for scales_group_t in scales_group_t_res
      ]
      data_x = np.array(list_sim_res)[list_bool_fit]
      data_y_group_x = [
        [
          scale
          for scale in scales_group_t
          if scale is not None
        ]
        for scales_group_t, bool_fit in zip(scales_group_t_res, list_bool_fit)
        if bool_fit
      ]
      ## fit measured scales at different resolution runs
      if check_mean_within_10_percent(data_y_group_x):
        stats_converge = np.mean(data_y_group_x[-1]), np.std(data_y_group_x[-1])
        ax.axhline(y=np.mean(data_y_group_x[-1]), color="black", ls=":", lw=1.5)
      else:
        stats_converge = fitLogisticModel(
          ax                 = ax,
          list_res           = data_x,
          scales_group_t_res = data_y_group_x,
          color              = "black",
          ls                 = ":"
        )
      return stats_converge
    ## fit scales
    self.k_p_stats_nres       = fitData(self.ax_k_p,       self.k_p_group_t_group_sim_res)
    self.k_eta_mag_stats_nres = fitData(self.ax_k_eta_mag, self.k_eta_mag_group_t_group_sim_res)
    self.k_eta_cur_stats_nres = fitData(self.ax_k_eta_cur, self.k_eta_cur_group_t_group_sim_res)
    self.k_nu_lgt_stats_nres  = fitData(self.ax_k_nu_lgt,  self.k_nu_lgt_group_t_group_sim_res)
    self.k_nu_trv_stats_nres  = fitData(self.ax_k_nu_trv,  self.k_nu_trv_group_t_group_sim_res)

  def __labelAxis(self):
    ## define helper function
    def labelAxis(ax, stats_converge):
      PlotFuncs.addBoxOfLabels(
        fig         = self.fig,
        ax          = ax,
        bbox        = (1.0, 0.0),
        xpos        = 0.95,
        ypos        = 0.05,
        alpha       = 0.85,
        fontsize    = 20,
        list_colors = [
          "black",
        ],
        list_labels = [
          createLabel_logisticModel(stats_converge),
        ],
      )
    ## define helper variables
    ## adjust axis
    for ax_index in range(len(self.list_axs)):
      self.list_axs[ax_index].set_xscale("log")
      self.list_axs[ax_index].set_yscale("log")
      self.list_axs[ax_index].set_xlim([ 10, 10**4 ])
      self.list_axs[ax_index].set_ylim([ 1,  3*10**2 ])
    ## label axis
    self.list_axs[-1].set_xlabel(r"$N_{\rm res}$")
    self.list_axs[-1].set_xlabel(r"$N_{\rm res}$")
    self.ax_k_p.set_ylabel(r"$k_{\rm p}$")
    self.ax_k_eta_mag.set_ylabel(r"$k_{\eta, \mathbf{B}}$")
    self.ax_k_eta_cur.set_ylabel(r"$k_{\eta, \nabla\times\mathbf{B}}$")
    self.ax_k_nu_lgt.set_ylabel(r"$k_{\nu, \parallel}$")
    self.ax_k_nu_trv.set_ylabel(r"$k_{\nu, \perp}$")
    ## annotate simulation parameters
    SimParams.addLabel_simInputs(
      filepath      = f"{self.filepath_sim}/288/",
      fig           = self.fig,
      ax            = self.list_axs[0],
      bbox          = (0.0, 1.0),
      vpos          = (0.05, 0.95),
      bool_show_res = False
    )
    ## annotate fitted scales
    labelAxis(self.ax_k_p,       self.k_p_stats_nres)
    labelAxis(self.ax_k_eta_mag, self.k_eta_mag_stats_nres)
    labelAxis(self.ax_k_eta_cur, self.k_eta_cur_stats_nres)
    labelAxis(self.ax_k_nu_lgt,  self.k_nu_lgt_stats_nres)
    labelAxis(self.ax_k_nu_trv,  self.k_nu_trv_stats_nres)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  list_sim_filepaths = SimParams.getListOfSimFilepaths(
    basepath           = BASEPATH,
    list_suite_folders = LIST_SUITE_FOLDERS,
    list_sonic_regimes = LIST_SONIC_REGIMES,
    list_sim_folders   = LIST_SIM_FOLDERS,
    list_sim_res       = [ "288" ]
  )
  for filepath_sim in list_sim_filepaths:
    obj = PlotConvergence(filepath_sim_288=filepath_sim)
    obj.readData()
    obj.plotData()
    obj.saveData()
    print(" ")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BASEPATH = "/scratch/ek9/nk7952/"

# ## PLASMA PARAMETER SET
# LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm3000" ]
# LIST_SONIC_REGIMES = [ "Mach5" ]
# LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
# LIST_SIM_RES       = [ "18", "36", "72", "144", "288", "576" ]

# ## MACH NUMBER SET
# LIST_SUITE_FOLDERS = [ "Re300" ]
# LIST_SONIC_REGIMES = [ "Mach0.3", "Mach1", "Mach5", "Mach10" ]
# LIST_SIM_FOLDERS   = [ "Pm4" ]
# LIST_SIM_RES       = [ "36", "72", "144", "288" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys, copy
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

## load user defined modules
from TheSimModule import SimParams
from TheUsefulModule import WWFnF, WWObjs
from TheLoadingModule import FileNames
from TheFittingModule import UserModels
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def convertNone2Nan(list_elems):
  return [ np.nan if elem is None else elem for elem in list_elems ]

def removeNones(list_elems):
  return [ elem for elem in list_elems if elem is not None ]

def createLabel_fromStats_converge(stats):
  return r"${} \pm {}$\;".format(
    str(round(stats[0], 2)),
    str(round(stats[1], 2))
  )

def createLabel_fromStats_powerlaw(stats):
  return r"${} k^{}$\;".format(
    str(round(stats[0], 2)),
    "{" + str(round(stats[1], 2)) + "}"
  )

def fitScales(
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
    sigma = [ np.nanstd(scales_group_t)+1e-2       for scales_group_t in scales_group_t_res ],
    absolute_sigma=True, bounds=bounds, maxfev=10**5
  )
  fit_std = np.sqrt(np.diag(fit_cov))[0] # confidence in fit
  data_x  = np.logspace(np.log10(1), np.log10(10**4), 100)
  data_y  = func(data_x, *fit_params)
  if ax is not None: ax.plot(data_x, data_y, color=color, ls=ls, lw=1.5)
  return data_y[-1], fit_std

def fitPowerLaw(
    list_res, scales_group_t_res,
    ax     = None,
    color  = "black",
    ls     = ":"
  ):
  func = UserModels.ListOfModels.powerlaw_linear
  fit_params, fit_cov = curve_fit(
    f     = func,
    xdata = list_res,
    ydata = [ np.nanpercentile(scales_group_t, 50) for scales_group_t in scales_group_t_res ],
    sigma = [ np.nanstd(scales_group_t)+1e-2       for scales_group_t in scales_group_t_res ],
    absolute_sigma=True, maxfev=10**5
  )
  fit_std = np.sqrt(np.diag(fit_cov))[0] # confidence in fit
  data_x  = np.logspace(np.log10(1), np.log10(10**4), 100)
  data_y  = func(data_x, *fit_params)
  if ax is not None: ax.plot(data_x, data_y, color=color, ls=ls, lw=1.5)
  return fit_params


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class PlotScaleConvergence():
  def __init__(
      self,
      filepath_sim, filepath_vis, sim_name
    ):
    self.filepath_sim = filepath_sim
    self.filepath_vis = filepath_vis
    self.sim_name     = sim_name
    self.__initialiseData()

  def __initialiseData(self):
    ## initialise input datasets
    self.list_sim_res         = []
    self.k_p_group_t_res      = []
    self.k_eta_group_t_res    = []
    self.k_nu_lgt_group_t_res = []
    self.k_nu_trv_group_t_res = []

  def readData(self):
    for sim_res in LIST_SIM_RES:
      try:
        dict_sim_outputs = SimParams.readSimOutputs(f"{self.filepath_sim}/{sim_res}/")
        self.k_p_group_t_res.append(dict_sim_outputs["k_p_group_t"])
        self.k_eta_group_t_res.append(dict_sim_outputs["k_eta_group_t"])
        self.k_nu_lgt_group_t_res.append(dict_sim_outputs["k_nu_trv_group_t"])
        self.k_nu_trv_group_t_res.append(dict_sim_outputs["k_nu_lgt_group_t"])
        self.list_sim_res.append(int(sim_res))
      except: continue

  def plotData(self):
    self.fig, fig_grid = PlotFuncs.createFigure_grid(
      fig_scale        = 1.0,
      fig_aspect_ratio = (5.0, 8.0),
      num_rows         = 4,
      num_cols         = 2
    )
    self.ax_k_p      = self.fig.add_subplot(fig_grid[0, 0])
    self.ax_k_eta    = self.fig.add_subplot(fig_grid[1, 0])
    self.ax_k_nu_lgt = self.fig.add_subplot(fig_grid[0, 1])
    self.ax_k_nu_trv = self.fig.add_subplot(fig_grid[1, 1])
    ## plot and fit data
    self.__plotScales()
    self.__fitScales()
    self.__labelAxis()
    ## save figure
    filepath_fig = f"{self.filepath_vis}/{self.sim_name}_nres_scales.png"
    PlotFuncs.saveFigure(self.fig, filepath_fig)

  def saveData(self):
    dict_converged_scales = {
      "k_p_stats_converge"      : self.k_p_stats_converge,
      "k_eta_stats_converge"    : self.k_eta_stats_converge,
      "k_nu_lgt_stats_converge" : self.k_nu_lgt_stats_converge,
      "k_nu_trv_stats_converge" : self.k_nu_trv_stats_converge,
    }
    WWObjs.saveDict2JsonFile(
      filepath_file = f"{self.filepath_sim}/{FileNames.FILENAME_SIM_SCALES}",
      input_dict    = dict_converged_scales
    )
    return

  def __plotScales(self):
    for res_index, sim_res in enumerate(self.list_sim_res):
      PlotFuncs.plotErrorBar_1D(
        ax      = self.ax_k_p,
        x       = sim_res,
        array_y = self.k_p_group_t_res[res_index],
        color   = "black"
      )
      PlotFuncs.plotErrorBar_1D(
        ax      = self.ax_k_eta,
        x       = sim_res,
        array_y = self.k_eta_group_t_res[res_index],
        color   = "black"
      )
      PlotFuncs.plotErrorBar_1D(
        ax      = self.ax_k_nu_lgt,
        x       = sim_res,
        array_y = self.k_nu_lgt_group_t_res[res_index],
        color   = "black"
      )
      PlotFuncs.plotErrorBar_1D(
        ax      = self.ax_k_nu_trv,
        x       = sim_res,
        array_y = self.k_nu_trv_group_t_res[res_index],
        color   = "black"
      )

  def __fitScales(self):
    ## define helper function
    def fitData(ax, scales_group_t_res, bool_extend=False):
      list_sim_res = copy.deepcopy(self.list_sim_res)
      if bool_extend:
        list_sim_res       += [ 2 * self.list_sim_res[-1] ]
        scales_group_t_res += [ scales_group_t_res[-1] ]
      ## check that there is at least five data points to fit to for each resolution run
      list_bool_fit = [
        5 < (len(scales_group_t) - sum([ 1 for scale in scales_group_t if scale is None ]))
        for scales_group_t in scales_group_t_res
      ]
      data_x = np.array(list_sim_res)[list_bool_fit]
      data_y_group_x = [
        [ scale for scale in scales_group_t if scale is not None ]
        for scales_group_t, bool_fit in zip(scales_group_t_res, list_bool_fit)
        if bool_fit
      ]
      ## check if the scales are saturated
      if (
          (np.abs(np.log10(np.mean(data_y_group_x[-1])) - np.log10(np.mean(data_y_group_x[1]))) < 0.035)
          or
          (np.abs(np.log10(np.mean(data_y_group_x[-1])) - np.log10(np.mean(data_y_group_x[-2]))) < 0.035)
          or
          (np.abs(np.log10(np.mean(data_y_group_x[-1])) - np.log10(np.mean(data_y_group_x[-3]))) < 0.035)
        ):
        stats_converge = np.mean(data_y_group_x[-1]), np.std(data_y_group_x[-1])
        ax.axhline(y=stats_converge[0], ls=":", c="k")
      else:
        ## fit measured scales at different resolution runs
        stats_converge = fitScales(
          ax                 = ax,
          list_res           = data_x,
          scales_group_t_res = data_y_group_x,
          color              = "black",
          ls                 = ":"
        )
      return stats_converge
    ## fit scales
    self.k_p_stats_converge      = fitData(self.ax_k_p, self.k_p_group_t_res, bool_extend=True)
    self.k_eta_stats_converge    = fitData(self.ax_k_eta, self.k_eta_group_t_res)
    self.k_nu_lgt_stats_converge = fitData(self.ax_k_nu_lgt, self.k_nu_lgt_group_t_res)
    self.k_nu_trv_stats_converge = fitData(self.ax_k_nu_trv, self.k_nu_trv_group_t_res)

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
          createLabel_fromStats_converge(stats_converge),

        ],
      )
    ## define helper variables
    bounds_nres = [ 1, 10**4 ]
    bounds_kscales = [ 1, 300 ]
    axs = [
      [ self.ax_k_p, self.ax_k_nu_lgt ],
      [ self.ax_k_eta, self.ax_k_nu_trv ]
    ]
    ## adjust axis
    for row_index in range(2):
      for col_index in range(2):
        axs[row_index][col_index].set_xscale("log")
        axs[row_index][col_index].set_yscale("log")
        axs[row_index][col_index].set_xlim(bounds_nres)
        # axs[row_index][col_index].set_ylim(bounds_kscales)
    ## label axis
    axs[-1][0].set_xlabel(r"$N_{\rm res}$")
    axs[-1][1].set_xlabel(r"$N_{\rm res}$")
    axs[0][0].set_ylabel(r"$k_{\rm p}$")
    axs[1][0].set_ylabel(r"$k_\eta$")
    axs[0][1].set_ylabel(r"$k_{\nu, \parallel}$")
    axs[1][1].set_ylabel(r"$k_{\nu, \perp}$")
    ## annotate simulation parameters
    SimParams.addLabel_simInputs(
      filepath      = f"{self.filepath_sim}/288/",
      fig           = self.fig,
      ax            = axs[0][0],
      bbox          = (0.0, 1.0),
      vpos          = (0.05, 0.95),
      bool_show_res = False
    )
    ## annotate fitted scales
    labelAxis(self.ax_k_p,      self.k_p_stats_converge)
    labelAxis(self.ax_k_eta,    self.k_eta_stats_converge)
    labelAxis(self.ax_k_nu_lgt, self.k_nu_lgt_stats_converge)
    labelAxis(self.ax_k_nu_trv, self.k_nu_trv_stats_converge)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## LOOK AT EACH SIMULATION FOLDER
  ## ------------------------------
  ## loop over the simulation suites
  for suite_folder in LIST_SUITE_FOLDER:
    ## COMMUNICATE PROGRESS
    ## --------------------
    str_message = f"Looking at suite: {suite_folder}, regime: {SONIC_REGIME}"
    print(str_message)
    print("=" * len(str_message))
    print(" ")
    ## loop over the simulation folders
    for sim_folder in LIST_SIM_FOLDER:
      ## define name of simulation dataset
      sim_name = f"{suite_folder}_{sim_folder}"
      ## define filepath to simulation
      filepath_sim = WWFnF.createFilepath([ 
        BASEPATH, suite_folder, SONIC_REGIME, sim_folder
      ])
      ## CHECK THE NRES=288 DATASET EXISTS
      ## ---------------------------------
      ## check that the simulation data exists at Nres=288
      if not os.path.isfile(f"{filepath_sim}/288/{FileNames.FILENAME_SIM_OUTPUTS}"): continue
      ## MAKE SURE A VISUALISATION FOLDER EXISTS
      ## ---------------------------------------
      ## where plots/dataset of converged data will be stored
      filepath_vis = f"{filepath_sim}/vis_folder/"
      WWFnF.createFolder(filepath_vis, bool_verbose=False)
      ## MEASURE HOW WELL SCALES ARE CONVERGED
      ## -------------------------------------
      obj = PlotScaleConvergence(filepath_sim, filepath_vis, sim_name)
      obj.readData()
      obj.plotData()
      obj.saveData()
      if BOOL_DEBUG: return
      ## create empty space
      print(" ")
    print(" ")


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BOOL_DEBUG        = 0
BASEPATH          = "/scratch/ek9/nk7952/"
SONIC_REGIME      = "sub_sonic"

# LIST_SUITE_FOLDER = [ "Re10", "Re500", "Rm3000" ]
# LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
LIST_SIM_RES      = [ "18", "36", "72", "144", "288", "576" ]

LIST_SUITE_FOLDER = [ "Re10" ]
LIST_SIM_FOLDER   = [ "Pm50" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
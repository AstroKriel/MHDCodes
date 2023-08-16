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
    bounds = ( (1.0, 0.1, 0.0), (1e4, 1e5, 5.0) ) # amplitude, turnover scale, turnover rate
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
  fit_std = np.sqrt(np.diag(fit_cov)) # confidence in fit
  fit_std[0] = np.nanmean([
    np.nanstd(sub_list)
    for sub_list in scales_group_t_res
  ])
  data_x  = np.logspace(np.log10(1), np.log10(10**4), 100)
  data_y  = func(data_x, *fit_params)
  if ax is not None: ax.plot(data_x, data_y, color=color, ls=ls, lw=1.5)
  return data_y[-1], fit_std, fit_params

def fitScales(ax, list_sim_res, scales_group_t_res, color="black"):
  list_sim_res = copy.deepcopy(list_sim_res)
  scales_group_t_res = copy.deepcopy(scales_group_t_res)
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
    val = np.mean(data_y_group_x[-1])
    std = np.std(data_y_group_x[-1])
    fit_params = val, list_sim_res[0], np.nan
    fit_std = std, np.nan, np.nan
    ax.axhline(y=np.mean(data_y_group_x[-1]), color=color, ls=":", lw=1.5)
  else:
    val, fit_std, fit_params = fitLogisticModel(
      ax                 = ax,
      list_res           = data_x,
      scales_group_t_res = data_y_group_x,
      color              = color,
      ls                 = ":"
    )
  return {
    "val" : fit_params[0],
    "std" : fit_std[0],
    "fit_params" : fit_params,
    "fit_std" : fit_std
  }

def getScaleStats(list_scales):
  list_vals = [
    val
    for val in list_scales
    if val is not None
  ]
  return {
    "val" : np.mean(list_vals) if len(list_vals) > 5 else None,
    "std" : np.std(list_vals)  if len(list_vals) > 5 else None
  }


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class PlotConvergence():
  def __init__(self, filepath_sim_288, bool_verbose):
    self.filepath_sim_288 = filepath_sim_288
    self.dict_sim_inputs  = SimParams.readSimInputs(self.filepath_sim_288, bool_verbose=False)
    suite_folder = self.dict_sim_inputs["suite_folder"]
    mach_regime  = self.dict_sim_inputs["mach_regime"]
    sim_folder   = self.dict_sim_inputs["sim_folder"]
    self.filepath_sim_wo_base = f"{suite_folder}/{mach_regime}/{sim_folder}/"
    self.filepath_vis = f"{filepath_sim_288}/../vis_folder/"
    WWFnF.createFolder(self.filepath_vis, bool_verbose=False)
    self.sim_name     = SimParams.getSimName(self.dict_sim_inputs)
    self.bool_verbose = bool_verbose

  def performRoutine(self):
    self.fig, self.axs = plt.subplots(nrows=5, figsize=(6, 5*4))
    self._readData()
    self._plotScales()
    self._fitScales()
    dict_stats_knu_vel  = {}
    dict_stats_knu_kin  = {}
    dict_stats_keta_cur = {}
    dict_stats_keta_mag = {}
    dict_stats_kp       = {}
    ## add resolution dependent scales
    for nres_index in range(len(self.list_nres)):
      dict_scales = self.dict_scales_group_nres[nres_index]
      sim_res = str(int(self.list_nres[nres_index]))
      dict_stats_knu_vel[sim_res]  = getScaleStats(dict_scales["k_nu_vel"])
      dict_stats_knu_kin[sim_res]  = getScaleStats(dict_scales["k_nu_kin"])
      dict_stats_keta_cur[sim_res] = getScaleStats(dict_scales["k_eta_cur"])
      dict_stats_keta_mag[sim_res] = getScaleStats(dict_scales["k_eta_mag"])
      dict_stats_kp[sim_res]       = getScaleStats(dict_scales["k_p"])
    ## add converged scales
    dict_stats_knu_vel["inf"]  = self.knu_vel_inf_stats
    dict_stats_knu_kin["inf"]  = self.knu_kin_inf_stats
    dict_stats_keta_cur["inf"] = self.keta_cur_inf_stats
    dict_stats_keta_mag["inf"] = self.keta_mag_inf_stats
    dict_stats_kp["inf"]       = self.kp_inf_stats
    ## create full dataset
    dict_sim_inputs_288  = SimParams.readSimInputs(self.filepath_sim_288, False)
    dict_sim_outputs_288 = SimParams.readSimOutputs(self.filepath_sim_288, False)
    self.dict_sim_stats = {}
    self.dict_sim_stats[self.sim_name] = {
      "Re"            : dict_sim_inputs_288["Re"],
      "Rm"            : dict_sim_inputs_288["Rm"],
      "Pm"            : dict_sim_inputs_288["Pm"],
      "nu"            : dict_sim_inputs_288["nu"],
      "eta"           : dict_sim_inputs_288["eta"],
      "Mach"          : dict_sim_outputs_288["Mach"],
      "E_growth_rate" : dict_sim_outputs_288["E_growth_rate"],
      "E_ratio_sat"   : dict_sim_outputs_288["E_ratio_sat"],
      "k_nu_vel"      : dict_stats_knu_vel,
      "k_nu_kin"      : dict_stats_knu_kin,
      "k_eta_cur"     : dict_stats_keta_cur,
      "k_eta_mag"     : dict_stats_keta_mag,
      "k_p"           : dict_stats_kp
    }

  def saveData(self):
    filepath_fig = f"{self.filepath_vis}/{self.sim_name}_nres_scales.png"
    PlotFuncs.saveFigure(self.fig, filepath_fig)
    WWObjs.saveDict2JsonFile(
      filepath_file = f"{FILEPATH_OUTPUT}/dataset.json",
      input_dict    = self.dict_sim_stats,
      bool_verbose  = False
    )

  def _readData(self):
    ## initialise lists to store simulation data
    self.list_nres = []
    self.dict_scales_group_nres = []
    for base_path in LIST_BASE_PATHS:
      for sim_res in LIST_SIM_RES:
        filepath_sim_res = f"{base_path}/{self.filepath_sim_wo_base}/{sim_res}"
        if not(os.path.isdir(filepath_sim_res)): continue
        ## extract simulation data
        dict_sim_outputs = SimParams.readSimOutputs(filepath_sim_res, bool_verbose=self.bool_verbose)
        index_growth_start, index_growth_end = dict_sim_outputs["index_bounds_growth"]
        ## extract scales in the kinematic regime
        k_nu_vel_group_t  = dict_sim_outputs["k_nu_vel_tot_group_t"][index_growth_start : index_growth_end]
        k_nu_kin_group_t  = dict_sim_outputs["k_nu_kin_group_t"][index_growth_start : index_growth_end]
        k_eta_cur_group_t = dict_sim_outputs["k_eta_cur_group_t"][index_growth_start : index_growth_end]
        k_eta_mag_group_t = dict_sim_outputs["k_eta_mag_group_t"][index_growth_start : index_growth_end]
        k_p_group_t       = dict_sim_outputs["k_p_mag_group_t"][index_growth_start : index_growth_end]
        ## store datasets
        self.list_nres.append(int(sim_res))
        self.dict_scales_group_nres.append({
          "k_nu_vel"  : k_nu_vel_group_t,
          "k_nu_kin"  : k_nu_kin_group_t,
          "k_eta_cur" : k_eta_cur_group_t,
          "k_eta_mag" : k_eta_mag_group_t,
          "k_p"       : k_p_group_t
        })

  def _plotScales(self):
    for nres_index, nres in enumerate(self.list_nres):
      dict_scales = self.dict_scales_group_nres[nres_index]
      PlotFuncs.plotErrorBar_1D(self.axs[0], nres, dict_scales["k_nu_vel"])
      PlotFuncs.plotErrorBar_1D(self.axs[1], nres, dict_scales["k_nu_kin"])
      PlotFuncs.plotErrorBar_1D(self.axs[2], nres, dict_scales["k_eta_cur"])
      PlotFuncs.plotErrorBar_1D(self.axs[3], nres, dict_scales["k_eta_mag"])
      PlotFuncs.plotErrorBar_1D(self.axs[4], nres, dict_scales["k_p"])

  def _fitScales(self):
    ## plot and fit scales
    self.knu_vel_inf_stats = fitScales(self.axs[0], self.list_nres, [
      dict_scales["k_nu_vel"]
      for dict_scales in self.dict_scales_group_nres
    ])
    self.knu_kin_inf_stats = fitScales(self.axs[1], self.list_nres, [
      dict_scales["k_nu_kin"]
      for dict_scales in self.dict_scales_group_nres
    ])
    self.keta_cur_inf_stats = fitScales(self.axs[2], self.list_nres, [
      dict_scales["k_eta_cur"]
      for dict_scales in self.dict_scales_group_nres
    ])
    self.keta_mag_inf_stats = fitScales(self.axs[3], self.list_nres, [
      dict_scales["k_eta_mag"]
      for dict_scales in self.dict_scales_group_nres
    ])
    self.kp_inf_stats = fitScales(self.axs[4], self.list_nres, [
      dict_scales["k_p"]
      for dict_scales in self.dict_scales_group_nres
    ])
    ## adjust figure axis
    for ax in self.axs:
      ax.set_xscale("log")
      ax.set_yscale("log")


## ###############################################################
## OPPERATOR HANDLING PLOT CALLS
## ###############################################################
def getSimData(
    filepath_sim_res,
    lock         = None,
    bool_verbose = True,
    **kwargs
  ):
  print("Looking at:", filepath_sim_res)
  obj = PlotConvergence(filepath_sim_288=filepath_sim_res, bool_verbose=bool_verbose)
  obj.performRoutine()
  ## SAVE FIGURE + DATASET
  ## ---------------------
  if lock is not None: lock.acquire()
  obj.saveData()
  if lock is not None: lock.release()
  if bool_verbose: print(" ")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  SimParams.callFuncForAllSimulations(
    func               = getSimData,
    bool_mproc         = BOOL_MPROC,
    list_base_paths    = LIST_BASE_PATHS,
    list_suite_folders = LIST_SUITE_FOLDERS,
    list_mach_regimes  = LIST_MACH_REGIMES,
    list_sim_folders   = LIST_SIM_FOLDERS,
    list_sim_res       = [ "288" ]
  )


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
BOOL_MPROC      = 1
FILEPATH_OUTPUT = "/home/586/nk7952/MHDCodes/kriel2023"
LIST_BASE_PATHS = [
  "/scratch/ek9/nk7952/",
  "/scratch/jh2/nk7952/"
]
LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Re2000", "Rm500", "Rm3000" ]
LIST_MACH_REGIMES  = [ "Mach0.3", "Mach1", "Mach5", "Mach10" ]
LIST_SIM_FOLDERS   = [
  "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm30", "Pm50", "Pm125", "Pm250", "Pm300"
]
LIST_SIM_RES = [ "18", "36", "72", "144", "288", "576", "1152" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

## load user defined modules
from TheUsefulModule import WWFnF, WWObjs
from TheFittingModule import UserModels
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def plotErrorBar_1D(ax, x, array_y, color="k", marker="o"):
  y_median = np.percentile(array_y, 50)
  y_p16    = np.percentile(array_y, 16)
  y_p84    = np.percentile(array_y, 84)
  y_1std   = np.vstack([
    y_median - y_p16,
    y_p84 - y_median
  ])
  ax.errorbar(
    x, y_median,
    yerr  = y_1std,
    color = color,
    fmt   = marker,
    markersize=7, elinewidth=2, linestyle="None", markeredgecolor="black", capsize=7.5, zorder=10
  )


def fitScales(
    ax, list_res, list_scales_group_res,
    bounds = ( (0.01, 1, 0), (50, 1000, 3) )
  ):
  ## check if measured scales increase or decrease with resolution
  if np.mean(list_scales_group_res[0 ]) < np.mean(list_scales_group_res[-1 ]):
    func = UserModels.ListOfModels.logistic_growth_increasing
  else: func = UserModels.ListOfModels.logistic_growth_decreasing
  fit_params, fit_cov = curve_fit(
    f     = func,
    xdata = list_res,
    ydata = [ np.median(list_scales) for list_scales in list_scales_group_res ],
    sigma = [ np.std(list_scales)    for list_scales in list_scales_group_res ],
    bounds=bounds, absolute_sigma=True, maxfev=10**5
  )
  fit_std = np.sqrt(np.diag(fit_cov))[0] # confidence in fit
  domain_array = np.logspace(np.log10(1), np.log10(5000), 100)
  list_scales_model = func(domain_array, *fit_params)
  return list_scales_model[-1], fit_std


## ###############################################################
## MEASURE, PLOT + SAVE CONVEREGED SCALES
## ###############################################################
class PlotSpectraConvergence():
  def __init__(
      self,
      filepath_sim, filepath_vis, sim_name
    ):
    self.filepath_sim             = filepath_sim
    self.filepath_vis             = filepath_vis
    self.sim_name                 = sim_name
    self.list_sim_res             = []
    self.list_alpha_kin_group_res = []
    self.list_k_nu_group_res      = []
    self.list_k_p_group_res       = []

  def readDataset(self):
    ## read in scales for each resolution run
    for sim_res in LIST_SIM_RES:
      ## load json-file into a dictionary
      try:
        dict_sim_data = WWObjs.loadJson2Dict(
          filepath = f"{self.filepath_sim}/{sim_res}",
          filename = f"{self.sim_name}_dataset.json",
          bool_hide_updates = True
        )
      except: continue
      ## pull out data
      self.list_sim_res.append(sim_res)
      self.list_alpha_kin_group_res.append(dict_sim_data["list_alpha_kin"])
      self.list_k_nu_group_res.append(dict_sim_data["list_k_nu"])
      self.list_k_p_group_res.append(dict_sim_data["list_k_p"])

  def createFigure(self):
    fig, fig_grid = PlotFuncs.createFigGrid(
      fig_scale        = 1.0,
      fig_aspect_ratio = (5.0, 8.0),
      num_rows         = 2,
      num_cols         = 2
    )
    self.ax_k_nu = fig.add_subplot(fig_grid[0, 0])
    self.ax_k_p  = fig.add_subplot(fig_grid[1, 0])
    ## plot and fit data
    self.__plotDataset()
    # self.__fitDataset()
    self.__annotateFigure()
    ## save figure
    filepath_fig = f"{self.filepath_vis}/{self.sim_name}_nres_study.png"
    plt.savefig(filepath_fig)
    print("Saved figure:", filepath_fig)
    ## close plot
    plt.close(fig)

  # def createDataset(self):
  #   spectra_converged_obj = FitMHDScales.SpectraConvergedScales(
  #     k_nu_converged  = self.k_nu_converged,
  #     k_eta_converged = self.k_eta_converged,
  #     k_p_converged   = self.k_p_converged,
  #     k_nu_std        = self.k_nu_std,
  #     k_eta_std       = self.k_eta_std,
  #     k_p_std         = self.k_p_std
  #   )
  #   WWObjs.saveObj2Json(
  #     obj      = spectra_converged_obj,
  #     filepath = WWFnF.createFilepath([ BASEPATH, self.sim_suite, SONIC_REGIME ]),
  #     filename = f"{self.sim_folder}_{FILENAME_CONVERGED}"
  #   )

  def __plotDataset(self):
    for res_index, sim_res in enumerate(self.list_sim_res):
      plotErrorBar_1D(
        ax      = self.ax_k_nu,
        x       = int(sim_res),
        array_y = self.list_k_nu_group_res[res_index]
      )
      plotErrorBar_1D(
        ax      = self.ax_k_p,
        x       = int(sim_res),
        array_y = self.list_k_p_group_res[res_index]
      )

  # def __fitDataset(self):
  #   self.k_nu_converged, self.k_nu_std = fitScales(self.ax_k_nu, self.list_sim_res, self.list_k_nu_group_res)
  #   self.k_p_converged, self.k_p_std   = fitScales(self.ax_k_p,  self.list_sim_res, self.list_k_p_group_res)

  def __annotateFigure(self):
    ## label k_nu
    self.ax_k_nu.set_ylabel(r"$k_\nu$")
    self.ax_k_nu.set_yscale("log")
    ## label k_p
    self.ax_k_p.set_ylabel(r"$k_{\rm p}$")
    self.ax_k_p.set_xlabel(r"$N_{\rm res}$")
    self.ax_k_p.set_yscale("log")


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
    str_message = f"Looking at suite: {suite_folder}"
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
      if not os.path.isfile(WWFnF.createFilepath([ 
          filepath_sim, "288", f"{sim_name}_dataset.json"
        ])): continue

      ## MAKE SURE A VISUALISATION FOLDER EXISTS
      ## ---------------------------------------
      ## where plots/dataset of converged data will be stored
      filepath_vis = WWFnF.createFilepath([ 
        filepath_sim, "vis_folder"
      ])
      WWFnF.createFolder(filepath_vis, bool_hide_updates=True)

      ## MEASURE HOW WELL SCALES ARE CONVERGED
      ## -------------------------------------
      obj = PlotSpectraConvergence(filepath_sim, filepath_vis, sim_name)
      obj.readDataset()
      obj.createFigure()
      # obj.createDataset()

      if BOOL_DEBUG: return
      ## create empty space
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


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
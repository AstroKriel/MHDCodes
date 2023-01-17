#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

## load user defined modules
from TheSimModule import SimParams
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
def addLabel_simInputs(
    filepath_sim_res,
    fig, ax,
    bbox          = (0.0, 0.0),
    vpos          = (0.05, 0.05),
    bool_show_res = True
  ):
  ## load simulation parameters
  dict_sim_inputs = SimParams.readSimInputs(filepath_sim_res)
  if bool_show_res:
    label_res = r"${\rm N}_{\rm res} = $ " + "{:d}".format(int(dict_sim_inputs["sim_res"]))
  else: label_res = None
  ## annotate simulation parameters
  PlotFuncs.addBoxOfLabels(
    fig, ax,
    box_alignment = bbox,
    xpos          = vpos[0],
    ypos          = vpos[1],
    alpha         = 0.5,
    fontsize      = 18,
    list_labels   = [
      label_res,
      r"${\rm Re} = $ " + "{:d}".format(int(dict_sim_inputs["Re"])),
      r"${\rm Rm} = $ " + "{:d}".format(int(dict_sim_inputs["Rm"])),
      r"${\rm Pm} = $ " + "{:d}".format(int(dict_sim_inputs["Pm"])),
    ]
  )

def createLabel_fromModes(input_val, input_std):
  return r"${} \pm {}$\;".format(
    str(round(input_val, 2)),
    str(round(input_std, 2))
  )

def fitScales(
    ax, list_res, list_scales_group_res,
    color  = "black",
    ls     = ":",
    bounds = ( (0.01, 1, 0), (500, 5000, 5) )
  ):
  ## check if measured scales increase or decrease with resolution
  if np.mean(list_scales_group_res[0]) < np.mean(list_scales_group_res[-1]):
    func = UserModels.ListOfModels.logistic_growth_increasing
  else: func = UserModels.ListOfModels.logistic_growth_decreasing
  fit_params, fit_cov = curve_fit(
    f     = func,
    xdata = list_res,
    ydata = [ np.percentile(list_scales, 50) for list_scales in list_scales_group_res ],
    sigma = [ np.std(list_scales)            for list_scales in list_scales_group_res ],
    bounds=bounds, absolute_sigma=True, maxfev=10**5
  )
  fit_std = np.sqrt(np.diag(fit_cov))[0] # confidence in fit
  data_x  = np.logspace(np.log10(1), np.log10(10**4), 100)
  data_y  = func(data_x, *fit_params)
  ax.plot(data_x, data_y, color=color, ls=ls, lw=1.5)
  return data_y[-1], fit_std

def getScaleFromParams_knu(fit_params_group_t):
  return [
    fit_params[2]
    for fit_params in fit_params_group_t
  ]


## ###############################################################
## OPERATOR CLASS: PLOT RESOLUTION STUDY
## ###############################################################
class PlotScaleConvergence():
  def __init__(
      self,
      filepath_sim, filepath_vis, sim_name
    ):
    self.filepath_sim = filepath_sim
    self.filepath_vis = filepath_vis
    self.sim_name     = sim_name
    self.__initialiseDatasets()

  def __initialiseDatasets(self):
    ## initialise input datasets
    self.list_sim_res            = []
    self.k_nu_lgt_group_t_res    = []
    self.k_nu_trv_group_t_res_p5 = []
    self.k_nu_trv_group_t_res_1  = []
    self.k_nu_trv_group_t_res_2  = []
    self.k_p_tot_group_t_res     = []
    ## initialise output datasets
    self.k_nu_tot_converged    = None
    self.k_nu_lgt_converged    = None
    self.k_nu_trv_converged_p5 = None
    self.k_nu_trv_converged_1  = None
    self.k_nu_trv_converged_2  = None
    self.k_p_tot_converged     = None
    self.k_nu_tot_std          = None
    self.k_nu_lgt_std          = None
    self.k_nu_trv_std_p5       = None
    self.k_nu_trv_std_1        = None
    self.k_nu_trv_std_2        = None
    self.k_p_tot_std           = None

  def readDataset(self):
    ## read in scales for each resolution run
    for sim_res in LIST_SIM_RES:
      ## load json-file into a dictionary
      try:
        dict_sim_outputs = SimParams.readSimOutputs(f"{self.filepath_sim}/{sim_res}/")
      except: continue
      ## extract data
      self.list_sim_res.append(sim_res)
      self.k_nu_lgt_group_t_res.append(dict_sim_outputs["k_nu_lgt_group_t"])
      self.k_nu_trv_group_t_res_p5.append(dict_sim_outputs["k_nu_trv_group_t_p5"])
      self.k_nu_trv_group_t_res_1.append(dict_sim_outputs["k_nu_trv_group_t_1"])
      self.k_nu_trv_group_t_res_2.append(dict_sim_outputs["k_nu_trv_group_t_2"])
      self.k_p_tot_group_t_res.append( dict_sim_outputs["k_p_tot_group_t"])

  def createFigure_scales(self):
    self.fig, fig_grid = PlotFuncs.createFigure_grid(
      fig_scale        = 1.0,
      fig_aspect_ratio = (5.0, 8.0),
      num_rows         = 2,
      num_cols         = 2
    )
    self.ax_k_nu = self.fig.add_subplot(fig_grid[0, 0])
    self.ax_k_p  = self.fig.add_subplot(fig_grid[1, 0])
    ## plot and fit data
    self.__plotScales()
    self.__fitDataset()
    self.__annotateFigure()
    ## save figure
    filepath_fig = f"{self.filepath_vis}/{self.sim_name}_nres_scales.png"
    PlotFuncs.saveFigure(self.fig, filepath_fig)

  def createDataset(self):
    dict_converged_scales = {
      "k_p_tot_converged"     : self.k_p_tot_converged,
      "k_nu_tot_converged"    : self.k_nu_tot_converged,
      "k_nu_lgt_converged"    : self.k_nu_lgt_converged,
      "k_nu_trv_converged_p5" : self.k_nu_trv_converged_p5,
      "k_nu_trv_converged_1"  : self.k_nu_trv_converged_1,
      "k_nu_trv_converged_2"  : self.k_nu_trv_converged_2,
      "k_p_tot_std"           : self.k_p_tot_std,
      "k_nu_tot_std"          : self.k_nu_tot_std,
      "k_nu_lgt_std"          : self.k_nu_lgt_std,
      "k_nu_trv_std_p5"       : self.k_nu_trv_std_p5,
      "k_nu_trv_std_1"        : self.k_nu_trv_std_1,
      "k_nu_trv_std_2"        : self.k_nu_trv_std_2,
    }
    WWObjs.saveDict2JsonFile(
      filepath_file = f"{self.filepath_sim}/scales.json",
      input_dict    = dict_converged_scales
    )

  def __plotScales(self):
    for res_index, sim_res in enumerate(self.list_sim_res):
      PlotFuncs.plotErrorBar_1D(
        ax      = self.ax_k_nu,
        x       = int(sim_res),
        array_y = self.k_nu_lgt_group_t_res[res_index],
        color   = "blue"
      )
      PlotFuncs.plotErrorBar_1D(
        ax      = self.ax_k_nu,
        x       = int(sim_res),
        array_y = self.k_nu_trv_group_t_res_p5[res_index],
        color   = "darkviolet",
        # marker  = "s"
      )
      PlotFuncs.plotErrorBar_1D(
        ax      = self.ax_k_nu,
        x       = int(sim_res),
        array_y = self.k_nu_trv_group_t_res_1[res_index],
        color   = "darkviolet",
        # marker  = "o"
      )
      PlotFuncs.plotErrorBar_1D(
        ax      = self.ax_k_nu,
        x       = int(sim_res),
        array_y = self.k_nu_trv_group_t_res_2[res_index],
        color   = "darkviolet",
        # marker  = "D"
      )
      PlotFuncs.plotErrorBar_1D(
        ax      = self.ax_k_p,
        x       = int(sim_res),
        array_y = self.k_p_tot_group_t_res[res_index],
        color   = "black"
      )

  def __fitDataset(self):
    self.k_nu_lgt_converged, self.k_nu_lgt_std = fitScales(
      ax                    = self.ax_k_nu,
      list_res              = self.list_sim_res,
      list_scales_group_res = self.k_nu_lgt_group_t_res,
      color                 = "blue"
    )
    self.k_nu_trv_converged_p5, self.k_nu_trv_std_p5 = fitScales(
      ax                    = self.ax_k_nu,
      list_res              = self.list_sim_res,
      list_scales_group_res = self.k_nu_trv_group_t_res_p5,
      color                 = "darkviolet",
      ls                    = ":"
    )
    self.k_nu_trv_converged_1, self.k_nu_trv_std_1 = fitScales(
      ax                    = self.ax_k_nu,
      list_res              = self.list_sim_res,
      list_scales_group_res = self.k_nu_trv_group_t_res_1,
      color                 = "darkviolet",
      ls                    = "-"
    )
    self.k_nu_trv_converged_2, self.k_nu_trv_std_2 = fitScales(
      ax                    = self.ax_k_nu,
      list_res              = self.list_sim_res,
      list_scales_group_res = self.k_nu_trv_group_t_res_2,
      color                 = "darkviolet",
      ls                    = "-."
    )
    self.k_p_tot_converged,  self.k_p_tot_std = fitScales(
      ax                    = self.ax_k_p,
      list_res              = self.list_sim_res,
      list_scales_group_res = self.k_p_tot_group_t_res,
      color                 = "black"
    )

  def __annotateFigure(self):
    bounds_nres = [ 10, 10**4 ]
    ## label k_nu
    self.ax_k_nu.set_xscale("log")
    self.ax_k_nu.set_yscale("log")
    self.ax_k_nu.set_xlim(bounds_nres)
    self.ax_k_nu.set_ylim(bottom=0.5)
    self.ax_k_nu.set_ylabel(r"$k_\nu$")
    addLabel_simInputs(
      filepath_sim_res = f"{self.filepath_sim}/288/",
      fig           = self.fig,
      ax            = self.ax_k_nu,
      bbox          = (1.0, 0.0),
      vpos          = (0.95, 0.05),
      bool_show_res = False
    )
    PlotFuncs.addBoxOfLabels(
      fig           = self.fig,
      ax            = self.ax_k_nu,
      box_alignment = (0.0, 1.0),
      xpos          = 0.05,
      ypos          = 0.95,
      alpha         = 0.85,
      fontsize      = 20,
      list_labels   = [
        r"$k_{\nu, \parallel} =$ "  + createLabel_fromModes(self.k_nu_lgt_converged, self.k_nu_lgt_std),
        r"$k_{\nu, \perp, 0.5} =$ " + createLabel_fromModes(self.k_nu_trv_converged_p5, self.k_nu_trv_std_p5),
        r"$k_{\nu, \perp, 1} =$ "   + createLabel_fromModes(self.k_nu_trv_converged_1, self.k_nu_trv_std_1),
        r"$k_{\nu, \perp, 2} =$ "   + createLabel_fromModes(self.k_nu_trv_converged_2, self.k_nu_trv_std_2),
      ],
      list_colors   = [
        "blue",
        "darkviolet",
        "darkviolet",
        "darkviolet"
      ]
    )
    ## label k_p
    PlotFuncs.addBoxOfLabels(
      fig           = self.fig,
      ax            = self.ax_k_p,
      box_alignment = (0.0, 1.0),
      xpos          = 0.05,
      ypos          = 0.95,
      alpha         = 0.85,
      fontsize      = 20,
      list_labels   = [
        r"$k_{\rm p, tot} =$ " + createLabel_fromModes(self.k_p_tot_converged, self.k_p_tot_std)
      ],
      list_colors   = [
        "black"
      ]
    )
    self.ax_k_p.set_xscale("log")
    self.ax_k_p.set_yscale("log")
    self.ax_k_p.set_xlim(bounds_nres)
    self.ax_k_p.set_ylim(bottom=1.0)
    self.ax_k_p.set_ylabel(r"$k_{\rm p}$")
    self.ax_k_p.set_xlabel(r"$N_{\rm res}$")


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
      if not os.path.isfile(f"{filepath_sim}/288/sim_outputs.json"): continue

      ## MAKE SURE A VISUALISATION FOLDER EXISTS
      ## ---------------------------------------
      ## where plots/dataset of converged data will be stored
      filepath_vis = f"{filepath_sim}/vis_folder/"
      WWFnF.createFolder(filepath_vis, bool_hide_updates=True)

      ## MEASURE HOW WELL SCALES ARE CONVERGED
      ## -------------------------------------
      obj = PlotScaleConvergence(filepath_sim, filepath_vis, sim_name)
      obj.readDataset()
      obj.createFigure_scales()
      obj.createDataset()

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

# LIST_SUITE_FOLDER = [ "Rm3000" ]
# LIST_SIM_RES      = [ "144", "288", "576" ]


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
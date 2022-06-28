#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit

from TheUsefulModule import WWFnF, WWObjs, WWLists
from TheFittingModule import FitMHDScales, UserModels
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
## use a non-interactive plotting backend
plt.ioff()
plt.switch_backend("agg")


## ###############################################################
## MEASURE, PLOT + SAVE CONVEREGED SCALES
## ###############################################################
class PlotSpectraConvergence():
  def __init__(
      self,
      sim_suite,
      sim_folder
    ):
    self.sim_suite  = sim_suite
    self.sim_folder = sim_folder
    ## initialise figure
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    ## create figure sub-axis
    gs = GridSpec(3, 2, figure=fig)
    ## 'Turb.dat' data
    self.ax_k_nu      = fig.add_subplot(gs[0, 0])
    self.ax_k_eta     = fig.add_subplot(gs[1, 0])
    self.ax_k_p       = fig.add_subplot(gs[2, 0])
    self.ax_alpha_kin = fig.add_subplot(gs[0, 1])
    self.ax_alpha_mag = fig.add_subplot(gs[1, 1])
    ## plot, fit and save data
    self.readScales()
    self.plotScales()
    self.fitScales()
    self.saveFig()
    self.saveConvergedScales()
    ## close plot
    plt.close(fig)

  def readScales(self):
    self.list_k_nu_group_res      = []
    self.list_k_eta_group_res     = []
    self.list_k_p_group_res       = []
    self.list_alpha_kin_group_res = []
    self.list_alpha_mag_group_res = []
    ## read in scales for each resolution run
    for sim_res in LIST_SIM_RES:
      try:
        ## load spectra-fit data as a dictionary
        fits_dict = WWObjs.loadJson2Dict(
          filepath = WWFnF.createFilepath([ BASEPATH, self.sim_suite, sim_res, SONIC_REGIME, self.sim_folder, "spect" ]),
          filename = FILENAME_SPECTRA,
          bool_hide_updates = True
        )
        ## store dictionary data in spectra-fit object
        fits_obj = FitMHDScales.SpectraFit(**fits_dict)
      except: continue
      ## check that a fit-range has been defined
      bool_kin_fit = (fits_obj.kin_fit_time_start is not None) and (fits_obj.kin_fit_time_end is not None)
      bool_mag_fit = (fits_obj.mag_fit_time_start is not None) and (fits_obj.mag_fit_time_end is not None)
      if not(bool_kin_fit) or not(bool_mag_fit):
        raise Exception("Fit range has not been defined for: {}{}{} energy spectra.".format(
          "kinetic"  if not(bool_kin_fit) else "",
          " and " if (not(bool_kin_fit) and not(bool_mag_fit)) else "",
          "magnetic" if not(bool_mag_fit) else ""
        ))
      ## find indices corresponding with the fit-range bounds
      kin_index_start = WWLists.getIndexClosestValue(fits_obj.kin_list_sim_times, fits_obj.kin_fit_time_start)
      mag_index_start = WWLists.getIndexClosestValue(fits_obj.mag_list_sim_times, fits_obj.mag_fit_time_start)
      kin_index_end   = WWLists.getIndexClosestValue(fits_obj.kin_list_sim_times, fits_obj.kin_fit_time_end)
      mag_index_end   = WWLists.getIndexClosestValue(fits_obj.mag_list_sim_times, fits_obj.mag_fit_time_end)
      ## save measured scales in fit time-range
      self.list_k_nu_group_res.append(
        fits_obj.k_nu_group_t[kin_index_start : kin_index_end]
      )
      self.list_k_eta_group_res.append(
        fits_obj.k_eta_group_t[mag_index_start : mag_index_end]
      )
      self.list_k_p_group_res.append(
        fits_obj.k_p_group_t[mag_index_start : mag_index_end]
      )
      self.list_alpha_kin_group_res.append([ 
        param[1]
        for index, param in enumerate(fits_obj.kin_list_fit_params_group_t)
        if (kin_index_start <= index) and (index <= kin_index_end)
      ])
      self.list_alpha_mag_group_res.append([ 
        param[1]
        for index, param in enumerate(fits_obj.mag_list_fit_params_group_t)
        if (mag_index_start <= index) and (index <= mag_index_end)
      ])

  def plotScales(self):
    for res_index, sim_res in enumerate(LIST_SIM_RES):
      ## plot measured scales 
      PlotFuncs.plotErrorBar(
        ax     = self.ax_k_nu,
        data_x = int(sim_res),
        data_y = self.list_k_nu_group_res[res_index],
        color  = "black"
      )
      PlotFuncs.plotErrorBar(
        ax     = self.ax_k_eta,
        data_x = int(sim_res),
        data_y = self.list_k_eta_group_res[res_index],
        color  = "black"
      )
      PlotFuncs.plotErrorBar(
        ax     = self.ax_k_p,
        data_x = int(sim_res),
        data_y = self.list_k_p_group_res[res_index],
        color  = "black"
      )
      ## plot measured alpha exponents
      PlotFuncs.plotErrorBar(
        ax     = self.ax_alpha_kin,
        data_x = int(sim_res),
        data_y = self.list_alpha_kin_group_res[res_index],
        color  = "black"
      )
      PlotFuncs.plotErrorBar(
        ax     = self.ax_alpha_mag,
        data_x = int(sim_res),
        data_y = self.list_alpha_mag_group_res[res_index],
        color  = "black"
      )

  def fitScales(self):
    ## remove Nres=72 simulation data for k_nu
    list_res_excl_Nres72 = [ int(res) for res in LIST_SIM_RES if not(int(res) == 72) ]
    list_knu_group_excl_Nres72 = [
      list_scales
      for res, list_scales in zip(LIST_SIM_RES, self.list_k_nu_group_res)
      if not(int(res) == 72)
    ]
    ## fit scales
    self.k_nu_converged, self.k_nu_std   = self.__fitScales(self.ax_k_nu,  list_res_excl_Nres72, list_knu_group_excl_Nres72)
    self.k_eta_converged, self.k_eta_std = self.__fitScales(self.ax_k_eta, LIST_SIM_RES, self.list_k_eta_group_res)
    self.k_p_converged, self.k_p_std     = self.__fitScales(self.ax_k_p,   LIST_SIM_RES, self.list_k_p_group_res)

  def __fitScales(
      self,
      ax, list_res, list_scales_group_res,
      bounds = ( (0.01, 1, 0), (50, 1000, 3) )
    ):
    ## check if scales increase or decrease with resolution
    if np.mean(list_scales_group_res[0 ]) < np.mean(list_scales_group_res[-1 ]):
      func = UserModels.ListOfModels.logistic_growth_increasing
    else:
      func = UserModels.ListOfModels.logistic_growth_decreasing
    ## fit scales
    fit_params, fit_cov = curve_fit(
      f     = func,
      xdata = list_res,
      ydata = [ np.median(list_scales) for list_scales in list_scales_group_res ],
      sigma = [ np.std(list_scales)    for list_scales in list_scales_group_res ],
      bounds=bounds, absolute_sigma=True, maxfev=10**5
    )
    fit_std = np.sqrt(np.diag(fit_cov))[0] # confidence in fit
    ## plot fitted model
    domain_array = np.logspace(np.log10(MIN_AX_NRES), np.log10(MAX_AX_NRES), 100)
    list_scales_model = func(domain_array, *fit_params)
    # ax.plot(
    #   domain_array,
    #   list_scales_model,
    #   color="k", linestyle="-.", linewidth=2
    # )
    ## plot fit uncertainty
    # ax.fill_between(
    #   domain_array,
    #   func(domain_array, *fit_params) - fit_std,
    #   func(domain_array, *fit_params) + fit_std,
    #   color="black", alpha=0.2
    # )
    return list_scales_model[-1], fit_std

  def saveFig(self):
    ## label k_nu
    self.ax_k_nu.set_ylabel(r"$k_\nu$")
    self.ax_k_nu.set_xscale("log")
    self.ax_k_nu.set_yscale("log")
    self.ax_k_nu.set_xlim([ MIN_AX_NRES, MAX_AX_NRES ])
    ## label k_eta
    self.ax_k_eta.set_ylabel(r"$k_\eta$")
    self.ax_k_eta.set_xscale("log")
    self.ax_k_eta.set_yscale("log")
    self.ax_k_eta.set_xlim([ MIN_AX_NRES, MAX_AX_NRES ])
    ## label k_p
    self.ax_k_p.set_ylabel(r"$k_{\rm p}$")
    self.ax_k_p.set_xlabel(r"Linear Resolution $N_{\rm res}$")
    self.ax_k_p.set_xscale("log")
    self.ax_k_p.set_yscale("log")
    self.ax_k_p.set_xlim([ MIN_AX_NRES, MAX_AX_NRES ])
    ## label alpha_kin
    self.ax_alpha_kin.set_ylabel(r"$\alpha_{\rm kin}$")
    self.ax_alpha_kin.set_xscale("log")
    self.ax_alpha_kin.set_xlim([ MIN_AX_NRES, MAX_AX_NRES ])
    ## label alpha_mag
    self.ax_alpha_mag.set_ylabel(r"$\alpha_{\rm mag}$")
    self.ax_alpha_mag.set_xlabel(r"Linear Resolution $N_{\rm res}$")
    self.ax_alpha_mag.set_xscale("log")
    self.ax_alpha_mag.set_xlim([ MIN_AX_NRES, MAX_AX_NRES ])
    ## save figure
    fig_name = f"{self.sim_folder}_scale_convergence{FILENAME_TAG}.pdf"
    plt.savefig(
      WWFnF.createFilepath([ BASEPATH, self.sim_suite, SONIC_REGIME, fig_name ])
    )
    print("Figure saved:", fig_name)

  def saveConvergedScales(self):
    spectra_converged_obj = FitMHDScales.SpectraConvergedScales(
      k_nu_converged  = self.k_nu_converged,
      k_eta_converged = self.k_eta_converged,
      k_p_converged   = self.k_p_converged,
      k_nu_std        = self.k_nu_std,
      k_eta_std       = self.k_eta_std,
      k_p_std         = self.k_p_std
    )
    WWObjs.saveObj2Json(
      obj      = spectra_converged_obj,
      filepath = WWFnF.createFilepath([ BASEPATH, self.sim_suite, SONIC_REGIME ]),
      filename = f"{self.sim_folder}_{FILENAME_CONVERGED}"
    )


## ###############################################################
## MAIN PROGRAM
## ###############################################################
BASEPATH           = "/scratch/ek9/nk7952/"
SONIC_REGIME       = "super_sonic"
FILENAME_SPECTRA   = "spectra_fits.json"
FILENAME_CONVERGED = "spectra_converged.json"
FILENAME_TAG       = ""
LIST_SIM_RES       = [ "18", "36", "72", "144", "288" ]
MIN_AX_NRES        = 10
MAX_AX_NRES        = 3000

def main():
  ## #############################################
  ## LOOK AT EACH SIMULATION GROUPED BY RESOLUTION
  ## #############################################
  ## loop over the simulation suites
  for suite_folder in [
      "Re10", "Re500", "Rm3000"
    ]: # "Re10", "Re500", "Rm3000", "keta"

    ## ######################################
    ## CHECK THE SUITE'S FIGURE FOLDER EXISTS
    ## ######################################
    filepath_suite_output = WWFnF.createFilepath([ 
      BASEPATH, suite_folder, SONIC_REGIME
    ])
    if not os.path.exists(filepath_suite_output):
      print("{} does not exist.".format(filepath_suite_output))
      continue
    str_message = "Looking at suite: {}".format(suite_folder)
    print(str_message)
    print("=" * len(str_message))
    print("Saving figures in:", filepath_suite_output)

    ## #####################
    ## LOOP OVER SIMULATIONS
    ## #####################
    ## loop over the simulation folders
    for sim_folder in [
        "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"
      ]: # "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"

      ## check that the simulation folder exists at Nres=288
      if not os.path.isfile(WWFnF.createFilepath([ 
          BASEPATH, suite_folder, "288", SONIC_REGIME, sim_folder, "spect", FILENAME_SPECTRA
        ])): continue

      ## measure converged scales
      PlotSpectraConvergence(sim_suite=suite_folder, sim_folder=sim_folder)

      ## create empty space
      print(" ")
    print(" ")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
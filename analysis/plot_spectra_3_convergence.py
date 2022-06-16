#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

from math import floor, ceil

from TheUsefulModule import WWFnF, WWObjs, WWLists
from TheFittingModule import FitMHDScales
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
#################################################################
os.system("clear") # clear terminal window
plt.ioff()
plt.switch_backend("agg") # use a non-interactive plotting backend


# ## ###############################################################
# ## FUNCTIONS
# ## ###############################################################
# def funcPlotConvergedScale(ax, label_scale, list_scales, converged_scale):
#   ## get panel dimensions
#   y_min, y_max = ax.get_ylim()
#   ## find what coordinate of label as a percentage of panel dimensions
#   scale_height_ax_percent = (
#     np.log10(converged_scale) - np.log10(y_min)
#   ) / (
#     np.log10(y_max) - np.log10(y_min)
#   )
#   ## create label
#   scale_label = label_scale+r"$(\infty) = {}_{}^{}$".format(
#     "{:0.2f}".format(
#       np.nanpercentile(list_scales, 50)
#     ),
#     "{" + "{:0.2f}".format(
#       np.nanpercentile(list_scales, 16)
#     ) + "}",
#     "{" + "{:0.2f}".format(
#       np.nanpercentile(list_scales, 84)
#     ) + "}"
#   )
#   ## annotate the converged scale
#   ax.text(
#     0.025, scale_height_ax_percent - 0.025,
#     scale_label,
#     ha="left", va="top", transform=ax.transAxes, fontsize=16, zorder=7
#   )

# def funcPrintForm(
#     val_median, val_error,
#     num_digits = 2
#   ):
#   str_median = ("{0:.2g}").format(val_median)
#   num_decimals = 1
#   ## if integer
#   if ("." not in str_median) and (len(str_median.replace("-", "")) < 2):
#     str_median = ("{0:.1f}").format(val_median)
#   ## if a float
#   if "." in str_median:
#     ## if integer component is 0
#     if ("0" in str_median.split(".")[0]) and (len(str_median.split(".")[1]) < 2):
#       str_median = ("{0:.2f}").format(val_median)
#     num_decimals = len(str_median.split(".")[1])
#   ## if integer > 9
#   elif len(str_median.split(".")[0].replace("-", "")) > 1:
#     num_decimals = 0
#   str_error = ("{0:."+str(num_decimals)+"f}").format(val_error)
#   return r"${} \pm {}$".format(
#     str_median,
#     str_error
#   )

# def funcFitScales(
#     ax, list_x, list_y_group, func,
#     p0          = None,
#     bounds      = None,
#     label_scale = None,
#     bool_debug  = False,
#     absolute_sigma = True
#   ):
#   ## #######################
#   ## FIT: GET COVERGED SCALE
#   ## #######################
#   ## fit data
#   fit_params, fit_cov = curve_fit(
#     func,
#     list_x,
#     [
#       np.median(list_y)
#       for list_y in list_y_group
#     ],
#     bounds = bounds,
#     sigma  = [
#       np.std(list_y)
#       for list_y in list_y_group
#     ],
#     absolute_sigma = absolute_sigma
#   )
#   ## get errors
#   fit_std = np.sqrt(np.diag(fit_cov))[0] # confidence in fit to medians
#   data_std = np.std(list_y_group[-1]) # fit inherets error in last data point
#   if fit_params[0] < 0.8:
#     rand_std = abs(fit_params[0] - random.uniform(
#       10**( np.log10(fit_params[0]) + 0.01 ),
#       10**( np.log10(fit_params[0]) + 0.045 )
#     ))
#   else:
#     rand_std = abs(fit_params[0] - random.uniform(
#       10**( np.log10(fit_params[0]) + 0.015 ),
#       10**( np.log10(fit_params[0]) + 0.085 )
#     ))
#   # print(
#   #     "& {} & {}".format(
#   #         funcPrintForm(fit_params[1], np.sqrt(np.diag(fit_cov))[1]),
#   #         funcPrintForm(fit_params[2], np.sqrt(np.diag(fit_cov))[2])
#   #     )
#   # )
#   ## ####################
#   ## PLOT CONVERGENCE FIT
#   ## ####################
#   ## create plot domain
#   domain_array = np.logspace(1, np.log10(1900), 300)
#   ## plot converging fit
#   ax.plot(
#     domain_array,
#     func(domain_array, *fit_params),
#     color="k", linestyle="-.", linewidth=2
#   )
#   # ## plot fit error
#   # ax.fill_between(
#   #     domain_array,
#   #     func(domain_array, *fit_params) - fit_std,
#   #     func(domain_array, *fit_params) + fit_std,
#   #     color="black", alpha=0.2
#   # )
#   ## plot converged scale
#   ax.axhline(y=fit_params[0], color="black", dashes=(7.5, 3.5), linewidth=2)
#   ## #####################################
#   ## PLOT DISTRIBUTION OF CONVERGED SCALES
#   ## #####################################
#   ## create distribution of scales
#   list_converged_scales = np.random.normal(
#     fit_params[0], # measured convergence scale
#     # min([fit_std, data_std], key=lambda x:abs(x-1)),
#     # fit_std + data_std,
#     data_std + rand_std,
#     # fit_std,
#     # fit_std if absolute_sigma else data_std if data_std > 0.05 else rand_std
#     # max([fit_std, rand_std], key=lambda x:abs(x-1)),
#     10**3 # number of samples
#   )
#   print(
#     "{:.3f}   {:.3f}   {:.3f}   {:.3f}".format(
#       fit_params[0],
#       fit_std,
#       data_std,
#       rand_std
#     )
#   )
#   print(
#     "{:0.3f}, {:0.3f}, {:0.3f}".format(
#       np.percentile(list_converged_scales, 16),
#       np.percentile(list_converged_scales, 50),
#       np.percentile(list_converged_scales, 84)
#     )
#   )
#   if np.percentile(list_converged_scales, 16) < 0:
#     print("FUCK..!")
#   print(" ")
#   ## ##############
#   ## LABEL THE PLOT
#   ## ##############
#   ## fix axis limits
#   ax.set_xlim([domain_array[0], domain_array[-1]])
#   ## fix axis scale
#   ax.set_xscale("log")
#   ax.set_yscale("log")
#   ## adjust axis tick labels
#   ax.xaxis.set_major_formatter(ScalarFormatter())
#   bool_small_domain_cross = ceil(np.log10(ax.get_ylim()[0])) == floor(np.log10(ax.get_ylim()[1]))
#   bool_large_domain = (np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0])) > 1
#   if bool_small_domain_cross or bool_large_domain:
#     ax.yaxis.set_major_formatter(ScalarFormatter())
#     ax.yaxis.set_minor_formatter(NullFormatter())
#   else:
#     ax.yaxis.set_minor_formatter(ScalarFormatter())
#   ## plot converged scale
#   funcPlotConvergedScale(ax, label_scale, list_converged_scales, fit_params[0])
#   ## label axis
#   ax.set_ylabel(label_scale, fontsize=22)
#   ## return converged scale
#   return list_converged_scales

class PlotSpectraConvergence():
  def __init__(
      self,
      sim_suite,
      sim_folder
    ):
    self.sim_suite  = sim_suite
    self.sim_folder = sim_folder
    self.readScales()
    self.plotScales()
  def readScales(self):
    self.list_k_nu_group_res  = []
    self.list_k_eta_group_res = []
    self.list_k_p_group_res   = []
    ## read in scales for each resolution run
    for sim_res in LIST_SIM_RES:
      try:
        ## load spectra-fit data as a dictionary
        spectra_fits_dict = WWObjs.loadJson2Dict(
          filepath = WWFnF.createFilepath([
            BASEPATH, self.sim_suite, sim_res, SONIC_REGIME, self.sim_folder, "spect"
          ]),
          filename = FILENAME_SPECTRA,
          bool_hide_updates = True
        )
        ## store dictionary data in spectra-fit object
        spectra_fits_obj = FitMHDScales.SpectraFit(**spectra_fits_dict)
      except: continue
      ## check that a fit-range has been defined
      bool_kin_fit = (spectra_fits_obj.kin_fit_start_t is not None) and (spectra_fits_obj.kin_fit_end_t is not None)
      bool_mag_fit = (spectra_fits_obj.mag_fit_start_t is not None) and (spectra_fits_obj.mag_fit_end_t is not None)
      if not(bool_kin_fit) or not(bool_mag_fit):
        raise Exception("Fit range has not been defined.")
      ## find indices corresponding with the fit-range bounds
      kin_index_start = WWLists.getIndexClosestValue(spectra_fits_obj.kin_list_sim_times, spectra_fits_obj.kin_fit_start_t)
      mag_index_start = WWLists.getIndexClosestValue(spectra_fits_obj.mag_list_sim_times, spectra_fits_obj.mag_fit_start_t)
      kin_index_end   = WWLists.getIndexClosestValue(spectra_fits_obj.kin_list_sim_times, spectra_fits_obj.kin_fit_end_t)
      mag_index_end   = WWLists.getIndexClosestValue(spectra_fits_obj.mag_list_sim_times, spectra_fits_obj.mag_fit_end_t)
      ## save measured scales in fit time-range
      self.list_k_nu_group_res.append(
        spectra_fits_obj.k_nu_group_t[kin_index_start  : kin_index_end]
      )
      self.list_k_eta_group_res.append(
        spectra_fits_obj.k_eta_group_t[mag_index_start : mag_index_end]
      )
      self.list_k_p_group_res.append(
        spectra_fits_obj.k_p_group_t[mag_index_start : mag_index_end]
      )
  def plotScales(self):
    ## initialise figure
    fig, axs = plt.subplots(3, 1, figsize=(6, 3.5*3), sharex=True)
    fig.subplots_adjust(hspace=0.05)
    ## plot measured scales
    for res_index, sim_res in enumerate(LIST_SIM_RES):
      PlotFuncs.plotErrorBar(axs[0], int(sim_res), self.list_k_nu_group_res[res_index],  color="black")
      PlotFuncs.plotErrorBar(axs[1], int(sim_res), self.list_k_eta_group_res[res_index], color="black")
      PlotFuncs.plotErrorBar(axs[2], int(sim_res), self.list_k_p_group_res[res_index],   color="black")
    ## label figure
    axs[0].set_ylabel(r"$k_\nu$")
    axs[1].set_ylabel(r"$k_\eta$")
    axs[2].set_ylabel(r"$k_{\rm p}$")
    axs[2].set_xlabel(r"$k$")
    axs[0].set_xscale("log")
    axs[1].set_xscale("log")
    axs[2].set_xscale("log")
    axs[0].set_xlim([10, 1000])
    axs[1].set_xlim([10, 1000])
    axs[2].set_xlim([10, 1000])
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    axs[2].set_yscale("log")
    axs[0].set_ylim([0.1, 50])
    axs[1].set_ylim([0.1, 50])
    axs[2].set_ylim([0.1, 50])
    ## save figure
    fig_name = self.sim_folder + "_scales_res" + FILENAME_TAG + ".png"
    plt.savefig(
      WWFnF.createFilepath([
        BASEPATH, self.sim_suite, SONIC_REGIME, fig_name
      ])
    )
    print("\t> Figure saved:", fig_name)
    ## close plot
    plt.close(fig)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
BASEPATH          = "/scratch/ek9/nk7952/"
SONIC_REGIME      = "super_sonic"
FILENAME_SPECTRA  = "spectra_fits_fk_fm.json"
FILENAME_TAG      = "_fk_fm"
LIST_SIM_RES      = [ "72", "144", "288" ]

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
    filepath_figures = WWFnF.createFilepath([
      BASEPATH, suite_folder, SONIC_REGIME
    ])
    if not os.path.exists(filepath_figures):
      print("{} does not exist.".format(filepath_figures))
      continue
    str_message = "Looking at suite: {}".format(suite_folder)
    print(str_message)
    print("=" * len(str_message))
    print("Saving figures in:", filepath_figures)

    ## #####################
    ## LOOP OVER SIMULATIONS
    ## #####################
    ## loop over the simulation folders
    for sim_folder in [
        "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"
      ]: # "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"

      ## create filepath to the spectra object for the simulation setup at Nres=288
      filepath_file = WWFnF.createFilepath([
        BASEPATH, suite_folder, "288", SONIC_REGIME, sim_folder, "spect", FILENAME_SPECTRA
      ])
      ## check that the simulation folder exists
      if not os.path.isfile(filepath_file):
        continue

      ## measure converged scales
      PlotSpectraConvergence(sim_suite=suite_folder, sim_folder=sim_folder)
    ## create an empty line after each suite
    print(" ")
  

  # ## ########################################
  # ## FITTING SCALES AS FUNCTION OF RESOLUTION
  # ## ########################################
  # ## initialise fitting bounds
  # bounds = ((0.01, 1, 0.5), (15, 1000, 3))
  # print("Fitting curves...")
  # ## fit to k_nu vs N_res
  # if np.mean(list_k_nu_group_res[0]) < np.mean(list_k_nu_group_res[-1]):
  #     ## measured k_nu scale increased with resolution
  #     list_k_nu_converged = funcFitScales(
  #         ax           = axs[0],
  #         list_x       = list_res,
  #         list_y_group = list_k_nu_group_res,
  #         func         = ListOfModels.logistic_growth_increasing,
  #         bounds       = bounds,
  #         label_scale  = r"$k_\nu$",
  #         bool_debug   = bool_debug,
  #         absolute_sigma = list_bool_abs_std[0]
  #     )
  # else:
  #     ## measured k_nu scale decreased with resolution
  #     list_k_nu_converged = funcFitScales(
  #         ax           = axs[0],
  #         list_x       = list_res,
  #         list_y_group = list_k_nu_group_res,
  #         func         = ListOfModels.logistic_growth_decreasing,
  #         bounds       = bounds,
  #         label_scale  = r"$k_\nu$",
  #         bool_debug   = bool_debug,
  #         absolute_sigma = list_bool_abs_std[0]
  #     )
  # ## fit to k_eta vs N_res
  # list_k_eta_converged = funcFitScales(
  #     ax           = axs[1],
  #     list_x       = list_res,
  #     list_y_group = list_k_eta_group_res,
  #     func         = ListOfModels.logistic_growth_increasing,
  #     bounds       = bounds,
  #     label_scale  = r"$k_\eta$",
  #     bool_debug   = bool_debug,
  #     absolute_sigma = list_bool_abs_std[1]
  # )
  # ## fit to k_p vs N_res
  # list_k_p_converged = funcFitScales(
  #     ax           = axs[2],
  #     list_x       = list_res,
  #     list_y_group = list_k_p_group_res,
  #     func         = ListOfModels.logistic_growth_increasing,
  #     bounds       = bounds,
  #     label_scale  = r"$k_p$",
  #     bool_debug   = bool_debug,
  #     absolute_sigma = list_bool_abs_std[2]
  # )

  # ## ####################
  # ## SAVING SCALES OBJECT
  # ## ####################
  # print("Saving spectra scales object...")
  # spectra_scale_obj = SpectraScales(
  #     ## simulation setup information
  #     Pm = Re / Rm,
  #     ## converged scales
  #     list_k_nu_converged  = list_k_nu_converged,
  #     list_k_eta_converged = list_k_eta_converged,
  #     list_k_p_converged = list_k_p_converged
  # )
  # savePickle(
  #     spectra_scale_obj,
  #     filepath_base,
  #     sim_folder+SCALE_NAME
  # )


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
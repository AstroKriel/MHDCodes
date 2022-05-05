#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import copy

# ## load old user defined modules
# from OldModules import the_fitting_library
# sys.modules["the_fitting_library"] = the_fitting_library

## 'tmpfile' needs to be loaded before 'matplotlib'
## This is so matplotlib's cache is stored in a temporary location when plotting (in parallel)
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
import matplotlib.pyplot as plt

## suppress "optimise" warnings
import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("ignore", OptimizeWarning)

## load user defined modules
from TheUsefulModule import *
from TheLoadingModule import LoadFlashData
from TheFittingModule import FitMHDScales
from ThePlottingModule import *


## ###############################################################
## PREPARE WORKSPACE
##################################################################
os.system("clear") # clear terminal window
plt.ioff()
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## FUNCTIONS
## ###############################################################
def funcPrintSimInfo(
    sim_index,
    sim_directory,
    sim_suite,
    sim_name,
    res_group_sim,
    kin_start_fit,
    kin_end_fit,
    mag_start_fit,
    mag_end_fit,
    Re,
    Rm
  ):
  print("({:d}) sim directory: ".format(sim_index).ljust(20) + sim_directory)
  print("\t> sim suite: {:}".format(sim_suite))
  print("\t> sim name: {:}".format(sim_name))
  print("\t> sim resolution: {:}".format(res_group_sim))
  print("\t> fit domain (kin): [{:}, {:}]".format(kin_start_fit, kin_end_fit))
  print("\t> fit domain (mag): [{:}, {:}]".format(mag_start_fit, mag_end_fit))
  print("\t> Re: {:}, Rm: {:}, Pm: {:}".format(Re, Rm, Rm/Re))

def funcCreateSpectraObj(
    ## directory to spectra
    list_data_filepaths,
    ## output: list of loaded spectra
    list_spectra_objs,
    ## input: list of spectra variables
    suite_name_group_sim,
    label_group_sim,
    res_group_sim,
    Re_group_sim,
    Rm_group_sim,
    kin_fit_start_t_group_sim,
    mag_fit_start_t_group_sim,
    kin_fit_end_t_group_sim,
    mag_fit_end_t_group_sim,
    ## optional fit parameters
    bool_fit_sub_Ek_range_group_sim = False,
    log_Ek_range_group_sim = 6,
    bool_kin_fit_fixed = False,
    bool_mag_fit_fixed = False,
    ## hide terminal output
    bool_hide_updates = True
  ):
  ## loop over each simulation dataset
  for filepath_data, sim_index in zip(
      list_data_filepaths,
      range(len(list_data_filepaths))
    ):
    print("Looking at: " + filepath_data)
    ## load spectra data
    print("\tLoading spectra...")
    list_kin_k_group, list_kin_power_group, list_kin_sim_times = LoadFlashData.loadListSpectra(
      filepath_data,
      str_spectra_type  = "vel", # kinetic energy spectra \sim |velocity|
      bool_hide_updates = bool_hide_updates
    )
    list_mag_k_group, list_mag_power_group, list_mag_sim_times = LoadFlashData.loadListSpectra(
      filepath_data,
      str_spectra_type  = "mag",
      bool_hide_updates = bool_hide_updates
    )
    ## fit kinetic energy and magnetic spectra
    print("\tFitting spectra...")
    kin_fit = FitMHDScales.FitVelSpectra(
      list_kin_k_group, list_kin_power_group, list_kin_sim_times,
      bool_fit_sub_Ek_range = bool_fit_sub_Ek_range_group_sim[sim_index],
      log_Ek_range          = log_Ek_range_group_sim[sim_index],
      bool_fit_fixed_model  = bool_kin_fit_fixed,
      bool_hide_updates     = bool_hide_updates
    )
    mag_fit = FitMHDScales.FitMagSpectra(
      list_mag_k_group, list_mag_power_group, list_mag_sim_times,
      bool_fit_fixed_model = bool_mag_fit_fixed,
      bool_hide_updates    = bool_hide_updates
    )
    ## store spectra object variables
    kin_args = kin_fit.getFitArgs()
    mag_args = mag_fit.getFitArgs()
    sim_args = {
      "sim_suite":suite_name_group_sim[sim_index],
      "sim_label":label_group_sim[sim_index],
      "sim_res":res_group_sim[sim_index],
      "Re":Re_group_sim[sim_index],
      "Rm":Rm_group_sim[sim_index],
      "kin_fit_start_t":kin_fit_start_t_group_sim[sim_index],
      "mag_fit_start_t":mag_fit_start_t_group_sim[sim_index],
      "kin_fit_end_t":kin_fit_end_t_group_sim[sim_index],
      "mag_fit_end_t":mag_fit_end_t_group_sim[sim_index]
    }
    ## create and save spectra object
    spectra_obj = FitMHDScales.SpectraFit(**sim_args, **kin_args, **mag_args)
    WWObjs.savePickleObject(spectra_obj, filepath_data, "spectra_obj.pkl")
    ## append object
    list_spectra_objs.append(spectra_obj)
    print(" ")

def funcLoadSpectraObj(
    ## directory to spectra
    list_data_filepaths,
    ## output: list of loaded spectra
    list_spectra_objs,
    ## input: list of spectra variables
    suite_name_group_sim,
    label_group_sim,
    res_group_sim,
    Re_group_sim,
    Rm_group_sim,
    kin_fit_start_t_group_sim,
    kin_fit_end_t_group_sim,
    mag_fit_start_t_group_sim,
    mag_fit_end_t_group_sim,
    ## show object attributes
    bool_print_obj_attrs
  ):
  ## loop over each simulation dataset
  print("Loading spectra objects...")
  if len(list_data_filepaths) == 0:
    raise Exception("ERROR: No data objects to look at.")
  for filepath_data, sim_index in zip(
      list_data_filepaths,
      range(len(list_data_filepaths))
    ):
    ## load spectra object
    spectra_obj = WWObjs.loadPickleObject(filepath_data, "spectra_obj.pkl")
    ## check if simulation variables need to be updated (if parameter is specified -- i.e. not(None))
    duplicate_obj = copy.deepcopy(spectra_obj)
    prev_obj = vars(duplicate_obj)
    bool_updated_obj = False
    ## list of spectra object attributes that can be changed after creation
    bool_updated_obj |= WWObjs.updateAttr(spectra_obj, "sim_suite",       suite_name_group_sim[sim_index])
    bool_updated_obj |= WWObjs.updateAttr(spectra_obj, "sim_label",       label_group_sim[sim_index])
    bool_updated_obj |= WWObjs.updateAttr(spectra_obj, "sim_res",         res_group_sim[sim_index])
    bool_updated_obj |= WWObjs.updateAttr(spectra_obj, "Re",              Re_group_sim[sim_index])
    bool_updated_obj |= WWObjs.updateAttr(spectra_obj, "Rm",              Rm_group_sim[sim_index])
    bool_updated_obj |= WWObjs.updateAttr(spectra_obj, "kin_fit_start_t", kin_fit_start_t_group_sim[sim_index])
    bool_updated_obj |= WWObjs.updateAttr(spectra_obj, "mag_fit_start_t", mag_fit_start_t_group_sim[sim_index])
    bool_updated_obj |= WWObjs.updateAttr(spectra_obj, "kin_fit_end_t",   kin_fit_end_t_group_sim[sim_index])
    bool_updated_obj |= WWObjs.updateAttr(spectra_obj, "mag_fit_end_t",   mag_fit_end_t_group_sim[sim_index])
    new_obj = vars(spectra_obj)
    ## if at least one of the attributes have been updated, then overwrite the spectra object
    if bool_updated_obj:
      ## keep a list of updated attributes
      list_updated_attr = []
      ## find which attributes have been changed
      for prev_attr, new_attr in zip(prev_obj, new_obj):
        ## don't compare list entries (i.e. list of data)
        if not isinstance(prev_obj[prev_attr], list):
          ## if attribute has been changed, then note it down
          if not(prev_obj[prev_attr] == new_obj[new_attr]):
            list_updated_attr.append(prev_attr)
      ## print updated attributes
      print("\t> Updated attributes:", ", ".join(list_updated_attr))
      ## save updated spectra obj
      WWObjs.savePickleObject(spectra_obj, filepath_data, "spectra_obj.pkl")
    ## display the object's attribute values
    if bool_print_obj_attrs:
      print("\t> Object attributes:")
      ## for each of the object attributes
      for attr in new_obj:
        ## if the attribute is a list
        if isinstance(new_obj[attr], list):
          print(
            ("\t\t> " + attr + ": ").ljust(25),
            type(new_obj[attr]),
            len(new_obj[attr])
          )
        ## otherwise, if the attribute is a value of sorts
        else: print(
          ("\t\t> " + attr + ": ").ljust(25),
          new_obj[attr]
        )
    ## if any information had been printed to the screen (for aesthetics)
    if bool_updated_obj or bool_print_obj_attrs:
      print(" ")
    ## append object
    list_spectra_objs.append(spectra_obj)
  ## for aesthetic reasons, do not print line if a line has already been drawn
  if not(bool_updated_obj or bool_print_obj_attrs):
    print(" ")

def funcPlotSpectra(
    ## list of spectra objects
    list_spectra_objs,
    ## plotting variables
    filepath_plot_spect,
    filepath_plot,
    plot_spectra_every,
    plot_spectra_from,
    ## workflow inputs
    bool_plot_spectra,
    bool_hide_updates
  ):
  print("Plotting spectra...")
  ## loop over each simulation dataset
  for spectra_obj in list_spectra_objs:
    ## create plotting object looking at simulation fit
    plot_spectra_obj = PlotSpectra.PlotSpectraFit(spectra_obj)
    ## create frames of spectra evolution
    if bool_plot_spectra:
      ## print information to the terminal
      print("\t> Plotting spectra from simulation '{:}' in '{:}'...".format(
          spectra_obj.sim_label,
          spectra_obj.sim_suite
      ))
      print("\t(Total of '{:d}' spectra fits. Plotting every '{:d}' fits from fit-index '{:d}')".format(
          len(WWLists.getCommonElements(spectra_obj.kin_sim_times, spectra_obj.mag_sim_times)),
          plot_spectra_every,
          plot_spectra_from
        )
      )
      ## plot spectra evolution
      plot_spectra_obj.plotSpectraEvolution(
        filepath_plot     = filepath_plot_spect,
        plot_index_start  = plot_spectra_from,
        plot_index_step   = plot_spectra_every,
        bool_hide_updates = bool_hide_updates
      )
    ## animate spectra evolution
    plot_spectra_obj.aniSpectra(
      filepath_frames    = filepath_plot_spect,
      filepath_ani_movie = filepath_plot,
      bool_hide_updates  = bool_hide_updates
    )


## ###############################################################
## DEFINE MAIN PROGRAM
## ###############################################################
def main():
  ## #############################
  ## DEFINE COMMAND LINE ARGUMENTS
  ## #############################
  parser = WWArgparse.MyParser()
  ## ------------------- DEFINE OPTIONAL ARGUMENTS
  args_opt = parser.add_argument_group(description="Optional processing arguments:")
  ## program workflow parameters
  opt_bool_args      = {"required":False, "type":WWArgparse.str2bool, "nargs":"?", "const":True}
  opt_list_bool_args = {"required":False, "type":WWArgparse.str2bool, "nargs":"+"}
  opt_list_args      = {"required":False, "default":[ None ], "nargs":"+"}
  args_opt.add_argument("-hide_updates",    default=False, **opt_bool_args)
  args_opt.add_argument("-print_obj_attrs", default=False, **opt_bool_args)
  args_opt.add_argument("-fit_spectra",     default=False, **opt_bool_args)
  args_opt.add_argument("-plot_spectra",    default=False, **opt_bool_args)
  ## directory information
  args_opt.add_argument("-vis_folder", type=str, default="vis_folder", required=False)
  args_opt.add_argument("-sub_folder", type=str, default="spect",      required=False)
  ## fit fixed spectra models
  args_opt.add_argument("-kin_fit_fixed",    default=False, **opt_bool_args)
  args_opt.add_argument("-mag_fit_fixed",    default=False, **opt_bool_args)
  ## energy range to fit kinetic energy spectra
  args_opt.add_argument("-fit_sub_Ek_range", default=[False], **opt_list_bool_args)
  args_opt.add_argument("-log_Ek_range",     type=float,      **opt_list_args)
  ## time range to fit spectra
  args_opt.add_argument("-kin_start_fits",   type=float, **opt_list_args)
  args_opt.add_argument("-kin_end_fits",     type=float, **opt_list_args)
  args_opt.add_argument("-mag_start_fits",   type=float, **opt_list_args)
  args_opt.add_argument("-mag_end_fits",     type=float, **opt_list_args)
  ## plotting parameters
  args_opt.add_argument("-plot_spectra_from",  type=int, default=0, required=False, help="first plot index")
  args_opt.add_argument("-plot_spectra_every", type=int, default=1, required=False, help="index step size")
  ## simulation information
  args_opt.add_argument("-sim_suites", type=str,   **opt_list_args)
  args_opt.add_argument("-sim_res",    type=int,   **opt_list_args)
  args_opt.add_argument("-Re",         type=float, **opt_list_args)
  args_opt.add_argument("-Rm",         type=float, **opt_list_args)
  args_opt.add_argument("-Pm",         type=float, **opt_list_args)
  ## ------------------- DEFINE REQUIRED ARGUMENTS
  args_req = parser.add_argument_group(description="Required processing arguments:")
  args_req.add_argument("-base_path",   type=str, required=True)
  args_req.add_argument("-sim_folders", type=str, required=True, nargs="+")

  ## #########################
  ## INTERPRET INPUT ARGUMENTS
  ## #########################
  ## ---------------------------- OPEN ARGUMENTS
  args = vars(parser.parse_args())
  ## ---------------------------- SAVE PARAMETERS
  ## (boolean) workflow parameters
  bool_hide_updates               = args["hide_updates"]
  bool_print_obj_attrs            = args["print_obj_attrs"]
  bool_fit_spectra                = args["fit_spectra"]
  bool_plot_spectra               = args["plot_spectra"]
  ## fit fixed spectra models
  bool_kin_fit_fixed              = args["kin_fit_fixed"]
  bool_mag_fit_fixed              = args["mag_fit_fixed"]
  ## energy range to fit kinetic energy spectra
  bool_fit_sub_Ek_range_group_sim = args["fit_sub_Ek_range"]
  log_Ek_range_group_sim          = args["log_Ek_range"]
  ## time range to fit spectra
  kin_fit_start_t_group_sim       = args["kin_start_fits"]
  kin_fit_end_t_group_sim         = args["kin_end_fits"]
  mag_fit_start_t_group_sim       = args["mag_start_fits"]
  mag_fit_end_t_group_sim         = args["mag_end_fits"]
  ## plotting parameters
  plot_spectra_from               = args["plot_spectra_from"]
  plot_spectra_every              = args["plot_spectra_every"]
  ## important directory information
  filepath_base                   = args["base_path"]
  folder_vis                      = args["vis_folder"]
  folder_sub                      = args["sub_folder"]
  suite_name_group_sim            = args["sim_suites"]
  label_group_sim                 = args["sim_folders"]
  res_group_sim                   = args["sim_res"]
  ## simulation parameters
  Re_group_sim                    = args["Re"]
  Rm_group_sim                    = args["Rm"]
  Pm_group_sim                    = args["Pm"]

  ## ######################
  ## INITIALISING VARIABLES
  ## ######################
  print("Interpreting user inputs...")
  num_sims = len(label_group_sim)
  ## if analysing (i.e. creating spectra object) and Re / Rm weren"t defined
  bool_missing_plasma_numbers = (
    ( (None in Re_group_sim) and (None in Pm_group_sim) ) or
    ( (None in Rm_group_sim) and (None in Pm_group_sim) ) or
    ( (None in Re_group_sim) and (None in Rm_group_sim) )
  )
  if bool_fit_spectra and bool_missing_plasma_numbers:
    raise Exception("> You need to define 2 of 'Re', 'Rm' and 'Pm' to fit to spectra.")
  ## check that each simulation has been given an Re & Rm
  WWLists.extendInputList(Re_group_sim, "Re_group_sim", num_sims)
  WWLists.extendInputList(Rm_group_sim, "Rm_group_sim", num_sims)
  WWLists.extendInputList(Pm_group_sim, "Pm_group_sim", num_sims)
  ## if simulation suite / labels weren"t specified
  WWLists.extendInputList(suite_name_group_sim, "suite_name_group_sim", num_sims)
  WWLists.extendInputList(res_group_sim,        "res_group_sim",        num_sims)
  ## if a time-range isn"t specified for one of the simulations, then use the default time-range
  WWLists.extendInputList(kin_fit_start_t_group_sim, "kin_fit_start_t_group_sim", num_sims)
  WWLists.extendInputList(mag_fit_start_t_group_sim, "mag_fit_start_t_group_sim", num_sims)
  WWLists.extendInputList(kin_fit_end_t_group_sim,   "kin_fit_end_t_group_sim",   num_sims)
  WWLists.extendInputList(mag_fit_end_t_group_sim,   "mag_fit_end_t_group_sim",   num_sims)
  ## check if user wants to fit to a subset of the kinetic energy spectrum
  WWLists.extendInputList(bool_fit_sub_Ek_range_group_sim, "bool_fit_sub_Ek_range_group_sim", num_sims)
  WWLists.extendInputList(log_Ek_range_group_sim,          "log_Ek_range_group_sim",          num_sims)
  print(" ")
  ## interpret missing plasma reynolds numbers
  if bool_fit_spectra:
    if None in Re_group_sim:
      Re_group_sim = []
      for sim_index in range(num_sims):
        Re_group_sim.append(
          Rm_group_sim[sim_index] / Pm_group_sim[sim_index]
        )
    elif None in Rm_group_sim:
      Rm_group_sim = []
      for sim_index in range(num_sims):
        Rm_group_sim.append(
          Re_group_sim[sim_index] * Pm_group_sim[sim_index]
        )

  ## #####################
  ## PREPARING DIRECTORIES
  ## #####################
  ## folders where spectra data is
  list_data_filepaths = []
  for sim_index in range(num_sims):
    list_data_filepaths.append( WWFnF.createFilepath([
      filepath_base, label_group_sim[sim_index], folder_sub
    ]) )
  ## folder where visualisations will be saved
  filepath_plot = WWFnF.createFilepath([filepath_base, folder_vis])
  ## folder where spectra plots will be saved
  filepath_plot_spect = WWFnF.createFilepath([filepath_base, folder_vis, "plotSpectra"])

  ## ##############
  ## CREATE FOLDERS
  ## ######
  if bool_plot_spectra:
    WWFnF.createFolder(filepath_plot)
    WWFnF.createFolder(filepath_plot_spect)

  ## ############################
  ## PRINT INFORMATION TO CONSOLE
  ## ############################
  P2Term.printInfo("Base filepath:", filepath_base)
  P2Term.printInfo("Figure folder:", filepath_plot)
  ## if fitting spectra data and making spectra objects
  if bool_fit_spectra:
    print("Fitting '{:d}' spectra simulation(s):".format(len(list_data_filepaths)))
    for sim_index in range(len(list_data_filepaths)):
      funcPrintSimInfo(
        sim_index      = sim_index,
        sim_directory  = list_data_filepaths[sim_index],
        sim_suite      = suite_name_group_sim[sim_index],
        sim_name       = label_group_sim[sim_index],
        res_group_sim  = res_group_sim[sim_index],
        kin_start_fit  = kin_fit_start_t_group_sim[sim_index],
        kin_end_fit    = kin_fit_end_t_group_sim[sim_index],
        mag_start_fit  = mag_fit_start_t_group_sim[sim_index],
        mag_end_fit    = mag_fit_end_t_group_sim[sim_index],
        Re             = Re_group_sim[sim_index],
        Rm             = Rm_group_sim[sim_index]
      )
    print(" ")
  ## otherwise if reading in spectra objects
  else:
    list_data_filepaths_remove = []
    for data_filepath in list_data_filepaths:
      ## check that the spectra object exists in the simulation folder
      try: WWObjs.loadPickleObject(data_filepath, "spectra_obj.pkl", bool_hide_updates=True)
      ## remove simulations that don't have spectra object to look at
      except: list_data_filepaths_remove.append(data_filepath)
    list_data_filepaths = [
      data_filepath
      for data_filepath in list_data_filepaths
      if data_filepath not in list_data_filepaths_remove
    ]
    ## indicate which simulations will be looked at
    print("Reading '{:d}' spectra object(s):".format(len(list_data_filepaths)))
    for sim_index in range(len(list_data_filepaths)):
      P2Term.printInfo("({:d}) sim directory:".format(sim_index), list_data_filepaths[sim_index], 20)
    print(" ")

  ## #######################
  ## LOAD & FIT SPECTRA DATA
  ## #######################
  ## initialise list of spectra objects
  list_spectra_objs = []
  ## if user wants to fit spectra
  if bool_fit_spectra:
    funcCreateSpectraObj(
      ## directory to spectra
      list_data_filepaths             = list_data_filepaths,
      ## output
      list_spectra_objs               = list_spectra_objs,
      ## input
      suite_name_group_sim            = suite_name_group_sim,
      label_group_sim                 = label_group_sim,
      res_group_sim                   = res_group_sim,
      Re_group_sim                    = Re_group_sim,
      Rm_group_sim                    = Rm_group_sim,
      kin_fit_start_t_group_sim       = kin_fit_start_t_group_sim,
      kin_fit_end_t_group_sim         = kin_fit_end_t_group_sim,
      mag_fit_start_t_group_sim       = mag_fit_start_t_group_sim,
      mag_fit_end_t_group_sim         = mag_fit_end_t_group_sim,
      ## optional fit parameters
      bool_fit_sub_Ek_range_group_sim = bool_fit_sub_Ek_range_group_sim,
      log_Ek_range_group_sim          = log_Ek_range_group_sim,
      bool_kin_fit_fixed              = bool_kin_fit_fixed,
      bool_mag_fit_fixed              = bool_mag_fit_fixed,
      ## hide terminal output
      bool_hide_updates               = bool_hide_updates
    )
  ## otherwise read in previously fitted spectra
  else:
    funcLoadSpectraObj(
      ## directory to spectra
      list_data_filepaths       = list_data_filepaths,
      ## output: list of loaded spectra
      list_spectra_objs         = list_spectra_objs,
      ## input: list of spectra variables
      suite_name_group_sim      = suite_name_group_sim,
      label_group_sim           = label_group_sim,
      res_group_sim             = res_group_sim,
      Re_group_sim              = Re_group_sim,
      Rm_group_sim              = Rm_group_sim,
      kin_fit_start_t_group_sim = kin_fit_start_t_group_sim,
      kin_fit_end_t_group_sim   = kin_fit_end_t_group_sim,
      mag_fit_start_t_group_sim = mag_fit_start_t_group_sim,
      mag_fit_end_t_group_sim   = mag_fit_end_t_group_sim,
      ## show object attributes
      bool_print_obj_attrs      = bool_print_obj_attrs
    )

  ## ######################
  ## PLOT SPECTRA EVOLUTION
  ## ######################
  if bool_plot_spectra:
    funcPlotSpectra(
      ## list of spectra objects
      list_spectra_objs   = list_spectra_objs,
      ## plotting variables
      filepath_plot_spect = filepath_plot_spect,
      filepath_plot       = filepath_plot,
      plot_spectra_every  = plot_spectra_every,
      plot_spectra_from   = plot_spectra_from,
      ## workflow inputs
      bool_plot_spectra   = bool_plot_spectra,
      bool_hide_updates   = bool_hide_updates
    )


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
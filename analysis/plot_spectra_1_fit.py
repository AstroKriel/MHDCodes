#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import copy

# ## load old user MHD analysis modules
# from OldModules import the_fitting_library
# sys.modules["the_fitting_library"] = the_fitting_library

## 'tmpfile' needs to be loaded before 'matplotlib'.
## This is so matplotlib's cache is stored in a temporary location when plotting.
## This is useful for plotting parallel
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


SPECTRA_OBJ_NAME = "spectra_obj.pkl"
## #########################################
## CLASS TO HANDLE CALLS TO FITTING ROUTINES
## #########################################
class CreateUpdateSpectraObject():
  def __init__(
      self,
      filepath_data,
      sim_suite, sim_label, sim_res,
      Re, Rm, Pm,
      kin_fit_start_time, kin_fit_end_time,
      mag_fit_start_time, mag_fit_end_time
    ):
    ## where the simulation spectra data is stored
    self.filepath_data      = filepath_data
    ## simulation parameters
    self.sim_suite          = sim_suite
    self.sim_label          = sim_label
    self.sim_res            = sim_res
    self.Re                 = Re
    self.Rm                 = Rm
    self.Pm                 = Pm
    ## fit domain
    self.kin_fit_start_time = kin_fit_start_time
    self.mag_fit_start_time = mag_fit_start_time
    self.kin_fit_end_time   = kin_fit_end_time
    self.mag_fit_end_time   = mag_fit_end_time
  def createSpectraObj(
      self,
      bool_kin_fit_sub_range = False,
      kin_num_decades_to_fit = 6,
      bool_kin_fit_fixed     = False,
      bool_mag_fit_fixed     = False,
      bool_hide_updates      = False
    ):
    print("\tLoading spectra...")
    ## load kinetic energy spectra
    list_kin_k_group_t, list_kin_power_group_t, list_kin_sim_times = LoadFlashData.loadListSpectra(
      filepath_data     = self.filepath_data,
      str_spectra_type  = "vel",
      bool_hide_updates = bool_hide_updates
    )
    ## load magnetic energy spectra
    list_mag_k_group_t, list_mag_power_group_t, list_mag_sim_times = LoadFlashData.loadListSpectra(
      filepath_data     = self.filepath_data,
      str_spectra_type  = "mag",
      bool_hide_updates = bool_hide_updates
    )
    print("\tFitting spectra...")
    ## fit kinetic energy spectra
    kin_fit_obj = FitMHDScales.FitVelSpectra(
      list_k_group_t        = list_kin_k_group_t,
      list_power_group_t    = list_kin_power_group_t,
      list_sim_times        = list_kin_sim_times,
      bool_fit_sub_Ek_range = bool_kin_fit_sub_range,
      log_Ek_range          = kin_num_decades_to_fit,
      bool_fit_fixed_model  = bool_kin_fit_fixed,
      bool_hide_updates     = bool_hide_updates
    )
    ## fit magnetic energy spectra
    mag_fit_obj = FitMHDScales.FitMagSpectra(
      list_k_group_t       = list_mag_k_group_t,
      list_power_group_t   = list_mag_power_group_t,
      list_sim_times       = list_mag_sim_times,
      bool_fit_fixed_model = bool_mag_fit_fixed,
      bool_hide_updates    = bool_hide_updates
    )
    ## extract spectra fit parameters
    kin_fit_args = kin_fit_obj.getFitArgs()
    mag_fit_args = mag_fit_obj.getFitArgs()
    ## store siulation parameters in a dictionary
    sim_args = {
      "sim_suite":self.sim_suite,
      "sim_label":self.sim_label,
      "sim_res":self.sim_res,
      "Re":self.Re,
      "Rm":self.Rm,
      "Pm":self.Pm,
      "kin_fit_start_t":self.kin_fit_start_time,
      "mag_fit_start_t":self.mag_fit_start_time,
      "kin_fit_end_t":self.kin_fit_end_time,
      "mag_fit_end_t":self.mag_fit_end_time
    }
    ## create spectra-object
    self.spectra_obj = FitMHDScales.SpectraFit(
      **sim_args,
      **kin_fit_args,
      **mag_fit_args
    )
    ## save the pickle object
    WWObjs.savePickleObject(self.spectra_obj, self.filepath_data, SPECTRA_OBJ_NAME)
    print(" ")
  def loadSpectraObj(
      self,
      bool_print_obj_attrs = False
    ):
    ## load spectra object
    spectra_obj = WWObjs.loadPickleObject(self.filepath_data, SPECTRA_OBJ_NAME)
    ## check if any simulation paarmeters need to be updated:
    ## assumed attribute should be updated if it is not 'None'
    duplicate_obj = copy.deepcopy(spectra_obj) # create a copy of the spectra-object for reference
    prev_obj = vars(duplicate_obj) # old attribute values
    bool_obj_was_updated = False
    ## #####################################################
    ## UPDATE SIMULATION ATTRIBUTES STORED IN SPECTRA OBJECT
    ## #####################################################
    ## list of simulation attributes that can be updated
    bool_obj_was_updated |= WWObjs.updateAttr(spectra_obj, "sim_suite",       self.sim_suite)
    bool_obj_was_updated |= WWObjs.updateAttr(spectra_obj, "sim_label",       self.sim_label)
    bool_obj_was_updated |= WWObjs.updateAttr(spectra_obj, "sim_res",         self.sim_res)
    bool_obj_was_updated |= WWObjs.updateAttr(spectra_obj, "Re",              self.Re)
    bool_obj_was_updated |= WWObjs.updateAttr(spectra_obj, "Rm",              self.Rm)
    bool_obj_was_updated |= WWObjs.updateAttr(spectra_obj, "Pm",              self.Rm)
    bool_obj_was_updated |= WWObjs.updateAttr(spectra_obj, "kin_fit_start_t", self.kin_fit_start_time)
    bool_obj_was_updated |= WWObjs.updateAttr(spectra_obj, "mag_fit_start_t", self.mag_fit_start_time)
    bool_obj_was_updated |= WWObjs.updateAttr(spectra_obj, "kin_fit_end_t",   self.kin_fit_end_time)
    bool_obj_was_updated |= WWObjs.updateAttr(spectra_obj, "mag_fit_end_t",   self.mag_fit_end_time)
    new_obj = vars(spectra_obj) # updated attribute values
    ## ########################################
    ## CHECK WHICH ATTRIBUTES HAVE BEEN UPDATED
    ## ########################################
    ## if any of the attributes have been updated
    if bool_obj_was_updated:
      ## save the updated spectra object
      WWObjs.savePickleObject(spectra_obj, self.filepath_data, SPECTRA_OBJ_NAME)
      ## keep a list of updated attributes
      list_updated_attrs = []
      for prev_attr_name, new_attr_name in zip(prev_obj, new_obj):
        ## don't compare list entries
        if not isinstance(prev_obj[prev_attr_name], list):
          ## note attribute name if it has been updated
          if not(prev_obj[prev_attr_name] == new_obj[new_attr_name]):
            list_updated_attrs.append(prev_attr_name)
      ## print updated attributes
      print("\t> Updated attributes:", ", ".join(list_updated_attrs))
    ## ###################################
    ## PRINT ALL SPECTRA OBJECT ATTRIBUTES
    ## ###################################
    if bool_print_obj_attrs:
      print("\t> Object attributes:")
      ## print each paramater name and value stored in the spectra object
      for attr_name in new_obj:
        if isinstance(new_obj[attr_name], list):
          print( # the attribute is a list
            ("\t\t> {}".format(attr_name)).ljust(25),
            "type: {}, # of entries: {}".format(
              type(new_obj[attr_name]),
              len(new_obj[attr_name])
            )
          )
        else:
          print( # the attribute is a value of sorts
            ("\t\t> {}".format(attr_name)).ljust(25),
            "= {}".format(new_obj[attr_name])
          )
    ## for aesthetics: if any information had been printed to the screen
    if bool_obj_was_updated or bool_print_obj_attrs:
      print(" ")
    ## store the updated spectra object
    self.spectra_obj
  def plotSpectraFits(
      self,
      filepath_vis,
      filepath_vis_frames,
      plot_spectra_from  = 1,
      plot_spectra_every = 1,
      bool_hide_updates  = False
    ):
    ## create plotting object looking at simulation fit
    plot_spectra_obj = PlotSpectra.PlotSpectraFit(self.spectra_obj)
    ## create frames of spectra evolution
    print("\t> Plotting spectra from simulation '{:}' in '{:}'...".format(
        self.spectra_obj.sim_label,
        self.spectra_obj.sim_suite
    ))
    print("\t(Total of '{:d}' spectra fits. Plotting every '{:d}' fit(s) from fit-index '{:d}')".format(
        len(WWLists.getCommonElements(self.spectra_obj.kin_sim_times, self.spectra_obj.mag_sim_times)),
        plot_spectra_every,
        plot_spectra_from
      )
    )
    ## plot spectra evolution
    plot_spectra_obj.plotSpectraEvolution(
      filepath_plot     = filepath_vis_frames,
      plot_index_start  = plot_spectra_from,
      plot_index_step   = plot_spectra_every,
      bool_hide_updates = bool_hide_updates
    )
    ## animate spectra evolution
    plot_spectra_obj.aniSpectra(
      filepath_frames    = filepath_vis_frames,
      filepath_ani_movie = filepath_vis,
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
  opt_bool_args = {"required":False, "type":WWArgparse.str2bool, "nargs":"?", "const":True}
  opt_args      = {"required":False, "default":None}
  args_opt.add_argument("-hide_updates",    default=False, **opt_bool_args)
  args_opt.add_argument("-print_obj_attrs", default=False, **opt_bool_args)
  args_opt.add_argument("-fit_spectra",     default=False, **opt_bool_args)
  args_opt.add_argument("-plot_spectra",    default=False, **opt_bool_args)
  ## directory information
  args_opt.add_argument("-vis_folder",  type=str, default="vis_folder", required=False)
  args_opt.add_argument("-data_folder", type=str, default="spect",      required=False)
  ## fit fixed spectra models
  args_opt.add_argument("-kin_fit_fixed",    default=False, **opt_bool_args)
  args_opt.add_argument("-mag_fit_fixed",    default=False, **opt_bool_args)
  ## energy range to fit kinetic energy spectra
  args_opt.add_argument("-kin_fit_sub_range",      default=False, **opt_bool_args)
  args_opt.add_argument("-kin_num_decades_to_fit", type=float,    **opt_args)
  ## time range to fit spectra
  args_opt.add_argument("-kin_start_fit",   type=float, **opt_args)
  args_opt.add_argument("-mag_start_fit",   type=float, **opt_args)
  args_opt.add_argument("-kin_end_fit",     type=float, **opt_args)
  args_opt.add_argument("-mag_end_fit",     type=float, **opt_args)
  ## plotting parameters
  args_opt.add_argument("-plot_spectra_from",  type=int, default=0, required=False, help="first plot index")
  args_opt.add_argument("-plot_spectra_every", type=int, default=1, required=False, help="index step size")
  ## simulation information
  args_opt.add_argument("-sim_suite", type=str,   **opt_args)
  args_opt.add_argument("-sim_res",   type=int,   **opt_args)
  args_opt.add_argument("-Re",        type=float, **opt_args)
  args_opt.add_argument("-Rm",        type=float, **opt_args)
  args_opt.add_argument("-Pm",        type=float, **opt_args)
  ## ------------------- DEFINE REQUIRED ARGUMENTS
  args_req = parser.add_argument_group(description="Required processing arguments:")
  args_req.add_argument("-base_path",  type=str, required=True)
  args_req.add_argument("-sim_folder", type=str, required=True)

  ## #########################
  ## INTERPRET INPUT ARGUMENTS
  ## #########################
  ## ---------------------------- OPEN ARGUMENTS
  args = vars(parser.parse_args())
  ## ---------------------------- SAVE PARAMETERS
  ## (boolean) workflow parameters
  bool_hide_updates     = args["hide_updates"]
  bool_print_obj_attrs  = args["print_obj_attrs"]
  bool_fit_spectra      = args["fit_spectra"]
  bool_plot_spectra     = args["plot_spectra"]
  ## fit fixed spectra models
  bool_kin_fit_fixed    = args["kin_fit_fixed"]
  bool_mag_fit_fixed    = args["mag_fit_fixed"]
  ## energy range to fit kinetic energy spectra
  bool_kin_fit_sub_range = args["kin_fit_sub_range"]
  kin_num_decades_to_fit = args["kin_num_decades_to_fit"]
  ## time range to fit spectra
  kin_fit_start_time    = args["kin_start_fit"]
  mag_fit_start_time    = args["mag_start_fit"]
  kin_fit_end_time      = args["kin_end_fit"]
  mag_fit_end_time      = args["mag_end_fit"]
  ## plotting parameters
  plot_spectra_from     = args["plot_spectra_from"]
  plot_spectra_every    = args["plot_spectra_every"]
  ## important directory information
  filepath_base         = args["base_path"]
  folder_vis            = args["vis_folder"]
  folder_data           = args["data_folder"]
  sim_suite             = args["sim_suite"]
  sim_label             = args["sim_folder"]
  sim_res               = args["sim_res"]
  ## simulation parameters
  Re                    = args["Re"]
  Rm                    = args["Rm"]
  Pm                    = args["Pm"]

  ## ######################
  ## INITIALISING VARIABLES
  ## ######################
  if bool_fit_spectra:
    ## check if any pair of plasma Reynolds numbers have not been defined
    bool_missing_plasma_numbers = (
      ( (Re == None) and (Pm == None) ) or
      ( (Rm == None) and (Pm == None) ) or
      ( (Re == None) and (Rm == None) )
    )
    if bool_missing_plasma_numbers:
      raise Exception("Error: Undefined plasma-Reynolds numbers. You need to define 2 of 'Re', 'Rm' and 'Pm'.")
    elif Re == None:
      Re = Rm / Pm
    elif Rm == None:
      Rm = Re * Pm
    elif Pm == None:
      Pm = Rm / Re

  ## #####################
  ## PREPARING DIRECTORIES
  ## #####################
  ## folders where spectra data is
  filepath_data = WWFnF.createFilepath([ filepath_base, sim_label, folder_data ])
  ## folder where visualisations will be saved
  filepath_vis = WWFnF.createFilepath([ filepath_base, folder_vis ])
  ## folder where spectra plots will be saved
  filepath_vis_frames = WWFnF.createFilepath([ filepath_vis, "plotSpectra" ])

  ## ##############
  ## CREATE FOLDERS
  ## ##############
  if bool_plot_spectra:
    WWFnF.createFolder(filepath_vis)
    WWFnF.createFolder(filepath_vis_frames)

  ## ######################################
  ## PRINT SIMULATION PARAMETERS TO CONSOLE
  ## ######################################
  P2Term.printInfo("Base filepath:", filepath_base)
  P2Term.printInfo("Data filepath:", filepath_data)
  P2Term.printInfo("Vis. filepath:", filepath_vis)
  ## print simulation parameters
  if bool_fit_spectra:
    print("\t> sim suite: {}".format(sim_suite))
    print("\t> sim label: {}".format(sim_label))
    print("\t> sim resolution: {}".format(sim_res))
    print("\t> fit domain (kin): [{}, {}]".format(kin_fit_start_time, kin_fit_end_time))
    print("\t> fit domain (mag): [{}, {}]".format(mag_fit_start_time, mag_fit_end_time))
    print("\t> Re: {}, Rm: {}, Pm: {}".format(Re, Rm, Pm))
    print(" ")
  ## otherwise check if the spectra pickle-file exists
  else:
    try: WWObjs.loadPickleObject(filepath_data, SPECTRA_OBJ_NAME, bool_hide_updates=True)
    except: raise Exception("Error: '{}' does not exist.".format(SPECTRA_OBJ_NAME))

  ## #########################
  ## LOAD FITTED / FIT SPECTRA
  ## #########################
  cuso = CreateUpdateSpectraObject(
    filepath_data      = filepath_data,
    sim_suite          = sim_suite,
    sim_label          = sim_label,
    sim_res            = sim_res,
    Re                 = Re,
    Rm                 = Rm,
    Pm                 = Pm,
    kin_fit_start_time = kin_fit_start_time,
    mag_fit_start_time = mag_fit_start_time,
    kin_fit_end_time   = kin_fit_end_time,
    mag_fit_end_time   = mag_fit_end_time
  )
  ## read and fit spectra data
  if bool_fit_spectra:
    cuso.createSpectraObj(
      bool_kin_fit_sub_range = bool_kin_fit_sub_range,
      kin_num_decades_to_fit = kin_num_decades_to_fit,
      bool_kin_fit_fixed     = bool_kin_fit_fixed,
      bool_mag_fit_fixed     = bool_mag_fit_fixed,
      bool_hide_updates      = bool_hide_updates
    )
  ## read in already fitted spectra
  else: cuso.loadSpectraObj(bool_print_obj_attrs)

  ## #############################
  ## PLOT EVOLUTION OF THE SPECTRA
  ## #############################
  if bool_plot_spectra:
    cuso.plotSpectraFits(
      filepath_vis        = filepath_vis,
      filepath_vis_frames = filepath_vis_frames,
      plot_spectra_from   = plot_spectra_from,
      plot_spectra_every  = plot_spectra_every,
      bool_hide_updates   = bool_hide_updates
    )


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
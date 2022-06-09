#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import copy

## 'tmpfile' needs to be loaded before 'matplotlib'.
## This is so matplotlib stores its cache in a temporary directory.
## (Useful for plotting parallel)
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
import matplotlib.pyplot as plt

## suppress "optimise" warnings
import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("ignore", OptimizeWarning)

## load user defined modules
from TheUsefulModule import WWArgparse, WWObjs, WWLists, WWFnF, P2Term
from TheLoadingModule import LoadFlashData
from TheFittingModule import FitMHDScales
from ThePlottingModule import PlotSpectra


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.ioff()
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## CLASS TO MANAGE CALLS TO SPECTRA FITTING ROUTINES
## ###############################################################
class SpectraObject():
  def __init__(
      self,
      filepath_data, filename_spectra_fits,
      sim_suite, sim_label, sim_res,
      Re, Rm, Pm,
      kin_fit_start_time, kin_fit_end_time,
      mag_fit_start_time, mag_fit_end_time
    ):
    ## where the simulation spectra data is stored
    self.filepath_data         = filepath_data
    self.filename_spectra_fits = filename_spectra_fits
    ## simulation parameters
    self.sim_suite             = sim_suite
    self.sim_label             = sim_label
    self.sim_res               = sim_res
    self.Re                    = Re
    self.Rm                    = Rm
    self.Pm                    = Pm
    ## fit domain
    self.kin_fit_start_time    = kin_fit_start_time
    self.mag_fit_start_time    = mag_fit_start_time
    self.kin_fit_end_time      = kin_fit_end_time
    self.mag_fit_end_time      = mag_fit_end_time
  def createSpectraFitsObj(
      self,
      bool_kin_fit_fixed_model = False,
      bool_mag_fit_fixed_model = False,
      bool_kin_fit_sub_y_range = False,
      kin_num_decades_to_fit   = 6,
      bool_hide_updates        = False
    ):
    ## extract the number of plt-files per eddy-turnover-time from 'Turb.log'
    plots_per_eddy = LoadFlashData.getPlotsPerEddy(self.filepath_data + "/../", bool_hide_updates=False)
    if plots_per_eddy is None:
      raise Exception("ERROR: # plt-files could not be read from 'Turb.log'.")
    print("\t> Loading spectra data...")
    ## load kinetic energy spectra
    list_kin_k_group_t, list_kin_power_group_t, list_kin_sim_times = LoadFlashData.loadListSpectra(
      filepath_data     = self.filepath_data,
      str_spectra_type  = "vel",
      plots_per_eddy    = plots_per_eddy,
      bool_hide_updates = bool_hide_updates
    )
    ## load magnetic energy spectra
    list_mag_k_group_t, list_mag_power_group_t, list_mag_sim_times = LoadFlashData.loadListSpectra(
      filepath_data     = self.filepath_data,
      str_spectra_type  = "mag",
      plots_per_eddy    = plots_per_eddy,
      bool_hide_updates = bool_hide_updates
    )
    print("\t> Fitting spectra data...")
    ## fit kinetic energy spectra
    kin_fit_obj = FitMHDScales.FitKinSpectra(
      list_sim_times       = list_kin_sim_times,
      list_k_group_t       = list_kin_k_group_t,
      list_power_group_t   = list_kin_power_group_t,
      bool_fit_fixed_model = bool_kin_fit_fixed_model,
      k_start              = 3, # exclude driving modes
      k_fit_from           = 7,
      k_step_size          = 1,
      bool_fit_sub_y_range = bool_kin_fit_sub_y_range,
      num_decades_to_fit   = kin_num_decades_to_fit,
      bool_hide_updates    = bool_hide_updates
    )
    ## fit magnetic energy spectra
    mag_fit_obj = FitMHDScales.FitMagSpectra(
      list_sim_times       = list_mag_sim_times,
      list_k_group_t       = list_mag_k_group_t,
      list_power_group_t   = list_mag_power_group_t,
      bool_fit_fixed_model = bool_mag_fit_fixed_model,
      k_start              = 1,
      k_fit_from           = 5,
      k_step_size          = 1,
      bool_hide_updates    = bool_hide_updates
    )
    ## extract spectra fit parameters
    kin_fit_dict = kin_fit_obj.getFitDict()
    mag_fit_dict = mag_fit_obj.getFitDict()
    ## store siulation parameters in a dictionary
    sim_fit_dict = {
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
    self.spectra_fits_obj = FitMHDScales.SpectraFit(
      **sim_fit_dict,
      **kin_fit_dict,
      **mag_fit_dict
    )
    ## save spectra-fit data in a json-file
    WWObjs.saveObj2Json(
      obj      = self.spectra_fits_obj,
      filepath = self.filepath_data,
      filename = self.filename_spectra_fits
    )
    print(" ")
  def loadSpectraFitsObj(
      self,
      bool_show_obj_attrs = False
    ):
    ## load spectra-fit data as a dictionary
    spectra_fits_dict = WWObjs.loadJson2Dict(
      filepath = self.filepath_data,
      filename = self.filename_spectra_fits
    )
    ## store dictionary data in spectra-fit object
    spectra_fits_obj = FitMHDScales.SpectraFit(**spectra_fits_dict)
    ## check if any simulation paarmeters need to be updated:
    ## assumed attribute should be updated if it is not 'None'
    bool_obj_was_updated  = False
    spectra_fits_obj_copy = copy.deepcopy(spectra_fits_obj)
    prev_dict = vars(spectra_fits_obj_copy) # copy of old attributes
    ## #########################################################
    ## UPDATE SIMULATION ATTRIBUTES STORED IN SPECTRA-FIT OBJECT
    ## #########################################################
    ## list of simulation attributes that can be updated
    bool_obj_was_updated |= WWObjs.updateObjAttr(spectra_fits_obj, "sim_suite",       self.sim_suite)
    bool_obj_was_updated |= WWObjs.updateObjAttr(spectra_fits_obj, "sim_label",       self.sim_label)
    bool_obj_was_updated |= WWObjs.updateObjAttr(spectra_fits_obj, "sim_res",         self.sim_res)
    bool_obj_was_updated |= WWObjs.updateObjAttr(spectra_fits_obj, "Re",              self.Re)
    bool_obj_was_updated |= WWObjs.updateObjAttr(spectra_fits_obj, "Rm",              self.Rm)
    bool_obj_was_updated |= WWObjs.updateObjAttr(spectra_fits_obj, "Pm",              self.Rm)
    bool_obj_was_updated |= WWObjs.updateObjAttr(spectra_fits_obj, "kin_fit_start_t", self.kin_fit_start_time)
    bool_obj_was_updated |= WWObjs.updateObjAttr(spectra_fits_obj, "mag_fit_start_t", self.mag_fit_start_time)
    bool_obj_was_updated |= WWObjs.updateObjAttr(spectra_fits_obj, "kin_fit_end_t",   self.kin_fit_end_time)
    bool_obj_was_updated |= WWObjs.updateObjAttr(spectra_fits_obj, "mag_fit_end_t",   self.mag_fit_end_time)
    new_dict = vars(spectra_fits_obj) # updated attributes
    ## ########################################
    ## CHECK WHICH ATTRIBUTES HAVE BEEN UPDATED
    ## ########################################
    ## if any of the attributes have been updated
    if bool_obj_was_updated:
      ## save the updated spectra object
      WWObjs.saveObj2Json(
        obj      = spectra_fits_obj,
        filepath = self.filepath_data,
        filename = self.filename_spectra_fits
      )
      ## keep a list of updated attributes
      list_updated_attrs = []
      for prev_attr_name, new_attr_name in zip(prev_dict, new_dict):
        ## don't compare list entries
        if not isinstance(prev_dict[prev_attr_name], list):
          ## note attribute name if it has been updated
          if not(prev_dict[prev_attr_name] == new_dict[new_attr_name]):
            list_updated_attrs.append(prev_attr_name)
      ## print updated attributes
      print("\t> Updated attributes:", ", ".join(list_updated_attrs))
    ## ###################################
    ## PRINT ALL SPECTRA OBJECT ATTRIBUTES
    ## ###################################
    if bool_show_obj_attrs:
      print("\t> Object attributes:")
      ## print each paramater name and value stored in the spectra object
      for attr_name in new_dict:
        if isinstance(new_dict[attr_name], list):
          print( # the attribute is a list
            ("\t\t> {}".format(attr_name)).ljust(35),
            "type: {}, # of entries: {}".format(
              type(new_dict[attr_name]),
              len(new_dict[attr_name])
            )
          )
        else:
          print( # the attribute is a value of sorts
            ("\t\t> {}".format(attr_name)).ljust(35),
            "= {}".format(new_dict[attr_name])
          )
    print(" ")
    ## store the updated spectra object
    self.spectra_fits_obj = spectra_fits_obj
  def plotSpectraFits(
      self,
      filepath_vis,
      filepath_vis_frames,
      plot_spectra_from  = 1,
      plot_spectra_every = 1,
      bool_hide_updates  = False
    ):
    ## create plotting object looking at simulation fit
    spectra_plot_obj = PlotSpectra.PlotSpectraFit(self.spectra_fits_obj)
    ## plot spectra evolution
    spectra_plot_obj.plotSpectraEvolution(
      filepath_plot          = filepath_vis_frames,
      plot_index_start       = plot_spectra_from,
      plot_index_step        = plot_spectra_every,
      bool_delete_old_frames = True,
      bool_hide_updates      = bool_hide_updates
    )
    print(" ")
    ## animate spectra evolution
    spectra_plot_obj.aniSpectra(
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
  parser = WWArgparse.MyParser(description="Fit kinetic and magnetic energy spectra.")
  ## ------------------- DEFINE OPTIONAL ARGUMENTS
  args_opt = parser.add_argument_group(description="Optional processing arguments:")
  ## program workflow parameters
  args_opt.add_argument("-v", "--verbose",        **WWArgparse.opt_bool_arg)
  args_opt.add_argument("-s", "--show_obj_attrs", **WWArgparse.opt_bool_arg)
  args_opt.add_argument("-f", "--fit_spectra",    **WWArgparse.opt_bool_arg)
  args_opt.add_argument("-p", "--plot_spectra",   **WWArgparse.opt_bool_arg)
  ## directory information
  args_opt.add_argument("-vis_folder",  type=str, default="vis_folder", **WWArgparse.opt_arg)
  args_opt.add_argument("-data_folder", type=str, default="spect",      **WWArgparse.opt_arg)
  ## fit fixed spectra models
  args_opt.add_argument("-kin_fit_fixed", **WWArgparse.opt_bool_arg)
  args_opt.add_argument("-mag_fit_fixed", **WWArgparse.opt_bool_arg)
  ## energy range to fit kinetic energy spectra
  args_opt.add_argument("-kin_fit_sub_y_range",    **WWArgparse.opt_bool_arg)
  args_opt.add_argument("-kin_num_decades_to_fit", type=float, default=6, **WWArgparse.opt_arg)
  ## time range to fit spectra
  args_opt.add_argument("-kin_start_fit", type=float, default=None, **WWArgparse.opt_arg)
  args_opt.add_argument("-mag_start_fit", type=float, default=None, **WWArgparse.opt_arg)
  args_opt.add_argument("-kin_end_fit",   type=float, default=None, **WWArgparse.opt_arg)
  args_opt.add_argument("-mag_end_fit",   type=float, default=None, **WWArgparse.opt_arg)
  ## plotting parameters
  args_opt.add_argument("-plot_spectra_from",  type=int, default=0, **WWArgparse.opt_arg)
  args_opt.add_argument("-plot_spectra_every", type=int, default=1, **WWArgparse.opt_arg)
  ## simulation information
  args_opt.add_argument("-sim_suite", type=str,   default=None, **WWArgparse.opt_arg)
  args_opt.add_argument("-sim_res",   type=int,   default=None, **WWArgparse.opt_arg)
  args_opt.add_argument("-Re",        type=float, default=None, **WWArgparse.opt_arg)
  args_opt.add_argument("-Rm",        type=float, default=None, **WWArgparse.opt_arg)
  args_opt.add_argument("-Pm",        type=float, default=None, **WWArgparse.opt_arg)
  ## ------------------- DEFINE REQUIRED ARGUMENTS
  args_req = parser.add_argument_group(description="Required processing arguments:")
  args_req.add_argument("-suite_path", type=str, required=True, help="type: %(type)s")
  args_req.add_argument("-sim_folder", type=str, required=True, help="type: %(type)s")

  ## #########################
  ## INTERPRET INPUT ARGUMENTS
  ## #########################
  ## ---------------------------- OPEN ARGUMENTS
  args = vars(parser.parse_args())
  ## ---------------------------- SAVE PARAMETERS
  ## (boolean) workflow parameters
  bool_hide_updates        = not(args["verbose"])
  bool_show_obj_attrs      = args["show_obj_attrs"]
  bool_fit_spectra         = args["fit_spectra"]
  bool_plot_spectra        = args["plot_spectra"]
  ## fit fixed spectra models
  bool_kin_fit_fixed_model = args["kin_fit_fixed"]
  bool_mag_fit_fixed_model = args["mag_fit_fixed"]
  ## energy range to fit kinetic energy spectra
  bool_kin_fit_sub_y_range = args["kin_fit_sub_y_range"]
  kin_num_decades_to_fit   = args["kin_num_decades_to_fit"]
  ## time range to fit spectra
  kin_fit_start_time       = args["kin_start_fit"]
  mag_fit_start_time       = args["mag_start_fit"]
  kin_fit_end_time         = args["kin_end_fit"]
  mag_fit_end_time         = args["mag_end_fit"]
  ## plotting parameters
  plot_spectra_from        = args["plot_spectra_from"]
  plot_spectra_every       = args["plot_spectra_every"]
  ## important directory information
  filepath_suite           = args["suite_path"]
  folder_vis               = args["vis_folder"]
  folder_data              = args["data_folder"]
  sim_suite                = args["sim_suite"]
  sim_label                = args["sim_folder"]
  sim_res                  = args["sim_res"]
  ## simulation parameters
  Re                       = args["Re"]
  Rm                       = args["Rm"]
  Pm                       = args["Pm"]

  ## ######################
  ## INITIALISING VARIABLES
  ## ######################
  ## define the file-name where spectra fit parameters are stored
  filename_spectra_fits = "spectra_fits"
  if bool_kin_fit_fixed_model:
    filename_spectra_fits += "_fk"
  if bool_mag_fit_fixed_model:
    filename_spectra_fits += "_fm"
  filename_spectra_fits += ".json"
  ## check if any pair of plasma Reynolds numbers have not been defined
  if bool_fit_spectra:
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
  filepath_data = WWFnF.createFilepath([ filepath_suite, sim_label, folder_data ])
  ## folder where visualisations will be saved
  filepath_vis = WWFnF.createFilepath([ filepath_suite, folder_vis ])
  ## folder where spectra plots will be saved
  sub_folder_vis      = "plotSpectraFits"
  filepath_vis_frames = WWFnF.createFilepath([ filepath_vis, sub_folder_vis ])

  ## ##############
  ## CREATE FOLDERS
  ## ##############
  if bool_plot_spectra:
    WWFnF.createFolder(filepath_vis)
    WWFnF.createFolder(filepath_vis_frames)

  ## ######################################
  ## PRINT SIMULATION PARAMETERS TO CONSOLE
  ## ######################################
  P2Term.printInfo("Suite filepath:", filepath_suite)
  P2Term.printInfo("Data filepath:",  filepath_data)
  P2Term.printInfo("Vis. filepath:",  filepath_vis)
  ## print simulation parameters
  if bool_fit_spectra:
    print("\t> sim suite: {}".format(sim_suite))
    print("\t> sim label: {}".format(sim_label))
    print("\t> sim resolution: {}".format(sim_res))
    print("\t> fit domain (kin): [{}, {}]".format(kin_fit_start_time, kin_fit_end_time))
    print("\t> fit domain (mag): [{}, {}]".format(mag_fit_start_time, mag_fit_end_time))
    print("\t> Re: {}, Rm: {}, Pm: {}".format(Re, Rm, Pm))
    print("\t> Fitting with {} kinetic energy spectra model.".format(
      "fixed" if bool_kin_fit_fixed_model else "complete"
    ))
    print("\t> Fitting with {} magnetic energy spectra model.".format(
      "fixed" if bool_mag_fit_fixed_model else "complete"
    ))
  ## check if the spectra-fit parameters have already been stored (in a json-file)
  else:
    try: WWObjs.loadJson2Dict(
      filepath = filepath_data,
      filename = filename_spectra_fits,
      bool_hide_updates = True
    )
    except: raise Exception("Error: '{}' does not exist.".format(filename_spectra_fits))
  print(" ")

  ## #########################
  ## LOAD FITTED / FIT SPECTRA
  ## #########################
  spec_obj = SpectraObject(
    filepath_data         = filepath_data,
    filename_spectra_fits = filename_spectra_fits,
    sim_suite             = sim_suite,
    sim_label             = sim_label,
    sim_res               = sim_res,
    Re                    = Re,
    Rm                    = Rm,
    Pm                    = Pm,
    kin_fit_start_time    = kin_fit_start_time,
    mag_fit_start_time    = mag_fit_start_time,
    kin_fit_end_time      = kin_fit_end_time,
    mag_fit_end_time      = mag_fit_end_time
  )
  ## read and fit spectra data
  if bool_fit_spectra:
    spec_obj.createSpectraFitsObj(
      bool_kin_fit_fixed_model = bool_kin_fit_fixed_model,
      bool_mag_fit_fixed_model = bool_mag_fit_fixed_model,
      bool_kin_fit_sub_y_range = bool_kin_fit_sub_y_range,
      kin_num_decades_to_fit   = kin_num_decades_to_fit,
      bool_hide_updates        = bool_hide_updates
    )
  ## read in already fitted spectra
  else: spec_obj.loadSpectraFitsObj(bool_show_obj_attrs)

  ## #############################
  ## PLOT EVOLUTION OF THE SPECTRA
  ## #############################
  if bool_plot_spectra:
    spec_obj.plotSpectraFits(
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
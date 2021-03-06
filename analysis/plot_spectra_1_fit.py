#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys, copy

## 'tmpfile' needs to be loaded before 'matplotlib'.
## This is so matplotlib stores cache in a temporary directory.
## (Useful for plotting in parallel)
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
import matplotlib.pyplot as plt

## suppress "optimise" warnings
import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter("ignore", OptimizeWarning)

## load user defined modules
from TheUsefulModule import WWArgparse, WWObjs, WWFnF, P2Term
from TheLoadingModule import LoadFlashData
from TheFittingModule import FitMHDScales
from ThePlottingModule import PlotSpectra


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
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
      kin_fit_time_start, kin_fit_time_end,
      mag_fit_time_start, mag_fit_time_end,
      bool_debug
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
    self.kin_fit_time_start    = kin_fit_time_start
    self.mag_fit_time_start    = mag_fit_time_start
    self.kin_fit_time_end      = kin_fit_time_end
    self.mag_fit_time_end      = mag_fit_time_end
    ## debug status
    self.bool_debug            = bool_debug

  def createSpectraFitsObj(
      self,
      kin_bool_fit_fixed_model = False,
      mag_bool_fit_fixed_model = False,
      bool_kin_fit_sub_y_range = False,
      kin_num_decades_to_fit   = 6,
      bool_hide_updates        = False
    ):
    ## extract the number of plt-files per eddy-turnover-time from 'Turb.log'
    plots_per_eddy = LoadFlashData.getPlotsPerEddy(self.filepath_data + "/../", bool_hide_updates=False)
    if plots_per_eddy is None:
      raise Exception("ERROR: # plt-files could not be read from 'Turb.log'.")
    print("Loading spectra data...")
    ## load kinetic energy spectra
    list_kin_k_group_t, list_kin_power_group_t, list_kin_list_sim_times = LoadFlashData.loadListSpectra(
      filepath_data     = self.filepath_data,
      str_spectra_type  = "vel",
      file_start_time   = 5,
      read_every        = 25 if self.bool_debug else 1,
      plots_per_eddy    = plots_per_eddy,
      bool_hide_updates = bool_hide_updates
    )
    ## load magnetic energy spectra
    list_mag_k_group_t, list_mag_power_group_t, list_mag_list_sim_times = LoadFlashData.loadListSpectra(
      filepath_data     = self.filepath_data,
      str_spectra_type  = "mag",
      file_start_time   = 5,
      read_every        = 25 if self.bool_debug else 1,
      plots_per_eddy    = plots_per_eddy,
      bool_hide_updates = bool_hide_updates
    )
    print(" ")
    print("Fitting spectra data...")
    ## fit magnetic energy spectra
    mag_fit_obj = FitMHDScales.FitMagSpectra(
      list_sim_times       = list_mag_list_sim_times,
      list_k_group_t       = list_mag_k_group_t,
      list_power_group_t   = list_mag_power_group_t,
      bool_fit_fixed_model = mag_bool_fit_fixed_model,
      k_index_fit_from     = 0,
      k_index_break_from   = 5,
      k_index_break_step   = 1,
      bool_hide_updates    = bool_hide_updates
    )
    ## fit kinetic energy spectra
    kin_fit_obj = FitMHDScales.FitKinSpectra(
      list_sim_times       = list_kin_list_sim_times,
      list_k_group_t       = list_kin_k_group_t,
      list_power_group_t   = list_kin_power_group_t,
      bool_fit_fixed_model = kin_bool_fit_fixed_model,
      k_index_fit_from     = 3, # exclude driving modes: k > 4
      k_index_break_from   = 5, # provide enough degrees of freedom
      k_index_break_step   = 1,
      bool_fit_sub_y_range = bool_kin_fit_sub_y_range,
      num_decades_to_fit   = kin_num_decades_to_fit,
      bool_hide_updates    = bool_hide_updates
    )
    ## extract spectra fit parameters
    kin_fit_dict = kin_fit_obj.getFitDict()
    mag_fit_dict = mag_fit_obj.getFitDict()
    ## store siulation parameters in a dictionary
    sim_fit_dict = {
      "sim_suite":          self.sim_suite,
      "sim_label":          self.sim_label,
      "sim_res":            self.sim_res,
      "Re":                 self.Re,
      "Rm":                 self.Rm,
      "Pm":                 self.Pm,
      "kin_fit_time_start": self.kin_fit_time_start,
      "mag_fit_time_start": self.mag_fit_time_start,
      "kin_fit_time_end":   self.kin_fit_time_end,
      "mag_fit_time_end":   self.mag_fit_time_end
    }
    ## create spectra-object
    self.fits_obj = FitMHDScales.SpectraFit(
      **sim_fit_dict,
      **kin_fit_dict,
      **mag_fit_dict
    )
    ## save spectra-fit data in a json-file
    WWObjs.saveObj2Json(
      obj      = self.fits_obj,
      filepath = self.filepath_data,
      filename = self.filename_spectra_fits
    )
    print(" ")

  def loadSpectraFitsObj(
      self,
      bool_show_obj_attrs = False
    ):
    ## load spectra-fit data as a dictionary
    try:
      fits_dict = WWObjs.loadJson2Dict(
        filepath = self.filepath_data,
        filename = self.filename_spectra_fits
      )
    except:
      raise Exception(f"Error: '{self.filename_spectra_fits}' does not exist.")
    ## store dictionary data in spectra-fit object
    fits_obj = FitMHDScales.SpectraFit(**fits_dict)
    ## check whether any simulation parameters need to be updated:
    ## it is assumed that an attribute should be updated if its value is not 'None'
    bool_obj_was_updated  = False
    fits_obj_copy = copy.deepcopy(fits_obj)
    old_dict = vars(fits_obj_copy) # create a copy of the old attribute values
    ## #########################################################
    ## UPDATE SIMULATION ATTRIBUTES STORED IN SPECTRA-FIT OBJECT
    ## #########################################################
    ## list of simulation attributes that can be updated
    bool_obj_was_updated |= WWObjs.updateObjAttr(fits_obj, "sim_suite",          self.sim_suite)
    bool_obj_was_updated |= WWObjs.updateObjAttr(fits_obj, "sim_label",          self.sim_label)
    bool_obj_was_updated |= WWObjs.updateObjAttr(fits_obj, "sim_res",            self.sim_res)
    bool_obj_was_updated |= WWObjs.updateObjAttr(fits_obj, "Re",                 self.Re)
    bool_obj_was_updated |= WWObjs.updateObjAttr(fits_obj, "Rm",                 self.Rm)
    bool_obj_was_updated |= WWObjs.updateObjAttr(fits_obj, "Pm",                 self.Pm)
    bool_obj_was_updated |= WWObjs.updateObjAttr(fits_obj, "kin_fit_time_start", self.kin_fit_time_start)
    bool_obj_was_updated |= WWObjs.updateObjAttr(fits_obj, "mag_fit_time_start", self.mag_fit_time_start)
    bool_obj_was_updated |= WWObjs.updateObjAttr(fits_obj, "kin_fit_time_end",   self.kin_fit_time_end)
    bool_obj_was_updated |= WWObjs.updateObjAttr(fits_obj, "mag_fit_time_end",   self.mag_fit_time_end)
    new_dict = vars(fits_obj) # updated attributes
    ## ########################################
    ## CHECK WHICH ATTRIBUTES HAVE BEEN UPDATED
    ## ########################################
    ## if any of the attributes have been updated
    if bool_obj_was_updated:
      ## save the updated spectra object
      WWObjs.saveObj2Json(
        obj      = fits_obj,
        filepath = self.filepath_data,
        filename = self.filename_spectra_fits
      )
      ## keep a list of updated attributes
      list_updated_attrs = []
      for old_attr_name, new_attr_name in zip(old_dict, new_dict):
        ## don't compare list entries
        if not isinstance(old_dict[old_attr_name], list):
          ## note attribute name if it has been updated
          if not(old_dict[old_attr_name] == new_dict[new_attr_name]):
            list_updated_attrs.append(old_attr_name)
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
    self.fits_obj = fits_obj

  def plotSpectraFits(
      self,
      filepath_vis,
      filepath_vis_frames,
      plot_spectra_from  = 1,
      plot_spectra_every = 1,
      bool_hide_updates  = False
    ):
    print("Plotting energy spectra...")
    ## create plotting object looking at simulation fit
    spectra_plot_obj = PlotSpectra.PlotSpectraFit(self.fits_obj)
    ## remove old frames
    list_filenames_to_delete = WWFnF.getFilesFromFolder(
      filepath     = filepath_vis_frames, 
      str_contains = f"_{self.fits_obj.sim_label}_"
    )
    if len(list_filenames_to_delete) > 0:
      print("\t> Removing old spectra frames...")
      for filename in list_filenames_to_delete:
        os.system(f"rm {filepath_vis_frames}/{filename}")
    ## plot both energy spectra
    print("\t> Plotting kinetic and magnetic energy spectra...")
    spectra_plot_obj.plotSpectraEvolution(
      filepath_frames        = filepath_vis_frames,
      filepath_movie         = filepath_vis,
      plot_index_start       = plot_spectra_from,
      plot_index_step        = plot_spectra_every,
      bool_plot_kin          = True,
      bool_plot_mag          = True,
      bool_adjust_y_axis     = True,
      bool_hide_updates      = bool_hide_updates
    )
    ## plot magnetic energy spectra only
    print("\t> Plotting magnetic energy spectra only...")
    spectra_plot_obj.plotSpectraEvolution(
      filepath_frames        = filepath_vis_frames,
      filepath_movie         = filepath_vis,
      plot_index_start       = plot_spectra_from,
      plot_index_step        = plot_spectra_every,
      bool_plot_kin          = False,
      bool_plot_mag          = True,
      bool_adjust_y_axis     = False,
      bool_hide_updates      = bool_hide_updates
    )
    ## plot magnetic energy spectra only
    print("\t> Plotting kinetic energy spectra only...")
    spectra_plot_obj.plotSpectraEvolution(
      filepath_frames        = filepath_vis_frames,
      filepath_movie         = filepath_vis,
      plot_index_start       = plot_spectra_from,
      plot_index_step        = plot_spectra_every,
      bool_plot_kin          = True,
      bool_plot_mag          = False,
      bool_adjust_y_axis     = False,
      bool_hide_updates      = bool_hide_updates
    )


## ###############################################################
## MAIN PROGRAM
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
  args_opt.add_argument("-d", "--debug",          **WWArgparse.opt_bool_arg)
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
  args_opt.add_argument("-sim_suite",  type=str,   default=None, **WWArgparse.opt_arg)
  args_opt.add_argument("-sim_res",    type=int,   default=None, **WWArgparse.opt_arg)
  args_opt.add_argument("-Re",         type=float, default=None, **WWArgparse.opt_arg)
  args_opt.add_argument("-Rm",         type=float, default=None, **WWArgparse.opt_arg)
  args_opt.add_argument("-Pm",         type=float, default=None, **WWArgparse.opt_arg)
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
  bool_debug               = args["debug"]
  bool_show_obj_attrs      = args["show_obj_attrs"]
  bool_fit_spectra         = args["fit_spectra"]
  bool_plot_spectra        = args["plot_spectra"]
  ## fit fixed spectra models
  kin_bool_fit_fixed_model = args["kin_fit_fixed"]
  mag_bool_fit_fixed_model = args["mag_fit_fixed"]
  ## energy range to fit kinetic energy spectra
  bool_kin_fit_sub_y_range = args["kin_fit_sub_y_range"]
  kin_num_decades_to_fit   = args["kin_num_decades_to_fit"]
  ## time range to fit spectra
  kin_fit_time_start       = args["kin_start_fit"]
  mag_fit_time_start       = args["mag_start_fit"]
  kin_fit_time_end         = args["kin_end_fit"]
  mag_fit_time_end         = args["mag_end_fit"]
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
  filename_spectra_fits = "spectra_fits{}{}.json".format(
    "_fk" if kin_bool_fit_fixed_model else "",
    "_fm" if mag_bool_fit_fixed_model else ""
  )
  ## check if any pair of plasma Reynolds numbers have not been defined
  bool_missing_plasma_numbers = (
    ( (Re == None) and (Pm == None) ) or
    ( (Rm == None) and (Pm == None) ) or
    ( (Re == None) and (Rm == None) )
  )
  if bool_fit_spectra or not(bool_missing_plasma_numbers):
    if bool_missing_plasma_numbers:
      raise Exception("Error: Undefined plasma-Reynolds numbers. Need to define two of the following: 'Re', 'Rm', and 'Pm'.")
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
  filepath_vis_frames = WWFnF.createFilepath([ filepath_vis, "plotSpectraFits" ])

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
    print(f"\t> sim suite: {sim_suite}")
    print(f"\t> sim label: {sim_label}")
    print(f"\t> sim resolution: {sim_res}")
    print(f"\t> fit domain (kin): [{kin_fit_time_start}, {kin_fit_time_end}]")
    print(f"\t> fit domain (mag): [{mag_fit_time_start}, {mag_fit_time_end}]")
    print(f"\t> Re: {Re}, Rm: {Rm}, Pm: {Pm}")
    print("\t> Fitting with the {} kinetic energy spectra model.".format(
      "fixed" if kin_bool_fit_fixed_model else "full"
    ))
    print("\t> Fitting with the {} magnetic energy spectra model.".format(
      "fixed" if mag_bool_fit_fixed_model else "full"
    ))
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
    kin_fit_time_start    = kin_fit_time_start,
    mag_fit_time_start    = mag_fit_time_start,
    kin_fit_time_end      = kin_fit_time_end,
    mag_fit_time_end      = mag_fit_time_end,
    bool_debug            = bool_debug
  )
  ## read and fit spectra data
  if bool_fit_spectra:
    spec_obj.createSpectraFitsObj(
      kin_bool_fit_fixed_model = kin_bool_fit_fixed_model,
      mag_bool_fit_fixed_model = mag_bool_fit_fixed_model,
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
## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import os, functools
import multiprocessing as mproc
import concurrent.futures as cfut

## import user defined modules
from TheUsefulModule import WWFnF, WWTerminal, WWVariables, WWObjs
from TheFlashModule import LoadData, FileNames
from ThePlottingModule import PlotFuncs


## ###############################################################
## CREATE STRINGS FROM SIMULATION PARAMETERS
## ###############################################################
def getSonicRegime(Mach):
  if Mach < 1: return f"Mach{float(Mach):.1f}"
  return f"Mach{int(Mach):d}"

def getJobTag(dict_sim_inputs, job_name):
  sonic_regime = getSonicRegime(dict_sim_inputs["desired_Mach"])
  suite_folder = dict_sim_inputs["suite_folder"]
  sim_folder   = dict_sim_inputs["sim_folder"]
  sim_res      = dict_sim_inputs["sim_res"]
  return f"{sonic_regime}{suite_folder}{sim_folder}{job_name}{sim_res}"

def getSimName(dict_sim_inputs):
  return "{}_{}_{}".format(
    dict_sim_inputs["sonic_regime"],
    dict_sim_inputs["suite_folder"],
    dict_sim_inputs["sim_folder"]
  )


## ###############################################################
## CREATE A LIST OF SIMULATION DIRECTORIES
## ###############################################################
def getListOfSimFilepaths(basepath, list_suite_folders, list_sonic_regimes, list_sim_folders, list_sim_res):
  list_filepath_sim_res = []
  ## LOOK AT EACH SIMULATION SUITE
  ## -----------------------------
  for suite_folder in list_suite_folders:
    ## LOOK AT EACH SONIC REGIME
    ## -------------------------
    for sonic_regime in list_sonic_regimes:
      ## LOOK AT EACH SIMULATION FOLDER
      ## -----------------------------
      for sim_folder in list_sim_folders:
        ## CHECK THE SIMULATION CONFIGURATION EXISTS
        ## -----------------------------------------
        filepath_sim = WWFnF.createFilepath([
          basepath, suite_folder, sonic_regime, sim_folder
        ])
        if not os.path.exists(filepath_sim): continue
        ## loop over the different resolution runs
        for sim_res in list_sim_res:
          ## CHECK THE RESOLUTION RUN EXISTS
          ## -------------------------------
          filepath_sim_res = f"{filepath_sim}/{sim_res}/"
          if not os.path.exists(filepath_sim_res): continue
          ## store for looking at later
          list_filepath_sim_res.append(filepath_sim_res)
  return list_filepath_sim_res


## ###############################################################
## APPLY FUNCTION OVER ALL SIMULATIONS
## ###############################################################
def callFuncForAllSimulations(
    func,
    basepath, list_suite_folders, list_sonic_regimes, list_sim_folders, list_sim_res,
    bool_mproc      = False,
    bool_check_only = False
  ):
  list_filepath_sim_res = getListOfSimFilepaths(
    basepath           = basepath,
    list_suite_folders = list_suite_folders,
    list_sonic_regimes = list_sonic_regimes,
    list_sim_folders   = list_sim_folders,
    list_sim_res       = list_sim_res
  )
  if bool_mproc:
    print(f"Looking at {len(list_filepath_sim_res)} simulation(s):")
    [
      print("\t> " + filepath_sim_res)
      for filepath_sim_res in list_filepath_sim_res
    ]
    print(" ")
    print("Processing...")
    with cfut.ProcessPoolExecutor() as executor:
      manager = mproc.Manager()
      lock = manager.Lock()
      ## loop over all simulation folders
      futures = [
        executor.submit(
          functools.partial(
            func,
            lock            = lock,
            bool_check_only = bool_check_only,
            bool_verbose    = False
          ),
          filepath_sim_res
        ) for filepath_sim_res in list_filepath_sim_res
      ]
      ## wait to ensure that all scheduled and running tasks have completed
      cfut.wait(futures)
      ## check if any tasks failed
      for future in cfut.as_completed(futures): future.result()
    print("Finished processing.")
  else: [
    func(
      filepath_sim_res = filepath_sim_res,
      bool_check_only  = bool_check_only,
      bool_verbose     = True
    )
    for filepath_sim_res in list_filepath_sim_res
  ]


## ###############################################################
## PLOT SIMULATION DETAILS
## ###############################################################
def addLabel_simInputs(
    fig, ax,
    dict_sim_inputs = None,
    filepath        = None,
    bbox            = (0,0),
    vpos            = (0.05, 0.05),
    bool_show_res   = True
  ):
  ## load simulation parameters
  if dict_sim_inputs is None:
    if filepath is None:
      raise Exception("Error: need to pass details about simulation inputs")
    dict_sim_inputs = readSimInputs(filepath)
  ## annotate simulation parameters
  PlotFuncs.addBoxOfLabels(
    fig, ax,
    bbox        = bbox,
    xpos        = vpos[0],
    ypos        = vpos[1],
    alpha       = 0.5,
    fontsize    = 18,
    list_labels = [
      r"${\rm N}_{\rm res} = $ " + "{:d}".format(int(dict_sim_inputs["sim_res"])) if bool_show_res else "",
      r"${\rm Re} = $ " + "{:d}".format(int(dict_sim_inputs["Re"])),
      r"${\rm Rm} = $ " + "{:d}".format(int(dict_sim_inputs["Rm"])),
      r"${\rm Pm} = $ " + "{:d}".format(int(dict_sim_inputs["Pm"])),
    ]
  )


## ###############################################################
## SAVE / READ SIMULALATION INPUT / OUTPUT FILES
## ###############################################################
def saveSimInputs(obj_sim_params, filepath):
  WWObjs.saveObj2JsonFile(
    obj      = obj_sim_params,
    filepath = filepath,
    filename = FileNames.FILENAME_SIM_INPUTS
  )

def readSimInputs(filepath, bool_verbose=True):
  dict_sim_inputs = WWObjs.readJsonFile2Dict(
    filepath     = filepath,
    filename     = FileNames.FILENAME_SIM_INPUTS,
    bool_verbose = bool_verbose
  )
  return dict_sim_inputs

def readSimOutputs(filepath, bool_verbose=True):
  dict_sim_outputs = WWObjs.readJsonFile2Dict(
    filepath     = filepath,
    filename     = FileNames.FILENAME_SIM_OUTPUTS,
    bool_verbose = bool_verbose
  )
  return dict_sim_outputs

def createSimInputs(
    filepath, suite_folder, sim_folder, sim_res, k_turb, desired_Mach,
    Re = None,
    Rm = None,
    Pm = None
  ):
  ## check that a valid driving scale is defined
  if k_turb is None: raise Exception(f"Error: you have provided a invalid driving scale = {k_turb}")
  ## number of cells per block that the flash4-exe was compiled with
  if   sim_res in [ "1152" ]:              num_blocks = [ 48, 48, 72 ]
  elif sim_res in [ "144", "288", "576" ]: num_blocks = [ 36, 36, 48 ]
  elif sim_res in [ "36", "72" ]:          num_blocks = [ 12, 12, 18 ]
  elif sim_res in [ "18" ]:                num_blocks = [ 6, 6, 6 ]
  num_procs = [
    int(int(sim_res) / num_blocks_in_dir)
    for num_blocks_in_dir in num_blocks
  ]
  ## create object to define simulation input parameters
  obj_sim_params = SimInputParams(
    suite_folder = suite_folder,
    sim_folder   = sim_folder,
    sim_res      = sim_res,
    num_blocks   = num_blocks,
    num_procs    = num_procs,
    k_turb       = k_turb,
    desired_Mach = desired_Mach,
    Re           = Re,
    Rm           = Rm,
    Pm           = Pm
  )
  obj_sim_params.defineParams()
  ## save input file
  saveSimInputs(obj_sim_params, filepath)
  return obj_sim_params


## ###############################################################
## COMPUTE ALL RELEVANT SIMULATION PARAMETERS
## ###############################################################
class SimInputParams():
  def __init__(
      self,
      suite_folder, sim_folder, sim_res, num_blocks, num_procs, k_turb, desired_Mach,
      t_turb = None,
      Re     = None,
      Rm     = None,
      Pm     = None,
      **kwargs # unused arguments
    ):
    ## required parameters
    self.suite_folder   = suite_folder
    self.sim_folder     = sim_folder
    self.sim_res        = sim_res
    self.num_blocks     = num_blocks
    self.num_procs      = num_procs
    self.k_turb         = k_turb
    self.desired_Mach   = desired_Mach
    self.max_num_t_turb = 100
    ## parameters that (may) need to be computed
    self.t_turb         = t_turb
    self.Re             = Re
    self.Rm             = Rm
    self.Pm             = Pm
    ## parameters that need to be computed
    self.sonic_regime   = None
    self.nu             = None
    self.eta            = None

  def defineParams(self):
    ## check input parameters are of the right type
    WWVariables.assertType("suite_folder", self.suite_folder, str)
    WWVariables.assertType("sim_folder",   self.sim_folder,   str)
    WWVariables.assertType("sim_res",      self.sim_res,      str)
    WWVariables.assertType("num_blocks",   self.num_blocks,   list)
    WWVariables.assertType("k_turb",       self.k_turb,       (int, float))
    WWVariables.assertType("Mach",         self.desired_Mach, (int, float))
    ## perform routines
    self.__defineSonicRegime()
    self.__definePlasmaParameters()
    self.__checkSimParamsDefined()
    self.__roundParams()

  def __defineSonicRegime(self):
    ## t_turb = ell_turb / (Mach * c_s)
    self.t_turb = 1 / (self.k_turb * self.desired_Mach)
    self.sonic_regime = getSonicRegime(self.desired_Mach)

  def __definePlasmaParameters(self):
    dict_params = LoadData.computePlasmaConstants(
      Mach   = self.desired_Mach,
      k_turb = self.k_turb,
      Re     = self.Re,
      Rm     = self.Rm,
      Pm     = self.Pm
    )
    self.nu  = dict_params["nu"]
    self.eta = dict_params["eta"]
    self.Re  = dict_params["Re"]
    self.Rm  = dict_params["Rm"]
    self.Pm  = dict_params["Pm"]

  def __checkSimParamsDefined(self):
    list_check_params_defined = [
      "sonic_regime" if self.sonic_regime is None else "",
      "t_turb"       if self.t_turb       is None else "",
      "Re"           if self.Re           is None else "",
      "Rm"           if self.Rm           is None else "",
      "Pm"           if self.Pm           is None else "",
      "nu"           if self.nu           is None else "",
      "eta"          if self.eta          is None else ""
    ]
    list_params_not_defined = [
      param_name
      for param_name in list_check_params_defined
      ## remove entry if it is empty
      if len(param_name) > 0
    ]
    if len(list_params_not_defined) > 0:
      raise Exception(f"Error: You have not defined the following parameter(s):", list_params_not_defined)

  def __roundParams(self, num_decimals=5):
    ## round numeric parameter values
    self.k_turb       = round(self.k_turb,       num_decimals)
    self.desired_Mach = round(self.desired_Mach, num_decimals)
    self.t_turb       = round(self.t_turb,       num_decimals)
    self.Re           = round(self.Re,           num_decimals)
    self.Rm           = round(self.Rm,           num_decimals)
    self.Pm           = round(self.Pm,           num_decimals)
    self.nu           = round(self.nu,           num_decimals)
    self.eta          = round(self.eta,          num_decimals)


## END OF LIBRARY
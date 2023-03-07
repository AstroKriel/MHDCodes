## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
from TheUsefulModule import WWVariables, WWObjs
from TheLoadingModule import LoadFlashData
from ThePlottingModule import PlotFuncs

## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def addLabel_simInputs(
    fig, ax,
    filepath_sim_res,
    bbox          = (0,0),
    vpos          = (0.05, 0.05),
    bool_show_res = True
  ):
  ## load simulation parameters
  dict_sim_inputs = readSimInputs(filepath_sim_res)
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

def getSonicRegime(Mach):
  if   Mach > 1.0: sonic_regime = "super_sonic"
  elif Mach < 1.0: sonic_regime = "sub_sonic"
  else:            sonic_regime = "trans_sonic"
  return sonic_regime

def saveSimInputs(obj_sim_params, filepath):
  WWObjs.saveObj2JsonFile(
    obj      = obj_sim_params,
    filepath = filepath,
    filename = "sim_inputs.json"
  )

def readSimInputs(filepath, bool_verbose=True):
  dict_sim_inputs = WWObjs.readJsonFile2Dict(
    filepath          = filepath,
    filename          = "sim_inputs.json",
    bool_verbose = bool_verbose
  )
  return dict_sim_inputs

def readSimOutputs(filepath, bool_verbose=True):
  dict_sim_outputs = WWObjs.readJsonFile2Dict(
    filepath          = filepath,
    filename          = "sim_outputs.json",
    bool_verbose = bool_verbose
  )
  return dict_sim_outputs

def createSimInputs(
    filepath_sim_res, suite_folder, sim_folder, sim_res, k_turb, des_mach,
    Re = None,
    Rm = None,
    Pm = None
  ):
  ## number of cells per block that the flash4-exe was compiled with
  if sim_res in [ "144", "288", "576" ]:
    num_blocks = [ 36, 36, 48 ]
  elif sim_res in [ "36", "72" ]:
    num_blocks = [ 12, 12, 18 ]
  elif sim_res in [ "18" ]:
    num_blocks = [ 6, 6, 6 ]
  num_procs = [
    float(sim_res) / num_blocks_in_dir
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
    desired_Mach = des_mach,
    Re           = Re,
    Rm           = Rm,
    Pm           = Pm
  )
  obj_sim_params.defineParams()
  ## save input file
  saveSimInputs(obj_sim_params, filepath_sim_res)
  return obj_sim_params


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class SimInputParams():
  def __init__(
      self,
      suite_folder, sim_folder, sim_res, num_blocks, num_procs, k_turb, desired_Mach,
      sonic_regime = None,
      t_turb       = None,
      Re           = None,
      Rm           = None,
      Pm           = None,
      nu           = None,
      eta          = None,
      **kwargs # unused arguments
    ):
    ## required parameters
    self.suite_folder  = suite_folder
    self.sim_folder    = sim_folder
    self.sim_res       = sim_res
    self.num_blocks    = num_blocks
    self.num_procs     = num_procs
    self.k_turb        = k_turb
    self.desired_Mach  = desired_Mach
    ## parameters that (may) need to be computed
    self.sonic_regime  = sonic_regime
    self.t_turb        = t_turb
    self.Re            = Re
    self.Rm            = Rm
    self.Pm            = Pm
    self.nu            = nu
    self.eta           = eta

  def getSimParams(self):
    return {
      "suite_folder" : self.suite_folder,
      "sim_folder"   : self.sim_folder,
      "sim_res"      : self.sim_res,
      "num_blocks"   : self.num_blocks,
      "k_turb"       : self.k_turb,
      "desired_Mach" : self.desired_Mach,
      "sonic_regime" : self.sonic_regime,
      "t_turb"       : self.t_turb,
      "Re"           : self.Re,
      "Rm"           : self.Rm,
      "Pm"           : self.Pm,
      "nu"           : self.nu,
      "eta"          : self.eta
    }

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
    self.t_turb       = 1 / (self.k_turb * self.desired_Mach)
    self.sonic_regime = getSonicRegime(self.desired_Mach)

  def __definePlasmaParameters(self):
    dict_params = LoadFlashData.getPlasmaNumbers_fromInputs(
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


## ###############################################################
## TEST SUITE
## ###############################################################
class OtherClass():
  def __init__(self):
    self.a = 10

def checkAssert(obj_sim_params : SimInputParams):
  WWVariables.assertType("obj_sim_params", obj_sim_params, SimInputParams)
  a = 10

def tests():
  print("Running tests...")
  filepath_demo  = "/scratch/ek9/nk7952/"
  ## create input istance
  print("Creating input-params obj...")
  obj_sim_params = SimInputParams()
  obj_sim_params.defineParams(
    suite_folder = "Re10",
    sim_folder   = "Pm3",
    sim_res      = "288",
    num_blocks   = [ 36, 36, 48 ],
    k_turb       = 2.0,
    desired_Mach = 5.0,
    Re           = 25,
    Rm           = 100,
    # Pm           = 4
  )
  print(" ")
  ## check what parameters are stored in the obj
  print("Printing params stored (for reference)...")
  print(obj_sim_params.__dict__)
  print(" ")
  ## save and read obj
  saveSimInputs(obj_sim_params, filepath_demo)
  obj_read_params = readSimInputs(filepath_demo)
  print(" ")
  print("Printing params stored (in read obj)...")
  print(obj_read_params.__dict__)
  print(" ")
  # ## check type can be asserted
  # obj_other_class = OtherClass()
  # checkAssert(obj_sim_params)
  # checkAssert(obj_other_class)
  ## success
  print("all tests passed...")


## ###############################################################
## RUN TEST SUITE
## ###############################################################
if __name__ == "__main__":
  tests()


## END OF LIBRARY
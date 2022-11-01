## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
from TheUsefulModule import WWVariables, WWObjs


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def saveSimInputParams(obj_sim_params, filepath):
  WWObjs.saveObj2JsonFile(
    obj      = obj_sim_params,
    filepath = filepath,
    filename = "sim_inputs.json"
  )

def readSimInputParams(filepath):
  dict_input_params = WWObjs.loadJsonFile2Dict(
    filepath = filepath,
    filename = "sim_inputs.json"
  )
  obj_input_params = SimInputParams(**dict_input_params)
  return obj_input_params


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class SimInputParams():
  def __init__(
      self,
      suite_folder = None,
      sim_folder   = None,
      sim_res      = None,
      num_blocks   = None,
      k_turb       = None,
      desired_Mach = None,
      sonic_regime = None,
      t_turb       = None,
      Re           = None,
      Rm           = None,
      Pm           = None,
      nu           = None,
      eta          = None,
      **kwargs # unused arguments
    ):
    ## parameters that should be provided
    self.suite_folder  = suite_folder
    self.sim_folder    = sim_folder
    self.sim_res       = sim_res
    self.num_blocks    = num_blocks
    self.k_turb        = k_turb
    self.desired_Mach  = desired_Mach
    ## parameters that will need to be computed
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
      "N_res"        : int(self.sim_res),
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

  def defineParams(
      self,
      suite_folder, sim_folder, sim_res,
      num_blocks, k_turb, desired_Mach,
      Re=None, Rm=None, Pm=None
    ):
    ## check input parameters are of the right type
    WWVariables.assertType("suite_folder", suite_folder, str)
    WWVariables.assertType("sim_folder",   sim_folder,   str)
    WWVariables.assertType("sim_res",      sim_res,      str)
    WWVariables.assertType("num_blocks",   num_blocks,   list)
    WWVariables.assertType("k_turb",       k_turb,       (int, float))
    WWVariables.assertType("Mach",         desired_Mach, (int, float))
    ## save input parameters
    self.suite_folder = suite_folder
    self.sim_folder   = sim_folder
    self.sim_res      = sim_res
    self.num_blocks   = num_blocks
    self.k_turb       = k_turb
    self.desired_Mach = desired_Mach
    ## save (optional) input parameters
    self.Re           = Re
    self.Rm           = Rm
    self.Pm           = Pm
    ## perform routines
    self.__defineSonicRegime()
    self.__definePlasmaParameters()
    self.__checkSimParamsDefined()
    self.__roundParams()

  def __defineSonicRegime(self):
    ## t_turb = ell_turb / (Mach * c_s)
    self.t_turb = 1 / (self.k_turb * self.desired_Mach)
    ## define sonic regime
    if self.desired_Mach > 1:
      self.sonic_regime     = "super_sonic"
    else: self.sonic_regime = "sub_sonic"

  def __definePlasmaParameters(self):
    ## Re and Pm have been defined
    if (self.Re is not None) and (self.Pm is not None):
      self.nu  = self.desired_Mach / (self.k_turb * self.Re)
      self.eta = self.nu / self.Pm
      self.Rm  = self.desired_Mach / (self.k_turb * self.eta)
    ## Rm and Pm have been defined
    elif (self.Rm is not None) and (self.Pm is not None):
      self.eta = self.desired_Mach / (self.k_turb * self.Rm)
      self.nu  = self.eta * self.Pm
      self.Re  = self.desired_Mach / (self.k_turb * self.nu)
    ## Re and Rm have been defined
    elif (self.Re is not None) and (self.Rm is not None):
      self.nu  = self.desired_Mach / (self.k_turb * self.Re)
      self.eta = self.desired_Mach / (self.k_turb * self.Rm)
      self.Pm  = self.Rm / self.Re
    ## error
    else: raise Exception(f"You have not defined enough plasma Reynolds numbers: Re = {self.Re}, Rm = {self.Rm}, and Pm = {self.Rm}.")

  def __checkSimParamsDefined(self):
    list_check_params_defined = [
      "sonic_regime" if self.sonic_regime is None else "",
      "Re"  if self.Re  is None else "",
      "Rm"  if self.Rm  is None else "",
      "Pm"  if self.Pm  is None else "",
      "nu"  if self.nu  is None else "",
      "eta" if self.eta is None else ""
    ]
    list_params_not_defined = [
      param_name
      for param_name in list_check_params_defined
      ## remove entry if it is empty
      if len(param_name) > 0
    ]
    if len(list_params_not_defined) > 0:
      raise Exception(f"ERROR: You have not defined the following parameter(s):", list_params_not_defined)

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
  saveSimInputParams(obj_sim_params, filepath_demo)
  obj_read_params = readSimInputParams(filepath_demo)
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
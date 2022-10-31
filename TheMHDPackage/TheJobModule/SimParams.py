## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
from TheUsefulModule import WWVariables


## ###############################################################
## HELPER FUNCTION: CALCULATE SIMULATION PARAMETERS
## ###############################################################
def getPlasmaNumbers(Mach, k_turb, Re=None, Rm=None, Pm=None):
  ## Re and Pm have been defined
  if (Re is not None) and (Pm is not None):
    nu  = round(Mach / (k_turb * Re), 5)
    eta = round(nu / Pm, 5)
    Rm  = round(Mach / (k_turb * eta))
  ## Rm and Pm have been defined
  elif (Rm is not None) and (Pm is not None):
    eta = round(Mach / (k_turb * Rm), 5)
    nu  = round(eta * Pm, 5)
    Re  = round(Mach / (k_turb * nu))
  ## error
  else: raise Exception(f"You have not defined enough plasma Reynolds numbers: Re = {Re}, Rm = {Rm}, and Pm = {Rm}.")
  return Re, Rm, Pm, nu, eta

def getPlasmaNumberFromName(name, name_ref):
  name_lower = name.lower()
  name_ref_lower = name_ref.lower()
  return float(name_lower.replace(name_ref_lower, "")) if name_ref_lower in name_lower else None


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class SimParams():
  def __init__(
      self,
      suite_folder, sim_folder, sim_res,
      num_blocks, k_turb, Mach,
      Re=None, Rm=None, Pm=None
    ):
    WWVariables.assertType("suite_folder", suite_folder, str)
    WWVariables.assertType("sim_folder",   sim_folder,   str)
    WWVariables.assertType("sim_res",      sim_res,      int)
    WWVariables.assertType("num_blocks",   num_blocks,   list)
    WWVariables.assertType("k_turb",       k_turb,       (int, float))
    WWVariables.assertType("Mach",         Mach,         (int, float))
    ## input parameters
    self.suite_folder = suite_folder
    self.sim_folder   = sim_folder
    self.sim_res      = sim_res
    self.num_blocks   = num_blocks
    self.k_turb       = k_turb
    self.Mach         = Mach
    ## parameters to define
    self.sonic_regime = None
    self.Re           = Re
    self.Rm           = Rm
    self.Pm           = Pm
    self.nu           = None
    self.eta          = None
    ## perform routines
    self.__defineSonicRegime()
    self.__definePlasmaNumbers()
    self.__checkSimParamsDefined()

  # def saveSimParams(self):
  #   a = 10

  def __defineSonicRegime(self):
    if self.Mach < 1:
      self.sonic_regime = "sub_sonic"
    else: self.sonic_regime = "super_sonic"

  def __definePlasmaNumbers(self):
    self.Re, self.Rm, self.Pm, self.nu, self.eta = getPlasmaNumbers(
      self.Mach,
      self.k_turb,
      self.Re,
      self.Rm,
      self.Pm
    )

  def __checkSimParamsDefined(self):
    list_param_names = [
      "sonic_regime",
      "Re", "Rm", "Pm",
      "nu", "eta"
    ]
    list_bool_params_defined = [
      self.sonic_regime is None,
      self.Re  is None,
      self.Rm  is None,
      self.Pm  is None,
      self.nu  is None,
      self.eta is None
    ]
    list_params_not_defined = [
      list_param_names[index]
      for index, bool_val in enumerate(list_bool_params_defined)
      if bool_val
    ]
    if len(list_params_not_defined) > 0:
      raise Exception(f"ERROR: You have not defined the following parameter(s):", list_params_not_defined)


## ###############################################################
## TEST SUITE
## ###############################################################
class OtherClass():
  def __init__(self):
    self.a = 10

def func(obj_sim_params : SimParams):
  WWVariables.assertType("obj_sim_params", obj_sim_params, SimParams)

def tests():
  print("Running tests...")
  obj_sim_params = SimParams(
    num_blocks = [ 36, 36, 48 ],
    sim_res    = 288,
    k_turb     = 2.0,
    Mach       = 5.0,
    Re         = 1,
    Pm         = 1
  )
  dict_sim_params = obj_sim_params.getSimParams()
  obj_other_class = OtherClass()
  func(obj_sim_params)
  func(obj_other_class)
  print("all tests passed...")


## ###############################################################
## RUN TEST SUITE
## ###############################################################
if __name__ == "__main__":
  tests()


## END OF LIBRARY
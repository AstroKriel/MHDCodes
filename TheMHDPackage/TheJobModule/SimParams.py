## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
from TheUsefulModule import WWVariables


## ###############################################################
## DEFINE SIMULATION PARAMETERS
## ###############################################################
class SimParams():
  def __init__(
      self,
      num_blocks, sim_res, k_turb, rms_mach,
      Re = None,
      Rm = None,
      Pm = None
    ):
    WWVariables.assertType("num_blocks", num_blocks, list)
    WWVariables.assertType("sim_res",    sim_res,    int)
    WWVariables.assertType("k_turb",     k_turb,     (int, float))
    WWVariables.assertType("rms_mach",   rms_mach,   (int, float))
    self.num_blocks   = num_blocks
    self.sim_res      = sim_res
    self.k_turb       = k_turb
    self.rms_mach     = rms_mach
    self.sonic_regime = None
    self.Re           = Re
    self.Rm           = Rm
    self.Pm           = Pm
    self.nu           = None
    self.eta          = None
    self.__defineSonicRegime()
    self.__definePlasmaNumbers()
    self.__checkSimParamsDefined()

  def getSimParams(self):
    return {
      "num_blocks"   : self.num_blocks,
      "sim_res"      : self.sim_res,
      "k_turb"       : self.k_turb,
      "sonic_regime" : self.sonic_regime,
      "Re"           : self.Re,
      "Rm"           : self.Rm,
      "Pm"           : self.Pm,
      "nu"           : self.nu,
      "eta"          : self.eta
    }

  def __defineSonicRegime(self):
    if self.rms_mach < 1:
      self.sonic_regime = "sub_sonic"
    else: self.sonic_regime = "super_sonic"

  def __checkSimParamsDefined(self):
    list_param_names = [
      "sonic_regime",
      "Re",
      "Rm",
      "Pm",
      "nu",
      "eta"
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
      raise Exception(f"ERROR: You have not defined the following parameter(s): {list_params_not_defined}")

  def __definePlasmaNumbers(self):
    ## Re and Pm have been defined
    if (self.Re is not None) and (self.Pm is not None):
      self.nu  = round(self.rms_mach / (self.k_turb * self.Re), 5)
      self.eta = round(self.nu / self.Pm, 5)
      self.Rm  = round(self.rms_mach / (self.k_turb * self.eta))
    ## Rm and Pm have been defined
    elif (self.Rm is not None) and (self.Pm is not None):
      self.eta = round(self.rms_mach / (self.k_turb * self.Rm), 5)
      self.nu  = round(self.eta * self.Pm, 5)
      self.Re  = round(self.rms_mach / (self.k_turb * self.nu))
    ## error
    else: raise Exception(f"You have not defined the required number of plasma Reynolds numbers: Re = {self.Re} or Rm = {self.Rm}, and Pm = {self.Rm}.")


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
    rms_mach   = 5.0,
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
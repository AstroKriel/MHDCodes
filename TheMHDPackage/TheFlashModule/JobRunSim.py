## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
from TheFlashModule import SimParams, FileNames
from TheUsefulModule import WWFnF


## ###############################################################
## HELPFUL CONSTANTS
## ###############################################################
## turbulence driving input file
DRIVING_INPUT_NSPACES_PRE_ASSIGN  = 18
DRIVING_INPUT_NSPACES_PRE_COMMENT = 28
## flash simulation input file
FLASH_INPUT_NSPACES_PRE_ASSIGN    = 30
FLASH_INPUT_NSPACES_PRE_COMMENT   = 45

## ###############################################################
## HELPER FUNCTION
## ###############################################################
def updateAssign(file_line, param_value):
  return file_line.replace(
    file_line.split("=")[1].split()[0],
    param_value
  )

def paramAssignLine(
    param_name, param_value,
    comment             = "",
    nspaces_pre_assign  = 1,
    nspaces_pre_comment = 1
  ):
  space_assign  = " " * (nspaces_pre_assign - len(param_name))
  param_assign  = f"{param_name}{space_assign}= {param_value}"
  if len(comment) > 0:
    space_comment = " " * (nspaces_pre_comment - len(param_assign))
    return f"{param_assign}{space_comment}# {comment}\n"
  else: return f"{param_assign}\n"

def addParamAssign(
    dict_assigns, param_name, param_value,
    comment             = "",
    nspaces_pre_assign  = 1,
    nspaces_pre_comment = 1
  ):
  assign_line = paramAssignLine(param_name, param_value, comment, nspaces_pre_assign, nspaces_pre_comment)
  dict_assigns[param_name] = {
    "assign_line"   : assign_line,
    "bool_assigned" : False
  }

def processLine(
    ref_line, dict_assigns,
    nspaces_pre_assign  = 1,
    nspaces_pre_comment = 1
  ):
  list_line_elems = ref_line.split()
  ## empty line
  if len(list_line_elems) == 0:
    return "\n"
  ## parameter who's value needs to be assigned
  elif list_line_elems[0] in dict_assigns:
    param_name = list_line_elems[0]
    dict_assigns[param_name]["bool_assigned"] = True
    return str(dict_assigns[param_name]["assign_line"])
  ## parameter who's value doesn't need to change
  elif not(list_line_elems[0] == "#") and (list_line_elems[1] == "="):
    ## line format: param_name = param_value # comment
    param_name  = list_line_elems[0]
    param_value = list_line_elems[2]
    if (len(list_line_elems) > 4) and (list_line_elems[3] == "#"):
      comment = " ".join(list_line_elems[4:])
    else: comment = ""
    return paramAssignLine(param_name, param_value, comment, nspaces_pre_assign, nspaces_pre_comment)
  ## comment that has overflowed to the next line
  ## (* are used as headings)
  elif (list_line_elems[0] == "#") and ("*" not in ref_line):
    space_pre_comment = " " * nspaces_pre_comment
    comment = " ".join(list_line_elems[1:])
    return f"{space_pre_comment}# {comment}\n"
  ## something else (e.g., heading)
  return ref_line


## ###############################################################
## WRITE TURBULENCE DRIVING FILE
## ###############################################################
def writeTurbDrivingFile(filepath_ref, filepath_to, dict_params):
  ## helper function
  def _addParamAssign(param_name, param_value, comment):
    addParamAssign(
      dict_assigns        = dict_assigns,
      param_name          = param_name,
      param_value         = param_value,
      comment             = comment,
      nspaces_pre_assign  = DRIVING_INPUT_NSPACES_PRE_ASSIGN,
      nspaces_pre_comment = DRIVING_INPUT_NSPACES_PRE_COMMENT
    )
  ## initialise dictionary storing parameter assignemnts
  dict_assigns = {}
  ## add parameter assignments
  _addParamAssign(
    param_name  = "velocity",
    param_value = "{:.3f}".format(dict_params["des_velocity"]),
    comment     = "Target turbulent velocity dispersion"
  )
  _addParamAssign(
    param_name  = "ampl_factor",
    param_value = "{:.5f}".format(dict_params["des_ampl_factor"]),
    comment     = "Used to achieve a target velocity dispersion; scales with velocity/velocity_measured"
  )
  _addParamAssign(
    param_name  = "k_driv",
    param_value = "{:.3f}".format(dict_params["des_k_driv"]),
    comment     = "Characteristic driving scale in units of 2pi / Lx"
  )
  _addParamAssign(
    param_name  = "k_min",
    param_value = "{:.3f}".format(dict_params["des_k_min"]),
    comment     = "Minimum driving wavnumber in units of 2pi / Lx"
  )
  _addParamAssign(
    param_name  = "k_max",
    param_value = "{:.3f}".format(dict_params["des_k_max"]),
    comment     = "Maximum driving wavnumber in units of 2pi / Lx"
  )
  _addParamAssign(
    param_name  = "sol_weight",
    param_value = "{:.3f}".format(dict_params["des_sol_weight"]),
    comment     = "1.0: solenoidal driving, 0.0: compressive driving, 0.5: natural mixture"
  )
  _addParamAssign(
    param_name  = "spect_form",
    param_value = "{:.3f}".format(dict_params["des_spect_form"]),
    comment     = "0: band/rectangle/constant, 1: paraboloid, 2: power law"
  )
  _addParamAssign(
    param_name  = "nsteps_per_t_turb",
    param_value = "{:d}".format(dict_params["des_nsteps_per_t_turb"]),
    comment     = "Number of turbulence driving pattern updates per turnover time"
  )
  ## read from reference file and write to new files
  filepath_ref  = f"{filepath_ref}/{FileNames.FILENAME_DRIVING_INPUT}"
  filepath_file = f"{filepath_to}/{FileNames.FILENAME_DRIVING_INPUT}"
  with open(filepath_ref, "r") as ref_file:
    with open(filepath_file, "w") as new_file:
      for ref_line in ref_file.readlines():
        new_line = processLine(
          ref_line, dict_assigns,
          nspaces_pre_assign  = DRIVING_INPUT_NSPACES_PRE_ASSIGN,
          nspaces_pre_comment = DRIVING_INPUT_NSPACES_PRE_COMMENT
        )
        new_file.write(new_line)
  ## check that every parameter has been successfully defined
  list_params_not_assigned = []
  for param_name in dict_assigns:
    if not(dict_assigns[param_name]["bool_assigned"]):
      list_params_not_assigned.append(param_name)
  if len(list_params_not_assigned) == 0:
    print(f"Successfully defined turbulence driving parameters")
  else: raise Exception("Error: failed to define the following turbulence driving parameters:", list_params_not_assigned)


## ###############################################################
## WRITE FLASH INPUT PARAMETER FILE
## ###############################################################
def writeFlashParamFile(filepath_ref, filepath_to, dict_sim_inputs, max_num_hours):
  ## helper function
  def _addParamAssign(param_name, param_value, comment=""):
    addParamAssign(
      dict_assigns        = dict_assigns,
      param_name          = param_name,
      param_value         = param_value,
      comment             = comment,
      nspaces_pre_assign  = FLASH_INPUT_NSPACES_PRE_ASSIGN,
      nspaces_pre_comment = FLASH_INPUT_NSPACES_PRE_COMMENT
    )
  ## initialise dictionary storing parameter assignemnts
  dict_assigns = {}
  ## add parameter assignments
  _addParamAssign(
    param_name  = "st_infilename",
    param_value = FileNames.FILENAME_DRIVING_INPUT
  )
  _addParamAssign(
    param_name  = "useViscosity",
    param_value = ".true."
  )
  _addParamAssign(
    param_name  = "useMagneticResistivity",
    param_value = ".true."
  )
  _addParamAssign(
    param_name  = "diff_visc_nu",
    param_value = dict_sim_inputs["nu"],
    comment     = "implies Re = {} with Mach = {}".format(
      dict_sim_inputs["Re"],
      dict_sim_inputs["desired_Mach"]
    )
  )
  _addParamAssign(
    param_name  = "resistivity",
    param_value = dict_sim_inputs["eta"],
    comment     = "implies Rm = {} and Pm = {}".format(
      dict_sim_inputs["Rm"],
      dict_sim_inputs["Pm"]
    )
  )
  _addParamAssign(
    param_name  = "iProcs",
    param_value = dict_sim_inputs["num_procs"][0]
  )
  _addParamAssign(
    param_name  = "jProcs",
    param_value = dict_sim_inputs["num_procs"][1]
  )
  _addParamAssign(
    param_name  = "kProcs",
    param_value = dict_sim_inputs["num_procs"][2]
  )
  _addParamAssign(
    param_name  = "restart",
    param_value = ".false."
  )
  _addParamAssign(
    param_name  = "checkpointFileNumber",
    param_value = "0"
  )
  _addParamAssign(
    param_name  = "plotFileNumber",
    param_value = "0"
  )
  _addParamAssign(
    param_name  = "checkpointFileIntervalTime",
    param_value = dict_sim_inputs["t_turb"],
    comment     = "1 t_turb"
  )
  _addParamAssign(
    param_name  = "plotFileIntervalTime",
    param_value = "{:.3f}".format(0.1 * dict_sim_inputs["t_turb"]),
    comment     = "0.1 t_turb"
  )
  _addParamAssign(
    param_name  = "tmax",
    param_value = "{:.3f}".format(
      dict_sim_inputs["max_num_t_turb"] * dict_sim_inputs["t_turb"]
    ),
    comment     = "{} turb".format(dict_sim_inputs["max_num_t_turb"])
  )
  _addParamAssign(
    param_name  = "wall_clock_time_limit",
    param_value = max_num_hours * 60 * 60 - 1000, # [seconds]
    comment     = "close and save sim after this time has elapsed"
  )
  ## read from reference file and write to new files
  filepath_ref  = f"{filepath_ref}/{FileNames.FILENAME_FLASH_INPUT}"
  filepath_file = f"{filepath_to}/{FileNames.FILENAME_FLASH_INPUT}"
  with open(filepath_ref, "r") as ref_file:
    with open(filepath_file, "w") as new_file:
      for ref_line in ref_file.readlines():
        new_line = processLine(
          ref_line, dict_assigns,
          nspaces_pre_assign  = FLASH_INPUT_NSPACES_PRE_ASSIGN,
          nspaces_pre_comment = FLASH_INPUT_NSPACES_PRE_COMMENT
        )
        new_file.write(new_line)
  ## check that every parameter has been successfully defined
  list_params_not_assigned = []
  for param_name in dict_assigns:
    if not(dict_assigns[param_name]["bool_assigned"]):
      list_params_not_assigned.append(param_name)
  if len(list_params_not_assigned) == 0:
    print(f"Successfully defined flash input parameters")
  else: raise Exception("Error: failed to define the following flash input parameters:", list_params_not_assigned)


## ###############################################################
## PREPARE ALL PARAMETER FILES FOR SIMULATION
## ###############################################################
class JobRunSim():
  def __init__(
      self,
      filepath_sim, dict_sim_inputs,
      run_index = 0
    ):
    self.filepath_sim    = filepath_sim
    self.dict_sim_inputs = dict_sim_inputs
    self.run_index       = run_index
    self.des_velocity    = self.dict_sim_inputs["desired_Mach"] # velocity = Mach / cs
    self.k_turb          = self.dict_sim_inputs["k_turb"]
    if self.k_turb < 2: raise Exception(f"Error: driving mode cannot be larger than 1/2 box size (k_turb < 2). You have requested k_turb = {self.k_turb}")
    self.dict_driving_params = {
      "des_velocity"          : self.des_velocity,
      "des_ampl_factor"       : 0.1,
      "des_k_driv"            : self.k_turb,
      "des_k_min"             : self.k_turb - 1,
      "des_k_max"             : self.k_turb + 1,
      "des_sol_weight"        : 1.0,
      "des_spect_form"        : 1.0,
      "des_nsteps_per_t_turb" : 10
    }
    self._calcJobParams()

  def fromTemplate(self, filepath_ref_folder):
    ## copy flash4 executable
    WWFnF.copyFile(
      directory_from = filepath_ref_folder,
      directory_to   = self.filepath_sim,
      filename       = self.filename_flash_exe
    )
    ## write driving parameter file
    writeTurbDrivingFile(
      filepath_ref = filepath_ref_folder,
      filepath_to  = self.filepath_sim,
      dict_params  = self.dict_driving_params
    )
    ## write flash parameter file
    writeFlashParamFile(
      filepath_ref    = filepath_ref_folder,
      filepath_to     = self.filepath_sim,
      dict_sim_inputs = self.dict_sim_inputs,
      max_num_hours   = self.max_num_hours
    )
    ## create job script
    self._createJob()

  def fromLowerNres(self, filepath_ref_sim):
    ## copy flash4 executable
    WWFnF.copyFile(
      directory_from = filepath_ref_sim,
      directory_to   = self.filepath_sim,
      filename       = self.filename_flash_exe
    )
    ## copy driving parameter file
    WWFnF.copyFile(
      directory_from = filepath_ref_sim,
      directory_to   = self.filepath_sim,
      filename       = FileNames.FILENAME_DRIVING_INPUT
    )
    ## copy and update flash parameter file
    self._copyFlashInputFromLower(filepath_ref_sim)
    ## create job script
    self._createJob()

  def _calcJobParams(self):
    self.iprocs, self.jprocs, self.kprocs = self.dict_sim_inputs["num_procs"]
    self.num_procs = int(self.iprocs * self.jprocs * self.kprocs)
    self.max_mem   = int(4 * self.num_procs)
    if self.num_procs > 1000:
      self.max_num_hours = 24
    else: self.max_num_hours = 48
    self.job_name    = FileNames.FILENAME_RUN_SIM_JOB
    self.job_tagname = SimParams.getJobTag(self.dict_sim_inputs, "sim")
    self.job_output  = FileNames.FILENAME_RUN_SIM_OUTPUT + str(self.run_index).zfill(2)
    self.filename_flash_exe = "flash4_nxb{}_nyb{}_nzb{}_3.0".format(
      self.dict_sim_inputs["num_blocks"][0],
      self.dict_sim_inputs["num_blocks"][1],
      self.dict_sim_inputs["num_blocks"][2]
    )

  def _copyFlashInputFromLower(self, filepath_ref_sim):
    ## helper function
    def _addParamAssign(param_name, param_value):
      addParamAssign(
        dict_assigns       = dict_assigns,
        param_name         = param_name,
        param_value        = param_value,
        nspaces_pre_assign = FLASH_INPUT_NSPACES_PRE_ASSIGN
      )
    ## initialise dictionary storing parameter assignemnts
    dict_assigns = {}
    ## add parameter assignments
    _addParamAssign(
      param_name  = "iProcs",
      param_value = self.iprocs
    )
    _addParamAssign(
      param_name  = "jProcs",
      param_value = self.jprocs
    )
    _addParamAssign(
      param_name  = "kProcs",
      param_value = self.kprocs
    )
    ## read from reference file and write to new files
    filepath_ref = f"{filepath_ref_sim}/{FileNames.FILENAME_FLASH_INPUT}"
    filepath_new = f"{self.filepath_sim}/{FileNames.FILENAME_FLASH_INPUT}"
    with open(filepath_ref, "r") as ref_file:
      with open(filepath_new, "w") as new_file:
        for ref_line in ref_file.readlines():
          new_line = processLine(
            ref_line, dict_assigns,
            nspaces_pre_assign  = FLASH_INPUT_NSPACES_PRE_ASSIGN,
            nspaces_pre_comment = FLASH_INPUT_NSPACES_PRE_COMMENT
          )
          new_file.write(new_line)
    ## check that every parameter has been successfully defined
    list_params_not_assigned = []
    for param_name in dict_assigns:
      if not(dict_assigns[param_name]["bool_assigned"]):
        list_params_not_assigned.append(param_name)
    if len(list_params_not_assigned) == 0:
      print(f"Successfully copied and updated flash's input parameter file")
    else: raise Exception("Error: failed to update the following flash input parameters:", list_params_not_assigned)

  def _createJob(self):
    ## create/overwrite job file
    with open(f"{self.filepath_sim}/{self.job_name}", "w") as job_file:
      ## write contents
      job_file.write("#!/bin/bash\n")
      job_file.write("#PBS -P ek9\n")
      job_file.write("#PBS -q normal\n")
      job_file.write(f"#PBS -l walltime={self.max_num_hours}:00:00\n")
      job_file.write(f"#PBS -l ncpus={self.num_procs}\n")
      job_file.write(f"#PBS -l mem={self.max_mem}GB\n")
      job_file.write("#PBS -l storage=scratch/ek9+gdata/ek9\n")
      job_file.write("#PBS -l wd\n")
      job_file.write(f"#PBS -N {self.job_tagname}\n")
      job_file.write("#PBS -j oe\n")
      job_file.write("#PBS -m bea\n")
      job_file.write(f"#PBS -M neco.kriel@anu.edu.au\n")
      job_file.write("\n")
      job_file.write(". ~/modules_flash\n")
      job_file.write(f"mpirun ./{self.filename_flash_exe} 1>{self.job_output} 2>&1\n")
    ## indicate progress
    print(f"Created PBS job:")
    print(f"\t> Job name:",  self.job_name)
    print(f"\t> Directory:", self.filepath_sim)


## END OF LIBRARY
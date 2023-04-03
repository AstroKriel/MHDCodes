## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
from TheFlashModule import SimParams, FileNames
from TheUsefulModule import WWFnF


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def createParamFileLine(
    str_variable, str_value, str_comment,
    num_spaces_assign  = 25,
    num_spaces_comment = 40
  ):
  ## define helper function
  def numSpaces(num_spaces, string):
    return " " * ( num_spaces - len(string) )
  ## create variable assignment
  str_variable_assign = "{}{}= {}".format(
    str_variable,
    numSpaces(num_spaces_assign, str_variable),
    str_value
  )
  ## append comment
  return "{}{}# {}\n".format(
    str_variable_assign,
    numSpaces(num_spaces_comment, str_variable_assign),
    str_comment
  )

def updateParam(file_line, updated_param_value):
  return file_line.replace(
    file_line.split("=")[1].split()[0],
    updated_param_value
  )


## ###############################################################
## FUNCTION: write turbulence driving generator file
## ###############################################################
def writeTurbDrivingFile(
    filepath_ref, filepath_to,
    des_velocity          = 5.0,
    des_ampl_factor       = 0.1,
    des_k_driv            = 2.0,
    des_k_min             = 1.0,
    des_k_max             = 3.0,
    des_sol_weight        = 1.0, # solenoidal driving
    des_spect_form        = 1.0, # paraboloid profile
    des_nsteps_per_t_turb = 10
  ):
  ## initialise flags
  bool_set_velocity          = False
  bool_set_ampl_factor       = False
  bool_set_k_driv            = False
  bool_set_k_min             = False
  bool_set_k_max             = False
  bool_set_sol_weight        = False
  bool_set_spect_form        = False
  bool_set_nsteps_per_t_turb = False
  ## define turbulence generator input parameters
  args_spaces = {
    "num_spaces_assign"  : 18,
    "num_spaces_comment" : 28
  }
  str_velocity = createParamFileLine(
    str_variable = "velocity",
    str_value    = f"{des_velocity:.3f}",
    str_comment  = "Target turbulent velocity dispersion",
    **args_spaces
  )
  str_ampl_factor = createParamFileLine(
    str_variable = "ampl_factor",
    str_value    = f"{des_ampl_factor:.5f}",
    str_comment  = "Used to achieve a target velocity dispersion; scales with velocity/velocity_measured",
    **args_spaces
  )
  str_k_driv = createParamFileLine(
    str_variable = "k_driv",
    str_value    = f"{des_k_driv:.3f}",
    str_comment  = "Characteristic driving scale in units of 2pi / Lx",
    **args_spaces
  )
  str_k_min = createParamFileLine(
    str_variable = "k_min",
    str_value    = f"{des_k_min:.3f}",
    str_comment  = "Minimum driving wavnumber in units of 2pi / Lx",
    **args_spaces
  )
  str_k_max = createParamFileLine(
    str_variable = "k_max",
    str_value    = f"{des_k_max:.3f}",
    str_comment  = "Maximum driving wavnumber in units of 2pi / Lx",
    **args_spaces
  )
  str_sol_weight = createParamFileLine(
    str_variable = "sol_weight",
    str_value    = f"{des_sol_weight:.3f}",
    str_comment  = "1.0: solenoidal driving, 0.0: compressive driving, 0.5: natural mixture",
    **args_spaces
  )
  str_spect_form = createParamFileLine(
    str_variable = "spect_form",
    str_value    = f"{des_spect_form:.3f}",
    str_comment  = "0: band/rectangle/constant, 1: paraboloid, 2: power law",
    **args_spaces
  )
  str_nsteps_per_t_turb = createParamFileLine(
    str_variable = "nsteps_per_t_turb",
    str_value    = f"{des_nsteps_per_t_turb:d}",
    str_comment  = "Number of turbulence driving pattern updates per turnover time",
    **args_spaces
  )
  ## open new file
  with open(f"{filepath_to}/{FileNames.FILENAME_DRIVING_INPUT}", "w") as new_file:
    ## open refernce file
    with open(f"{filepath_ref}/{FileNames.FILENAME_DRIVING_INPUT}", "r") as ref_file:
      ## loop over lines in reference file
      for ref_line in ref_file.readlines():
        list_ref_line_elems = ref_line.split()
        ## handle empty lines
        ## ------------------
        if len(list_ref_line_elems) == 0:
          new_file.write("\n")
          continue
        ## setting velocity dispersion
        ## ---------------------------
        elif list_ref_line_elems[0] == "velocity":
          new_file.write(str_velocity)
          bool_set_velocity = True
        elif list_ref_line_elems[0] == "ampl_factor":
          new_file.write(str_ampl_factor)
          bool_set_ampl_factor = True
        ## setting driving profile
        ## -----------------------
        elif list_ref_line_elems[0] == "k_driv":
          new_file.write(str_k_driv)
          bool_set_k_driv = True
        elif list_ref_line_elems[0] == "k_min":
          new_file.write(str_k_min)
          bool_set_k_min = True
        elif list_ref_line_elems[0] == "k_max":
          new_file.write(str_k_max)
          bool_set_k_max = True
        elif list_ref_line_elems[0] == "sol_weight":
          new_file.write(str_sol_weight)
          bool_set_sol_weight = True
        elif list_ref_line_elems[0] == "spect_form":
          new_file.write(str_spect_form)
          bool_set_spect_form = True
        elif list_ref_line_elems[0] == "nsteps_per_t_turb":
          new_file.write(str_nsteps_per_t_turb)
          bool_set_nsteps_per_t_turb = True
        ## found line where comment overflows
        ## ----------------------------------
        elif (list_ref_line_elems[0] == "#") and ("*" not in list_ref_line_elems[1]):
          new_file.write("{}# {}\n".format(
            " " * args_spaces["num_spaces_comment"],
            " ".join(list_ref_line_elems[1:])
          ))
        ## otherwise write line contents
        else: new_file.write(ref_line)
  ## check that all parameters have been defined
  list_bools = [
    bool_set_velocity,
    bool_set_ampl_factor,
    bool_set_k_driv,
    bool_set_k_min,
    bool_set_k_max,
    bool_set_sol_weight,
    bool_set_spect_form,
    bool_set_nsteps_per_t_turb
  ]
  if all(list_bools):
    print(f"Successfully modified turbulence generator in:", filepath_to)
  else: raise Exception("ERROR: failed to write turbulence generator in:", filepath_to, list_bools)


## ###############################################################
## FUNCTION: write flash input parameter file
## ###############################################################
def writeFlashParamFile(filepath_ref, filepath_to, dict_sim_inputs, max_hours):
  max_wall_time_sec = max_hours * 60 * 60 - 1000 # [seconds]
  ## initialise flags
  bool_set_driving       = False
  bool_use_visc          = False
  bool_use_resis         = False
  bool_set_nu            = False
  bool_set_eta           = False
  bool_set_iproc         = False
  bool_set_jproc         = False
  bool_set_kproc         = False
  bool_set_restart       = False
  bool_set_chk_num       = False
  bool_set_plt_num       = False
  bool_set_chk_rate      = False
  bool_set_plt_rate      = False
  bool_set_max_sim_time  = False
  bool_set_max_wall_time = False
  ## define flash input parameters
  str_set_driving        = f"st_infilename = {FileNames.FILENAME_DRIVING_INPUT}\n"
  str_use_visc           = "useViscosity = .true.\n"
  str_use_resis          = "useMagneticResistivity = .true.\n"
  str_set_nu             = "diff_visc_nu = {} # implies Re = {} with Mach = {}\n".format(
    dict_sim_inputs["nu"],
    dict_sim_inputs["Re"],
    dict_sim_inputs["desired_Mach"]
  )
  str_set_eta            = "resistivity = {} # implies Rm = {} and Pm = {}\n".format(
    dict_sim_inputs["eta"],
    dict_sim_inputs["Rm"],
    dict_sim_inputs["Pm"]
  )
  str_set_iproc          = "iProcs = {}\n".format(dict_sim_inputs["num_procs"][0])
  str_set_jproc          = "jProcs = {}\n".format(dict_sim_inputs["num_procs"][1])
  str_set_kproc          = "kProcs = {}\n".format(dict_sim_inputs["num_procs"][2])
  str_set_restart        = "restart = .false.\n"
  str_set_chk_num        = "checkpointFileNumber = 0\n"
  str_set_plt_num        = "plotFileNumber = 0\n"
  str_set_chk_rate       = "checkpointFileIntervalTime = {} # 1 t_turb\n".format(dict_sim_inputs["t_turb"])
  str_set_plt_rate       = "plotFileIntervalTime = {} # 0.1 t_turb\n".format(dict_sim_inputs["t_turb"] / 10)
  str_set_max_sim_time   = "tmax = {} # {} t_turb\n".format(
    dict_sim_inputs["num_t_turb"] * dict_sim_inputs["t_turb"], # TODO
    dict_sim_inputs["num_t_turb"]
  )
  str_set_max_wall_time  = f"wall_clock_time_limit = {max_wall_time_sec} # closes sim and saves state\n"
  ## open new file
  with open(f"{filepath_to}/{FileNames.FILENAME_FLASH_INPUT}", "w") as new_file:
    ## open reference file
    with open(f"{filepath_ref}/{FileNames.FILENAME_FLASH_INPUT}", "r") as ref_file_lines:
      ## set cfl condition sufficiently low to resolve low Re dynamics
      if (dict_sim_inputs["Re"] < 50): new_file.write("hy_diffuse_cfl = 0.2\n\n")
      ## loop over reference file
      for ref_line_elems in ref_file_lines:
        ## split line contents into words
        list_ref_line_elems = ref_line_elems.split()
        ## handle empty lines
        ## -----------------
        if len(list_ref_line_elems) == 0:
          new_file.write("\n")
          continue
        ## extract parameter name
        ## ----------------------
        param_name = list_ref_line_elems[0]
        ## define driving parameter file
        ## -----------------------------
        if param_name == "st_infilename":
          new_file.write(str_set_driving)
          bool_set_driving = True
        ## turn physical dissipation on
        ## ----------------------------
        elif param_name == "useViscosity":
          new_file.write(str_use_visc)
          bool_use_visc = True
        elif param_name == "useMagneticResistivity":
          new_file.write(str_use_resis)
          bool_use_resis = True
        elif param_name == "diff_visc_nu":
          new_file.write(str_set_nu)
          bool_set_nu = True
        elif param_name == "resistivity":
          new_file.write(str_set_eta)
          bool_set_eta = True
        ## define number of processors
        ## ---------------------------
        elif param_name == "iProcs":
          new_file.write(str_set_iproc)
          bool_set_iproc = True
        elif param_name == "jProcs":
          new_file.write(str_set_jproc)
          bool_set_jproc = True
        elif param_name == "kProcs":
          new_file.write(str_set_kproc)
          bool_set_kproc = True
        ## initialise output file index
        ## ----------------------------
        elif param_name == "restart":
          new_file.write(str_set_restart)
          bool_set_restart = True
        elif param_name == "checkpointFileNumber":
          new_file.write(str_set_chk_num)
          bool_set_chk_num = True
        elif param_name == "plotFileNumber":
          new_file.write(str_set_plt_num)
          bool_set_plt_num = True
        ## define file output rates
        ## ------------------------
        elif param_name == "checkpointFileIntervalTime":
          new_file.write(str_set_chk_rate)
          bool_set_chk_rate = True
        elif param_name == "plotFileIntervalTime":
          new_file.write(str_set_plt_rate)
          bool_set_plt_rate = True
        ## define max times
        ## ----------------
        elif param_name == "tmax":
          new_file.write(str_set_max_sim_time)
          bool_set_max_sim_time = True
        elif param_name == "wall_clock_time_limit":
          new_file.write(str_set_max_wall_time)
          bool_set_max_wall_time = True
        ## write other line contents
        else: new_file.write(ref_line_elems)
  ## check that all parameters have been defined
  list_bools = [
    bool_set_driving,
    bool_use_visc,
    bool_use_resis,
    bool_set_nu,
    bool_set_eta,
    bool_set_iproc,
    bool_set_jproc,
    bool_set_kproc,
    bool_set_restart,
    bool_set_chk_num,
    bool_set_plt_num,
    bool_set_chk_rate,
    bool_set_plt_rate,
    bool_set_max_sim_time,
    bool_set_max_wall_time
  ]
  if all(list_bools):
    print("Successfully modified flash input parameter file in:", filepath_to)
  else: raise Exception("ERROR: failed to write flash parameter file in:", filepath_to, list_bools)


## ###############################################################
## PREPARE ALL PARAMETER FILES FOR SIMULATION
## ###############################################################
class PrepSimJob():
  def __init__(
      self,
      filepath_sim, dict_sim_inputs,
    ):
    self.filepath_sim = filepath_sim
    self.dict_sim_inputs = dict_sim_inputs
    self._calcJobParams()

  def fromTemplate(self, filepath_ref_folder):
    ## copy flash4 executable
    WWFnF.copyFileFromNTo(
      directory_from = filepath_ref_folder,
      directory_to   = self.filepath_sim,
      filename       = self.filename_flash_exe
    )
    ## write driving parameter file
    writeTurbDrivingFile(
      filepath_ref          = filepath_ref_folder,
      filepath_to           = self.filepath_sim,
      des_velocity          = 5.0,
      des_ampl_factor       = 0.1,
      des_k_driv            = 2.0,
      des_k_min             = 1.0,
      des_k_max             = 3.0,
      des_sol_weight        = 1.0,
      des_spect_form        = 1.0,
      des_nsteps_per_t_turb = 10
    )
    ## write flash parameter file
    writeFlashParamFile(
      filepath_ref    = filepath_ref_folder,
      filepath_to     = self.filepath_sim,
      dict_sim_inputs = self.dict_sim_inputs,
      max_hours       = self.max_hours
    )
    ## create job script
    self._createJob()

  def fromLowerNres(self, filepath_ref_sim):
    ## copy flash4 executable
    WWFnF.copyFileFromNTo(
      directory_from = filepath_ref_sim,
      directory_to   = self.filepath_sim,
      filename       = self.filename_flash_exe
    )
    ## copy driving parameter file
    WWFnF.copyFileFromNTo(
      directory_from = filepath_ref_sim,
      directory_to   = self.filepath_sim,
      filename       = FileNames.FILENAME_DRIVING_INPUT
    )
    ## copy and update flash parameter file
    self._copyFlashInputFromLower(filepath_ref_sim)
    ## create job script
    self._createJob()

  def _copyFlashInputFromLower(self, filepath_ref_sim):
    bool_update_iproc = False
    bool_update_jproc = False
    bool_update_kproc = False
    with open(f"{self.filepath_sim}/{FileNames.FILENAME_FLASH_INPUT}", "w") as new_file:
      with open(f"{filepath_ref_sim}/{FileNames.FILENAME_FLASH_INPUT}", "r") as ref_file:
        for ref_line in ref_file.readlines():
          ## handle empty lines
          ## ------------------
          if len(ref_line.split()) == 0:
            new_file.write("\n")
            continue
          ## update number of processors per dimension
          ## -----------------------------------------
          elif "iProcs" in ref_line:
            new_file.write(updateParam(ref_line, str(int(self.iprocs))))
            bool_update_iproc = True
          elif "jProcs"  in ref_line:
            new_file.write(updateParam(ref_line, str(int(self.jprocs))))
            bool_update_jproc = True
          elif "kProcs"  in ref_line:
            new_file.write(updateParam(ref_line, str(int(self.kprocs))))
            bool_update_kproc = True
          else: new_file.write(ref_line)
    list_bools = [
      bool_update_iproc,
      bool_update_jproc,
      bool_update_kproc
    ]
    if all(list_bools):
      print(f"Successfully copied flash input parameter file")
    else: raise Exception("ERROR: failed to copy flash input parameter file", list_bools)

  def _calcJobParams(self):
    nxb, nyb, nzb = self.dict_sim_inputs["num_blocks"]
    self.iprocs   = int(self.dict_sim_inputs["sim_res"]) // nxb
    self.jprocs   = int(self.dict_sim_inputs["sim_res"]) // nyb
    self.kprocs   = int(self.dict_sim_inputs["sim_res"]) // nzb
    self.num_procs = int(self.iprocs * self.jprocs * self.kprocs)
    self.max_mem  = int(4 * self.num_procs)
    if self.num_procs > 1000:
      self.max_hours = 24
    else: self.max_hours = 48
    self.job_name    = FileNames.FILENAME_RUN_SIM_JOB
    self.job_tagname = SimParams.getJobTag(self.dict_sim_inputs, "sim")
    self.filename_flash_exe = "flash4_nxb{}_nyb{}_nzb{}_3.0".format(
      self.dict_sim_inputs["num_blocks"][0],
      self.dict_sim_inputs["num_blocks"][1],
      self.dict_sim_inputs["num_blocks"][2]
    )

  def _createJob(self):
    ## create/overwrite job file
    with open(f"{self.filepath_sim}/{self.job_name}", "w") as job_file:
      ## write contents
      job_file.write("#!/bin/bash\n")
      job_file.write("#PBS -P ek9\n")
      job_file.write("#PBS -q normal\n")
      job_file.write(f"#PBS -l walltime={self.max_hours}:00:00\n")
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
      job_file.write(f"mpirun ./{self.filename_flash_exe} 1>shell_sim.out00 2>&1\n")
    ## indicate progress
    print(f"Created PBS job:")
    print(f"\t> Job name:",  self.job_name)
    print(f"\t> Directory:", self.filepath_sim)


## END OF LIBRARY
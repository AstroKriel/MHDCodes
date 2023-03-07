## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
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
  str_variable_assign = "{}{}={}".format(
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


FILENAME_TURB_PAR = "turbulence_generator.inp"
## ###############################################################
## FUNCTION: write turbulence driving generator file
## ###############################################################
def writeTurbGenFile(
    filepath_ref, filepath_to,
    des_velocity         = 5.0,
    des_ampl_coeff       = 0.1,
    des_k_driv           = 2.0,
    des_k_min            = 1.0,
    des_k_max            = 3.0,
    des_sol_weight       = 1.0, # solenoidal driving
    des_spect_form       = 1.0, # paraboloid profile
    des_nsteps_per_teddy = 10
  ):
  ## initialise flags
  bool_set_velocity         = False
  bool_set_ampl_coeff       = False
  bool_set_k_driv           = False
  bool_set_k_min            = False
  bool_set_k_max            = False
  bool_set_sol_weight       = False
  bool_set_spect_form       = False
  bool_set_nsteps_per_teddy = False
  ## define turbulence generator input parameters
  args_spaces = {
    "num_spaces_assign"  : 25,
    "num_spaces_comment" : 40
  }
  str_velocity = createParamFileLine(
    str_variable = "velocity",
    str_value    = f"{des_velocity:.3f}",
    str_comment  = "Target turbulent velocity dispersion",
    **args_spaces
  )
  str_ampl_coeff = createParamFileLine(
    str_variable = "ampl_coeff",
    str_value    = f"{des_ampl_coeff:.5f}",
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
  str_nsteps_per_teddy = createParamFileLine(
    str_variable = "nsteps_per_turnover_time",
    str_value    = f"{des_nsteps_per_teddy:d}",
    str_comment  = "Number of turbulence driving pattern updates per turnover time",
    **args_spaces
  )
  ## open new file
  with open(f"{filepath_to}/{FILENAME_TURB_PAR}", "w") as new_file:
    ## open refernce file
    with open(f"{filepath_ref}/{FILENAME_TURB_PAR}", "r") as ref_file_lines:
      ## loop over lines in reference file
      for ref_line in ref_file_lines:
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
        elif list_ref_line_elems[0] == "ampl_coeff":
          new_file.write(str_ampl_coeff)
          bool_set_ampl_coeff = True
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
        elif list_ref_line_elems[0] == "nsteps_per_turnover_time":
          new_file.write(str_nsteps_per_teddy)
          bool_set_nsteps_per_teddy = True
        ## found line where comment overflows
        ## ----------------------------------
        elif (list_ref_line_elems[0] == "#") and ("*" not in list_ref_line_elems[1]):
          new_file.write("{}# {}\n".format(
            " " * 40,
            " ".join(list_ref_line_elems[1:])
          ))
        ## otherwise write line contents
        else: new_file.write(ref_line)
  ## check that all parameters have been defined
  list_bools = [
    bool_set_velocity,
    bool_set_ampl_coeff,
    bool_set_k_driv,
    bool_set_k_min,
    bool_set_k_max,
    bool_set_sol_weight,
    bool_set_spect_form,
    bool_set_nsteps_per_teddy
  ]
  if all(list_bools):
    print(f"Successfully modified turbulence generator in:", filepath_to)
  else: raise Exception("ERROR: failed to write turbulence generator in:", filepath_to, list_bools)


FILENAME_FLASH_PAR = "flash.par"
## ###############################################################
## FUNCTION: write flash input parameter file
## ###############################################################
def writeFlashParamFile(
    filepath_ref, filepath_to,
    Re, Rm, Pm,
    nu, eta, Mach, t_turb,
    num_procs, max_hours
  ):
  max_wall_time_sec = max_hours * 60 * 60 - 1000 # [seconds]
  ## initialise flags
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
  str_use_visc           = "useViscosity = .true.\n"
  str_use_resis          = "useMagneticResistivity = .true.\n"
  str_set_nu             = f"diff_visc_nu = {nu} # implies Re = {Re} with Mach = {Mach}\n"
  str_set_eta            = f"resistivity = {eta} # implies Rm = {Rm} and Pm = {Pm}\n"
  str_set_iproc          = f"iProcs = {num_procs[0]}\n"
  str_set_jproc          = f"jProcs = {num_procs[1]}\n"
  str_set_kproc          = f"kProcs = {num_procs[2]}\n"
  str_set_restart        = "restart = .false.\n"
  str_set_chk_num        = "checkpointFileNumber = 0\n"
  str_set_plt_num        = "plotFileNumber = 0\n"
  str_set_chk_rate       = f"checkpointFileIntervalTime = {t_turb} # 1 t_turb\n"
  str_set_plt_rate       = f"plotFileIntervalTime = {t_turb / 10} # 0.1 t_turb\n"
  str_set_max_sim_time   = f"tmax = {100 * t_turb} # 100 t_turb\n"
  str_set_max_wall_time  = f"wall_clock_time_limit = {max_wall_time_sec} # closes sim and saves state\n"
  ## open new file
  with open(f"{filepath_to}/{FILENAME_FLASH_PAR}", "w") as new_file:
    ## open reference file
    with open(f"{filepath_ref}/{FILENAME_FLASH_PAR}", "r") as ref_file_lines:
      ## set cfl condition sufficiently low to resolve low Re dynamics
      if (Re < 50): new_file.write("hy_diffuse_cfl = 0.2\n\n")
      ## loop over lines in reference 'flash.par'
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
        ## turn physical dissipation on
        ## ----------------------------
        if param_name == "useViscosity":
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
## CLASS 
## ###############################################################
class PrepSimJob():
  def __init__(
      self,
      filepath_ref, filepath_sim, dict_sim_inputs,
    ):
    self.filepath_ref = filepath_ref
    self.filepath_sim = filepath_sim
    self.suite_folder = dict_sim_inputs["suite_folder"]
    self.sonic_regime = dict_sim_inputs["sonic_regime"]
    self.sim_folder   = dict_sim_inputs["sim_folder"]
    self.sim_res      = dict_sim_inputs["sim_res"]
    self.num_blocks   = dict_sim_inputs["num_blocks"]
    self.num_procs    = dict_sim_inputs["num_procs"]
    self.k_turb       = dict_sim_inputs["k_turb"]
    self.desired_Mach = dict_sim_inputs["desired_Mach"]
    self.t_turb       = dict_sim_inputs["t_turb"]
    self.nu           = dict_sim_inputs["nu"]
    self.eta          = dict_sim_inputs["eta"]
    self.Re           = dict_sim_inputs["Re"]
    self.Rm           = dict_sim_inputs["Rm"]
    self.Pm           = dict_sim_inputs["Pm"]
    self.__calcJobParams()

  def fromLowerNres(self, filepath_ref_sim):
    self.__copyFilesFromLowerNres(filepath_ref_sim)
    WWFnF.copyFileFromNTo(
        directory_from = filepath_ref_sim,
        directory_to   = self.filepath_sim,
        filename       = FILENAME_TURB_PAR
    )
    WWFnF.copyFileFromNTo(
        directory_from = filepath_ref_sim,
        directory_to   = self.filepath_sim,
        filename       = FILENAME_FLASH_PAR
    )
    self.__createJob()

  def fromTemplate(self):
    ## copy template flash4 executable
    WWFnF.copyFileFromNTo(
      directory_from = self.filepath_ref,
      directory_to   = self.filepath_sim,
      filename       = self.filename_flash_exe
    )
    ## modify simulation parameter files
    writeTurbGenFile(
      filepath_ref         = self.filepath_ref,
      filepath_to          = self.filepath_sim,
      des_velocity         = 5.0,
      des_ampl_coeff       = 0.1,
      des_k_driv           = 2.0,
      des_k_min            = 1.0,
      des_k_max            = 3.0,
      des_sol_weight       = 1.0,
      des_spect_form       = 1.0,
      des_nsteps_per_teddy = 10
    )
    writeFlashParamFile(
      filepath_ref = self.filepath_ref,
      filepath_to  = self.filepath_sim,
      Re           = self.Re,
      Rm           = self.Rm,
      Pm           = self.Pm,
      nu           = self.nu,
      eta          = self.eta,
      Mach         = self.desired_Mach,
      t_turb       = self.t_turb,
      num_procs    = self.num_procs,
      max_hours    = self.max_hours
    )
    self.__createJob()

  def __copyFilesFromLowerNres(self, filepath_ref_sim):
    ## copy flash4 executable from the home directory
    WWFnF.copyFileFromNTo(
      directory_from = filepath_ref_sim,
      directory_to   = self.filepath_sim,
      filename       = self.filename_flash_exe
    )
    ## copy forcing input file from the Nres=144 directory
    WWFnF.copyFileFromNTo(
      directory_from = filepath_ref_sim,
      directory_to   = self.filepath_sim,
      filename       = "forcing_generator.inp"
    )
    ## copy forcing data file from the Nres=144 directory
    WWFnF.copyFileFromNTo(
      directory_from = filepath_ref_sim,
      directory_to   = self.filepath_sim,
      filename       = "turb_driving.dat"
    )

  def __calcJobParams(self):
    nxb, nyb, nzb = self.num_blocks
    self.iprocs   = int(self.sim_res) // nxb
    self.jprocs   = int(self.sim_res) // nyb
    self.kprocs   = int(self.sim_res) // nzb
    self.num_cpus = int(self.iprocs * self.jprocs * self.kprocs)
    self.max_mem  = int(4 * self.num_cpus)
    if self.num_cpus > 1000:
      self.max_hours = 24
    else: self.max_hours = 48
    self.job_name    = "job_run_sim.sh"
    self.job_tagname = "{}{}{}sim{}".format(
      self.sonic_regime.split("_")[0],
      self.suite_folder,
      self.sim_folder,
      self.sim_res
    )
    self.filename_flash_exe = "flash4_nxb{}_nyb{}_nzb{}_3.0".format(
      self.num_blocks[0],
      self.num_blocks[1],
      self.num_blocks[2]
    )

  def __createJob(self):
    ## create/overwrite job file
    with open(f"{self.filepath_sim}/{self.job_name}", "w") as job_file:
      ## write contents
      job_file.write("#!/bin/bash\n")
      job_file.write("#PBS -P ek9\n")
      job_file.write("#PBS -q normal\n")
      job_file.write(f"#PBS -l walltime={self.max_hours}:00:00\n")
      job_file.write(f"#PBS -l ncpus={self.num_cpus}\n")
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
    print(f"\t> Job name: {self.job_name}")
    print(f"\t> Directory: {self.filepath_sim}")


## END OF LIBRARY
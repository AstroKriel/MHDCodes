## START OF LIBRARY

from TheUsefulModule import WWFnF

class PrepSimJob():
  def __init__(
      self,
      filepath_ref, filepath_sim, dict_sim_params
    ):
    self.filepath_ref = filepath_ref
    self.filepath_sim = filepath_sim
    self.suite_folder = dict_sim_params["suite_folder"]
    self.sonic_regime = dict_sim_params["sonic_regime"]
    self.sim_folder   = dict_sim_params["sim_folder"]
    self.sim_res      = dict_sim_params["sim_res"]
    self.num_blocks   = dict_sim_params["num_blocks"]
    self.k_turb       = dict_sim_params["k_turb"]
    self.desired_Mach = dict_sim_params["desired_Mach"]
    self.t_turb       = dict_sim_params["t_turb"]
    self.nu           = dict_sim_params["nu"]
    self.eta          = dict_sim_params["eta"]
    self.Re           = dict_sim_params["Re"]
    self.Rm           = dict_sim_params["Rm"]
    self.Pm           = dict_sim_params["Pm"]
    self.__calcJobParams()

  def fromLowerNres(self, filepath_ref_sim):
    self.__copyFilesFromLowerNres(filepath_ref_sim)
    self.__modifyFlashParamFile(self.filepath_ref)
    self.__createJob()

  def __copyFilesFromTemplate(self):
    ## copy flash4 executable from the home directory
    WWFnF.copyFileFromNTo(
      directory_from = self.filepath_ref,
      directory_to   = self.filepath_sim,
      filename       = self.filename_flash_exe
    )
    ## copy forcing input file from the base directory
    WWFnF.copyFileFromNTo(
        directory_from = self.filepath_ref,
        directory_to   = self.filepath_sim,
        filename       = "forcing_generator.inp"
    )

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
    self.filename_flash_exe = "flash4_nxb{}_nyb{}_nzb{}_2.0".format(
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
    print(f"\t> Filename: {self.job_name}")
    print(f"\t> Directory: {self.filepath_sim}")

  def __modifyFlashParamFile(self, filepath_ref):
    bool_switched_nu     = False
    bool_switched_eta    = False
    bool_defined_nu      = False
    bool_defined_eta     = False
    bool_defined_runtime = False
    bool_defined_chk     = False
    bool_defined_plt     = False
    bool_defined_tmax    = False
    bool_defined_iproc   = False
    bool_defined_jproc   = False
    bool_defined_kproc   = False
    bool_no_restart      = False
    bool_reset_chk_num   = False
    bool_reset_plt_num   = False
    ## open new 'flash.par' file
    with open(f"{self.filepath_sim}/flash.par", "w") as new_file:
      ## open the reference 'flash.par' file
      with open(f"{filepath_ref}/flash.par", "r") as ref_file_lines:
        # ## make sure the cfl value is sufficient to resolve low Re dynamics
        # if (self.Re < 50):
        #   new_file.write("hy_diffuse_cfl = 0.2\n\n")
        #   bool_defined_cfl = True
        ## loop over lines in reference 'flash.par'
        for ref_line_elems in ref_file_lines:
          ## split line contents into words
          list_ref_line_elems = ref_line_elems.split()
          ## handle empty lines
          if len(list_ref_line_elems) == 0:
            new_file.write("\n")
            continue
          ## extract parameter name
          param_name = list_ref_line_elems[0]
          ## turn viscosity on
          if param_name == "useViscosity":
            bool_switched_nu = True
            new_file.write("useViscosity = .true.\n")
          ## define nu
          elif param_name == "diff_visc_nu":
            bool_defined_nu = True
            new_file.write("diff_visc_nu = {self.nu} # implies Re = {self.Re}\n")
          ## turn resistivity on
          elif param_name == "useMagneticResistivity":
            bool_switched_eta = True
            new_file.write("useMagneticResistivity = .true.\n")
          ## define eta
          elif param_name == "resistivity":
            bool_defined_eta = True
            new_file.write(f"resistivity = {self.eta} # implies Rm = {self.Rm} and Pm = {self.Pm}\n")
          ## define wall clock limit
          elif param_name == "wall_clock_time_limit":
            bool_defined_runtime = True
            new_file.write("wall_clock_time_limit = {} # closes sim and saves state\n".format(
              self.max_hours * 60 * 60 - 1000 # [seconds]
            ))
          ## define chk-file interval [t_sim]
          elif param_name == "checkpointFileIntervalTime":
            bool_defined_chk = True
            new_file.write(f"checkpointFileIntervalTime = {self.t_turb} # 1 t_turb\n")
          ## define plt-file interval [t_sim]
          elif param_name == "plotFileIntervalTime":
            bool_defined_plt = True
            new_file.write(f"plotFileIntervalTime = {self.t_turb / 10} # 0.1 t_turb\n")
          ## define t_max [t_sim]
          elif param_name == "tmax":
            bool_defined_tmax = True
            new_file.write(f"tmax = {100 * self.t_turb} # 100 t_turb\n")
          ## define number of procs in i-direction
          elif param_name == "iProcs":
            bool_defined_iproc = True
            new_file.write(f"iProcs = {self.iprocs}\n")
          ## define number of procs in j-direction
          elif param_name == "jProcs":
            bool_defined_jproc = True
            new_file.write(f"jProcs = {self.jprocs}\n")
          ## define number of procs in k-direction
          elif param_name == "kProcs":
            bool_defined_kproc = True
            new_file.write(f"kProcs = {self.kprocs}\n")
          ## reset restart flag
          elif param_name == "restart":
            bool_no_restart = True
            new_file.write("restart = .false.\n")
          ## define chk-file number
          elif param_name == "checkpointFileNumber":
            bool_reset_chk_num = True
            new_file.write("checkpointFileNumber = 0\n")
          ## define plt-file number
          elif param_name == "plotFileNumber":
            bool_reset_plt_num = True
            new_file.write("plotFileNumber = 0\n")
          ## write other line contents
          else: new_file.write(ref_line_elems)
    ## check that all parameters have been defined
    list_bool = [
      bool_switched_nu,
      bool_switched_eta,
      bool_defined_nu,
      bool_defined_eta,
      bool_defined_runtime,
      bool_defined_chk,
      bool_defined_plt,
      bool_defined_tmax,
      bool_defined_iproc,
      bool_defined_jproc,
      bool_defined_kproc,
      bool_no_restart,
      bool_reset_chk_num,
      bool_reset_plt_num
    ]
    if not (False in list_bool):
      print("> Successfully modified 'flash.par'")
    else: raise Exception("ERROR: 'flash.par' failed to write:", list_bool)

## END OF LIBRARY
from TheUsefulModule import WWFnF
from TheJobModule.SimParams import SimParams

class PrepSimJob():
  def __init__(
      self,
      filepath_ref, filepath_sim,
      obj_sim_params : SimParams
    ):
    self.filepath_ref   = filepath_ref
    self.filepath_sim   = filepath_sim
    self.obj_sim_params = obj_sim_params
    self.suite_folder   = obj_sim_params.suite_folder
    self.sonic_regime   = obj_sim_params.sonic_regime
    self.sim_folder     = obj_sim_params.sim_folder
    self.sim_res        = obj_sim_params.sim_res
    self.num_blocks     = obj_sim_params.num_blocks
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
    ## t_turb [t_sim] = 1 / (k_turb * Mach * c_s)
    self.t_turb   = 1 / (self.obj_sim_params.k_turb * self.obj_sim_params.Mach)
    nxb, nyb, nzb = self.num_blocks
    self.iprocs   = int(self.sim_res) // nxb
    self.jprocs   = int(self.sim_res) // nyb
    self.kprocs   = int(self.sim_res) // nzb
    self.num_cpus = int(self.iprocs * self.jprocs * self.kprocs)
    self.max_mem  = int(4 * self.num_cpus)
    if self.num_cpus > 1000:
      self.max_hours = 24
    else: self.max_hours = 48
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
    ## define job details
    job_name = "job_run_sim.sh"
    ## create/overwrite job file
    with open(f"{self.filepath_sim}/{job_name}", "w") as job_file:
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
    print(f"\t> Created job '{job_name}' to run a FLASH simulation in:\n\t", self.filepath_sim)

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
          ## define viscosity is turned on
          elif list_ref_line_elems[0] == "useViscosity":
            new_file.write("useViscosity = .true.\n")
            bool_switched_nu = True
          ## define 'nu'
          elif list_ref_line_elems[0] == "diff_visc_nu":
            new_file.write("diff_visc_nu = {} # implies Re = {}\n".format(
              self.obj_sim_params.nu,
              self.obj_sim_params.Re
            ))
            bool_defined_nu = True
          ## define resistivity is turned on
          elif list_ref_line_elems[0] == "useMagneticResistivity":
            new_file.write("useMagneticResistivity = .true.\n")
            bool_switched_eta = True
          ## define 'eta'
          elif list_ref_line_elems[0] == "resistivity":
            new_file.write("resistivity = {} # implies Rm = {} and Pm = {}\n".format(
              self.obj_sim_params.eta,
              self.obj_sim_params.Rm,
              self.obj_sim_params.Pm
            ))
            bool_defined_eta = True
          ## define wall clock timelimit
          elif list_ref_line_elems[0] == "wall_clock_time_limit":
            new_file.write("wall_clock_time_limit = {} # closes sim and saves state\n".format(
              self.max_hours * 60 * 60 - 1000 # [seconds]
            ))
            bool_defined_runtime = True
          elif list_ref_line_elems[0] == "checkpointFileIntervalTime":
            new_file.write(f"checkpointFileIntervalTime = {self.t_turb} # 1 t_turb\n")
            bool_defined_chk = True
          elif list_ref_line_elems[0] == "plotFileIntervalTime":
            new_file.write(f"plotFileIntervalTime = {self.t_turb / 10} # 0.1 t_turb\n")
            bool_defined_plt = True
          elif list_ref_line_elems[0] == "tmax":
            new_file.write(f"tmax = {100 * self.t_turb} # 100 t_turb\n")
            bool_defined_tmax = True
          ## define 'iProcs'
          elif list_ref_line_elems[0] == "iProcs":
            new_file.write(f"iProcs = {self.iprocs}\n")
            bool_defined_iproc = True
          ## define 'jProcs'
          elif list_ref_line_elems[0] == "jProcs":
            new_file.write(f"jProcs = {self.jprocs}\n")
            bool_defined_jproc = True
          ## define 'kProcs'
          elif list_ref_line_elems[0] == "kProcs":
            new_file.write(f"kProcs = {self.kprocs}\n")
            bool_defined_kproc = True
          ## reset 'restart'
          elif list_ref_line_elems[0] == "restart":
            new_file.write("restart = .false.\n")
            bool_no_restart = True
          elif list_ref_line_elems[0] == "checkpointFileNumber":
            new_file.write("checkpointFileNumber = 0\n")
            bool_reset_chk_num = True
          elif list_ref_line_elems[0] == "plotFileNumber":
            new_file.write("plotFileNumber = 0\n")
            bool_reset_plt_num = True
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
      print("\t> Successfully modified 'flash.par'")
    else: raise Exception("ERROR: 'flash.par' failed to write:", list_bool)

## END OF LIBRARY
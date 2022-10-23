from TheUsefulModule import WWFnF
from TheJobModule.SimParams import SimParams

class PrepSimJob():
  def __init__(
      self,
      filepath_home, filepath_sim,
      suite_folder, sim_res, sim_folder,
      obj_sim_params : SimParams
    ):
    self.filepath_home   = filepath_home
    self.filepath_sim    = filepath_sim
    self.suite_folder    = suite_folder
    self.sim_res         = sim_res
    self.sim_folder      = sim_folder
    self.obj_sim_params  = obj_sim_params
    self.filename_flash4 = "flash4_nxb{}_nyb{}_nzb{}".format(
      self.obj_sim_params.num_blocks[0],
      self.obj_sim_params.num_blocks[1],
      self.obj_sim_params.num_blocks[2]
    )

  def copyFilesFromHome(self):
    ## copy flash4 executable from the home directory
    WWFnF.copyFileFromNTo(
      directory_from = self.filepath_home,
      directory_to   = self.filepath_sim,
      filename       = self.filename_flash4
    )
    ## copy forcing input file from the base directory
    WWFnF.copyFileFromNTo(
        directory_from = self.filepath_home,
        directory_to   = self.filepath_sim,
        filename       = "forcing_generator.inp"
    )

  def copyFilesFromLowerResSim(self):
    filepath_low_res_sim = WWFnF.createFilepath([
      self.filepath_home, self.suite_folder, "144", self.obj_sim_params.sonic_regime, self.sim_folder
    ])
    ## copy flash4 executable from the home directory
    WWFnF.copyFileFromNTo(
      directory_from = self.filepath_home,
      directory_to   = self.filepath_sim,
      filename       = self.filename_flash4
    )
    ## copy forcing input file from the Nres=144 directory
    WWFnF.copyFileFromNTo(
      directory_from = filepath_low_res_sim,
      directory_to   = self.filepath_sim,
      filename       = "forcing_generator.inp"
    )
    ## copy forcing data file from the Nres=144 directory
    WWFnF.copyFileFromNTo(
      directory_from = filepath_low_res_sim,
      directory_to   = self.filepath_sim,
      filename       = "turb_driving.dat"
    )

  def createJob(self):
    ## define job details
    job_name     = "job_run_sim.sh"
    job_tagname  = "{}{}{}sim{}".format(
      self.obj_sim_params.sonic_regime.split("_")[0],
      self.suite_folder,
      self.sim_folder,
      self.sim_res
    )
    nxb, nyb, nzb = self.obj_sim_params.num_blocks
    iprocs   = int(self.sim_res) // nxb
    jprocs   = int(self.sim_res) // nyb
    kprocs   = int(self.sim_res) // nzb
    num_cpus = int(iprocs * jprocs * kprocs)
    max_mem  = int(4 * num_cpus)
    if num_cpus > 1000: max_hours = 24
    else: max_hours = 48
    ## create/overwrite job file
    with open(f"{self.filepath_sim}/{job_name}", "w") as job_file:
      ## write contents
      job_file.write("#!/bin/bash\n")
      job_file.write("#PBS -P ek9\n")
      job_file.write("#PBS -q normal\n")
      job_file.write(f"#PBS -l walltime={max_hours}:00:00\n")
      job_file.write(f"#PBS -l ncpus={num_cpus}\n")
      job_file.write(f"#PBS -l mem={max_mem}GB\n")
      job_file.write("#PBS -l storage=scratch/ek9+gdata/ek9\n")
      job_file.write("#PBS -l wd\n")
      job_file.write(f"#PBS -N {job_tagname}\n")
      job_file.write("#PBS -j oe\n")
      job_file.write("#PBS -m bea\n")
      job_file.write(f"#PBS -M neco.kriel@anu.edu.au\n")
      job_file.write("\n")
      job_file.write(f"mpirun ./{self.filename_flash4} 1>shell_sim.out00 2>&1\n")
    ## print to terminal that job file has been created
    print(f"\t> Created job '{job_name}' to run a FLASH simulation")

  def modifyFlashParamFile(self):
    filepath_sim_flash_file = f"{self.filepath_sim}/flash.par"
    filepath_ref_flash_file = f"{self.filepath_home}/flash_template.par"
    bool_nu_turned_on  = False
    bool_set_nu        = False
    bool_eta_turned_on = False
    bool_set_eta       = False
    bool_set_runtime   = False
    bool_set_iproc     = False
    bool_set_jproc     = False
    bool_set_kproc     = False
    ## open new "flash.par" file
    with open(filepath_sim_flash_file, "w") as new_file:
      ## open the reference "flash.par" file
      with open(filepath_ref_flash_file, "r") as ref_file_lines:
        if (self.Re < 50):
          new_file.write("hy_diffuse_cfl = 0.2\n\n")
        ## loop over lines in "flash.par"
        for ref_line_elems in ref_file_lines:
          ## split contents (i.e. words) in the line
          list_ref_line_elems = ref_line_elems.split()
          ## handle empty lines
          if len(list_ref_line_elems) == 0:
            new_file.write("\n")
            continue
          ## found row where viscosity is turned on
          elif list_ref_line_elems[0] == "useViscosity":
            new_file.write("useViscosity = .true.\n")
            bool_nu_turned_on = True
          ## found row where 'nu' is defined
          elif list_ref_line_elems[0] == "diff_visc_nu":
            new_file.write(f"diff_visc_nu = {self.nu} # implies Re = {self.Re}\n")
            bool_set_nu = True
          ## found row where resistivity is turned on
          elif list_ref_line_elems[0] == "useMagneticResistivity":
            new_file.write("useMagneticResistivity = .true.\n")
            bool_eta_turned_on = True
          ## found row where 'eta' is defined
          elif list_ref_line_elems[0] == "resistivity":
            new_file.write(f"resistivity = {self.eta} # implies Rm = {self.Rm} and Pm = {self.Pm}\n")
            bool_set_eta = True
          ## found row where wall clock timelimit is defined
          elif list_ref_line_elems[0] == "wall_clock_time_limit":
            new_file.write("wall_clock_time_limit = {:} # closes sim and saves state\n".format(
              self.max_hours * 60 * 60 - 1000 # number of seconds
            ))
            bool_set_runtime = True
          ## found row where 'iProcs' is defined
          elif list_ref_line_elems[0] == "iProcs":
            new_file.write(f"iProcs = {self.iprocs} # num procs in i direction\n")
            bool_set_iproc = True
          ## found row where 'jProcs' is defined
          elif list_ref_line_elems[0] == "jProcs":
            new_file.write(f"jProcs = {self.jprocs} # num procs in i direction\n")
            bool_set_jproc = True
          ## found row where 'kProcs' is defined
          elif list_ref_line_elems[0] == "kProcs":
            new_file.write(f"kProcs = {self.kprocs} # num procs in i direction\n")
            bool_set_kproc = True
          ## otherwise write line contents
          else: new_file.write(ref_line_elems)
    ## check that all parameters have been defined
    if (bool_nu_turned_on  and
        bool_set_nu        and
        bool_eta_turned_on and
        bool_set_eta       and
        bool_set_runtime   and
        bool_set_iproc     and
        bool_set_jproc     and
        bool_set_kproc
      ):
      ## indicate function executed successfully
      print("\t> 'flash.par' has been successfully written.")
    else: raise Exception("ERROR: 'flash.par' failed to write correctly.")


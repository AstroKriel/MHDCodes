class PrepSpectCalcJob():
  def __init__(
      self,
      filepath_plt, suite_folder, sonic_regime, sim_folder, sim_res
    ):
    self.filepath_plt = filepath_plt
    self.suite_folder = suite_folder
    self.sonic_regime = sonic_regime
    self.sim_folder   = sim_folder
    self.sim_res      = sim_res
    self.max_hours    = int(8)
    self.num_cpus     = int(6)
    self.max_mem      = int(4 * self.num_cpus)
    self.program_name = "script_calc_spectra.py"
    self.job_name     = "job_calc_spect.sh"
    self.job_tagname  = "{}{}{}sim{}".format(
      self.sonic_regime.split("_")[0],
      suite_folder,
      sim_folder,
      sim_res
    )
    self.__createJob()

  def __createJob(self):
    ## create/overwrite job file
    with open(f"{self.filepath_plt}/{self.job_name}", "w") as job_file:
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
      job_file.write("{} -data_path {} -num_proc {} -check_only 1>shell_calc.out00 2>&1\n".format(
        self.program_name,
        self.filepath_plt, # path to plt files
        self.num_cpus
      ))
    ## print to terminal that job file has been created
    print(f"\t> Created job '{self.job_name}' to run '{self.program_name}'")


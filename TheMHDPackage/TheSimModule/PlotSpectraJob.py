## START OF LIBRARY
import os

class PlotSpectraJob():
  def __init__(
      self,
      filepath_sim, dict_sim_inputs,
      bool_verbose = True
    ):
    self.filepath_sim   = filepath_sim
    self.filepath_plt   = f"{self.filepath_sim}/plt"
    self.filepath_spect = f"{self.filepath_sim}/spect"
    self.suite_folder   = dict_sim_inputs["suite_folder"]
    self.sonic_regime   = dict_sim_inputs["sonic_regime"]
    self.sim_folder     = dict_sim_inputs["sim_folder"]
    self.sim_res        = dict_sim_inputs["sim_res"]
    self.bool_verbose   = bool_verbose
    if not os.path.exists(self.filepath_plt):
      if self.bool_verbose: print(self.filepath_plt, "does not exist.")
      return
    if not os.path.exists(self.filepath_spect):
      if self.bool_verbose: print(self.filepath_spect, "does not exist.")
      return
    self.max_hours    = int(3)
    self.num_cpus     = int(1)
    self.max_mem      = int(4 * self.num_cpus)
    self.program_name = "plot_spectra.py"
    self.job_name     = "job_plot_spect.sh"
    self.job_tagname  = "{}{}{}plot{}".format(
      self.sonic_regime.split("_")[0],
      self.suite_folder,
      self.sim_folder,
      self.sim_res
    )
    ## perform routine
    self.__createJob()

  def __createJob(self):
    ## create job file
    with open(f"{self.filepath_spect}/{self.job_name}", "w") as job_file:
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
      job_file.write(f"{self.program_name} -suite_path {self.filepath_sim}/.. -sim_folder {self.sim_folder} 1>shell_plot_spect.out00 2>&1\n")
    ## indicate progress
    if self.bool_verbose:
      print(f"Created PBS job:")
      print(f"\t> Job name: {self.job_name}")
      print(f"\t> Directory: {self.filepath_spect}")

## END OF LIBRARY
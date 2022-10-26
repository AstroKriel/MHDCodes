import os

class PrepPlotSpectra():
  def __init__(
      self,
      filepath_sim, suite_folder, sonic_regime, sim_folder, sim_res
    ):
    self.filepath_sim   = filepath_sim
    self.filepath_plt   = f"{self.filepath_sim}/plt"
    self.filepath_spect = f"{self.filepath_sim}/spect"
    if not os.path.exists(self.filepath_plt):
      print(self.filepath_plt, "does not exist.")
      return
    if not os.path.exists(self.filepath_spect):
      print(self.filepath_spect, "does not exist.")
      return
    self.max_hours    = int(3)
    self.num_cpus     = int(1)
    self.max_mem      = int(4 * self.num_cpus)
    self.program_name = "plot_spectra.py"
    self.job_name     = "job_plot_spect.sh"
    self.job_tagname  = "{}{}{}plot{}".format(
      sonic_regime.split("_")[0],
      self.suite_folder,
      self.sim_folder,
      self.sim_res
    )
    self.suite_folder = suite_folder
    self.sim_res      = sim_res
    self.sim_folder   = sim_folder
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
      job_file.write("{} -suite_path {} -sim_folder {} 1>shell_plot.out00 2>&1\n".format(
        self.program_name,
        f"{self.filepath_sim}/..", # path to simulation suite
        self.sim_folder
      ))
    ## print to terminal that job file has been created
    print(f"\t> Created job '{self.job_name}' to run '{self.program_name}'")


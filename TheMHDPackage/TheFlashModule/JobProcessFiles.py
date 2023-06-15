## START OF LIBRARY

from TheFlashModule import SimParams, FileNames

class JobProcessFiles():
  def __init__(
      self,
      filepath_plt, dict_sim_inputs,
      file_start_index = None,
      file_end_index   = None,
      bool_verbose     = True
    ):
    self.filepath_plt = filepath_plt
    self.bool_verbose = bool_verbose
    self.max_hours    = int(24)
    self.num_procs    = int(dict_sim_inputs["sim_res"]) // 6
    self.max_mem      = int(4 * self.num_procs)
    ## check group project
    if   "ek9" in self.filepath_plt: self.group_project = "ek9"
    elif "jh2" in self.filepath_plt: self.group_project = "jh2"
    else: raise Exception("Error: undefined group project.")
    self.program_name = FileNames.FILENAME_PROCESS_PLT_SCRIPT
    self.job_name     = FileNames.FILENAME_PROCESS_PLT_JOB
    self.job_output   = FileNames.FILENAME_PROCESS_PLT_OUTPUT
    self.job_tagname  = SimParams.getJobTag(dict_sim_inputs, "plt")
    self.command      = self.program_name
    self.command      += f" -data_path {self.filepath_plt}"
    self.command      += f" -num_procs {self.num_procs}"
    self.command      += f" -check_only"
    if file_start_index is not None: self.command += f" -file_start_index {int(file_start_index)}"
    if file_end_index   is not None: self.command += f" -file_end_index {int(file_end_index)}"
    ## perform routine
    self.__createJob()

  def __createJob(self):
    ## create/overwrite job file
    with open(f"{self.filepath_plt}/{self.job_name}", "w") as job_file:
      ## write contents
      job_file.write("#!/bin/bash\n")
      job_file.write(f"#PBS -P {self.group_project}\n")
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
      job_file.write(f"{self.command} 1>{self.job_output} 2>&1\n")
    ## indicate progress
    if self.bool_verbose:
      print(f"Created PBS job:")
      print(f"\t> Job name:",  self.job_name)
      print(f"\t> Directory:", self.filepath_plt)

## END OF LIBRARY
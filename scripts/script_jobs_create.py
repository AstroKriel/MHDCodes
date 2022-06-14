#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import shutil

## load old user defined modules
from TheUsefulModule import WWFnF


## ###############################################################
## FUNCTION: Create a job file that runs simulation
## ###############################################################
def funcPrepSimulation(
    filepath_home, filepath_sim,
    suite_folder, sim_res, sim_folder,
    num_blocks,
    Re = None,
    Rm = None,
    Pm = None
  ):
  ## ###################################
  ## CALCULATE NU AND ETA FOR SIMULATION
  ## ###################################
  if (Re is not None) and (Pm is not None):
    ## Re and Pm have been defined
    nu  = round(float(MACH) / (K_TURB * float(Re)), 5)
    eta = round(nu / float(Pm), 5)
    Rm  = round(float(MACH) / (K_TURB * eta))
  elif (Rm is not None) and (Pm is not None):
    ## Rm and Pm have been defined
    eta = round(float(MACH) / (K_TURB * float(Rm)), 5)
    nu  = round(eta * float(Pm), 5)
    Re  = round(float(MACH) / (K_TURB * nu))
  else:
    Exception("You have not defined the required number of plasma Reynolds numbers (Re = {:} or Rm = {:}, and Pm = {:}).".format(
      Re, Rm, Pm
    ))
    return
  print("\t> Re = {}, Rm = {}, Pm = {}, nu = {}, eta = {}".format(
    Re, Rm, Pm,
    nu, eta
  ))
  # ## #####################################
  # ## COPY SETUP FILES TO SIMULATION FOLDER
  # ## #####################################
  ## copy flash4 executable from the home directory
  filename_flash4 = "flash4_nxb{}_nyb{}_nzb{}_2.0".format(
    num_blocks[0],
    num_blocks[1],
    num_blocks[2]
  )
  funcCopy(
    directory_from = filepath_home,
    directory_to   = filepath_sim,
    filename       = filename_flash4
  )
  # # ## copy forcing input file from the base directory
  # # funcCopy(
  # #     directory_from = filepath_home,
  # #     directory_to   = filepath_sim,
  # #     filename       = "forcing_generator.inp"
  # # )
  ## copy forcing input file from the Nres=144 directory
  funcCopy(
    directory_from = WWFnF.createFilepath([
      filepath_home, suite_folder, "144", SONIC_REGIME, sim_folder
    ]),
    directory_to   = filepath_sim,
    filename       = "forcing_generator.inp"
  )
  ## copy forcing data file from the Nres=144 directory
  funcCopy(
    directory_from = WWFnF.createFilepath([
      filepath_home, suite_folder, "144", SONIC_REGIME, sim_folder
    ]),
    directory_to   = filepath_sim,
    filename       = "turb_driving.dat"
  )
  ## #################################
  ## CREATE JOB FILE TO RUN SIMULATION
  ## #################################
  max_hours, iprocs, jprocs, kprocs = funcCreateSimJob(
    filepath_sim,
    suite_folder, sim_res, sim_folder,
    filename_flash4,
    num_blocks
  )
  ## #######################################
  ## WRITE FLASH.PAR WITH DESIRED PARAMETERS
  ## #######################################
  funcWriteFlashParamFile(
    filepath_sim, filepath_home,
    nu, Re,
    eta, Rm, Pm,
    max_hours,
    iprocs, jprocs, kprocs
  )

def funcCreateSimJob(
    filepath_sim,
    suite_folder, sim_res, sim_folder,
    filename_flash4,
    num_blocks
  ):
  ## define job details
  job_name     = "job_run_sim.sh"
  filepath_job = filepath_sim + "/" + job_name
  job_tagname  = suite_folder + sim_folder + "sim" + sim_res
  ## number of processors required to run simulation with block setup [36, 36, 48] at a given linear resolution
  nxb, nyb, nzb = num_blocks
  iprocs   = int(sim_res) // nxb
  jprocs   = int(sim_res) // nyb
  kprocs   = int(sim_res) // nzb
  num_cpus = int(iprocs * jprocs * kprocs)
  max_mem  = int(4 * num_cpus)
  if num_cpus > 1000:
    max_hours = 24
  else: max_hours = 48
  ## create/overwrite job file
  with open(filepath_job, "w") as job_file:
    ## write contents
    job_file.write("#!/bin/bash\n")
    job_file.write("#PBS -P ek9\n")
    job_file.write("#PBS -q normal\n")
    job_file.write("#PBS -l walltime={}:00:00\n".format(max_hours))
    job_file.write("#PBS -l ncpus={}\n".format(num_cpus))
    job_file.write("#PBS -l mem={}GB\n".format(max_mem))
    job_file.write("#PBS -l storage=scratch/ek9+gdata/ek9\n")
    job_file.write("#PBS -l wd\n")
    job_file.write("#PBS -N {}\n".format(job_tagname))
    job_file.write("#PBS -j oe\n")
    job_file.write("#PBS -m bea\n")
    job_file.write("#PBS -M {}\n".format(EMAIL_ADDRESS))
    job_file.write("\n")
    job_file.write("mpirun ./{} 1>shell_sim.out00 2>&1\n".format(
      filename_flash4
    ))
  ## print to terminal that job file has been created
  print("\t> Created job '{}' to run a FLASH simulation".format( job_name ))
  return max_hours, iprocs, jprocs, kprocs

def funcWriteFlashParamFile(
    filepath_sim, filepath_home,
    nu, Re,
    eta, Rm, Pm,
    max_hours,
    iprocs, jprocs, kprocs
  ):
  filepath_sim_flash_file = filepath_sim  + "/flash.par"
  filepath_ref_flash_file = filepath_home + "/flash_template.par"
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
      if (Re < 50):
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
          new_file.write("diff_visc_nu = {:} # implies Re = {:}\n".format(
            nu, Re
          ))
          bool_set_nu = True
        ## found row where resistivity is turned on
        elif list_ref_line_elems[0] == "useMagneticResistivity":
          new_file.write("useMagneticResistivity = .true.\n")
          bool_eta_turned_on = True
        ## found row where 'eta' is defined
        elif list_ref_line_elems[0] == "resistivity":
          new_file.write("resistivity = {:} # implies Rm = {:} and Pm = {:}\n".format(
            eta, Rm, Pm
          ))
          bool_set_eta = True
        ## found row where wall clock timelimit is defined
        elif list_ref_line_elems[0] == "wall_clock_time_limit":
          new_file.write("wall_clock_time_limit = {:} # closes sim and saves state\n".format(
            max_hours*60*60 - 1000 # number of seconds
          ))
          bool_set_runtime = True
        ## found row where 'iProcs' is defined
        elif list_ref_line_elems[0] == "iProcs":
          new_file.write("iProcs = {:d} # num procs in i direction\n".format( int(iprocs)) )
          bool_set_iproc = True
        ## found row where 'jProcs' is defined
        elif list_ref_line_elems[0] == "jProcs":
          new_file.write("jProcs = {:d} # num procs in i direction\n".format( int(jprocs)) )
          bool_set_jproc = True
        ## found row where 'kProcs' is defined
        elif list_ref_line_elems[0] == "kProcs":
          new_file.write("kProcs = {:d} # num procs in i direction\n".format( int(kprocs)) )
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
  else: print("\t> ERROR: 'flash.par' failed to write correctly.")

def funcCopy(directory_from, directory_to, filename):
  shutil.copy( # copy the file and it's permissions (i.e. executable)
    directory_from + "/" + filename,
    directory_to   + "/" + filename
  )
  print("\t> Successfully coppied: {}".format( filename ))
  print("\t\t From: {}".format( directory_from ))
  print("\t\t To: {}".format( directory_to ))


## ###############################################################
## FUNCTION: Create a job file that calculates spectra
## ###############################################################
def funcCreateCalcSpectraJob(
    filepath_sim,
    suite_folder, sim_res, sim_folder
  ):
  program_name = "script_calc_spectra.py"
  job_name     = "job_calc_spect.sh"
  filepath_job = filepath_sim + "/" + job_name
  job_tagname  = "{}{}calc{}".format(
    suite_folder,
    sim_folder,
    sim_res
  )
  max_hours    = int(8)
  num_cpus     = int(6)
  max_mem      = int(4 * num_cpus)
  ## create/overwrite job file
  with open(filepath_job, "w") as job_file:
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
    job_file.write(f"#PBS -M {EMAIL_ADDRESS}\n")
    job_file.write("\n")
    job_file.write("{} -base_path {} -num_proc {} -check_only 1 1>shell_calc.out00 2>&1\n".format(
      program_name, # fitting program
      filepath_sim, # path to simulation suite
      num_cpus
    ))
  ## print to terminal that job file has been created
  print("\t> Created job '{}' to run '{}'".format(
    job_name,
    program_name
  ))


## ###############################################################
## CREATE JOB: FIT SPECTRA
## ###############################################################
class PrepFitSpectra():
  def __init__(
      self,
      filepath_scratch,
      filepath_sim,
      suite_folder,
      sim_res,
      sim_folder
    ):
    ## check simulation sub-folders exist
    self.filepath_sim     = filepath_sim
    self.filepath_plt     = self.filepath_sim + "/plt"
    self.filepath_spect   = self.filepath_sim + "/spect"
    if not os.path.exists(self.filepath_plt):
      print(self.filepath_plt, "does not exist.")
      return
    if not os.path.exists(self.filepath_spect):
      print(self.filepath_spect, "does not exist.")
      return
    ## store provided information
    self.filepath_scratch = filepath_scratch
    self.suite_folder     = suite_folder
    self.sim_res          = sim_res
    self.sim_folder       = sim_folder
    self.filepath_suite   = WWFnF.createFilepath([
      self.filepath_scratch,
      self.suite_folder,
      self.sim_res,
      SONIC_REGIME
    ])
  def getPlasmaNumbers(self):
    nu  = None
    eta = None
    with open(self.filepath_sim + "/flash.par") as file_lines:
      for line in file_lines:
        list_line_elems = line.split()
        ## ignore empty lines
        if len(list_line_elems) == 0:
          continue
        ## read value for 'diff_visc_nu'
        if list_line_elems[0] == "diff_visc_nu":
          nu = float(list_line_elems[2])
        ## read value for 'resistivity'
        if list_line_elems[0] == "resistivity":
          eta = float(list_line_elems[2])
        ## stop searching if both parameters have been identified
        if (nu is not None) and (eta is not None):
          break
    ## display parameter values found
    if (nu is not None) and (eta is not None):
      self.nu  = nu
      self.eta = eta
      self.Re  = int(MACH / (K_TURB * self.nu))
      self.Rm  = int(MACH / (K_TURB * self.eta))
      self.Pm  = int(self.nu / self.eta)
      print("\t> Re = {:}, Rm = {:}, Pm = {:}, nu = {:0.2e}, eta = {:0.2e},".format(
        self.Re, self.Rm, self.Pm,
        self.nu, self.eta
      ))
    else: Exception("\t> ERROR: Could not find {}{}{}.".format(
      "nu"   if (nu is None)  else "",
      " or " if (nu is None) and (eta is None) else "",
      "eta"  if (eta is None) else ""
    ))
  def createJob(self):
    program_name   = "plot_spectra_1_fit.py"
    if BOOL_FIT_FIXED:
      job_name     = "job_fit_spect_fixed.sh"
    else: job_name = "job_fit_spect.sh"
    job_tagname    = "{}{}{}fit{}".format(
      SONIC_REGIME.split("_")[0],
      self.suite_folder,
      self.sim_folder,
      self.sim_res
    )
    max_hours      = int(8)
    num_cpus       = int(2)
    max_mem        = int(4 * num_cpus)
    str_fit_args   = "-f -p"
    str_fit_args   += " -k_turb_end {}".format(K_TURB + K_TURB_WIDTH)
    str_fit_args   += " -kin_fit_sub_y_range -kin_num_decades_to_fit {}".format(6)
    if BOOL_FIT_FIXED:
      str_fit_args += " -kin_fit_fixed -mag_fit_fixed"
    ## create job file
    filepath_job = self.filepath_spect + "/" + job_name
    with open(filepath_job, "w") as job_file:
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
      job_file.write(f"#PBS -M {EMAIL_ADDRESS}\n")
      job_file.write("\n")
      job_file.write("{} -suite_path {} -sim_folder {} -sim_suite {}_{} -sim_res {} -Re {} -Rm {} {} 1>shell_fit.out00 2>&1\n".format(
        program_name,               # fitting program
        self.filepath_suite,        # path to simulation suite
        self.sim_folder,            # simulation name
        SONIC_REGIME.split("_")[0], # sonic regime (sub / super)
        self.suite_folder,          # suite name
        self.sim_res,               # simulation linear resolution
        self.Re, self.Rm,           # plasma Reynolds numbers
        str_fit_args                # extra arguments
      ))
    ## print to terminal that job file has been created
    print("\t> Created job '{}' to run '{}'".format(
      job_name,
      program_name
    ))


## ###############################################################
## CREATE JOB: PLOT SPECTRA
## ###############################################################
class PrepPlotSpectra():
  def __init__(
      self,
      filepath_scratch,
      filepath_sim,
      suite_folder,
      sim_res,
      sim_folder
    ):
    ## check simulation sub-folders exist
    self.filepath_sim     = filepath_sim
    self.filepath_plt     = self.filepath_sim + "/plt"
    self.filepath_spect   = self.filepath_sim + "/spect"
    if not os.path.exists(self.filepath_plt):
      print(self.filepath_plt, "does not exist.")
      return
    if not os.path.exists(self.filepath_spect):
      print(self.filepath_spect, "does not exist.")
      return
    ## store provided information
    self.filepath_scratch = filepath_scratch
    self.suite_folder     = suite_folder
    self.sim_res          = sim_res
    self.sim_folder       = sim_folder
    self.filepath_suite   = WWFnF.createFilepath([
      self.filepath_scratch,
      self.suite_folder,
      self.sim_res,
      SONIC_REGIME
    ])
  def createJob(self):
    program_name   = "plot_spectra.py"
    job_name       = "job_plot_spect.sh"
    job_tagname    = "{}{}{}plot{}".format(
      SONIC_REGIME.split("_")[0],
      self.suite_folder,
      self.sim_folder,
      self.sim_res
    )
    max_hours      = int(3)
    num_cpus       = int(1)
    max_mem        = int(4 * num_cpus)
    ## create job file
    filepath_job = self.filepath_spect + "/" + job_name
    with open(filepath_job, "w") as job_file:
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
      job_file.write(f"#PBS -M {EMAIL_ADDRESS}\n")
      job_file.write("\n")
      job_file.write("{} -suite_path {} -sim_folder {} 1>shell_plot.out00 2>&1\n".format(
        program_name,        # fitting program
        self.filepath_suite, # path to simulation suite
        self.sim_folder,     # simulation name
      ))
    ## print to terminal that job file has been created
    print("\t> Created job '{}' to run '{}'".format(
      job_name,
      program_name
    ))


## ###############################################################
## MAIN PROGRAM
## ###############################################################
EMAIL_ADDRESS      = "neco.kriel@anu.edu.au"
BASEPATH           = "/scratch/ek9/nk7952/"
BOOL_PREP_SIM      = 0
BOOL_CALC_SPECTRA  = 0
BOOL_PLOT_SPECTRA  = 0
BOOL_FIT_SPECTRA   = 1
BOOL_FIT_FIXED     = 0
NUM_BLOCKS         = [ 36, 36, 48 ]
SONIC_REGIME       = "super_sonic"
MACH               = 5.0
K_TURB             = 2
K_TURB_WIDTH       = 1

def main():
  ## ##############################
  ## LOOK AT EACH SIMULATION FOLDER
  ## ##############################
  ## loop over the simulation suites
  for suite_folder in [
      "Rm3000"
    ]: # "Re10", "Re500", "Rm3000", "keta"

    ## loop over the different resolution runs
    for sim_res in [
        "144"
      ]: # "18", "36", "72", "144", "288", "576"

      ## print to the terminal what suite is being looked at
      str_msg = "Looking at suite: {}, Nres = {}".format( suite_folder, sim_res )
      print(str_msg)
      print("=" * len(str_msg))

      ## loop over the simulation folders
      for sim_folder in [
          "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"
        ]: # "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"

        ## ##################################
        ## CHECK THE SIMULATION FOLDER EXISTS
        ## ##################################
        ## create filepath to simulation directory
        filepath_sim = WWFnF.createFilepath([
          BASEPATH, suite_folder, sim_res, SONIC_REGIME, sim_folder
        ])
        ## check that the simulation directory exists
        if not os.path.exists(filepath_sim):
          print(filepath_sim, "does not exist.")
          continue
        ## indicate which folder is being worked on
        print("Looking at: {}".format(filepath_sim))

        ## #################################
        ## CREATE JOB FILE TO RUN SIMULATION
        ## #################################
        if BOOL_PREP_SIM:
          funcPrepSimulation(
            filepath_home = BASEPATH,
            filepath_sim  = filepath_sim,
            suite_folder  = suite_folder,
            sim_res       = sim_res,
            sim_folder    = sim_folder,
            num_blocks    = NUM_BLOCKS,
            Re = float(
              suite_folder.replace("Re", "")
            ) if "Re" in suite_folder else None,
            Rm = float(
              suite_folder.replace("Rm", "")
            ) if "Rm" in suite_folder else None,
            Pm = float(
              sim_folder.replace("Pm", "")
            ) if "Pm" in sim_folder else None
          )

        ## ####################################
        ## CREATE JOB FILE TO CALCULATE SPECTRA
        ## ####################################
        if BOOL_CALC_SPECTRA:
          funcCreateCalcSpectraJob(
            filepath_sim = filepath_sim,
            suite_folder = suite_folder,
            sim_res      = sim_res,
            sim_folder   = sim_folder
          )

        ## ###############################
        ## CREATE JOB FILE TO PLOT SPECTRA
        ## ###############################
        if BOOL_PLOT_SPECTRA:
          plot_spectra_obj = PrepPlotSpectra(
            filepath_scratch = BASEPATH,
            filepath_sim     = filepath_sim,
            suite_folder     = suite_folder,
            sim_res          = sim_res,
            sim_folder       = sim_folder
          )
          plot_spectra_obj.createJob()

        ## ##############################
        ## CREATE JOB FILE TO FIT SPECTRA
        ## ##############################
        if BOOL_FIT_SPECTRA:
          fit_spectra_obj = PrepFitSpectra(
            filepath_scratch = BASEPATH,
            filepath_sim     = filepath_sim,
            suite_folder     = suite_folder,
            sim_res          = sim_res,
            sim_folder       = sim_folder
          )
          fit_spectra_obj.getPlasmaNumbers()
          fit_spectra_obj.createJob()

        ## clear line if things have been printed
        print(" ")
      print(" ")
    print(" ")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
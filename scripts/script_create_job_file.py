#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import shutil
import argparse
## load old user defined modules
from OldModules.the_useful_library import *


## ###############################################################
## FUNCTION: Create a job file that runs simulation
## ###############################################################
def funcPrepSimulation(
    filepath_home, filepath_sim,
    suite_folder, sim_res, sonic_regime, sim_folder,
    num_blocks,
    rms_Mach = 5.0,
    ell_turb = 0.5, # ell_turb = 1/k_turb = 1/2
    Re = None,
    Rm = None,
    Pm = None
  ):
  ## ###################################
  ## CALCULATE NU AND ETA FOR SIMULATION
  ## ###################################
  if (Re is not None) and (Pm is not None):
    nu  = round(float(rms_Mach) * ell_turb / float(Re), 5)
    eta = round(nu / float(Pm), 5)
    Rm  = round(float(rms_Mach) * ell_turb / eta)
  elif (Rm is not None) and (Pm is not None):
    eta = round(float(rms_Mach) * ell_turb / float(Rm), 5)
    nu  = round(eta * float(Pm), 5)
    Re  = round(float(rms_Mach) * ell_turb / nu)
  else:
    print("You have not defined the required number of plasma Reynolds numbers (Re = {:}, Rm = {:}, Pm = {:}).".format(
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
    directory_from = createFilepath([
      filepath_home, suite_folder, "144", sonic_regime, sim_folder
    ]),
    directory_to   = filepath_sim,
    filename       = "forcing_generator.inp"
  )
  ## copy forcing data file from the Nres=144 directory
  funcCopy(
    directory_from = createFilepath([
      filepath_home, suite_folder, "144", sonic_regime, sim_folder
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
    bool_set_kproc):
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
  # define job details
  filename_execute_program = "calc_spectra_data.py"
  job_name     = "job_calc_spect.sh"
  filepath_job = filepath_sim + "/" + job_name
  job_tagname  = suite_folder + sim_folder + "spectra" + sim_res
  max_hours    = int(8)
  num_cpus     = int(6)
  max_mem      = int(4*num_cpus)
  ## create/overwrite job file
  with open(filepath_job, "w") as job_file:
    ## write contents
    job_file.write("#!/bin/bash\n")
    job_file.write("#PBS -P ek9\n")
    job_file.write("#PBS -q normal\n")
    job_file.write("#PBS -l walltime={}:00:00\n".format( max_hours ))
    job_file.write("#PBS -l ncpus={}\n".format( num_cpus ))
    job_file.write("#PBS -l mem={}GB\n".format( max_mem ))
    job_file.write("#PBS -l storage=scratch/ek9+gdata/ek9\n")
    job_file.write("#PBS -l wd\n")
    job_file.write("#PBS -N {}\n".format( job_tagname ))
    job_file.write("#PBS -j oe\n")
    job_file.write("#PBS -m bea\n")
    job_file.write("#PBS -M {}\n".format( EMAIL_ADDRESS ))
    job_file.write("\n")
    job_file.write("{} -base_path {} -num_proc {} -check_only 1 1>shell_calc.out00 2>&1\n".format(
      filename_execute_program, # program
      filepath_sim,
      num_cpus
    ))
  ## print to terminal that job file has been created
  print("\t> Created job '{}' to run '{}'".format(
    job_name,
    filename_execute_program
  ))


## ###############################################################
## FUNCTION: Create a job file that runs spectra fitting
## ###############################################################
def funcGetPlasmaNumbers(filepath_sim, rms_Mach):
  filepath_flash_param_file = filepath_sim + "/flash.par"
  nu = None
  eta = None
  with open(filepath_flash_param_file) as file_lines:
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
    Re = int(rms_Mach / nu)
    Rm = int(rms_Mach / eta)
    Pm = int(nu / eta)
    print("\t> Re = {:}, Rm = {:}, Pm = {:}, nu = {:0.2e}, eta = {:0.2e},".format(
      Re, Rm, Pm,
      nu, eta
    ))
    return Re, Rm, Pm, nu, eta
  else: Exception("\t> ERROR: Could not find {}{}{}.".format(
    "nu"   if (nu is None)  else "",
    " or " if (nu is None)  and (eta is None) else "",
    "eta"  if (eta is None) else ""
  ))

def funcCreateFitJob(
    filepath_base, filepath_sim_spect,
    suite_folder, sim_res, sonic_regime, sim_folder,
    Re, Rm
  ):
  # define job details
  filename_execute_program = "plot_spectra_1_fit.py"
  job_name     = "job_fit_spect.sh"
  filepath_job = filepath_sim_spect + "/" + job_name
  job_tagname  = suite_folder + sim_folder + "fit" + sim_res
  max_hours    = int(8)
  num_cpus     = int(2)
  max_mem      = int(4*num_cpus)
  ## create/overwrite job file
  with open(filepath_job, "w") as job_file:
    ## write contents
    job_file.write("#!/bin/bash\n")
    job_file.write("#PBS -P ek9\n")
    job_file.write("#PBS -q normal\n")
    job_file.write("#PBS -l walltime={}:00:00\n".format( max_hours ))
    job_file.write("#PBS -l ncpus={}\n".format( num_cpus ))
    job_file.write("#PBS -l mem={}GB\n".format( max_mem ))
    job_file.write("#PBS -l storage=scratch/ek9+gdata/ek9\n")
    job_file.write("#PBS -l wd\n")
    job_file.write("#PBS -N {}\n".format( job_tagname ))
    job_file.write("#PBS -j oe\n")
    job_file.write("#PBS -m bea\n")
    job_file.write("#PBS -M {}\n".format( EMAIL_ADDRESS ))
    job_file.write("\n")
    job_file.write("{} -base_path {} -sim_folders {} -analyse 1 -Re {} -Rm {} -sim_suites super_{} -plot_spectra 1 -hide_updates 1 -fit_sub_Ek_range 1 -log_Ek_range 6 1>shell_fit.out00 2>&1\n".format(
      filename_execute_program, # program
      createFilepath([
        filepath_base, suite_folder, sim_res, sonic_regime
      ]),
      sim_folder,  # simulation name
      Re, Rm,      # plasma Reynolds numbers
      suite_folder # suite name
    ))
  ## print to terminal that job file has been created
  print("\t> Created job '{}' to run '{}'".format(
    job_name,
    filename_execute_program
  ))

def funcPrepSpectraFit(
    filepath_base, filepath_sim,
    suite_folder, sim_res, sonic_regime, sim_folder,
    rms_Mach = 5.0 # TODO: read from Turb.dat
  ):
  filepath_sim_plt   = filepath_sim + "/plt"
  filepath_sim_spect = filepath_sim + "/spect"
  ## check that the 'plt' folder exists
  if not os.path.exists(filepath_sim_plt):
    print(filepath_sim_plt, "does not exist.")
    return
  ## check that the 'spect' folder exists
  if not os.path.exists(filepath_sim_spect):
    print(filepath_sim_spect, "does not exist.")
    return
  ## #############################
  ## READ RE AND RM FROM FLASH.PAR
  ## #############################
  Re, Rm, _, _, _ = funcGetPlasmaNumbers(filepath_sim, rms_Mach)
  ## ################################
  ## CREATE JOB FILE: FITTING SPECTRA
  ## ################################
  funcCreateFitJob(
    filepath_base, filepath_sim_spect,
    suite_folder, sim_res, sonic_regime, sim_folder,
    Re, Rm
  )


## ###############################################################
## MAIN PROGRAM
## ###############################################################
EMAIL_ADDRESS = "neco.kriel@anu.edu.au"
def main():
  ## #############################
  ## DEFINE COMMAND LINE ARGUMENTS
  ## #############################
  parser = MyParser()
  ## ------------------- DEFINE OPTIONAL ARGUMENTS
  args_opt = parser.add_argument_group(description='Optional processing arguments:')
  ## define typical input requirements
  bool_args = {"required":False, "type":str2bool, "nargs":"?", "const":True}
  ## program inputs
  args_opt.add_argument("-make_sim",     default=False, **bool_args)
  args_opt.add_argument("-calc_spectra", default=False, **bool_args)
  args_opt.add_argument("-fit_spectra",  default=False, **bool_args)
  ## information about the directory
  args_opt.add_argument("-sub_folder",   required=False, type=str, default="")
  args_opt.add_argument("-sonic_regime", required=False, type=str, default="super_sonic")
  ## simulation details
  args_opt.add_argument("-num_blocks",   required=False, type=int, default=[36, 36, 48], nargs="+")
  ## ------------------- DEFINE REQUIRED ARGUMENTS
  args_req = parser.add_argument_group(description='Required processing arguments:')
  ## required inputs
  args_req.add_argument("-base_path",    type=str, required=True)
  args_req.add_argument("-sim_suites",   type=str, required=True, nargs="+")
  args_req.add_argument("-sim_res",      type=str, required=True, nargs="+")
  args_req.add_argument("-sim_folders",  type=str, required=True, nargs="+")

  ## #########################
  ## INTERPRET INPUT ARGUMENTS
  ## #########################
  ## ---------------------------- OPEN ARGUMENTS
  args = vars(parser.parse_args())
  ## ---------------------------- SAVE PARAMETERS
  ## booleans to determine what jobs the program creates
  bool_make_sim      = args["make_sim"]
  bool_calc_spectra  = args["calc_spectra"]
  bool_fit_spectra   = args["fit_spectra"]
  ## directory information
  filepath_base      = args["base_path"]
  list_suite_folders = args["sim_suites"]
  list_sim_res       = args["sim_res"]
  sonic_regime       = args["sonic_regime"]
  list_sim_folders   = args["sim_folders"]
  sub_folder         = args["sub_folder"]
  ## simulation details
  num_blocks         = args["num_blocks"]

  ## ####################
  ## PROCESS MAIN PROGRAM
  ## ####################
  ## loop over the simulation suites
  for suite_folder in list_suite_folders:
    ## loop over the different resolution runs
    for sim_res in list_sim_res:
      ## print to the terminal what suite is being looked at
      str_msg = "Looking at suite: {}, Nres = {}".format( suite_folder, sim_res )
      print(str_msg)
      print("=" * len(str_msg))
      ## loop over the simulation folders
      for sim_folder in list_sim_folders:

        ## ##################################
        ## CHECK THE SIMULATION FOLDER EXISTS
        ## ##################################
        ## create filepath to simulation folder (on GADI)
        filepath_sim = createFilepath([
          filepath_base, suite_folder, sim_res, sonic_regime, sim_folder, sub_folder
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
        if bool_make_sim:
          funcPrepSimulation(
            ## directories
            filepath_base, # home (SCRATCH) directory
            filepath_sim,  # simulation directory
            ## simulation details
            suite_folder, sim_res, sonic_regime, sim_folder,
            num_blocks = num_blocks,
            ## simulation parameters
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
        if bool_calc_spectra:
          funcCreateCalcSpectraJob(
            filepath_sim,
            suite_folder, sim_res, sim_folder
          )

        ## ##############################
        ## CREATE JOB FILE TO FIT SPECTRA
        ## ##############################
        if bool_fit_spectra:
          funcPrepSpectraFit(
            filepath_base,
            createFilepath([
              filepath_base, suite_folder, sim_res, sonic_regime, sim_folder
            ]),
            suite_folder, sim_res, sonic_regime, sim_folder
          )

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
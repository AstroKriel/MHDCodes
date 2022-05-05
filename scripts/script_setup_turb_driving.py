#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import subprocess
import shutil

from os import path
from datetime import datetime

## load old user defined modules
from OldModules.the_useful_library import *
from OldModules.the_loading_library import *


## ###############################################################
## FUNCTION: writing 'forcing_generator.inp'
## ###############################################################
def funcWriteForceGenInput(
    filepath_ref_file, filepath_new_file,
    des_velocity         = 5.0,
    des_k_driv           = 2.0,
    des_k_min            = 1.0,
    des_k_max            = 3.0,
    des_solweight        = 1.0,    # solenoidal driving
    des_spectform        = 1.0,    # paraboloid driving profile
    des_energy_coeff     = 5.0e-3, # energy prefactor
    des_end_time         = 100,
    des_nsteps_per_teddy = 10,
    num_space_pad        = 35
  ):
  ## initialise boolean variables
  bool_set_velocity         = False
  bool_set_k_driv           = False
  bool_set_k_min            = False
  bool_set_k_max            = False
  bool_set_solweight        = False
  bool_set_spectform        = False
  bool_set_energy_coeff     = False
  bool_set_end_time         = False
  bool_set_nsteps_per_teddy = False
  ## initialise lines to write in file
  str_velocity         = "velocity                 = {}".format(des_velocity)
  str_k_driv           = "k_driv                   = {}".format(des_k_driv)
  str_k_min            = "k_min                    = {}".format(des_k_min)
  str_k_max            = "k_max                    = {}".format(des_k_max)
  str_solweight        = "st_solweight             = {}".format(des_solweight)
  str_spectform        = "st_spectform             = {}".format(des_spectform)
  str_energy_coeff     = "st_energy_coeff          = {:0.4f}".format(des_energy_coeff)
  str_end_time         = "end_time                 = {}".format(des_end_time)
  str_nsteps_per_teddy = "nsteps_per_turnover_time = {}".format(des_nsteps_per_teddy)
  ## open new file
  with open(filepath_new_file, "w") as new_file:
    ## open refernce file
    with open(filepath_ref_file, "r") as ref_file_lines:
      ## loop over lines in reference file
      for ref_line_elems in ref_file_lines:
        ## split contents
        list_ref_line_elems = ref_line_elems.split()
        ## handle empty lines
        if len(list_ref_line_elems) == 0:
          new_file.write("\n")
          continue
        ## found row where 'velocity' is defined
        elif list_ref_line_elems[0] == "velocity":
          new_file.write("{}{} ! Target velocity dispersion\n".format(
            str_velocity,
            " " * ( num_space_pad - len(str_velocity) )
          ))
          bool_set_velocity = True
        ## found row where 'k_driv' is defined
        elif list_ref_line_elems[0] == "k_driv":
          new_file.write("{}{} ! Characteristic driving scale in units of 2pi / Lx.\n".format(
            str_k_driv,
            " " * ( num_space_pad - len(str_k_driv) )
          ))
          bool_set_k_driv = True
        ## found row where 'k_min' is defined
        elif list_ref_line_elems[0] == "k_min":
          new_file.write("{}{} ! Minimum driving wavnumber in units of 2pi / Lx\n".format(
            str_k_min,
            " " * ( num_space_pad - len(str_k_min) )
          ))
          bool_set_k_min = True
        ## found row where 'k_max' is defined
        elif list_ref_line_elems[0] == "k_max":
          new_file.write("{}{} ! Maximum driving wavnumber in units of 2pi / Lx\n".format(
            str_k_max,
            " " * ( num_space_pad - len(str_k_max) )
          ))
          bool_set_k_max = True
        ## found row where 'st_solweight' is defined
        elif list_ref_line_elems[0] == "st_solweight":
          new_file.write("{}{} ! 1.0: solenoidal driving, 0.0: compressive driving, 0.5: natural mixture\n".format(
            str_solweight,
            " " * ( num_space_pad - len(str_solweight) )
          ))
          bool_set_solweight = True
        ## found row where 'st_spectform' is defined
        elif list_ref_line_elems[0] == "st_spectform":
          new_file.write("{}{} ! 0: band, 1: paraboloid, 2: power law\n".format(
            str_spectform,
            " " * ( num_space_pad - len(str_spectform) )
          ))
          bool_set_spectform = True
        ## found row where 'st_energy_coeff' is defined
        elif list_ref_line_elems[0] == "st_energy_coeff":
          new_file.write("{}{} ! Used to adjust to target velocity; scales with (velocity/velocity_measured)^3.\n".format(
            str_energy_coeff,
            " " * ( num_space_pad - len(str_energy_coeff) )
          ))
          bool_set_energy_coeff = True
        ## found row where 'end_time' is defined
        elif list_ref_line_elems[0] == "end_time":
          new_file.write("{}{} ! End time of forcing sequence in units of turnover times\n".format(
            str_end_time,
            " " * ( num_space_pad - len(str_end_time) )
          ))
          bool_set_end_time = True
        ## found row where 'nsteps_per_turnover_time' is defined
        elif list_ref_line_elems[0] == "nsteps_per_turnover_time":
          new_file.write("{}{} ! number of forcing patterns per turnover time\n".format(
            str_nsteps_per_teddy,
            " " * ( num_space_pad - len(str_nsteps_per_teddy) )
          ))
          bool_set_nsteps_per_teddy = True
        ## found line where comment overflows
        elif (list_ref_line_elems[0] == "!") and ("*" not in list_ref_line_elems[1]):
          new_file.write("{}! {}\n".format(
            " " * num_space_pad,
            " ".join(list_ref_line_elems[1:])
          ))
        ## otherwise write line contents
        else: new_file.write(ref_line_elems)
  ## check that all parameters have been defined
  if ((bool_set_velocity         is not None) and
    (bool_set_k_driv           is not None) and
    (bool_set_k_min            is not None) and
    (bool_set_k_max            is not None) and
    (bool_set_solweight        is not None) and
    (bool_set_spectform        is not None) and
    (bool_set_energy_coeff     is not None) and
    (bool_set_end_time         is not None) and
    (bool_set_nsteps_per_teddy is not None)):
    ## indicate function executed successfully
    print("\t> The '{}' has been successfully written.".format( FILENAME_GEN_INPUT ))
    return True
  else:
    print("\t> ERROR: '{}' failed to write correctly.".format( FILENAME_GEN_INPUT ))
    return False


## ###############################################################
## FUNCTION: read 'energy prefactor' in 'forcing_generator.inp'
## ###############################################################
def funcReadEnergyPrefactor(filepath_file):
  ## check the file exists
  if not path.isfile(filepath_file):
    raise Exception("\t> ERROR: the input force generator '{}' does not exist.".format( FILENAME_GEN_INPUT ))
  ## open file
  with open(filepath_file) as file_lines:
    ## loop over file lines
    for line in file_lines:
      ## split line into elements seperated by a space
      list_line_elems = line.split()
      ## ignore empty lines
      if len(list_line_elems) == 0:
        continue
      ## read value for 'st_energy_coeff'
      if list_line_elems[0] == "st_energy_coeff":
        return float(list_line_elems[2])
  ## if the prefactor wasn't found
  raise Exception("\t> ERROR: The energy prefactory 'st_energy_coeff' was not found in '{}'".format( FILENAME_GEN_INPUT ))


FILENAME_GEN_INPUT        = "forcing_generator.inp"
FILENAME_GEN_INTERMEDIATE = "turb_v5.00E+00_zeta1.0_seed140281.dat"
FILENAME_GEN_OUTPUT       = "turb_driving.dat"
FILENAME_SIM_OUTPUT       = "Turb.dat"
FILENAME_DRIVING_HISTORY  = "driving_history.txt"
## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## #############################
  ## DEFINE COMMAND LINE ARGUMENTS
  ## #############################
  parser = MyParser()
  ## ------------------- DEFINE INPUT ARGUMENTS
  args_input = parser.add_argument_group(description="Processing arguments:")
  args_input.add_argument("-base_path",    type=str,   required=True) # home directory
  args_input.add_argument("-sim_suites",   type=str,   required=True, nargs="+") # list of simulation suites
  args_input.add_argument("-sim_res",      type=str,   required=True, nargs="+") # list of simulation resolutions
  args_input.add_argument("-sonic_regime", type=str,   required=True) # sub folder
  args_input.add_argument("-sim_folders",  type=str,   required=True, nargs="+") # list of simulation folders
  args_input.add_argument("-des_Mach",     type=float, required=True) # desired Mach number
  args_input.add_argument("-des_k_driv",   type=float, required=True) # driving scale
  args_input.add_argument("-des_k_min",    type=float, required=True) # min driving scale
  args_input.add_argument("-des_k_max",    type=float, required=True) # max driving scale

  ## #########################
  ## INTERPRET INPUT ARGUMENTS
  ## #########################
  ## ---------------------------- OPEN ARGUMENTS
  args = vars(parser.parse_args())
  ## ---------------------------- SAVE PARAMETERS
  ## directory information
  filepath_base      = args["base_path"]
  list_suite_folders = args["sim_suites"]
  list_sim_res       = args["sim_res"]
  sonic_regime       = args["sonic_regime"]
  list_sim_folders   = args["sim_folders"]
  ## driving parameters
  des_Mach           = args["des_Mach"]
  des_k_driv         = args["des_k_driv"]
  des_k_min          = args["des_k_min"]
  des_k_max          = args["des_k_max"]

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

        ## #######################################
        ## CHECK THAT THE SIMULATION FOLDER EXISTS
        ## #######################################
        ## create filepath to simulation folder (on GADI)
        filepath_sim  = createFilepath([
          filepath_base, suite_folder, sim_res, sonic_regime, sim_folder
        ])
        ## check that the simulation filepath exists
        if not path.exists(filepath_sim):
          print(filepath_sim, "does not exist.")
          continue
        ## indicate which folder is being worked on
        print("Looking at: {}".format(filepath_sim))

        ## ########################################################
        ## CHECK THAT THE DRIVING PARAMETERS HAVE NOT CONVERGED YET
        ## ########################################################
        bool_update_turb    = False
        bool_run_simulation = False
        bool_done           = False
        list_prev_Mach  = []
        list_prev_coeff = []
        if path.isfile(filepath_sim + "/" + FILENAME_DRIVING_HISTORY):
          ## look for the term "done" in the file
          with open(filepath_sim + "/" + FILENAME_DRIVING_HISTORY, "r") as history_file_lines:
            for line in history_file_lines:
              list_prev_Mach.append(  line.split("\t")[1] )
              list_prev_coeff.append( line.split("\t")[2].split("->")[1] )
              if ("done" in line.lower()):
                bool_done = True
          print("\t> Previous Mach numbers:", list_prev_Mach)
          print("\t> Previous energy coefficients:", list_prev_coeff)
          ## if the term "done" appears in the file then skip this simulation
          if bool_done:
            print("\t> The '{}' indicates that the driving parameter has converged.\n".format(
              FILENAME_DRIVING_HISTORY
            ))
            continue

        ## ###############################################
        ## CHECK IF THE SIMULATION OUTPUT DATA FILE EXISTS
        ## ###############################################
        ## keep a history of the turbulent forcing parameters
        with open(filepath_sim + "/" + FILENAME_DRIVING_HISTORY, "a") as history_file:
          ## if the simulation has not been run yet (i.e. there is no 'Turb.dat' file in the simulation folder)
          if not path.isfile(filepath_sim + "/" + FILENAME_SIM_OUTPUT):
            print("\t> '{}' does not exist (i.e. there is no data to look at).".format( FILENAME_SIM_OUTPUT ))
            ## check if a previous energy coefficient can be used (i.e. exists)
            if (len(list_prev_coeff) > 0) and (list_prev_coeff[-1] is not None):
              des_energy_coeff = float(list_prev_coeff[-1])
            ## default starting energy coefficient
            else: des_energy_coeff = 5.0e-3
            print("\t> Using energy coefficient: {}".format( des_energy_coeff ))
            ## indicate that the setup needs to happen
            bool_update_turb = True
            bool_run_simulation = True
            history_file.write("{}\t{}\t{}->{}\t\n".format(
              ## current time
              datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
              ## previous Mach number
              None,
              ## previous -> new prefactor
              None, des_energy_coeff
            ))
          ## if the driving file does not exist
          elif not path.isfile(filepath_sim + "/" + FILENAME_GEN_OUTPUT):
            print("\t> '{}' does not exist.".format( FILENAME_GEN_OUTPUT ))
            ## default starting energy coefficient
            des_energy_coeff = 5.0e-3
            print("\t> Using energy coefficient: {}".format( des_energy_coeff ))
            ## indicate that the setup needs to happen
            bool_update_turb = True
            bool_run_simulation = True
            history_file.write("{}\t{}\t{}->{}\t\n".format(
              ## current time
              datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
              ## previous Mach number
              None,
              ## previous -> new prefactor
              None, des_energy_coeff
            ))
          ## if the simulation has been run: check if the Mach number is close to the desired value
          else:
            ## read previous energy prefactor
            if (len(list_prev_coeff) > 0) and (list_prev_coeff[-1] is not None):
              prev_energy_coeff = float(list_prev_coeff[-1])
            else: prev_energy_coeff = funcReadEnergyPrefactor(
              filepath_sim  + "/" + FILENAME_GEN_INPUT
            )
            ## load 'Mach' data
            print(1 / (des_k_driv * des_Mach))
            data_time, data_Mach = loadTurbData(
              filepath_data = filepath_sim,
              var_y      = 13,
              t_eddy     = 1 / (des_k_driv * des_Mach),
              time_start = 2,
              time_end   = 15
            )
            ## check that there is sufficient data to look at
            if (len(data_time) > 0) and (data_time[-1] < 4):
              print("\t> The data has insufficient time range.")
              continue
            elif len(data_time) == 0:
              print("\t> There is no data to look at.")
              continue
            ## measure the average Mach number
            prev_Mach_number = round(np.mean(data_Mach), 4)
            print("\t> The measured mean Mach number is: {}".format( round(prev_Mach_number, 3) ))
            ## if the Mach number is significantly different from the desired value then update the energy prefactor
            if (abs(prev_Mach_number - des_Mach) / des_Mach) > 0.05:
              bool_run_simulation = True
              ## check that the Mach number has changed between executions of this program
              if "{}".format(prev_Mach_number) in list_prev_Mach:
                bool_update_turb = False
                print("\t> No need to update the energy coefficient.")
                print("\t  (The measured Mach number is the same as in the {}).".format( FILENAME_DRIVING_HISTORY ))
              ## if the Mach number and energy coefficients have not appeared in the history file before
              else:
                ## calculate desired energy prefactor
                des_energy_coeff = round((prev_energy_coeff * ( des_Mach / prev_Mach_number )**3), 4)
                ## check that the energy coefficient has changed between executions of this program
                if "{}".format(des_energy_coeff) in list_prev_coeff:
                  bool_update_turb = False
                  print("\t> No need to update the energy coefficient.")
                  print("\t  (The new energy coefficient is the same as in the {}).".format( FILENAME_DRIVING_HISTORY ))
                else:
                  ## indicate that the setup needs to happen
                  bool_update_turb = True
                  history_file.write("{}\t{}\t{}->{}\t\n".format(
                    ## current time
                    datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    ## previous Mach number
                    prev_Mach_number,
                    ## previous -> new prefactor
                    prev_energy_coeff, des_energy_coeff
                  ))
            ## if the Mach number is sufficiently close to the desired value
            else:
              ## indicate that the setup does not need to happen
              bool_update_turb = False
              bool_run_simulation = False
              history_file.write("{}\t{}\t{}->{}\t\n".format(
                datetime.now().strftime("%d/%m/%Y %H:%M:%S"), # current time
                prev_Mach_number,                             # previous Mach number
                prev_energy_coeff, "Done."                    # previous -> new prefactor
              ))

        ## ######################################
        ## TUNE THE FORCING GENERATOR / DATA FILE
        ## ######################################
        if bool_update_turb:
          ## delete all ("Turb*" and "*.dat") simulation output files
          print("\t> Removing any existing simulation outputs...")
          os.system("rm {}/Turb*".format( filepath_sim ))
          os.system("rm {}/*.o*".format( filepath_sim ))
          os.system("rm {}/stir.dat".format( filepath_sim ))
          ## update the energy prefactor in the forcing generator
          funcWriteForceGenInput(
            filepath_ref_file = filepath_base + "/" + FILENAME_GEN_INPUT, # refernce (home) folder
            filepath_new_file = filepath_sim  + "/" + FILENAME_GEN_INPUT, # simulation folder
            des_velocity      = des_Mach,
            des_k_driv        = des_k_driv,
            des_k_min         = des_k_min,
            des_k_max         = des_k_max,
            des_energy_coeff  = des_energy_coeff # desired energy prefactor
          )
          ## generate the turbulent forcing data file
          print("\t> Generating a new forcing file...")
          p = subprocess.Popen([ "forcing_generator" ], cwd=filepath_sim)
          p.wait()
          ## rename the turbulent forcing data file
          os.rename(
            filepath_sim  + "/" + FILENAME_GEN_INTERMEDIATE, # old name
            filepath_sim  + "/" + FILENAME_GEN_OUTPUT        # new name
          )
          print("\t> Renamed the forcing file to: '{}'".format( FILENAME_GEN_OUTPUT ))
        elif bool_run_simulation:
          print("\t> No need to update the forcing file.")
        else: print("\t> The forcing file is sufficient to achieve a Mach number close to '{}'.".format( des_Mach ))

        ## ##################
        ## RUN THE SIMULATION
        ## ##################
        if bool_run_simulation:
          ## submit the simulation PBS job file
          print("\t> Submitting the simulation job:")
          p = subprocess.Popen([ "qsub", "job_run_sim.sh" ], cwd=filepath_sim)
          p.wait()

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
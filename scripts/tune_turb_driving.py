#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys, subprocess
from datetime import datetime

## load user defined modules
from TheUsefulModule import WWFnF



class TuneForceGenInput():
  def __init__(self):
    a = 10

  def checkDrivingConvergence(self):
    a = 10

  def editForceGen(self):
    a = 10




## ###############################################################
## FUNCTION: read 'energy prefactor' in 'forcing_generator.inp'
## ###############################################################
def readEnergyPrefactor(filepath_file):
  ## check the file exists
  if not os.path.isfile(filepath_file):
    raise Exception("ERROR: turbulence generator input file does not exist:", FILENAME_DRIVING_INPUT)
  ## open file
  with open(filepath_file) as file_lines:
    ## loop over file lines
    for line in file_lines:
      ## split line into elements seperated by a space
      list_line_elems = line.split()
      ## ignore empty lines
      if len(list_line_elems) == 0:
        continue
      ## read currenty amplitude coefficient
      if list_line_elems[0] == "ampl_coeff":
        return float(list_line_elems[2])
  ## if the coefficient wasn't found
  raise Exception(f"ERROR: Could not read the energy prefactory 'st_energy_coeff' in '{FILENAME_DRIVING_INPUT}'")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
FILEPATH_BASE            = "/scratch/ek9/nk7952/"
FILENAME_DRIVING_INPUT   = "turbulence_generator.inp"
FILENAME_DRIVING_HISTORY = "driving_history.txt"
FILENAME_SIM_OUTPUT      = "Turb.dat"

# def main():
#   ## ##############################
#   ## LOOK AT EACH SIMULATION FOLDER
#   ## ##############################
#   ## loop over the simulation suites
#   for suite_folder in [
#       "test_sim"
#     ]: # "Re10", "Re500", "Rm3000", "keta"

#     ## loop over the different resolution runs
#     for sim_res in [
#         ""
#       ]: # "18", "36", "72", "144", "288", "576"

#       ## print to the terminal what suite is being looked at
#       str_msg = "Looking at suite: {}, Nres = {}".format( suite_folder, sim_res )
#       print(str_msg)
#       print("=" * len(str_msg))

#       ## loop over the simulation folders
#       for sim_folder in [
#           ""
#         ]: # "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"

#         ## #######################################
#         ## CHECK THAT THE SIMULATION FOLDER EXISTS
#         ## #######################################
#         ## create filepath to simulation folder
#         filepath_sim  = WWFnF.createFilepath([
#           FILEPATH_BASE, suite_folder
#           # FILEPATH_BASE, suite_folder, sim_res, SONIC_REGIME, sim_folder
#         ])
#         ## check that the simulation filepath exists
#         if not os.path.exists(filepath_sim):
#           print(filepath_sim, "does not exist.")
#           continue
#         ## indicate which folder is being worked on
#         print("Looking at: {}".format(filepath_sim))

#         ## ########################################################
#         ## CHECK THAT THE DRIVING PARAMETERS HAVE NOT CONVERGED YET
#         ## ########################################################
#         bool_update_turb    = False
#         bool_run_simulation = False
#         bool_done           = False
#         list_prev_Mach  = []
#         list_prev_coeff = []
#         if os.path.isfile(filepath_sim + "/" + FILENAME_DRIVING_HISTORY):
#           ## look for the term "done" in the file
#           with open(filepath_sim + "/" + FILENAME_DRIVING_HISTORY, "r") as history_file_lines:
#             for line in history_file_lines:
#               list_prev_Mach.append(  line.split("\t")[1] )
#               list_prev_coeff.append( line.split("\t")[2].split("->")[1] )
#               if ("done" in line.lower()):
#                 bool_done = True
#           print("\t> Previous Mach numbers:", list_prev_Mach)
#           print("\t> Previous energy coefficients:", list_prev_coeff)
#           ## if the term "done" appears in the file then skip this simulation
#           if bool_done:
#             print("\t> The '{}' indicates that the driving parameter has converged.\n".format(
#               FILENAME_DRIVING_HISTORY
#             ))
#             continue

#         ## ###############################################
#         ## CHECK IF THE SIMULATION OUTPUT DATA FILE EXISTS
#         ## ###############################################
#         ## keep a history of the turbulent forcing parameters
#         with open(filepath_sim + "/" + FILENAME_DRIVING_HISTORY, "a") as history_file:
#           ## if the simulation has not been run yet (i.e. there is no 'Turb.dat' file in the simulation folder)
#           if not path.isfile(filepath_sim + "/" + FILENAME_SIM_OUTPUT):
#             print("\t> '{}' does not exist (i.e. there is no data to look at).".format( FILENAME_SIM_OUTPUT ))
#             ## check if a previous energy coefficient can be used (i.e. exists)
#             if (len(list_prev_coeff) > 0) and (list_prev_coeff[-1] is not None):
#               des_energy_coeff = float(list_prev_coeff[-1])
#             ## default starting energy coefficient
#             else: des_energy_coeff = 5.0e-3
#             print("\t> Using energy coefficient: {}".format( des_energy_coeff ))
#             ## indicate that the setup needs to happen
#             bool_update_turb = True
#             bool_run_simulation = True
#             history_file.write("{}\t{}\t{}->{}\t\n".format(
#               ## current time
#               datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
#               ## previous Mach number
#               None,
#               ## previous -> new prefactor
#               None, des_energy_coeff
#             ))
#           ## if the driving file does not exist
#           elif not path.isfile(filepath_sim + "/" + FILENAME_GEN_OUTPUT):
#             print("\t> '{}' does not exist.".format( FILENAME_GEN_OUTPUT ))
#             ## default starting energy coefficient
#             des_energy_coeff = 5.0e-3
#             print("\t> Using energy coefficient: {}".format( des_energy_coeff ))
#             ## indicate that the setup needs to happen
#             bool_update_turb = True
#             bool_run_simulation = True
#             history_file.write("{}\t{}\t{}->{}\t\n".format(
#               ## current time
#               datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
#               ## previous Mach number
#               None,
#               ## previous -> new prefactor
#               None, des_energy_coeff
#             ))
#           ## if the simulation has been run: check if the Mach number is close to the desired value
#           else:
#             ## read previous energy prefactor
#             if (len(list_prev_coeff) > 0) and (list_prev_coeff[-1] is not None):
#               prev_energy_coeff = float(list_prev_coeff[-1])
#             else: prev_energy_coeff = readEnergyPrefactor(
#               filepath_sim  + "/" + FILENAME_GEN_INPUT
#             )
#             ## load 'Mach' data
#             print(1 / (des_k_driv * des_Mach))
#             data_time, data_Mach = loadTurbData(
#               filepath_data = filepath_sim,
#               var_y      = 13,
#               t_turb     = 1 / (des_k_driv * des_Mach),
#               time_start = 2,
#               time_end   = 15
#             )
#             ## check that there is sufficient data to look at
#             if (len(data_time) > 0) and (data_time[-1] < 4):
#               print("\t> The data has insufficient time range.")
#               continue
#             elif len(data_time) == 0:
#               print("\t> There is no data to look at.")
#               continue
#             ## measure the average Mach number
#             prev_Mach_number = round(np.mean(data_Mach), 4)
#             print("\t> The measured mean Mach number is: {}".format( round(prev_Mach_number, 3) ))
#             ## if the Mach number is significantly different from the desired value then update the energy prefactor
#             if (abs(prev_Mach_number - des_Mach) / des_Mach) > 0.05:
#               bool_run_simulation = True
#               ## check that the Mach number has changed between executions of this program
#               if "{}".format(prev_Mach_number) in list_prev_Mach:
#                 bool_update_turb = False
#                 print("\t> No need to update the energy coefficient.")
#                 print("\t  (The measured Mach number is the same as in the {}).".format( FILENAME_DRIVING_HISTORY ))
#               ## if the Mach number and energy coefficients have not appeared in the history file before
#               else:
#                 ## calculate desired energy prefactor
#                 des_energy_coeff = round((prev_energy_coeff * ( des_Mach / prev_Mach_number )**3), 4)
#                 ## check that the energy coefficient has changed between executions of this program
#                 if "{}".format(des_energy_coeff) in list_prev_coeff:
#                   bool_update_turb = False
#                   print("\t> No need to update the energy coefficient.")
#                   print("\t  (The new energy coefficient is the same as in the {}).".format( FILENAME_DRIVING_HISTORY ))
#                 else:
#                   ## indicate that the setup needs to happen
#                   bool_update_turb = True
#                   history_file.write("{}\t{}\t{}->{}\t\n".format(
#                     ## current time
#                     datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
#                     ## previous Mach number
#                     prev_Mach_number,
#                     ## previous -> new prefactor
#                     prev_energy_coeff, des_energy_coeff
#                   ))
#             ## if the Mach number is sufficiently close to the desired value
#             else:
#               ## indicate that the setup does not need to happen
#               bool_update_turb = False
#               bool_run_simulation = False
#               history_file.write("{}\t{}\t{}->{}\t\n".format(
#                 datetime.now().strftime("%d/%m/%Y %H:%M:%S"), # current time
#                 prev_Mach_number,                             # previous Mach number
#                 prev_energy_coeff, "Done."                    # previous -> new prefactor
#               ))

#         ## ######################################
#         ## TUNE THE FORCING GENERATOR / DATA FILE
#         ## ######################################
#         if bool_update_turb:
#           ## delete all ("Turb*" and "*.dat") simulation output files
#           print("\t> Removing any existing simulation outputs...")
#           os.system("rm {}/Turb*".format( filepath_sim ))
#           os.system("rm {}/*.o*".format( filepath_sim ))
#           os.system("rm {}/stir.dat".format( filepath_sim ))
#           ## update the energy prefactor in the forcing generator
#           writeForceGenInput(
#             filepath_ref_file = filepath_base + "/" + FILENAME_GEN_INPUT, # refernce (home) folder
#             filepath_new_file = filepath_sim  + "/" + FILENAME_GEN_INPUT, # simulation folder
#             des_velocity      = des_Mach,
#             des_k_driv        = des_k_driv,
#             des_k_min         = des_k_min,
#             des_k_max         = des_k_max,
#             des_energy_coeff  = des_energy_coeff # desired energy prefactor
#           )
#           ## generate the turbulent forcing data file
#           print("\t> Generating a new forcing file...")
#           p = subprocess.Popen([ "forcing_generator" ], cwd=filepath_sim)
#           p.wait()
#           ## rename the turbulent forcing data file
#           os.rename(
#             filepath_sim  + "/" + FILENAME_GEN_INTERMEDIATE, # old name
#             filepath_sim  + "/" + FILENAME_GEN_OUTPUT        # new name
#           )
#           print("\t> Renamed the forcing file to: '{}'".format( FILENAME_GEN_OUTPUT ))
#         elif bool_run_simulation:
#           print("\t> No need to update the forcing file.")
#         else: print("\t> The forcing file is sufficient to achieve a Mach number close to '{}'.".format( des_Mach ))

#         ## ##################
#         ## RUN THE SIMULATION
#         ## ##################
#         if bool_run_simulation:
#           ## submit the simulation PBS job file
#           print("\t> Submitting the simulation job:")
#           p = subprocess.Popen([ "qsub", "job_run_sim.sh" ], cwd=filepath_sim)
#           p.wait()

#         ## create an empty line after each suite
#         print(" ")
#       print(" ")
#     print(" ")


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
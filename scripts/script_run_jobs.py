#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import sys, subprocess
from TheUsefulModule import WWFnF


BASEPATH       = "/scratch/ek9/nk7952/"
SONIC_REGIME   = "super_sonic"
DATA_SUBFOLDER = "spect"
JOB_NAME       = "job_fit_spect.sh"
## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## ##############################
  ## LOOK AT EACH SIMULATION FOLDER
  ## ##############################
  ## loop over the simulation suites
  for suite_folder in [
      "Re10", "Re500", "Rm3000"
    ]: # "Re10", "Re500", "Rm3000", "keta"

    ## loop over the different resolution runs
    for sim_res in [
        "72", "144", "288"
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
        ## create filepath to simulation folder (on GADI)
        filepath_sim = WWFnF.createFilepath([
          BASEPATH, suite_folder, sim_res, SONIC_REGIME, sim_folder, DATA_SUBFOLDER
        ])
        ## check that the simulation filepath exists
        if not os.path.exists(filepath_sim):
          print(filepath_sim, "does not exist.")
          continue

        ## #########################
        ## CHECK THE JOB FILE EXISTS
        ## #########################
        ## check that the job exists in the folder
        if not os.path.isfile(filepath_sim + "/" + JOB_NAME):
          print(JOB_NAME, "does not exist in", filepath_sim)
          continue
        ## indicate which folder is being worked on
        print("Looking at: {}".format(filepath_sim))
        print("\t> Submitting the simulation job:")
        p = subprocess.Popen([ "qsub", JOB_NAME ], cwd=filepath_sim)
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
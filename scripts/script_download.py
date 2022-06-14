#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import matplotlib.pyplot as plt

from os import path


## ###############################################################
## PREPARE WORKSPACE
##################################################################
os.system("clear") # clear terminal window
plt.ioff()
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## FUNCTIONS
## ###############################################################
def funcCheckSpectraObjNeedsUpdate(mac_filepath_data, spectra_name):
  ## load spectra object
  try:
    spectra_obj = loadPickle(
      mac_filepath_data,
      spectra_name,
      bool_check = True,
      bool_hide_updates = True
    )
  except EOFError : 
    return True
  ## if no file
  if spectra_obj == -1:
    return True
  ## otherwise, do not update
  else: return False


SONIC_REGIME = "super_sonic"
## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  my_filepath_base = "/Users/dukekriel/Documents/Studies/TurbulentDynamo/data/" + SONIC_REGIME
  for suite_folder in [
      "Re10", "Re500", "Rm3000"
    ]: # "Re10", "Re500", "keta", "Rm3000"

    ## ##############################################
    ## DOWNLOAD PLOTS CHECKING CONVERGENCE + PKL FILE
    ## ##############################################
    # mac_filepath_suite = createFilepath([my_filepath_base, suite_folder])
    # gadi_filepath_suite = createFilepath(["gadi:/scratch/ek9/nk7952/", suite_folder])
    # mac_filepath_figures = createFilepath([my_filepath_base, suite_folder, "vis_full"])
    # gadi_filepath_figures = createFilepath(["gadi:/scratch/ek9/nk7952/", suite_folder, "vis_folder"])
    # os.system("scp " + gadi_filepath_suite + "/*_full.pkl " + mac_filepath_suite + "/.")
    # os.system("scp " + gadi_filepath_figures + "/*_scales_res.pdf " + mac_filepath_figures + "/.")

    for sim_res in [
        "18", "36", "72", "144", "288"
      ]: # "72", "144", "288", "576"
      ## ################################
      ## CREATE FILEPATH TO FOLDER ON MAC
      ## ################################
      mac_filepath_figures = createFilepath([
        my_filepath_base, suite_folder, sim_res, "vis_folder"
      ])
      ## check that the filepath exists on MAC
      if not path.exists(mac_filepath_figures):
        print("{} does not exist.".format( mac_filepath_figures ))
        continue
      str_message = "Downloading from suite: {}, Nres = {}".format( suite_folder, sim_res )
      print(str_message)
      print("=" * len(str_message))

      # ## ##############################
      # ## DOWNLOAD FIT FIGURES FROM GADI
      # ## ##############################
      # gadi_filepath_figures = createFilepath([
      #     "gadi:/scratch/ek9/nk7952/", suite_folder, sim_res, SONIC_REGIME, "vis_folder"
      # ])
      # print("Downloading from:", gadi_filepath_figures)
      # ## download plots checking fits
      # os.system("scp " + gadi_filepath_figures + "/*_check_MeasuredScales.pdf " + mac_filepath_figures + "/.")
      # os.system("scp " + gadi_filepath_figures + "/check_fits/*.pdf " + mac_filepath_figures + "/check_fits/.")

      # #############################
      # DOWNLOAD FIT VIDEOS FROM GADI
      # #############################
      gadi_filepath_figures = createFilepath([
        "gadi:/scratch/ek9/nk7952/", suite_folder, sim_res, SONIC_REGIME, "vis_folder"
      ])
      print("Downloading from:", gadi_filepath_figures)
      ## download plots + animations
      os.system("scp " + gadi_filepath_figures + "/*.mp4 " + mac_filepath_figures + "/.")

      # ###################
      # DOWNLOAD DATA FILES
      # ###################
      for sim_folder in [
          "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250"
        ]:
        ## create filepath to folder on MAC
        mac_filepath_data = createFilepath([ my_filepath_base, suite_folder, sim_res, sim_folder ])
        ## check if the filepath exists on MAC
        if not path.exists(mac_filepath_data):
          continue
        ## create filepath to data folder on GADI
        gadi_filepath_data = createFilepath([
          "gadi:/scratch/ek9/nk7952/", suite_folder, sim_res, SONIC_REGIME, sim_folder
        ])
        print("Downloading from:", gadi_filepath_data)
        ## download spectra object
        # if funcCheckSpectraObjNeedsUpdate(mac_filepath_data, "spectra_obj_full.pkl"):
        os.system("scp {}/spect/{} {}/.".format(
          gadi_filepath_data,
          "spectra_obj_full.pkl",
          mac_filepath_data
        ))
        # if funcCheckSpectraObjNeedsUpdate(mac_filepath_data, "spectra_obj_mixed.pkl"):
        #     os.system("scp {}/spect/{} {}/.".format(
        #         gadi_filepath_data,
        #         "spectra_obj_mixed.pkl",
        #         mac_filepath_data
        #     ))
      print(" ")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
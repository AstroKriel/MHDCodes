import os, sys
# from distutils import dir_util
import shutil
from TheUsefulModule import WWFnF

def main():
  for suite_folder in LIST_SUITE_FOLDER:
    for sim_folder in LIST_SIM_FOLDER:
      for sim_res in LIST_SIM_RES:
        a = 10
        # ## check that the simulation folder exists
        # filepath_sim_from = WWFnF.createFilepath([ 
        #   BASEPATH, suite_folder, sim_res, SONIC_REGIME, sim_folder
        # ])
        # if not os.path.exists(filepath_sim_from): continue
        # ## create new filepath
        # filepath_sim_to = WWFnF.createFilepath([ 
        #   BASEPATH, suite_folder, SONIC_REGIME, sim_folder, sim_res
        # ])
        # WWFnF.createFolder(filepath_sim_to)
        # # ## copy directory from old to new place
        # os.system(f"mv {filepath_sim_from}/* {filepath_sim_to}/.")
        # ## create empty space
      print(" ")
    print(" ")

BASEPATH          = "/scratch/ek9/nk7952/"
SONIC_REGIME      = "sub_sonic"
LIST_SUITE_FOLDER = [ "Rm3000" ]
LIST_SIM_RES      = [ "288" ]
LIST_SIM_FOLDER   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]

if __name__ == "__main__":
  main()
  sys.exit()

## END OF PROGRAM
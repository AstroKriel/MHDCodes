#!/usr/bin/env python3

from TheLoadingModule import LoadFlashData

def checkSpectra(filepath):
  ## load kinetic energy spectra
  list_kin_k_group_t, list_kin_power_group_t, list_kin_sim_times = LoadFlashData.loadListSpectra(
    filepath_data     = filepath,
    str_spectra_type  = "vel"
  )
  ## load magnetic energy spectra
  list_mag_k_group_t, list_mag_power_group_t, list_mag_sim_times = LoadFlashData.loadListSpectra(
    filepath_data     = filepath,
    str_spectra_type  = "mag"
  )
  ## check integrated spectra
  print("\t> P_kin:", round(sum(list_kin_power_group_t[0]), 10))
  print("\t> P_mag:", round(sum(list_mag_power_group_t[0]), 10))
  print(" ")


def main():
  filepath = "/scratch/ek9/nk7952/Rm3000/72/super_sonic/Pm1/spect/test"
  print("Old spectra:")
  checkSpectra(filepath+"/old/")
  print("New spectra:")
  checkSpectra(filepath+"/new/")


if __name__ == "__main__":
  main()

## END OF PROGRAM
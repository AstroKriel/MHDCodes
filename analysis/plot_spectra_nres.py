#!/bin/env python3

## MODULES
import numpy as np
import matplotlib.pyplot as plt
from TheLoadingModule import LoadFlashData

## HELPER FUNCTION
def getAveNormKinSpectra(filepath_data):
  plots_per_eddy = LoadFlashData.getPlotsPerEddy(
    f"{filepath_data}/../",
    bool_hide_updates=True
  )
  ## load kinetic energy spectra
  list_kin_k_group_t, list_kin_power_group_t, _ = LoadFlashData.loadAllSpectraData(
    filepath_data     = filepath_data,
    str_spectra_type  = "vel",
    file_start_time   = 10,
    file_end_time     = 20,
    plots_per_eddy    = plots_per_eddy,
    bool_hide_updates = True
  )
  list_kin_power_norm_group_t = [
    np.array(list_power) / sum(list_power)
    for list_power in list_kin_power_group_t
  ]
  list_kin_power_ave = np.mean(list_kin_power_norm_group_t, axis=0)
  return list_kin_k_group_t[0], list_kin_power_ave

## MAIN PROGRAM
def main():
  basepath = "/scratch/ek9/nk7952/"
  filepath_sim = f"{basepath}/Rm3000/super_sonic/Pm1"
  filepath_fig = f"{filepath_sim}/kin_spect_comparison.png"
  list_k_18, list_spectra_18 = getAveNormKinSpectra(f"{filepath_sim}/18/spect/")
  list_k_36, list_spectra_36 = getAveNormKinSpectra(f"{filepath_sim}/36/spect/")
  list_k_72, list_spectra_72 = getAveNormKinSpectra(f"{filepath_sim}/72/spect/")
  list_k_144, list_spectra_144 = getAveNormKinSpectra(f"{filepath_sim}/144/spect/")
  list_k_288, list_spectra_288 = getAveNormKinSpectra(f"{filepath_sim}/288/spect/")
  list_k_576, list_spectra_576 = getAveNormKinSpectra(f"{filepath_sim}/576/spect/")
  _, ax = plt.subplots()
  ax.plot(list_k_18, list_spectra_18, "r--", label="18")
  ax.plot(list_k_36, list_spectra_36, "g--", label="36")
  ax.plot(list_k_72, list_spectra_72, "b--", label="72")
  ax.plot(list_k_144, list_spectra_144, "r-", label="144")
  ax.plot(list_k_288, list_spectra_288, "g-", label="288")
  ax.plot(list_k_576, list_spectra_576, "b-", label="576")
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlabel("k")
  ax.set_ylabel("kin spectra")
  ax.legend(loc="upper right")
  plt.savefig(filepath_fig)
  plt.close()
  print("Figure saved:", filepath_fig)

## START OF PROGRAM
if __name__ == "__main__":
  main()

## END OF PROGRAM
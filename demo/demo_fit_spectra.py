## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from plot_sim_data import fitKinSpectra

from TheUsefulModule import WWLists, WWObjs
from ThePlottingModule import PlotFuncs
from TheFittingModule import FitMHDScales


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## MAGNETIC ENERGY SPECTRUM MODEL
## ###############################################################
class MagSpectrum():
  def linear(k, A, alpha_1, alpha_2, k_eta):
    return A * np.array(k)**(alpha_1) * np.exp( -(np.array(k) / k_eta)**(alpha_2) )

  def loge(k, **params):
    return np.log(MagSpectrum.linear(k, **params))


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def getAveSpectra(filepath_sim_res):
  dict_sim_data = WWObjs.loadJsonFile2Dict(
    filepath = filepath_sim_res,
    filename = f"sim_outputs.json",
    bool_hide_updates = False
  )
  list_k         = dict_sim_data["list_k"]
  list_power_ave = dict_sim_data["list_kin_power_ave"]
  return list_k, list_power_ave


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  filepath_scratch = "/scratch/ek9/nk7952/"
  filepath_sim_res = f"{filepath_scratch}/Rm3000/super_sonic/Pm125/288/"
  fig, ax          = plt.subplots(figsize=(7,8))
  ## fit normalise and time-average
  list_k, list_power_ave = getAveSpectra(filepath_sim_res)
  ax.plot(list_k, list_power_ave, color="r", ls="", marker="o", ms=3)
  end_index_kin = WWLists.getIndexClosestValue(list_power_ave, 10**(-8))
  fitKinSpectra(
    ax          = ax,
    list_k      = list_k[1:end_index_kin],
    list_power  = list_power_ave[1:end_index_kin],
    bool_plot   = True
  )
  ## label figure
  ax.legend(loc="lower left")
  ax.set_xlabel(r"$k$")
  ax.set_ylabel(r"$\widehat{\mathcal{P}}_{\rm kin}(k)$")
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlim(left=10**(-1))
  ax.set_ylim(bottom=10**(-10), top=10**(5))
  ## save figure
  PlotFuncs.saveFigure(fig, "fit_spectra_kin.png")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF DEMO PROGRAM
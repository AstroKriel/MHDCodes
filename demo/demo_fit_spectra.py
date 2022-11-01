## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
import ThePlottingModule.TheMatplotlibStyler


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## MAGNETIC SPECTRUM MODEL
## ###############################################################
class MagSpectrum():
  def linear(k, A, alpha_1, alpha_2, k_eta):
    return A * np.array(k)**(alpha_1) * np.exp( -(np.array(k) / k_eta)**(alpha_2) )

  def loge(k, **params):
    return np.log(MagSpectrum.linear(k, **params))


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def loadSpectraData(filepath):
  data_k_group_t     = []
  data_power_group_t = []
  ## loop over spectra files in t/t_turb = [10, 30]
  for file_index in range(100, 300):
    filepath_file = f"{filepath}/Turb_hdf5_plt_cnt_{str(file_index).zfill(4)}_spect_mags.dat"
    data          = open(filepath_file).readlines()
    data_matrix   = np.array([ x.strip().split() for x in data[6:] ])
    data_k        = np.array(list(map(float, data_matrix[:, 1])))
    data_power    = np.array(list(map(float, data_matrix[:, 15])))
    data_k_group_t.append(data_k)
    data_power_group_t.append(data_power)
  return data_k_group_t, data_power_group_t

def AveNormSpectraData(data_power_group_t):
  data_power_norm_group_t = [
    np.array(data_power) / np.sum(data_power) for data_power in data_power_group_t
  ]
  return np.median(data_power_norm_group_t, axis=0)

def fitMagSpectra(ax, data_k, data_power):
  my_model = Model(MagSpectrum.loge)
  my_model.set_param_hint("A",       min = 1e-2, value = 1e-1, max = 1e2)
  my_model.set_param_hint("alpha_1", min = 0.5,  value = 1.5,  max = 2.0)
  my_model.set_param_hint("alpha_2", min = 0.1,  value = 1.0,  max = 2.0)
  my_model.set_param_hint("k_eta",   min = 1e-2, value = 5.0,  max = 10.0)
  input_params = my_model.make_params()
  fit_results  = my_model.fit(
    k      = data_k,
    data   = np.log(data_power),
    params = input_params
  )
  fit_params = [
    fit_results.params["A"].value,
    fit_results.params["alpha_1"].value,
    fit_results.params["alpha_2"].value,
    fit_results.params["k_eta"].value
  ]
  print("Fitted paramters:")
  print(fit_params)
  data_k_plot = np.logspace(0, 2, 1000)
  data_power_plot = MagSpectrum.linear(data_k_plot, *fit_params)
  ax.plot(data_k_plot, data_power_plot, c="black", ls="--", lw=3, zorder=7)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  filepath_scratch = "/scratch/ek9/nk7952/"
  filepath_data    = f"{filepath_scratch}/Rm3000/288/super_sonic/Pm5/spect"
  fig, ax = plt.subplots(figsize=(7,8))
  ## load magnetic energy spectrum
  data_k_group_t, data_power_group_t = loadSpectraData(filepath_data)
  ## normalise and time-average
  data_power_ave = AveNormSpectraData(data_power_group_t)
  ax.plot(data_k_group_t[0], data_power_ave, c="r", ls="", marker="o", ms=3)
  fitMagSpectra(ax, data_k_group_t[0], data_power_ave)
  ## save figure
  ax.set_xlabel(r"$k$")
  ax.set_ylabel(r"$\widehat{\mathcal{P}}_{\rm mag}(k)$")
  ax.set_xscale("log")
  ax.set_yscale("log")
  fig.savefig(f"{filepath_scratch}/mag_spect_fit.png")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
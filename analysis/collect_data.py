## IMPORT MODULES
import os, sys
import numpy as np

from TheSimModule import SimParams
from TheUsefulModule import WWObjs
from TheLoadingModule import LoadFlashData
from ThePlottingModule import PlotFuncs

os.system("clear")


## OPPERATOR FUNCTIONS
def getSimData(fig, ax, filepath_sim_res, sim_name, Re, Pm):
  print("Loading simulation data...")
  ## load sim outputs file
  dict_sim_inputs  = SimParams.readSimInputs(filepath_sim_res)
  dict_sim_outputs = SimParams.readSimOutputs(filepath_sim_res)
  ## extract simulation parameters
  sim_res           = dict_sim_inputs["sim_res"]
  Rm                = Re * Pm
  ## extract measured quantities
  time_growth_start = 5.0
  time_growth_end   = 10.0
  plots_per_eddy    = dict_sim_outputs["plots_per_eddy"]
  ## load magnetic energy spectra
  dict_spect_data = LoadFlashData.loadAllSpectraData(
    filepath          = f"{filepath_sim_res}/spect",
    spect_field       = "vel",
    file_start_time   = time_growth_start,
    file_end_time     = time_growth_end,
    plots_per_eddy    = plots_per_eddy,
    bool_verbose = True
  )
  ## extract spectra data
  list_k_group_t     = dict_spect_data["list_k_group_t"]
  list_power_group_t = dict_spect_data["list_power_group_t"]
  list_time          = dict_spect_data["list_time"]
  ## plot spectra
  print("Plotting kintic energy spectra...")
  ## initialise colormap
  cmap, norm = PlotFuncs.createCmap(
    cmap_name = "Blues",
    vmin      = time_growth_start,
    vmax      = time_growth_end
  )
  ## plot each time realisation in the kinematic phase
  for time_index, time_val in enumerate(list_time):
    ax.plot(
      list_k_group_t[0],
      list_power_group_t[time_index],
      color=cmap(norm(time_val)), ls="-", alpha=0.5, zorder=1
    )
  ## plot time-averaged spectrum
  list_power_ave = np.mean(list_power_group_t, axis=0)
  ax.plot(
    list_k_group_t[0],
    list_power_ave,
    label=r"$\langle \widehat{\mathcal{P}}_{\rm kin}(k, t) \rangle_{\forall t}$",
    color="black", ls="-", lw=5, zorder=3
  )
  ## add colorbar
  PlotFuncs.addColorbar_fromCmap(
    fig, ax, cmap, norm,
    label = r"$t = t_{\rm sim} / t_{\rm turb}$"
  )
  ## annotate parameters
  PlotFuncs.addBoxOfLabels(
    fig, ax,
    bbox        = (0.0, 0.0),
    xpos        = 0.05,
    ypos        = 0.05,
    alpha       = 0.5,
    fontsize    = 18,
    list_labels = [
      r"${\rm N}_{\rm res} = $ " + "{:d}".format(int(sim_res)),
      r"${\rm Re} = $ "          + "{:d}".format(int(Re)),
      r"${\rm Rm} = $ "          + "{:d}".format(int(Rm)),
      r"${\rm Pm} = $ "          + "{:d}".format(int(Pm))
    ]
  )
  ## label figure
  ax.set_xlabel(r"$k$")
  ax.set_xscale("log")
  ax.set_yscale("log")
  PlotFuncs.addAxisTicks_log10(
    ax,
    bool_major_ticks = True,
    num_major_ticks  = 10
  )
  ## save axis
  dict_params = {
    "Re"                 : Re,
    "Rm"                 : Rm,
    "Pm"                 : Pm,
    "list_k"             : list_k_group_t[0],
    "list_kin_power_ave" : list_power_ave
  }
  # sim_name = f"Re{Re:.0f}Pm{Pm:.0f}_{sim_res}"
  WWObjs.saveDict2JsonFile(f"{sim_name}_kin_spectra.json", dict_params)
  print(" ")


## MAIN PROGRAM
def main():
  ## Re3600Pm1, Re1700Pm2, Re600Pm5
  ## initialise figure
  fig, fig_grid = PlotFuncs.createFigure_grid(num_cols=3, fig_aspect_ratio=(5,4), fig_scale=1.5)
  ax0 = fig.add_subplot(fig_grid[0])
  ax1 = fig.add_subplot(fig_grid[1])
  ax2 = fig.add_subplot(fig_grid[2])
  ## load and plot data
  getSimData(fig, ax0, "/scratch/ek9/nk7952/Rm3000/sub_sonic/Pm1/288/", "Re3600Pm1", 3600, 1)
  getSimData(fig, ax1, "/scratch/ek9/nk7952/Rm3000/sub_sonic/Pm2/288/", "Re1700Pm2", 1700, 2)
  getSimData(fig, ax2, "/scratch/ek9/nk7952/Rm3000/sub_sonic/Pm5/288/", "Re600Pm5",  600,  5)
  ## label plot
  ax0.set_ylabel(r"$\widehat{\mathcal{P}}_{\rm kin}(k, t)$")
  ax0.legend(loc="center left", fontsize=20)
  ## save figure
  PlotFuncs.saveFigure(fig, "dataset_kin_spectra.png")


## PROGRAM ENTRY POINT
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
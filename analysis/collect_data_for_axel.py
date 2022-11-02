## IMPORT MODULES
import os, sys
import numpy as np
import cmasher as cmr
import matplotlib.colors as cols
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

from TheUsefulModule import WWObjs
from TheJobModule import SimInputParams
from TheLoadingModule import LoadFlashData
from ThePlottingModule import PlotFuncs

os.system("clear")


## OPPERATOR FUNCTIONS
def getSimData(fig, ax, filepath_sim, sim_name, Re, Pm, bool_ref_plot=False):
  print("Loading simulation data...")
  ## load sim outputs file
  dict_sim_outputs = WWObjs.loadJsonFile2Dict(
    filepath = filepath_sim,
    filename = "sim_outputs.json"
  )
  ## extract simulation parameters
  Gamma             = dict_sim_outputs["Gamma"]
  time_growth_start = 5.0
  time_growth_end   = 10.0
  list_k            = dict_sim_outputs["list_k"]
  plots_per_eddy    = dict_sim_outputs["plots_per_eddy"]
  ## load magnetic energy spectra
  _, list_mag_power_group_t, list_mag_time = LoadFlashData.loadAllSpectraData(
      filepath_data     = f"{filepath_sim}/spect",
      str_spectra_type  = "mag",
      file_start_time   = time_growth_start,
      file_end_time     = time_growth_end,
      plots_per_eddy    = plots_per_eddy,
      bool_hide_updates = True
    )
  ## plot spectra
  print("Plotting magnetic energy spectra...")
  ## initialise colormap
  cmap = cmr.ember_r
  norm = cols.Normalize(vmin=time_growth_start, vmax=time_growth_end)
  ## plot each time realisation in the kinematic phase
  list_mag_power_comp_group_t = []
  for time_index, time_val in enumerate(list_mag_time):
    list_mag_power_comp = np.array(list_mag_power_group_t[time_index]) / sum(list_mag_power_group_t[time_index])
    ax.plot(
      list_k,
      list_mag_power_comp,
      color=cmap(norm(time_val)), ls="-", alpha=0.5, zorder=1
    )
    list_mag_power_comp_group_t.append(list_mag_power_comp)
  ## plot time-averaged, compensated spectrum
  list_mag_power_comp_ave = np.mean(list_mag_power_comp_group_t, axis=0)
  ax.plot(
    list_k,
    list_mag_power_comp_ave,
    label=r"$\langle \widehat{\mathcal{P}}_{\rm mag}(k, t) \rangle_{\forall t}$",
    color="cornflowerblue", ls="-", lw=5, zorder=3
  )
  ## add colorbar
  list_ticks = range(5, 11)
  smap = ScalarMappable(cmap=cmap, norm=norm)
  div  = make_axes_locatable(ax)
  cax  = div.new_vertical(size="5%", pad=0.1)
  fig.add_axes(cax)
  fig.colorbar(mappable=smap, cax=cax, ticks=list_ticks, orientation="horizontal")
  cax.xaxis.set_ticks_position("top")
  if bool_ref_plot:
    cax.set_title(r"$t = t_{\rm sim}/t_{\rm turb}$")
  else: cax.set_title(r"$t$")
  ## annotate parameters
  PlotFuncs.addBoxOfLabels(
    fig, ax,
    box_alignment = (0.0, 0.0),
    xpos          = 0.05,
    ypos          = 0.2 if bool_ref_plot else 0.05,
    alpha         = 0.5,
    fontsize      = 18,
    list_labels   = [
      r"${\rm N}_{\rm res} = $ " + "{:d}".format(288),
      r"${\rm Re} = $ "          + "{:d}".format(int(Re)),
      r"${\rm Rm} = $ "          + "{:d}".format(int(Re * Pm)),
      r"${\rm Pm} = $ "          + "{:d}".format(int(Pm)),
      r"$\Gamma = $ "            + "{:.1f}".format(float(Gamma)),
    ]
  )
  ## label figure
  ax.set_xlabel(r"$k$")
  ax.set_xscale("log")
  ax.set_yscale("log")
  PlotFuncs.addLogAxisTicks(
    ax,
    bool_major_ticks = True,
    num_major_ticks  = 10
  )
  ## save dataset
  dict_params = {
    "Re"                      : Re,
    "Rm"                      : Re * Pm,
    "Pm"                      : Pm,
    "Gamma"                   : Gamma,
    "list_mag_power_comp_ave" : list_mag_power_comp_ave
  }
  WWObjs.saveDict2JsonFile(f"dataset_axel_{sim_name}.json", dict_params)
  print(" ")


## MAIN PROGRAM
def main():
  ## Re3600Pm1, Re1700Pm2, Re600Pm5
  filepath_Re3600Pm1 = "/scratch/ek9/nk7952/Rm3000/sub_sonic/Pm1/288/"
  filepath_Re1700Pm2 = "/scratch/ek9/nk7952/Rm3000/sub_sonic/Pm2/288/"
  filepath_Re600Pm5  = "/scratch/ek9/nk7952/Rm3000/sub_sonic/Pm5/288/"
  ## initialise figure
  fig, fig_grid = PlotFuncs.createFigGrid(1, 3, fig_aspect_ratio=(5,4))
  ax_Re3600Pm1 = fig.add_subplot(fig_grid[0])
  ax_Re1700Pm2 = fig.add_subplot(fig_grid[1])
  ax_Re600Pm5  = fig.add_subplot(fig_grid[2])
  ## load and plot data
  getSimData(fig, ax_Re3600Pm1, filepath_Re3600Pm1, "Re3600Pm1", 3600, 1, True)
  getSimData(fig, ax_Re1700Pm2, filepath_Re1700Pm2, "Re1700Pm2", 1700, 2)
  getSimData(fig, ax_Re600Pm5,  filepath_Re600Pm5,  "Re600Pm5",  600,  5)
  ## save figure
  ax_Re3600Pm1.set_ylabel(r"$\widehat{\mathcal{P}}_{\rm mag}(k, t) = \mathcal{P}_{\rm mag}(k, t) / \int{\rm d}k \mathcal{P}_{\rm mag}(k, t)$")
  ax_Re3600Pm1.legend(loc="lower left", fontsize=20)
  print("Saving figure...")
  filepath_fig = f"dataset_axel.png"
  plt.savefig(filepath_fig)
  plt.close()
  print("Saved figure:", filepath_fig)


## PROGRAM ENTRY POINT
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
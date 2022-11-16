#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os

## 'tmpfile' needs to be loaded before any 'matplotlib' libraries,
## so matplotlib stores its cache in a temporary directory.
## (necessary when plotting in parallel)
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
import matplotlib.pyplot as plt

## load user defined modules
from TheSimModule import SimParams
from TheLoadingModule import LoadFlashData
from ThePlottingModule import PlotFuncs


## ##############################################################
## PREPARE WORKSPACE
## ##############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


def plotSpectra_timeEvolve(
    fig, ax, list_sim_times, list_k_group_t, list_power_group_t, cmap_name
  ):
  cmap, norm = PlotFuncs.createCmap(
    cmap_name = cmap_name,
    vmin      = min(list_sim_times),
    vmax      = max(list_sim_times)
  )
  for time_index, time_val in enumerate(list_sim_times):
    ax.plot(
      list_k_group_t[0],
      list_power_group_t[time_index],
      color=cmap(norm(time_val)), ls="-", alpha=0.25, zorder=1
    )
  PlotFuncs.addColorbar_fromCmap(
    fig, ax, cmap, norm,
    label = r"$t = t_{\rm sim} / t_{\rm turb}$"
  )


class PlotSpectra():
  def __init__(self,
      fig, axs, filepath_sim_res
    ):
    self.fig              = fig
    self.axs              = axs
    self.filepath_sim_res = filepath_sim_res
  
  def performRoutines(self):
    self.__loadData()
    self.__plotData()
    self.__labelAxis()

  def __loadData(self):
    ## load simulation parameters
    dict_sim_outputs = SimParams.readSimOutputs(self.filepath_sim_res)
    plots_per_eddy  = dict_sim_outputs["plots_per_eddy"]
    ## load energy spectra
    print("Loading kinetic energy spectra...")
    self.list_kin_k_group_t, self.list_kin_power_group_t, self.list_kin_sim_times = LoadFlashData.loadAllSpectraData(
      filepath          = f"{self.filepath_sim_res}/spect/",
      str_spectra_type  = "vel",
      plots_per_eddy    = plots_per_eddy,
      bool_hide_updates = True
    )
    print("Loading magnetic energy spectra...")
    self.list_mag_k_group_t, self.list_mag_power_group_t, self.list_mag_sim_times = LoadFlashData.loadAllSpectraData(
      filepath          = f"{self.filepath_sim_res}/spect/",
      str_spectra_type  = "mag",
      plots_per_eddy    = plots_per_eddy,
      bool_hide_updates = True
    )

  def __plotData(self):
    plotSpectra_timeEvolve(
      fig                = self.fig,
      ax                 = self.axs[0],
      list_sim_times     = self.list_kin_sim_times,
      list_k_group_t     = self.list_kin_k_group_t,
      list_power_group_t = self.list_kin_power_group_t,
      cmap_name          = "Blues"
    )
    plotSpectra_timeEvolve(
      fig                = self.fig,
      ax                 = self.axs[1],
      list_sim_times     = self.list_mag_sim_times,
      list_k_group_t     = self.list_mag_k_group_t,
      list_power_group_t = self.list_mag_power_group_t,
      cmap_name          = "Reds"
    )

  def __labelAxis(self):
    for index_axs in range(len(self.axs)):
      self.axs[index_axs].set_xlabel(r"$k$")
      self.axs[index_axs].set_xscale("log")
      self.axs[index_axs].set_yscale("log")
      PlotFuncs.addAxisTicks_log10(
        self.axs[index_axs],
        bool_major_ticks = True,
        num_major_ticks  = 10
      )


## ##############################################################
## MAIN PROGRAM
## ##############################################################
def main():
  filepath_sim_res = "/scratch/ek9/nk7952/Re10/super_sonic/Pm25/288/"
  fig, fig_grid = PlotFuncs.createFigure_grid(num_rows=1, num_cols=2)
  ax00 = fig.add_subplot(fig_grid[0])
  ax10 = fig.add_subplot(fig_grid[1])
  axs = [ ax00, ax10 ]
  obj_plot_spectra = PlotSpectra(fig, axs, filepath_sim_res)
  obj_plot_spectra.performRoutines()
  PlotFuncs.saveFigure(fig, "demo.png")


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()


## END OF PROGRAM
#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import sys
import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit

## load user defined modules
from TheFlashModule import FileNames, LoadData, SimParams
from TheUsefulModule import WWFnF
from TheAnalysisModule import WWFields, StatsStuff
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def initFigure(ncols=1, nrows=1):
  return plt.subplots(
    ncols   = ncols,
    nrows   = nrows,
    figsize = (7*ncols, 7*nrows)
  )

def plotFieldPDF(
    ax, field,
    num_bins    = 100,
    color       = "black",
    weights     = None,
    label_xaxis = None
  ):
  mask = np.isfinite(field)
  field = field[mask]
  if weights is not None:
    weights = weights[mask]
  bin_edges, pdf = StatsStuff.computePDF(field, num_bins, weights)
  peak_index = np.argmax(pdf)
  bin_peak = (bin_edges[peak_index] + bin_edges[peak_index-1]) / 2
  ax.plot(bin_edges, pdf, drawstyle="steps", color=color)
  ax.axvline(x=bin_peak, color="black", ls="--", lw=2.0)
  ax.set_xlabel(label_xaxis)

def plotScatter(ax, list_x, list_y, color=None, cutofff=None, bool_fit=False):
  list_x = np.array(list_x).flatten()
  list_y = np.array(list_y).flatten()
  xy_stack = np.vstack([ list_x, list_y ])
  if color is None:
    density = gaussian_kde(xy_stack)(xy_stack)
    color = density
  sns.kdeplot(list_x, list_y, ax=ax, color=color, levels=5)
  # ax.scatter(list_x, list_y, c=color, s=1)
  if bool_fit:
    func_shallow = lambda x, a0: a0 - 0.5 * x
    func_steep   = lambda x, a0: a0 - 7 * x
    x_range = np.linspace(np.nanmin(list_x), np.nanmax(list_x), 100)
    params_shallow, _ = curve_fit(func_shallow, list_x, list_y)
    y_shallow = func_shallow(x_range, *params_shallow)
    ax.plot(x_range, y_shallow, color="black", ls="--", lw=2.0)
    if cutofff is not None:
      mask = (density >= cutofff * np.max(density))
      params_steep, _ = curve_fit(func_steep, list_x[mask], list_y[mask])
      y_steep = func_steep(x_range, *params_steep)
      ax.plot(x_range, y_steep, color="black", ls="-", lw=2.0)


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class PlotTNBBasis():
  def __init__(self, filepath_sim_res):
    self.filepath_sim_res = filepath_sim_res
    self.filepath_vis = f"{self.filepath_sim_res}/vis_folder/"
    WWFnF.createFolder(self.filepath_vis, bool_verbose=False)
    ## read simulation parameters
    self.dict_sim_inputs  = SimParams.readSimInputs(self.filepath_sim_res, False)
    self.dict_sim_outputs = SimParams.readSimOutputs(self.filepath_sim_res, False)
    self.index_bounds_growth, self.index_end_growth = self.dict_sim_outputs["index_bounds_growth"]
    self.sim_name = SimParams.getSimName(self.dict_sim_inputs)

  def _plotContours(self, ax, scalar_field, color):
    num_cells = self.dict_sim_inputs["num_blocks"][0] * self.dict_sim_inputs["num_procs"][0]
    x = np.linspace(-1.0, 1.0, num_cells)
    y = np.linspace(-1.0, 1.0, num_cells)
    X, Y = np.meshgrid(x, -y)
    ax.contour(
      X, Y,
      np.log10(scalar_field[:,:,0]),
      linestyles = "-",
      linewidths = 2.0,
      levels     = 20,
      colors     = color,
      alpha      = 0.5,
      zorder     = 3
    )

  def performRoutine(self):
    # self.fig, self.axs = initFigure(ncols=3)
    self.fig, self.axs = plt.subplots(figsize=(6,6))
    cmap, norm = PlotFuncs.createCmap(
      cmr.cm.emerald,
      vmin = 5,
      vmax = self.index_end_growth
    )
    file_index = (self.index_end_growth - 5) // 2
    filename = FileNames.FILENAME_FLASH_PLT_FILES + str(int(file_index)).zfill(4)
    filepath_file = f"{self.filepath_sim_res}/plt/{filename}"
    b_field = LoadData.loadFlashDataCube(
      filepath_file = filepath_file,
      num_blocks    = self.dict_sim_inputs["num_blocks"],
      num_procs     = self.dict_sim_inputs["num_procs"],
      field_name    = "mag"
    )
    b_magn = WWFields.fieldMagnitude(b_field)
    b_rms = WWFields.fieldRMS(b_magn)
    ## magnetic field slice
    print("Plotting magnetic field...")
    _, _, _, kappa_B = WWFields.computeTNBBasis(b_field)
    ## magnetic field curvature
    print("Plotting field curvature...")
    plotScatter(
      ax      = self.axs,
      list_x  = np.log10(kappa_B)[:,:,0],
      list_y  = np.log10(b_magn/b_rms)[:,:,0],
      color = cmap(norm(file_index))
    )
    # self.axs.set_xlim([ -1, 3.5 ])
    # self.axs.set_ylim([ -2.5, 1 ])
    self.axs.set_xlabel(r"$\kappa$")
    self.axs.set_ylabel(r"$b$")
    ## save figure
    filepath_fig = f"{self.filepath_sim_res}/vis_folder/{self.sim_name}_kappa.png"
    self.fig.savefig(filepath_fig, dpi=200)
    plt.close(self.fig)
    print("Saved figure:", filepath_fig)
    print(" ")


## ###############################################################
## OPPERATOR HANDLING PLOT CALLS
## ###############################################################
def plotSimData(filepath_sim_res, bool_verbose=True, lock=None, **kwargs):
  obj = PlotTNBBasis(filepath_sim_res)
  obj.performRoutine()


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  SimParams.callFuncForAllSimulations(
    func               = plotSimData,
    bool_mproc         = BOOL_MPROC,
    list_base_paths    = LIST_BASE_PATHS,
    list_suite_folders = LIST_SUITE_FOLDERS,
    list_mach_regimes  = LIST_MACH_REGIMES,
    list_sim_folders   = LIST_SIM_FOLDERS,
    list_sim_res       = LIST_SIM_RES
  )


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
BOOL_MPROC      = 0
LIST_BASE_PATHS = [
  "/scratch/ek9/nk7952/",
  # "/scratch/jh2/nk7952/"
]

LIST_SUITE_FOLDERS = [ "Rm3000" ]
LIST_MACH_REGIMES  = [ "Mach0.3", "Mach5" ]
LIST_SIM_FOLDERS   = [ "Pm1" ]
LIST_SIM_RES       = [ "144" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
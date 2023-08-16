#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import sys
import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from matplotlib.patches import FancyArrowPatch

## load user defined modules
from TheFlashModule import FileNames, LoadData, SimParams
from TheUsefulModule import WWFnF
from TheAnalysisModule import WWFields
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def initFigure(ncols=1, nrows=1):
  scale = 0.8
  return plt.subplots(
    ncols   = ncols,
    nrows   = nrows,
    sharex  = True,
    sharey  = True,
    figsize = (7*ncols*scale, 7*nrows*scale),
    constrained_layout = True
  )

def getSimLabel(dict_sim_inputs):
  Re = dict_sim_inputs["Re"]
  Pm = dict_sim_inputs["Pm"]
  return "$" + dict_sim_inputs["mach_regime"].replace("Mach", "\mathcal{M}") + "\\text{Re}" + f"{Re:.0f}" + "\\text{Pm}" + f"{Pm:.0f}" + "$"

def addText(ax, pos, text, color="black", ha="left", rotation=0, fontsize=24):
  ax.text(
    pos[0], pos[1],
    text,
    va        = "bottom",
    ha        = ha,
    color     = color,
    rotation  = rotation,
    fontsize  = fontsize,
    zorder    = 10
  )

def plotLinePassingThroughPoint(
    ax, domain, slope, coord,
    ls = "-",
    lw = 2.0
  ):
  x1, y1 = coord
  a0 = y1 - slope * x1
  x = np.linspace(domain[0], domain[1], 100)
  y = a0 + slope * x
  PlotFuncs.plotData_noAutoAxisScale(ax, x, y, ls=ls, lw=lw, zorder=10)

def compute2DHistogramBins(field_x, field_y, num_bins):
  _, bins_x, bins_y = np.histogram2d(
    x       = field_x,
    y       = field_y,
    bins    = num_bins,
    density = True
  )
  ## extend bins
  median_bin_x = np.median(bins_x) # find the middle
  bins_x = np.linspace(
    median_bin_x - 3*np.abs(median_bin_x - bins_x[0]),
    median_bin_x + 3*np.abs(median_bin_x - bins_x[0]),
    3*num_bins
  )
  ## do the same for y
  median_bin_y = np.median(bins_y) # find the middle
  bins_y = np.linspace(
    median_bin_y - 3*np.abs(median_bin_y - bins_y[0]),
    median_bin_y + 3*np.abs(median_bin_y - bins_y[0]),
    3*num_bins
  )
  return bins_x, bins_y

def plotScatter(ax, list_x, list_y, cutofff=0.75, bool_fit=False):
  list_x = np.array(list_x).flatten()
  list_y = np.array(list_y).flatten()
  xy_stack = np.vstack([ list_x, list_y ])
  density = gaussian_kde(xy_stack)(xy_stack)
  ax.scatter(list_x, list_y, c=density, s=1)
  if bool_fit:
    func_shallow = lambda x, a0: a0 - 1/2 * x
    func_steep   = lambda x, a0: a0 - 10 * x
    x_range = np.linspace(np.nanmin(list_x), np.nanmax(list_x), 100)
    ## fit shallow
    params_shallow, _ = curve_fit(func_shallow, list_x, list_y)
    y_shallow = func_shallow(x_range, *params_shallow)
    ax.plot(x_range, y_shallow, color="black", ls="--", lw=2.0)
    ## fit steep
    mask = (density >= cutofff * np.max(density))
    params_steep, _ = curve_fit(func_steep, list_x[mask], list_y[mask])
    y_steep = func_steep(x_range, *params_steep)
    ax.plot(x_range, y_steep, color="black", ls="-", lw=2.0)


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class PlotTNBBasis():
  def __init__(self, fig, ax, filepath_sim_res):
    self.fig = fig
    self.ax = ax
    self.filepath_sim_res = filepath_sim_res
    self.filepath_vis = f"{self.filepath_sim_res}/vis_folder/"
    WWFnF.createFolder(self.filepath_vis, bool_verbose=False)
    ## read simulation parameters
    self.dict_sim_inputs  = SimParams.readSimInputs(self.filepath_sim_res, False)
    self.dict_sim_outputs = SimParams.readSimOutputs(self.filepath_sim_res, False)
    self.index_bounds_growth, self.index_end_growth = self.dict_sim_outputs["index_bounds_growth"]
    self.sim_name = SimParams.getSimName(self.dict_sim_inputs)
    self.num_cells = self.dict_sim_inputs["num_blocks"][0] * self.dict_sim_inputs["num_procs"][0]

  def getKinematicRange(self):
    step_size = (self.index_end_growth - 5) // 10
    return range(5, self.index_end_growth, step_size)

  def getSaturatedRange(self):
    index_start_sat = self.dict_sim_outputs["index_start_sat"]
    index_end_sat = np.max([
      int(filename.split("_")[-1])
      for filename in list_filenames
    ])
    step_size = (index_end_sat - index_start_sat) // 10
    list_filenames = WWFnF.getFilesInDirectory(
      directory             = f"{self.filepath_sim_res}/plt/",
      filename_starts_with  = FileNames.FILENAME_FLASH_PLT_FILES,
      filename_not_contains = "spect",
      loc_file_index        = -1
    )
    return range(index_start_sat, index_end_sat, step_size)

  def performRoutine(self):
    bins_defined = False
    list_pdfs = []
    print("Loading data...")
    for file_index in self.getKinematicRange():
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
      _, _, _, kappa = WWFields.computeTNBBasis(b_field)
      ## magnetic field curvature
      if not(bins_defined):
        bins_x, bins_y = compute2DHistogramBins(
          field_x  = np.log10(kappa.flatten()),
          field_y  = np.log10(b_magn.flatten()/b_rms),
          num_bins = self.num_cells
        )
      pdf, _, _ = np.histogram2d(
        x       = np.log10(kappa.flatten()),
        y       = np.log10(b_magn.flatten()/b_rms),
        bins    = [bins_x, bins_y],
        density = True
      )
      list_pdfs.append(pdf)
    ## plot pdf
    print("Plotting average PDF...")
    X, Y = np.meshgrid(bins_x, bins_y)
    ave_pdf = np.log10(np.mean(list_pdfs, axis=0))
    obj_plot = self.ax.pcolormesh(
      X, Y,
      ave_pdf,
      cmap = "cmr.ocean_r",
      norm = colors.Normalize(vmin=-4.05, vmax=0.05)
    )
    plotLinePassingThroughPoint(
      ax     = self.ax,
      domain = (-2, 6),
      slope  = -1/4,
      coord  = (-2, 1),
      ls     = "-",
      lw     = 2.0
    )
    plotLinePassingThroughPoint(
      ax     = self.ax,
      domain = (-2, 6),
      slope  = -1/2,
      coord  = (-2, 1),
      ls     = "-",
      lw     = 2.0
    )
    plotLinePassingThroughPoint(
      ax     = self.ax,
      domain = (-2, 6),
      slope  = -1,
      coord  = (-2, 1),
      ls     = "-",
      lw     = 2.0
    )
    self.ax.text(
      0.95, 0.95,
      getSimLabel(self.dict_sim_inputs),
      va        = "top",
      ha        = "right",
      transform = self.ax.transAxes,
      color     = "black",
      fontsize  = 24,
      zorder    = 10
    )
    ## log_10(kappa ell) = log_10(ell/R), where R = ell_box / factor -> log_10(factor)
    self.ax.axvline(x=np.log10(2), ls="--", lw=2.0, color="red", zorder=5)
    return obj_plot


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  WWFnF.createFolder(PLOT_PATH)
  # list_filepaths = SimParams.getListOfSimFilepaths(
  #   list_base_paths    = LIST_BASE_PATHS,
  #   list_suite_folders = LIST_SUITE_FOLDERS,
  #   list_mach_regimes  = LIST_MACH_REGIMES,
  #   list_sim_folders   = LIST_SIM_FOLDERS,
  #   list_sim_res       = LIST_SIM_RES
  # )
  fig, fig_grid = PlotFuncs.createFigure_grid(2, 3, 0.8, (7,7))
  axs = [
    fig.add_subplot(fig_grid[0, 0]),
    fig.add_subplot(fig_grid[0, 1]),
    fig.add_subplot(fig_grid[0, 2]),
    fig.add_subplot(fig_grid[1, 0]),
    fig.add_subplot(fig_grid[1, 1]),
    fig.add_subplot(fig_grid[1, 2])
  ]
  for plot_index, filepath_sim_res in enumerate([
      "/scratch/ek9/nk7952/Rm3000/Mach0.3/Pm125/144",
      "/scratch/ek9/nk7952/Rm3000/Mach0.3/Pm5/144",
      "/scratch/ek9/nk7952/Rm3000/Mach0.3/Pm1/144",
      "/scratch/ek9/nk7952/Rm3000/Mach5/Pm125/144",
      "/scratch/ek9/nk7952/Rm3000/Mach5/Pm5/144",
      "/scratch/ek9/nk7952/Rm3000/Mach5/Pm1/144"
    ]):
    obj = PlotTNBBasis(fig, axs[plot_index], filepath_sim_res)
    obj_plot = obj.performRoutine()
    axs[plot_index].set_xlim([ -2.25, 4.75 ])
    axs[plot_index].set_ylim([ -3.1, 1.1 ])
    print(" ")
  axs[0].set_ylabel(r"$\log_{10}\big(b / b_\mathrm{rms}\big)$", fontsize=28)
  axs[3].set_ylabel(r"$\log_{10}\big(b / b_\mathrm{rms}\big)$", fontsize=28)
  axs[3].set_xlabel(r"$\log_{10}\big(\kappa \ell_\mathrm{box}\big)$", fontsize=28)
  axs[4].set_xlabel(r"$\log_{10}\big(\kappa \ell_\mathrm{box}\big)$", fontsize=28)
  axs[5].set_xlabel(r"$\log_{10}\big(\kappa \ell_\mathrm{box}\big)$", fontsize=28)
  ax_cbar = fig.add_axes([ 0.068, 1.01, 0.93, 0.035 ]) # 655 255
  fig.colorbar(mappable=obj_plot, cax=ax_cbar, orientation="horizontal")
  ax_cbar.set_title(r"$\log_{10}\big(\mathrm{PDF}\big)$", fontsize=28, pad=12.5)
  ax_cbar.xaxis.set_ticks_position("top")
  addText(axs[0], (3.25, -0.25), r"$\kappa^{-1/4}$", fontsize=28)
  addText(axs[0], (3.25, -2.75), r"$\kappa^{-1/2}$", fontsize=28)
  addText(axs[0], (0.75, -3), r"$\kappa^{-1}$", fontsize=28)
  addText(axs[1], (0.5, -2.65), r"field reversals", fontsize=24, color="red")
  axs[1].arrow(
    x  = np.log10(2),
    y  = -2.75,
    dx = np.log10(2)+1.75,
    dy = 0,
    color = "red",
    head_width = 0.15
  )
  addText(axs[1], (np.log10(2)-0.1, -2.95), r"$\log_{10}(2)$", fontsize=24, rotation=90, color="red", ha="right")
  ## top row
  axs[0].set_xticklabels([ ])
  axs[1].set_xticklabels([ ])
  axs[1].set_yticklabels([ ])
  axs[2].set_xticklabels([ ])
  axs[2].set_yticklabels([ ])
  ## bottom row
  axs[4].set_yticklabels([ ])
  axs[5].set_yticklabels([ ])
  ## save figure
  filepath_fig = f"{PLOT_PATH}/kappa_correlation.pdf"
  print("Saving figure...")
  fig.savefig(filepath_fig, dpi=200)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
PLOT_PATH  = "/home/586/nk7952/MHDCodes/kriel2023/kappa"
BOOL_MPROC = 0
LIST_BASE_PATHS = [
  "/scratch/ek9/nk7952/",
  # "/scratch/jh2/nk7952/"
]

LIST_SUITE_FOLDERS = [ "Rm3000" ]
LIST_MACH_REGIMES  = [ "Mach0.3", "Mach5" ]
LIST_SIM_FOLDERS   = [ "Pm1", "Pm125" ]
LIST_SIM_RES       = [ "144" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
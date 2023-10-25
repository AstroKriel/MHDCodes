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

def addText(ax, pos, text, color="black", ha="left", va="bottom", rotation=0, fontsize=28):
  ax.text(
    pos[0], pos[1],
    text,
    ha        = ha,
    va        = va,
    color     = color,
    rotation  = rotation,
    fontsize  = fontsize,
    zorder    = 10
  )

def plotLinePassingThroughPoint(
    ax, domain, coord, slope,
    ls = "-",
    lw = 2.0
  ):
  x1, y1 = coord
  a0 = y1 - slope * x1
  x = np.linspace(domain[0], domain[1], 100)
  y = a0 + slope * x
  PlotFuncs.plotData_noAutoAxisScale(ax, x, y, ls=ls, lw=lw, zorder=10)

# def plotPoint(ax, x_median, y_median, y_1sig, marker, label=None, color="black"):
#   ax.errorbar(
#     x_median, y_median,
#     yerr       = y_1sig,
#     fmt        = marker,
#     mfc        = color,
#     zorder     = 5,
#     markersize = 8,
#     label      = label,
#     ecolor = color, mec="black", elinewidth=1.5, capsize=7.5, linestyle="None"
#   )

# def something():
  # ## e_i e_j: (component-j, component-i, x, y, z)
  # basis_t_tensor = np.einsum("ixyz,jxyz->jixyz", basis_t, basis_t)
  # basis_n_tensor = np.einsum("ixyz,jxyz->jixyz", basis_n, basis_n)
  # ## dv_j/dx_i: (component-j, gradient-direction-i, x, y, z)
  # gradient_tensor = np.array([
  #   WWFields.fieldGradient(field_component)
  #   for field_component in u_field
  # ])
  # qq_t = np.einsum("jixyz,jixyz->xyz", basis_t_tensor, gradient_tensor)
  # qq_n = np.einsum("jixyz,jixyz->xyz", basis_n_tensor, gradient_tensor)
  # time_val = file_index / self.outputs_per_t_turb
  # if not(bool_added_label):
  #   label_n = r"$\hat{\bm{e}}_\mathrm{n}\otimes\hat{\bm{e}}_\mathrm{n} : \nabla\otimes\bm{u}$"
  #   label_t = r"$\hat{\bm{e}}_\mathrm{t}\otimes\hat{\bm{e}}_\mathrm{t} : \nabla\otimes\bm{u}$"
  #   bool_added_label = True
  # else: label_t = label_n = None
  # plotPoint(self.ax, time_val, np.mean(qq_n), np.std(qq_n), "D", color="orangered", label=label_n)
  # plotPoint(self.ax, time_val, np.mean(qq_t), np.std(qq_t), "o", color="royalblue", label=label_t)


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class PlotTNBBasis():
  def __init__(self, ax_kappa, ax_bfield, filepath_sim_res, color):
    self.ax_kappa  = ax_kappa
    self.ax_bfield = ax_bfield
    self.color = color
    self.filepath_sim_res = filepath_sim_res
    self.filepath_vis = f"{self.filepath_sim_res}/vis_folder/"
    WWFnF.createFolder(self.filepath_vis, bool_verbose=False)
    ## read simulation parameters
    self.dict_sim_inputs  = SimParams.readSimInputs(self.filepath_sim_res, False)
    self.dict_sim_outputs = SimParams.readSimOutputs(self.filepath_sim_res, False)
    self.outputs_per_t_turb = self.dict_sim_outputs["outputs_per_t_turb"]
    self.index_bounds_growth, self.index_end_growth = self.dict_sim_outputs["index_bounds_growth"]
    self.sim_name = SimParams.getSimName(self.dict_sim_inputs)
    self.num_cells = self.dict_sim_inputs["num_blocks"][0] * self.dict_sim_inputs["num_procs"][0]

  def getKinematicRange(self):
    step_size = (self.index_end_growth - 5*self.outputs_per_t_turb) // 3
    return range(5*self.outputs_per_t_turb, self.index_end_growth, step_size)

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
    print("Loading data...")
    num_bins = 50
    bins_kappa  = np.logspace(-3, 5, num=num_bins)
    bins_bfield = np.logspace(-4, 2, num=num_bins)
    pdf_kappa   = np.zeros(num_bins-1)
    pdf_bfield  = np.zeros(num_bins-1)
    count = 0
    for file_index in self.getKinematicRange():
      filename = FileNames.FILENAME_FLASH_PLT_FILES + str(int(file_index)).zfill(4)
      filepath_file = f"{self.filepath_sim_res}/plt/{filename}"
      bfield = LoadData.loadFlashDataCube(
        filepath_file = filepath_file,
        num_blocks    = self.dict_sim_inputs["num_blocks"],
        num_procs     = self.dict_sim_inputs["num_procs"],
        field_name    = "mag"
      )
      _, _, _, kappa = WWFields.computeTNBBasis(bfield)
      bmag = WWFields.fieldMagnitude(bfield)
      brms = WWFields.fieldRMS(bmag)
      count_kappa, _  = np.histogram(np.sort(kappa.flatten()), bins=bins_kappa)
      count_bfield, _ = np.histogram(np.sort(bmag.flatten() / brms), bins=bins_bfield)
      pdf_kappa  += count_kappa / np.sum(count_kappa)
      pdf_bfield += count_bfield / np.sum(count_bfield)
      count += 1
    label = getSimLabel(self.dict_sim_inputs)
    self.ax_kappa.plot(
      np.log10(bins_kappa[1:]),
      np.log10(pdf_kappa / count),
      "o-", color=self.color, ms=5, label=label
    )
    self.ax_bfield.plot(
      np.log10(bins_bfield[1:]),
      np.log10(pdf_bfield / count),
      "o-", color=self.color, ms=5, label=label
    )


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def saveKappaPlot(fig, ax):
  ## annotate figure
  ax.set_xlim([ -2-0.2, 4+0.2 ])
  ax.set_ylim([ -6-0.2, -0.5+0.2 ])
  ax.axvline(x=np.log10(2), ls="--", lw=2.0, color="red", zorder=1)
  addText(ax, (np.log10(2)-0.1, -5.75), r"$\log_{10}(2)$", rotation=90, color="red", ha="right", fontsize=26)
  plotLinePassingThroughPoint(ax, (-3, 5), (-0.65, -3), 2, ls="-")
  plotLinePassingThroughPoint(ax, (-3, 5), (3, -2.5), -13/7, ls="--")
  addText(ax, (-0.25, -1.5), r"$\kappa^{2}$", color="black", ha="right", va="top")
  addText(ax, (2.85, -1.5), r"$\kappa^{-13/7}$", color="black", ha="left", va="top")
  ax.legend(bbox_to_anchor=(0.375, 0.3), bbox_transform=ax.transAxes, fontsize=22)
  ax.set_xlabel(r"$\log_{10}\big(\kappa \ell_{\rm box}\big)$")
  ax.set_ylabel(r"$\log_{10}\big($PDF$(\kappa \ell_{\rm box})\big)$")
  addText(ax, (0.4, -3.95), r"field reversals", fontsize=22, color="red", va="top")
  ax.arrow(
    x  = np.log10(2),
    y  = -3.75,
    dx = 1,
    dy = 0,
    color = "red",
    width = 0.02,
    head_width = 0.15,
    head_length = 0.125
  )
  ## save figure
  filepath_fig = f"{PLOT_PATH}/pdfs_kappa.pdf"
  print("Saving figure...")
  fig.savefig(filepath_fig, dpi=200)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")

def saveBPlot(fig, ax):
  ## annotate figure
  ax.set_xlim([ -3-0.2, 1+0.2 ])
  ax.set_ylim([ -5-0.2, -0.5+0.2 ])
  ax.axvline(x=0, ls="--", lw=2.0, color="red", zorder=1)
  plotLinePassingThroughPoint(ax, (-4, 2), (-2, -2.25), 5/2, ls="-")
  addText(ax, (-2.1, -2), r"$b^{5/2}$", color="black", ha="right", va="top")
  ax.set_xlabel(r"$\log_{10}\big(b / b_{\rm rms}\big)$")
  ax.set_ylabel(r"$\log_{10}\big($PDF$(b / b_{\rm rms})\big)$")
  ## save figure
  filepath_fig = f"{PLOT_PATH}/pdfs_bfield.pdf"
  print("Saving figure...")
  fig.savefig(filepath_fig, dpi=200)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")

def main():
  WWFnF.createFolder(PLOT_PATH)
  # list_filepaths = SimParams.getListOfSimFilepaths(
  #   list_base_paths    = LIST_BASE_PATHS,
  #   list_suite_folders = LIST_SUITE_FOLDERS,
  #   list_mach_regimes  = LIST_MACH_REGIMES,
  #   list_sim_folders   = LIST_SIM_FOLDERS,
  #   list_sim_res       = LIST_SIM_RES
  # )
  fig_kappa, ax_kappa = plt.subplots()
  fig_bfield, ax_bfield = plt.subplots()
  list_colours = [ "#B80EF6", "#F4A123" ]
  for plot_index, filepath_sim_res in enumerate([
      "/scratch/ek9/nk7952/Rm3000/Mach0.3/Pm5/144",
      "/scratch/ek9/nk7952/Rm3000/Mach5/Pm5/144"
    ]):
    obj = PlotTNBBasis(ax_kappa, ax_bfield, filepath_sim_res, list_colours[plot_index])
    obj.performRoutine()
    print(" ")
  saveKappaPlot(fig_kappa, ax_kappa)
  saveBPlot(fig_bfield, ax_bfield)


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
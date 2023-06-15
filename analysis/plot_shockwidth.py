#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np

## 'tmpfile' needs to be loaded before any 'matplotlib' libraries,
## so matplotlib stores its cache in a temporary directory.
## (necessary when plotting in parallel)
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
import matplotlib.pyplot as plt

## load user defined modules
from TheFlashModule import FileNames, LoadData, SimParams
from TheUsefulModule import WWFnF, WWObjs
from TheAnalysisModule import WWFields, StatsStuff
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def initFigure(ncols, nrows):
  return plt.subplots(
    ncols   = ncols,
    nrows   = nrows,
    figsize = (7*ncols, 7*nrows)
  )

def measurePeakPDF(field, num_bins=100, weights=None):
  mask = np.isfinite(field)
  field = field[mask]
  if weights is not None:
    weights = weights[mask]
  bin_edges, pdf = StatsStuff.computePDF(field, num_bins, weights)
  peak_index = np.argmax(pdf)
  bin_peak = (bin_edges[peak_index] + bin_edges[peak_index-1]) / 2
  return bin_peak

def plotPowerLawPassingThroughPoint(
    ax, x_domain, slope, coord,
    ls = "--",
    lw = 1.0
  ):
  x1, y1 = coord
  a0 = y1 / x1**(slope)
  x = np.logspace(np.log10(x_domain[0]), np.log10(x_domain[1]), 100)
  y = a0 * x**(slope)
  PlotFuncs.plotData_noAutoAxisScale(ax, x, y, ls=ls, lw=lw, zorder=10)


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class PlotShockWidth():
  def __init__(self, filepath_sim_res):
    self.filepath_sim_res = filepath_sim_res
    self.filepath_vis = f"{self.filepath_sim_res}/vis_folder/"
    WWFnF.createFolder(self.filepath_vis, bool_verbose=False)
    ## read simulation parameters
    self.dict_sim_inputs  = SimParams.readSimInputs(self.filepath_sim_res, False)
    self.dict_sim_outputs = SimParams.readSimOutputs(self.filepath_sim_res, False)
    self.index_bounds_growth, self.index_end_growth = self.dict_sim_outputs["index_bounds_growth"]
    self.sim_name = SimParams.getSimName(self.dict_sim_inputs)
    print("Looking at:", self.sim_name)

  def performRoutine(self, ax, cmap, norm):
    dict_scales_group_sim = WWObjs.readJsonFile2Dict(
      filepath     = FILEPATH_PAPER,
      filename     = "dataset.json",
      bool_verbose = False
    )
    kscale = dict_scales_group_sim[self.sim_name]["k_eta_cur"]["inf"]["val"]
    Re = self.dict_sim_inputs["Re"]
    Mach = self.dict_sim_inputs["desired_Mach"]
    model = Re * (1 - 1/Mach**2)
    index_file = (self.index_bounds_growth + self.index_end_growth) // 3
    filename = FileNames.FILENAME_FLASH_PLT_FILES + str(int(index_file)).zfill(4)
    filepath_file = f"{self.filepath_sim_res}/plt/{filename}"
    b_field = LoadData.loadFlashDataCube(
      filepath_file = filepath_file,
      num_blocks    = self.dict_sim_inputs["num_blocks"],
      num_procs     = self.dict_sim_inputs["num_procs"],
      field_name    = "mag"
    )
    dens_field = LoadData.loadFlashDataCube(
      filepath_file = filepath_file,
      num_blocks    = self.dict_sim_inputs["num_blocks"],
      num_procs     = self.dict_sim_inputs["num_procs"],
      field_name    = "dens"
    )
    Bmagn = WWFields.fieldMagnitude(b_field)
    gradBmagn = WWFields.fieldGradient(Bmagn)
    gradDens = WWFields.fieldGradient(dens_field)
    ## compute tnb-basis
    t_basis, n_basis, b_basis, kappa = WWFields.computeTNBBasis(gradDens)
    ## project gradient of the field magnitude onto basis
    gradBmagn_proj_onto_t = WWFields.vectorDotProduct(gradBmagn, t_basis)
    ## measure shockwidth
    shockwidth = measurePeakPDF(
      field   = np.log10(np.abs(gradBmagn_proj_onto_t)),
      weights = dens_field
    )
    ## plot data
    if   Mach == 0.3: marker = "D"
    elif Mach == 1:   marker = "s"
    elif Mach == 5:   marker = "o"
    elif Mach == 10:  marker = "^"
    ax.plot(
      kscale, 10**(shockwidth),
      color  = cmap(norm(np.log10(model))),
      marker = marker,
      markersize = 8,
      markeredgecolor = "black"
    )


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  list_filepath_sim_res = SimParams.getListOfSimFilepaths(
    list_base_paths    = LIST_BASE_PATHS,
    list_suite_folders = LIST_SUITE_FOLDERS,
    list_mach_regimes  = LIST_MACH_REGIMES,
    list_sim_folders   = LIST_SIM_FOLDERS,
    list_sim_res       = LIST_SIM_RES
  )
  fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
  cmap, norm = PlotFuncs.createCmap(
    cmap_name = "coolwarm",
    vmin = 0.5,
    vmid = 2.0,
    vmax = 3.5
  )
  for filepath_sim_res in list_filepath_sim_res:
    obj = PlotShockWidth(filepath_sim_res)
    obj.performRoutine(ax, cmap, norm)
  ax.set_xscale("log")
  ax.set_yscale("log")
  # ax.set_xlim([ 6, 110 ])
  # ax.set_ylim([ 3e-3, 3e-1 ])
  # plotPowerLawPassingThroughPoint(ax, (1, 500), 1.5, (10, 0.01))
  # ax.text(
  #   0.05, 0.95,
  #   r"$\propto k_{\rm p}^{3/2}$",
  #   transform = ax.transAxes,
  #   va        = "top",
  #   ha        = "left",
  #   fontsize  = 20
  # )
  ax.set_xlabel(r"$k_{\rm scale}$")
  ax.set_ylabel(r"peak PDF[$(\nabla|B|)\cdot\hat{e}_{\rm t}$]")
  PlotFuncs.addColorbar_fromCmap(
    fig        = ax.get_figure(),
    ax         = ax,
    cmap       = cmap,
    norm       = norm,
    cbar_title = r"$\log_{10}\big[ {\rm Re} \, (1 - 1/\mathcal{M}^2) \big]$",
    cbar_title_pad=12, orientation="horizontal", size=8, fontsize=16
  )
  PlotFuncs.addLegend_fromArtists(
    ax,
    list_artists       = [
      "D", "s", "o", "^"
    ],
    list_marker_colors = [
      "whitesmoke",
      "black", "black", "black"
    ],
    list_legend_labels = [
      r"$\mathcal{M} = 0.3$",
      r"$\mathcal{M} = 1$",
      r"$\mathcal{M} = 5$",
      r"$\mathcal{M} = 10$",
    ],
    label_color        = "black",
    loc                = "right",
    bbox               = (1.0, 0.5),
    lw                 = 1,
    rspacing           = 0.75
  )
  PlotFuncs.saveFigure(fig, f"{FILEPATH_PAPER}/shockwidth.png")

## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
FILEPATH_PAPER  = "/home/586/nk7952/MHDCodes/kriel2023"
LIST_BASE_PATHS = [
  "/scratch/ek9/nk7952/",
  "/scratch/jh2/nk7952/"
]
LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm500", "Rm3000", "Re2000" ]
LIST_MACH_REGIMES  = [ "Mach0.3", "Mach1", "Mach5", "Mach10" ]
LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm30", "Pm50", "Pm125", "Pm250", "Pm300" ]
LIST_SIM_RES       = [ "144" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
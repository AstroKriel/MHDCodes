#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys, functools
import numpy as np

## 'tmpfile' needs to be loaded before any 'matplotlib' libraries,
## so matplotlib stores its cache in a temporary directory.
## (necessary when plotting in parallel)
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.optimize import curve_fit

## load user defined modules
from TheFlashModule import FileNames, LoadData, SimParams
from TheUsefulModule import WWFnF
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def divergence(field_group_dim):
  """
  Purpose: compute the divergence of the vector field F, corresponding with dFx/dx + dFy/dy + ...
  Argument: list of ndarrays, where each item of the list is one component (dimension) of the vector field
  Output: single ndarray (scalar field) of the same shape as each of the items in F
  """
  num_dims = len(field_group_dim)
  return np.ufunc.reduce(
    np.add, [
      np.gradient(field_group_dim[dim], axis=dim)
      for dim in range(num_dims)
    ]
  )

# def curl(field_group_dim):
#   dummy, dFx_dy, dFx_dz = np.gradient(u, dx, dy, dz, axis=[1,0,2])
#   dFy_dx, dummy, dFy_dz = np.gradient(v, dx, dy, dz, axis=[1,0,2])
#   dFz_dx, dFz_dy, dummy = np.gradient(w, dx, dy, dz, axis=[1,0,2])
#   rot_x = dFz_dy - dFy_dz
#   rot_y = dFx_dz - dFz_dx
#   rot_z = dFy_dx - dFx_dy
#   return rot_x, rot_y, rot_z

def gaussian(x, amplitude, x_mean, x_std):
  return amplitude * np.exp(-(x-x_mean)**2 / (2*x_std**2))

def plotPDF(
    ax, field,
    bool_fit     = True,
    bool_flip_ax = False
  ):
  field_flat = field.flatten()
  field_mean = np.mean(field_flat)
  field_std  = np.std(field_flat)
  bounds = [
    field_mean - 5*field_std,
    field_mean + 5*field_std
  ]
  list_bin_edges, list_dens_norm = PlotFuncs.plotPDF(
    ax           = ax,
    list_data    = field_flat[
      (field_mean - 3*field_std < field_flat) & (field_flat < field_mean + 3*field_std)
    ],
    num_bins     = 30,
    bool_flip_ax = bool_flip_ax
  )
  if bool_fit:
    list_bin_centers = [
      (list_bin_edges[bin_index] + list_bin_edges[bin_index+1]) / 2
      for bin_index in range(len(list_bin_edges)-1)
    ]
    ## fit bin centers
    fit_params, _ = curve_fit(
      f      = gaussian,
      xdata  = list_bin_centers,
      ydata  = list_dens_norm[1:],
      maxfev = 10**5,
      p0     = [
        max(list_dens_norm),
        field_mean,
        field_std
      ],
      bounds = (
        [ 1e-5, -10.0, 1e-5 ],
        [ 1.0,   10.0, 10.0 ]
      )
    )
    ## plot fits
    x = np.linspace(bounds[0], bounds[1], 1000)
    plot_params = { "color":"red", "ls":"--", "lw":2, "drawstyle":"steps" }
    if bool_flip_ax: ax.plot(gaussian(x, *fit_params), x, **plot_params)
    else: ax.plot(x, gaussian(x, *fit_params), **plot_params)
  ## adjust axis
  if bool_flip_ax: ax.set_ylim(bounds)
  else: ax.set_xlim(bounds)


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class PlotBoxData():
  def __init__(
      self,
      filepath_file, filepath_vis, dict_sim_inputs,
      bool_verbose = True
    ):
    ## save input arguments
    self.filepath_file   = filepath_file
    self.filepath_vis    = filepath_vis
    self.dict_sim_inputs = dict_sim_inputs
    self.bool_verbose    = bool_verbose

  def performRoutine(self, lock):
    ## save figure
    if self.bool_verbose: print("Initialising figure...")
    self.fig_fields, self.axs_fields = plt.subplots(ncols=5, nrows=3, figsize=(5*5, 3*5.5))
    self.fig_comps,  self.axs_comps  = plt.subplots(ncols=3, nrows=3, figsize=(3*5, 3*5.5))
    ## plot data
    self._plotMagneticField()
    self._plotCurrent()
    self._plotVelocityField()
    self._plotDensityField()
    ## save figure
    if lock is not None: lock.acquire()
    sim_name = SimParams.getSimName(self.dict_sim_inputs)
    PlotFuncs.saveFigure(self.fig_fields, f"{self.filepath_vis}/{sim_name}_field_stats.png", bool_draft=True)
    PlotFuncs.saveFigure(self.fig_comps,  f"{self.filepath_vis}/{sim_name}_comp_stats.png", bool_draft=True)
    if lock is not None: lock.release()
    if self.bool_verbose: print(" ")

  def __plotScatterAgainstBField(self, fig, ax, field, cbar_title=None):
    PlotFuncs.plotScatter(
      fig               = fig,
      ax                = ax,
      list_x            = field[:,:,0].flatten(),
      list_y            = np.log(self.mag_magnitude[:,:,0].flatten()),
      cbar_title        = cbar_title,
      bool_add_colorbar = True
    )

  def _plotMagneticField(self):
    if self.bool_verbose: print("Plotting magntic fields...")
    mag_x, mag_y, mag_z = LoadData.loadFlashDataCube(
      filepath_file = self.filepath_file,
      num_blocks    = self.dict_sim_inputs["num_blocks"],
      num_procs     = self.dict_sim_inputs["num_procs"],
      field_name    = "mag",
      bool_norm_rms = True
    )
    self.mag_magnitude = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)**2
    PlotFuncs.plotScalarField(
      field_slice       = np.log(self.mag_magnitude[:,:,0]),
      fig               = self.fig_fields,
      ax                = self.axs_fields[0][0],
      cmap_name         = "cmr.iceburn",
      NormType          = functools.partial(PlotFuncs.MidpointNormalize, midpoint=0),
      cbar_title        = r"$\ln\big(b^2 / b_{\rm rms}^2\big)$",
      bool_add_colorbar = True
    )
    plotPDF(self.axs_fields[1][0], np.log(self.mag_magnitude), bool_flip_ax=True)
    self.axs_fields[1][0].set_ylabel(r"$\ln\big(b^2 / b_{\rm rms}^2\big)$")
    plotPDF(self.axs_comps[0][0], mag_x)
    plotPDF(self.axs_comps[1][0], mag_y)
    plotPDF(self.axs_comps[2][0], mag_z)
    self.axs_comps[0][0].set_xlabel(r"$b_x / b_{x, {\rm rms}}$")
    self.axs_comps[1][0].set_xlabel(r"$b_y / b_{y, {\rm rms}}$")
    self.axs_comps[2][0].set_xlabel(r"$b_z / b_{z, {\rm rms}}$")

  def _plotCurrent(self):
    if self.bool_verbose: print("Plotting current density...")
    cur_x, cur_y, cur_z = LoadData.loadFlashDataCube(
      filepath_file = self.filepath_file,
      num_blocks    = self.dict_sim_inputs["num_blocks"],
      num_procs     = self.dict_sim_inputs["num_procs"],
      field_name    = "cur",
      bool_norm_rms = True
    )
    cur_magnitude = np.sqrt(cur_x**2 + cur_y**2 + cur_z**2)**2
    PlotFuncs.plotScalarField(
      field_slice       = np.log(cur_magnitude[:,:,0]),
      fig               = self.fig_fields,
      ax                = self.axs_fields[0][1],
      cmap_name         = "cmr.watermelon",
      NormType          = functools.partial(PlotFuncs.MidpointNormalize, midpoint=0),
      cbar_title        = r"$\ln\big(j^2 / j_{\rm rms}^2\big)$",
      bool_add_colorbar = True
    )
    x = np.linspace(-100, 100, 100)
    PlotFuncs.plotData_noAutoAxisScale(self.axs_fields[1][1], x, x, ls=":", lw=2)
    self.__plotScatterAgainstBField(
      fig        = self.fig_fields,
      ax         = self.axs_fields[1][1],
      field      = np.log(cur_magnitude),
      cbar_title = r"$\mathcal{P}(j^2,b^2)$"
    )
    self.axs_fields[1][1].set_xlim([ -20, 10 ])
    self.axs_fields[1][1].set_ylim([ -20, 10 ])
    plotPDF(self.axs_fields[2][1], np.log(cur_magnitude))
    self.axs_fields[2][1].set_xlabel(r"$\ln\big(j^2 / j_{\rm rms}^2\big)$")
    plotPDF(self.axs_comps[0][1], cur_x)
    plotPDF(self.axs_comps[1][1], cur_y)
    plotPDF(self.axs_comps[2][1], cur_z)
    self.axs_comps[0][1].set_xlabel(r"$j_x / j_{x, {\rm rms}}$")
    self.axs_comps[1][1].set_xlabel(r"$j_y / j_{y, {\rm rms}}$")
    self.axs_comps[2][1].set_xlabel(r"$j_z / j_{z, {\rm rms}}$")

  def _plotVelocityField(self):
    if self.bool_verbose: print("Plotting velocity fields...")
    ## plot velocity field
    vel_x, vel_y, vel_z = LoadData.loadFlashDataCube(
      filepath_file = self.filepath_file,
      num_blocks    = self.dict_sim_inputs["num_blocks"],
      num_procs     = self.dict_sim_inputs["num_procs"],
      field_name    = "vel",
      bool_norm_rms = True
    )
    vel_magnitude = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)**2
    PlotFuncs.plotScalarField(
      field_slice       = np.log(vel_magnitude[:,:,0]),
      fig               = self.fig_fields,
      ax                = self.axs_fields[0][2],
      cmap_name         = "cmr.viola",
      NormType          = functools.partial(PlotFuncs.MidpointNormalize, midpoint=0),
      cbar_title        = r"$\ln\big(u^2 / u_{\rm rms}^2\big)$",
      bool_add_colorbar = True
    )
    x = np.linspace(-100, 100, 100)
    PlotFuncs.plotData_noAutoAxisScale(self.axs_fields[1][2], x, x, ls=":", lw=2)
    self.__plotScatterAgainstBField(
      fig        = self.fig_fields,
      ax         = self.axs_fields[1][2],
      field      = np.log(vel_magnitude),
      cbar_title = r"$\mathcal{P}(u^2,b^2)$"
    )
    self.axs_fields[1][2].set_xlim([ -10, 10 ])
    self.axs_fields[1][2].set_ylim([ -20, 10 ])
    plotPDF(self.axs_fields[2][2], np.log(vel_magnitude))
    self.axs_fields[2][2].set_xlabel(r"$\ln\big(u^2 / u_{\rm rms}^2\big)$")
    plotPDF(self.axs_comps[0][2], vel_x)
    plotPDF(self.axs_comps[1][2], vel_y)
    plotPDF(self.axs_comps[2][2], vel_z)
    self.axs_comps[0][2].set_xlabel(r"$u_x / u_{x, {\rm rms}}$")
    self.axs_comps[1][2].set_xlabel(r"$u_y / u_{y, {\rm rms}}$")
    self.axs_comps[2][2].set_xlabel(r"$u_z / u_{z, {\rm rms}}$")
    ## plot divergence of velocity field
    vel_x, vel_y, vel_z = LoadData.loadFlashDataCube(
      filepath_file = self.filepath_file,
      num_blocks    = self.dict_sim_inputs["num_blocks"],
      num_procs     = self.dict_sim_inputs["num_procs"],
      field_name    = "vel"
    )
    vel_divergence = divergence([ vel_x, vel_y, vel_z ])
    PlotFuncs.plotScalarField(
      field_slice       = vel_divergence[:,:,0],
      fig               = self.fig_fields,
      ax                = self.axs_fields[0][3],
      cmap_name         = "cmr.viola",
      NormType          = functools.partial(PlotFuncs.MidpointNormalize, midpoint=0),
      cbar_title        = r"$\nabla\cdot\vec{u}$",
      bool_add_colorbar = True
    )
    self.__plotScatterAgainstBField(
      fig        = self.fig_fields,
      ax         = self.axs_fields[1][3],
      field      = vel_divergence, 
      cbar_title = r"$\mathcal{P}(\nabla\cdot\vec{u},b^2)$"
    )
    self.axs_fields[1][3].set_xlim([ -15, 15 ])
    self.axs_fields[1][3].set_ylim([ -20, 10 ])
    plotPDF(self.axs_fields[2][3], vel_divergence)
    self.axs_fields[2][3].set_xlabel(r"$\nabla\cdot\vec{u}$")

  def _plotDensityField(self):
    if self.bool_verbose: print("Plotting density field...")
    dens = LoadData.loadFlashDataCube(
      filepath_file = self.filepath_file,
      num_blocks    = self.dict_sim_inputs["num_blocks"],
      num_procs     = self.dict_sim_inputs["num_procs"],
      field_name    = "dens",
      bool_norm_rms = True
    )
    dens_magnitude = dens**2
    PlotFuncs.plotScalarField(
      field_slice       = np.log(dens_magnitude[:,:,0]),
      fig               = self.fig_fields,
      ax                = self.axs_fields[0][4],
      cmap_name         = "cmr.seasons",
      NormType          = colors.Normalize,
      cbar_title        = r"$\ln\big(\rho^2 / \rho_{\rm rms}^2\big)$",
      bool_add_colorbar = True
    )
    x = np.linspace(-100, 100, 100)
    PlotFuncs.plotData_noAutoAxisScale(self.axs_fields[1][4], x, x, ls=":", lw=2)
    self.__plotScatterAgainstBField(
      fig        = self.fig_fields,
      ax         = self.axs_fields[1][4],
      field      = np.log(dens_magnitude),
      cbar_title = r"$\mathcal{P}(\rho^2,b^2)$"
    )
    self.axs_fields[1][4].set_xlim([ -20, 10 ])
    self.axs_fields[1][4].set_ylim([ -20, 10 ])
    plotPDF(self.axs_fields[2][4], np.log(dens_magnitude))
    self.axs_fields[2][4].set_xlabel(r"$\ln\big(\rho^2 / \rho_{\rm rms}^2\big)$")


## ###############################################################
## OPPERATOR HANDLING PLOT CALLS
## ###############################################################
def plotSimData(filepath_sim_res, bool_verbose=True, lock=None, **kwargs):
  ## read simulation input parameters
  dict_sim_inputs  = SimParams.readSimInputs(filepath_sim_res, bool_verbose)
  dict_sim_outputs = SimParams.readSimOutputs(filepath_sim_res, bool_verbose)
  index_bounds_growth, index_end_growth = dict_sim_outputs["index_bounds_growth"]
  if bool_verbose: print(f"Index range in growth regime: [{index_bounds_growth}, {index_end_growth}]")
  index_file    = int((index_bounds_growth + index_end_growth) // 3)
  filename      = FileNames.FILENAME_FLASH_PLT_FILES + str(index_file).zfill(4)
  filepath_file = f"{filepath_sim_res}/plt/{filename}"
  print("Looking at:", filepath_file)
  ## make sure a visualisation folder exists
  filepath_vis = f"{filepath_sim_res}/vis_folder/"
  WWFnF.createFolder(filepath_vis, bool_verbose=False)
  ## plot quantities
  obj_plot_box = PlotBoxData(
    filepath_file   = filepath_file,
    filepath_vis    = filepath_vis,
    dict_sim_inputs = dict_sim_inputs,
    bool_verbose    = bool_verbose
  )
  obj_plot_box.performRoutine(lock)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  SimParams.callFuncForAllSimulations(
    func               = plotSimData,
    bool_mproc         = BOOL_MPROC,
    basepath           = PATH_SCRATCH,
    list_suite_folders = LIST_SUITE_FOLDERS,
    list_sonic_regimes = LIST_MACH_REGIMES,
    list_sim_folders   = LIST_SIM_FOLDERS,
    list_sim_res       = LIST_SIM_RES
  )


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
BOOL_MPROC   = 0
PATH_SCRATCH = "/scratch/ek9/nk7952/"
# PATH_SCRATCH = "/scratch/jh2/nk7952/"

## PLASMA PARAMETER SET
# LIST_SUITE_FOLDERS = [ "Re10", "Re500", "Rm3000" ]
# LIST_MACH_REGIMES = [ "Mach5" ]
# LIST_SIM_FOLDERS   = [ "Pm1", "Pm2", "Pm4", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
# LIST_SIM_RES       = [ "288" ]

LIST_SUITE_FOLDERS = [ "Rm3000" ]
LIST_MACH_REGIMES = [ "Mach5" ]
LIST_SIM_FOLDERS   = [ "Pm10" ]
LIST_SIM_RES       = [ "144" ]

# ## MACH NUMBER SET
# LIST_SUITE_FOLDERS = [ "Re300" ]
# LIST_MACH_REGIMES = [ "Mach0.3", "Mach1", "Mach5", "Mach10" ]
# LIST_SIM_FOLDERS   = [ "Pm4" ]
# LIST_SIM_RES       = [ "144" ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
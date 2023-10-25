#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import cmasher as cmr

## load user defined modules
from TheFlashModule import SimParams, LoadData
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
def plotVectorFieldLogged(ax, field_x1, field_x2, step=1):
  data_vecs_x1 = field_x1[::step, ::step]
  data_vecs_x2 = field_x2[::step, ::step]
  x = np.linspace(-1.0, 1.0, len(data_vecs_x1[0,:]))
  y = np.linspace(-1.0, 1.0, len(data_vecs_x2[:,0]))
  X, Y = np.meshgrid(x, -y)
  f = lambda field : np.sign(field) * np.log10(1 + field**2)
  plot_vecs_x1 = f(data_vecs_x1)
  plot_vecs_x2 = f(data_vecs_x2)
  ax.quiver(
    X, Y,
    plot_vecs_x1,
    plot_vecs_x2,
    color          = "red",
    width          = 2e-3,
    headaxislength = 0.0,
    headlength     = 0.0,
    alpha          = 0.35,
    zorder         = 5
  )


## ###############################################################
## OPERATOR CLASS
## ###############################################################
def plotSimData(ax, plot_args, filepath_sim_res):
  filepath_vis       = f"{PATH_PLOT}/field_slices/"
  WWFnF.createFolder(filepath_vis, bool_verbose=False)
  filepath_plt_files = f"{filepath_sim_res}/plt"
  filename = "Turb_hdf5_plt_cnt_0100"
  dict_sim_inputs = SimParams.readSimInputs(filepath_sim_res)
  num_cells = dict_sim_inputs["num_blocks"][0] * dict_sim_inputs["num_procs"][0]
  x = np.linspace(-1.0, 1.0, num_cells)
  y = np.linspace(-1.0, 1.0, num_cells)
  X, Y = np.meshgrid(x, -y)
  ## ------------- VELOCITY FIELD
  print("Loading velocity field data...")
  vel_field = LoadData.loadFlashDataCube(
    filepath_file = f"{filepath_plt_files}/{filename}",
    num_blocks    = dict_sim_inputs["num_blocks"],
    num_procs     = dict_sim_inputs["num_procs"],
    field_name    = "vel",
    bool_norm_rms = True
  )
  vel_magn = WWFields.fieldMagnitude(vel_field)
  ax.imshow(
    np.log10(vel_magn[:,:,0]),
    extent = [-1, 1, -1, 1],
    cmap   = plot_args["cmap_vel"],
    norm   = plot_args["norm_vel"]
  )
  ## ------------- MAGNETIC FIELD
  print("Loading magnetic field data...")
  mag_field = LoadData.loadFlashDataCube(
    filepath_file = f"{filepath_plt_files}/{filename}",
    num_blocks    = dict_sim_inputs["num_blocks"],
    num_procs     = dict_sim_inputs["num_procs"],
    field_name    = "mag",
    bool_norm_rms = True
  )
  logBmagn_slice = np.log10(WWFields.fieldMagnitude(mag_field))[:,:,0]
  logBmagn_slice[logBmagn_slice < -0.25] = np.nan
  ax.scatter(
    X, Y,
    s      = 2,
    c      = logBmagn_slice,
    cmap   = plot_args["cmap_mag"],
    norm   = plot_args["norm_mag"],
    zorder = 5,
    alpha  = 0.8,
  )
  ## ------------- DENSITY FIELD
  print("Loading density field data...")
  rho = LoadData.loadFlashDataCube(
    filepath_file = f"{filepath_plt_files}/{filename}",
    num_blocks    = dict_sim_inputs["num_blocks"],
    num_procs     = dict_sim_inputs["num_procs"],
    field_name    = "dens"
  )
  ax.contour(
    X, Y,
    np.log10(rho[:,:,0]),
    linestyles = "-",
    linewidths = 2.0,
    zorder     = 10,
    levels     = 20,
    colors     = "black",
    alpha      = 0.5,
  )


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  fig, ax = plt.subplots()
  cmap_vel, norm_vel = PlotFuncs.createCmap(
    cmap_name = "cmr.freeze",
    cmin      = 0.1,
    cmax      = 0.9,
    vmin      = 0.1,
    vmax      = 2.0,
    NormType  = colors.LogNorm
  )
  cmap_mag, norm_mag = PlotFuncs.createCmap(
    cmap_name = "Reds",
    cmin      = 0.2,
    vmin      = 0.5,
    vmax      = 50.0,
    NormType  = colors.LogNorm
  )
  plot_args = {
    "cmap_vel" : cmap_vel,
    "norm_vel" : norm_vel,
    "cmap_mag" : cmap_mag,
    "norm_mag" : norm_mag,
  }
  plotSimData(ax, plot_args, f"{PATH_SCRATCH_EK9}/Rm3000/Mach0.3/Pm5/144/")
  ax.set_xticklabels([ ])
  ax.set_yticklabels([ ])
  ## velocity colorbar
  ax_vel = fig.add_axes([ 0.068, 1.01, 0.93, 0.05 ])
  fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap_vel, norm=norm_vel), cax=ax_vel, orientation="horizontal")
  ax_vel.xaxis.set_ticks_position("top")
  ## magnetic colorbar
  ax_mag = fig.add_axes([ 0.068, 0.01, 0.93, 0.05 ])
  fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap_mag, norm=norm_mag), cax=ax_mag, orientation="horizontal")
  ax_mag.xaxis.set_ticks_position("bottom")
  ## save figure
  print("Saving figure...")
  fig_name = f"field_slices.pdf"
  filepath_fig = f"{PATH_PLOT}/{fig_name}"
  fig.savefig(filepath_fig, dpi=200)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
PATH_PLOT = "/home/586/nk7952/MHDCodes/kriel2023/"
PATH_SCRATCH_EK9 = "/scratch/ek9/nk7952/"
PATH_SCRATCH_JH2 = "/scratch/jh2/nk7952/"


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF PROGRAM
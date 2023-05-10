#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

## load user defined modules
from TheFlashModule import SimParams, LoadData
from TheUsefulModule import WWFnF
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
  # plot_vecs_log_magn = np.log10(np.sqrt(plot_vecs_x1**2 + plot_vecs_x2**2))
  # plot_vecs_x1[plot_vecs_log_magn < -1] = np.nan
  # plot_vecs_x2[plot_vecs_log_magn < -1] = np.nan
  ax.quiver(
    X, Y,
    plot_vecs_x1,
    plot_vecs_x2,
    color          = "red",
    width          = 2e-3,
    headaxislength = 0.0,
    headlength     = 0.0,
    alpha          = 0.35
  )

def plotContours(
    ax, field,
    levels = None,
    vmin   = None,
    vmax   = None
  ):
  x = np.linspace(-1.0, 1.0, len(field[0,:]))
  y = np.linspace(-1.0, 1.0, len(field[:,0]))
  X, Y = np.meshgrid(x, -y)
  ax.contour(
    X, Y,
    field,
    levels     = levels,
    vmin       = np.min(field) if (vmin is None) else vmin,
    vmax       = np.min(field) if (vmax is None) else vmax,
    colors     = "black",
    linestyles = "-",
    alpha      = 0.3
  )


## ###############################################################
## OPERATOR CLASS
## ###############################################################
def plotSimData(filepath_sim_res):
  filepath_vis       = f"{PATH_PLOT}/field_slices/"
  WWFnF.createFolder(filepath_vis, bool_verbose=False)
  filepath_plt_files = f"{filepath_sim_res}/plt"
  filename           = "Turb_hdf5_plt_cnt_0100"
  dict_sim_inputs    = SimParams.readSimInputs(filepath_sim_res)
  fig, ax = plt.subplots(figsize=(12, 12))
  ## ------------- VELOCITY FIELD
  print("Loading velocity field data...")
  vel_x, vel_y, vel_z = LoadData.loadFlashDataCube(
    filepath_file = f"{filepath_plt_files}/{filename}",
    num_blocks    = dict_sim_inputs["num_blocks"],
    num_procs     = dict_sim_inputs["num_procs"],
    field_name    = "vel",
    bool_norm_rms = True
  )
  vel_magn = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
  PlotFuncs.plotScalarField(
    field_slice          = np.log10(vel_magn[:,:,0]),
    fig                  = fig,
    ax                   = ax,
    cbar_orientation     = "horizontal",
    cmap_name            = "cmr.freeze",
    NormType             = colors.Normalize,
    cbar_bounds          = [ -2, 1 ],
    cbar_title           = None,
    bool_label_axis      = False
  )
  ## ------------- MAGNETIC FIELD
  print("Loading magnetic field data...")
  mag_x, mag_y, mag_z = LoadData.loadFlashDataCube(
    filepath_file = f"{filepath_plt_files}/{filename}",
    num_blocks    = dict_sim_inputs["num_blocks"],
    num_procs     = dict_sim_inputs["num_procs"],
    field_name    = "mag",
    bool_norm_rms = True
  )
  plotVectorFieldLogged(
    ax       = ax,
    field_x1 = mag_x[:,:,0],
    field_x2 = mag_y[:,:,0],
    step     = 1
  )
  # plotContours(
  #   ax     = ax,
  #   field  = np.log10(np.sqrt(mag_x**2 + mag_y**2 + mag_z**2))[:,:,0],
  #   levels = 7
  #   vmin   = -2,
  #   vmax   = 1
  # )
  ## ------------- DENSITY FIELD
  print("Loading density field data...")
  rho = LoadData.loadFlashDataCube(
    filepath_file = f"{filepath_plt_files}/{filename}",
    num_blocks    = dict_sim_inputs["num_blocks"],
    num_procs     = dict_sim_inputs["num_procs"],
    field_name    = "dens"
  )
  plotContours(
    ax     = ax,
    field  = np.log10(rho[:,:,0]),
    levels = 10
  )
  ## save figure
  print("Saving figure...")
  sim_name     = SimParams.getSimName(dict_sim_inputs)
  fig_name     = f"{sim_name}_field_slices.png"
  filepath_fig = f"{filepath_vis}/{fig_name}"
  fig.savefig(filepath_fig, dpi=100)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  plotSimData(f"{PATH_SCRATCH}/Rm3000/Mach0.3/Pm125/288/")
  # plotSimData(f"{PATH_SCRATCH}/Rm3000/Mach0.3/Pm10/288/")
  # plotSimData(f"{PATH_SCRATCH}/Rm3000/Mach0.3/Pm5/288/")
  plotSimData(f"{PATH_SCRATCH}/Rm3000/Mach0.3/Pm1/288/")
  plotSimData(f"{PATH_SCRATCH}/Rm3000/Mach5/Pm125/288/")
  # plotSimData(f"{PATH_SCRATCH}/Rm3000/Mach5/Pm10/288/")
  # plotSimData(f"{PATH_SCRATCH}/Rm3000/Mach5/Pm5/288/")
  plotSimData(f"{PATH_SCRATCH}/Rm3000/Mach5/Pm1/288/")


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
PATH_SCRATCH = "/scratch/ek9/nk7952/"
# PATH_SCRATCH = "/scratch/jh2/nk7952/"
PATH_PLOT    = "/home/586/nk7952/MHDCodes/kriel2023/"


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF PROGRAM
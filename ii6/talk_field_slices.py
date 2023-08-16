#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
def plotSimData(filepath_sim_res, bool_mag=True):
  filepath_vis       = f"{PATH_PLOT}/field_slices/"
  WWFnF.createFolder(filepath_vis, bool_verbose=False)
  filepath_plt_files = f"{filepath_sim_res}/plt"
  filename = "Turb_hdf5_plt_cnt_0100"
  dict_sim_inputs = SimParams.readSimInputs(filepath_sim_res)
  num_cells = dict_sim_inputs["num_blocks"][0] * dict_sim_inputs["num_procs"][0]
  x = np.linspace(-1.0, 1.0, int(num_cells))
  y = np.linspace(-1.0, 1.0, int(num_cells))
  X, Y = np.meshgrid(x, -y)
  fig, ax = plt.subplots(figsize=(12, 12))
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
  cmap_vel, norm_vel = PlotFuncs.createCmap(
    cmap_name = "cmr.freeze",
    vmin      = -1.0,
    vmid      =  0.0,
    vmax      =  0.5
  )
  ax.imshow(
    np.log10(vel_magn[:,:,0]),
    extent = [-1, 1, -1, 1],
    cmap   = cmap_vel,
    norm   = norm_vel,
    aspect = "equal"
  )
  ## ------------- MAGNETIC FIELD
  if bool_mag:
    print("Loading magnetic field data...")
    mag_field = LoadData.loadFlashDataCube(
      filepath_file = f"{filepath_plt_files}/{filename}",
      num_blocks    = dict_sim_inputs["num_blocks"],
      num_procs     = dict_sim_inputs["num_procs"],
      field_name    = "mag",
      bool_norm_rms = True
    )
    ax.streamplot(
      X, Y,
      mag_field[0][:,:,0],
      mag_field[1][:,:,0],
      color      = "red",
      arrowstyle = "->",
      linewidth  = 4.0,
      density    = 1,
      arrowsize  = 1,
      zorder     = 5
    )
    # logBmagn_slice = np.log10(WWFields.fieldMagnitude(mag_field))[:,:,0]
    # logBmagn_slice[logBmagn_slice < -0.25] = np.nan
    # cmap_mag, norm_mag = PlotFuncs.createCmap(
    #   cmap_name = "Reds",
    #   vmin      = -2.0,
    #   vmid      =  0.0,
    #   vmax      =  1.5,
    #   NormType  = colors.Normalize
    # )
    # ax.scatter(
    #   X, Y,
    #   s      = 2,
    #   c      = logBmagn_slice,
    #   cmap   = cmap_mag,
    #   norm   = norm_mag,
    #   zorder = 5,
    #   alpha  = 0.8,
    # )
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
  ax.set_xticks([ ])
  ax.set_yticks([ ])
  ## save figure
  print("Saving figure...")
  ax.axis("off")
  sim_name = SimParams.getSimName(dict_sim_inputs)
  # if bool_mag: sim_name += "_combined"
  fig_name = f"{sim_name}_field_streamlines.png"
  filepath_fig = f"{filepath_vis}/{fig_name}"
  fig.savefig(filepath_fig, dpi=200)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  # for bool_mag in [True, False]:
  bool_mag = True
  plotSimData(f"{PATH_SCRATCH_EK9}/Rm3000/Mach0.3/Pm125/288/", bool_mag)
  # plotSimData(f"{PATH_SCRATCH_EK9}/Rm3000/Mach0.3/Pm5/288/", bool_mag)
  plotSimData(f"{PATH_SCRATCH_EK9}/Rm3000/Mach0.3/Pm1/288/", bool_mag)
  plotSimData(f"{PATH_SCRATCH_EK9}/Rm3000/Mach5/Pm125/576/", bool_mag)
  # plotSimData(f"{PATH_SCRATCH_EK9}/Rm3000/Mach5/Pm5/288/", bool_mag)
  plotSimData(f"{PATH_SCRATCH_EK9}/Rm3000/Mach5/Pm1/288/", bool_mag)


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
PATH_PLOT = "/home/586/nk7952/MHDCodes/ii6/"
PATH_SCRATCH_EK9 = "/scratch/ek9/nk7952/"
PATH_SCRATCH_JH2 = "/scratch/jh2/nk7952/"


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()


## END OF PROGRAM
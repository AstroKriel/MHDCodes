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

def duplicateScalarField(scalar_field):
  scalar_field = np.concatenate([scalar_field, scalar_field], axis=1)
  scalar_field = np.concatenate([scalar_field, scalar_field], axis=0)
  return scalar_field


## ###############################################################
## OPERATOR CLASS
## ###############################################################
def plotSimData(filepath_sim_res):
  filepath_vis       = f"{PATH_PLOT}/field_slices/"
  WWFnF.createFolder(filepath_vis, bool_verbose=False)
  filepath_plt_files = f"{filepath_sim_res}/plt"
  filename = "Turb_hdf5_plt_cnt_0100"
  dict_sim_inputs = SimParams.readSimInputs(filepath_sim_res)
  num_cells = dict_sim_inputs["num_blocks"][0] * dict_sim_inputs["num_procs"][0]
  fig, ax = plt.subplots(figsize=(12, 12))
  ## ------------- MAGNETIC FIELD
  print("Loading magnetic field data...")
  mag_field = LoadData.loadFlashDataCube(
    filepath_file = f"{filepath_plt_files}/{filename}",
    num_blocks    = dict_sim_inputs["num_blocks"],
    num_procs     = dict_sim_inputs["num_procs"],
    field_name    = "mag",
    bool_norm_rms = True
  )
  mag_field_norm = mag_field / WWFields.fieldMagnitude(mag_field)
  ## ------------- DENSITY FIELD
  print("Loading density field data...")
  rho = LoadData.loadFlashDataCube(
    filepath_file = f"{filepath_plt_files}/{filename}",
    num_blocks    = dict_sim_inputs["num_blocks"],
    num_procs     = dict_sim_inputs["num_procs"],
    field_name    = "dens"
  )
  grad_rho = WWFields.fieldGradient(rho)
  grad_rho_norm = grad_rho / WWFields.fieldMagnitude(grad_rho)
  ## ------------- PLOT FIELD ALIGNMENT
  field_alignment = WWFields.vectorDotProduct(mag_field_norm, grad_rho_norm)
  cmap, norm = PlotFuncs.createCmap(
    cmap_name = "cmr.iceburn",
    vmin = -1.0,
    vmid =  0.0,
    vmax = +1.0,
  )
  ax.imshow(
    duplicateScalarField(field_alignment[:,:,0]),
    extent = [-1, 1, -1, 1],
    cmap   = cmap,
    norm   = norm
  )
  x = np.linspace(-1.0, 1.0, 2*num_cells)
  y = np.linspace(-1.0, 1.0, 2*num_cells)
  X, Y = np.meshgrid(x, -y)
  ax.contour(
    X, Y,
    duplicateScalarField(np.log10(rho[:,:,0])),
    linestyles = "-",
    linewidths = 2.0,
    zorder     = 10,
    levels     = 20,
    colors     = "white",
    alpha      = 0.35,
  )
  ax.set_xticks([ ])
  ax.set_yticks([ ])
  # ax.set_xticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"], fontsize=30)
  # ax.set_yticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"], fontsize=30)
  ## save figure
  print("Saving figure...")
  sim_name = SimParams.getSimName(dict_sim_inputs)
  fig_name = f"{sim_name}_field_slice.png"
  filepath_fig = f"{filepath_vis}/{fig_name}"
  fig.savefig(filepath_fig, dpi=200)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  plotSimData(f"{PATH_SCRATCH_EK9}/Rm3000/Mach0.3/Pm5/288/")
  plotSimData(f"{PATH_SCRATCH_EK9}/Rm3000/Mach5/Pm5/288/")


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
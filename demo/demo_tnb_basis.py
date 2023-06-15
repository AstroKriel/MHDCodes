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
import matplotlib.colors as colors

## load user defined modules
from TheFlashModule import FileNames, LoadData, SimParams
from TheUsefulModule import WWFnF
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTION
## ###############################################################
def vectorCrossProduct(vector1, vector2):
  vector3 = np.array([
    vector1[1] * vector2[2] - vector1[2] * vector2[1],
    vector1[2] * vector2[0] - vector1[0] * vector2[2],
    vector1[0] * vector2[1] - vector1[1] * vector2[0]
  ])
  return vector3

def vectorDotProduct(vector1, vector2):
  scalar = np.sum([
    comp1*comp2
    for comp1, comp2 in zip(vector1, vector2)
  ], axis=0)
  return scalar

def fieldMagnitude(field):
  return np.sqrt(np.sum(field**2, axis=0))

def gradient_2ocd(field, cell_width, gradient_dir):
  F = -1 # shift forwards
  B = +1 # shift backwards
  return (
    np.roll(field, F, axis=gradient_dir) - np.roll(field, B, axis=gradient_dir)
  ) / (2*cell_width)

def computeTNBBasis(field):
  ## format: (component, x, y, z)
  field = np.array(field)
  grid_size = 1 / field[0].shape[0]
  field_magn = fieldMagnitude(field)
  ## compute tangent basis
  basis_t = field / field_magn
  ## df_j/dx_i: (component-j, gradient-direction-i, x, y, z)
  gradient_tensor = np.array([
    [
      gradient_2ocd(field_component, grid_size, gradient_dir)
      for gradient_dir in [0, 1, 2]
    ] for field_component in field
  ])
  ## compute normal basis
  ## f_i df_j/dx_i
  basis_n_term1 = np.einsum("ixyz,jixyz->jxyz", field, gradient_tensor)
  ## f_i f_j f_m df_m/dx_i
  basis_n_term2 = np.einsum("ixyz,jxyz,mxyz,mixyz->jxyz", field, field, field, gradient_tensor)
  basis_n = (basis_n_term1 / field_magn**2) - (basis_n_term2 / field_magn**4)
  kappa = fieldMagnitude(basis_n)
  basis_n /= kappa
  ## compute binormal basis: orthogonal to both t- and b-basis
  basis_b = vectorCrossProduct(basis_t, basis_n)
  return basis_t, basis_n, basis_b, kappa


## ###############################################################
## OPPERATOR HANDLING PLOT CALLS
## ###############################################################
def plotTNB():
  ## load magnetic field cube-data
  print("Loading data...")
  filepath_sim_res = "/scratch/ek9/nk7952/Rm3000/Mach5/Pm5/144/"
  dict_sim_inputs  = SimParams.readSimInputs(filepath_sim_res, True)
  mag_field = LoadData.loadFlashDataCube(
    filepath_file = f"{filepath_sim_res}/plt/Turb_hdf5_plt_cnt_0350",
    num_blocks    = dict_sim_inputs["num_blocks"],
    num_procs     = dict_sim_inputs["num_procs"],
    field_name    = "mag"
  )
  ## compute tnb-basis
  print("Computing tnb-basis...")
  basis_t, basis_n, basis_b, kappa = computeTNBBasis(mag_field)
  ## compute field projections onto tnb-basis
  mag_proj_basis_t = vectorDotProduct(mag_field, basis_t)
  mag_proj_basis_n = vectorDotProduct(mag_field, basis_n)
  mag_proj_basis_b = vectorDotProduct(mag_field, basis_b)
  ## diagnostic plots
  print("Plotting data...")
  ncols = 4
  nrows = 2
  fig, axs = plt.subplots(
    ncols   = ncols,
    nrows   = nrows,
    figsize = (ncols*5, nrows*5),
    sharex  = True,
    sharey  = True
  )
  ## field slice
  PlotFuncs.plotVectorField(
    field_slice_x1        = mag_field[0][:,:,0],
    field_slice_x2        = mag_field[1][:,:,0],
    fig                   = fig,
    ax                    = axs[0,0],
    # bool_plot_streamlines = True,
    bool_plot_magnitude   = True,
    bool_add_colorbar     = True,
    cmap_name             = "cmr.iceburn",
    cbar_title            = "Field slice",
    bool_label_axis       = True
  )
  ## kappa
  PlotFuncs.plotScalarField(
    field_slice          = np.abs(kappa[:,:,0]),
    fig                  = fig,
    ax                   = axs[1,0],
    bool_add_colorbar    = True,
    bool_log_center_cbar = False,
    cbar_orientation     = "horizontal",
    cmap_name            = "cmr.arctic",
    cbar_bounds          = None,
    cbar_title           = r"$\kappa$",
    bool_label_axis      = True
  )
  ## t-basis
  PlotFuncs.plotVectorField(
    field_slice_x1        = basis_t[0][:,:,0],
    field_slice_x2        = basis_t[1][:,:,0],
    fig                   = fig,
    ax                    = axs[0,1],
    bool_plot_streamlines = True,
    bool_plot_magnitude   = True,
    bool_add_colorbar     = True,
    cmap_name             = "Reds",
    cbar_title            = "t-basis",
    bool_label_axis       = True
  )
  PlotFuncs.plotScalarField(
    field_slice          = np.abs(mag_proj_basis_t[:,:,0]),
    fig                  = fig,
    ax                   = axs[1,1],
    bool_add_colorbar    = True,
    bool_log_center_cbar = False,
    cbar_orientation     = "horizontal",
    cmap_name            = "cmr.arctic",
    cbar_bounds          = None,
    cbar_title           = "field proj. t-basis",
    bool_label_axis      = True
  )
  ## n-basis
  PlotFuncs.plotVectorField(
    field_slice_x1        = basis_n[0][:,:,0],
    field_slice_x2        = basis_n[1][:,:,0],
    fig                   = fig,
    ax                    = axs[0,2],
    bool_plot_streamlines = True,
    bool_plot_magnitude   = True,
    bool_add_colorbar     = True,
    cmap_name             = "Reds",
    cbar_title            = "n-basis",
    bool_label_axis       = True
  )
  PlotFuncs.plotScalarField(
    field_slice          = np.abs(mag_proj_basis_n[:,:,0]),
    fig                  = fig,
    ax                   = axs[1,2],
    bool_add_colorbar    = True,
    bool_log_center_cbar = False,
    cbar_orientation     = "horizontal",
    cmap_name            = "cmr.arctic",
    cbar_bounds          = None,
    cbar_title           = "field proj. n-basis",
    NormType             = colors.Normalize,
    bool_label_axis      = True
  )
  ## b-basis
  PlotFuncs.plotVectorField(
    field_slice_x1        = basis_b[0][:,:,0],
    field_slice_x2        = basis_b[1][:,:,0],
    fig                   = fig,
    ax                    = axs[0,3],
    bool_plot_streamlines = True,
    bool_plot_magnitude   = True,
    bool_add_colorbar     = True,
    cmap_name             = "Reds",
    cbar_title            = "b-basis",
    bool_label_axis       = True
  )
  PlotFuncs.plotScalarField(
    field_slice          = np.abs(mag_proj_basis_b[:,:,0]),
    fig                  = fig,
    ax                   = axs[1,3],
    bool_add_colorbar    = True,
    bool_log_center_cbar = False,
    cbar_orientation     = "horizontal",
    cmap_name            = "cmr.arctic",
    cbar_bounds          = None,
    cbar_title           = "field proj. b-basis",
    NormType             = colors.Normalize,
    bool_label_axis      = True
  )
  ## save figure
  print("Saving figure...")
  PlotFuncs.saveFigure(fig, f"{FILEPATH_PLOT}/tnb_basis.png", bool_draft=False)


FILEPATH_PLOT = "/home/586/nk7952/MHDCodes/demo/"
## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  plotTNB()
  sys.exit()


## END OF PROGRAM
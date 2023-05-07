#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np
import collections.abc

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
def fieldCurvature(fig, axs, field_group_comp):
  ## normalise vector field
  ## format: (component, x, y, z)
  field_group_comp = np.array(field_group_comp)
  # field_norm_group_comp = field_group_comp / np.sqrt(np.sum(field_group_comp**2, axis=0))
  grid_size = 1 / field_group_comp[0].shape[0]
  ## compute components of the normalised directional derivative
  ## format: (component, gradient-direction, x, y, z)
  field_gradient_group_dir_comp = np.array([
    [
      np.gradient(field_group_comp[0], grid_size, axis=1),
      np.gradient(field_group_comp[0], grid_size, axis=0),
      np.gradient(field_group_comp[0], grid_size, axis=2),
    ],
    [
      np.gradient(field_group_comp[1][::-1], grid_size, axis=1)[::-1],
      np.gradient(field_group_comp[1][::-1], grid_size, axis=0)[::-1],
      np.gradient(field_group_comp[1][::-1], grid_size, axis=2)[::-1],
    ],
    [
      np.gradient(field_group_comp[2], grid_size, axis=1),
      np.gradient(field_group_comp[2], grid_size, axis=0),
      np.gradient(field_group_comp[2], grid_size, axis=2),
    ]
  ])
  ## compute components of characteristic curvature
  # kappa_comp = np.einsum("cijk,cdijk->cijk", field_group_comp_rearranged, field_gradient_group_dir_comp)
  ## x
  field_x_x_dx = field_group_comp[0] * field_gradient_group_dir_comp[0][0]
  field_y_x_dy = field_group_comp[1] * field_gradient_group_dir_comp[0][1]
  field_z_x_dz = field_group_comp[2] * field_gradient_group_dir_comp[0][2]
  ## y
  field_x_y_dx = field_group_comp[0] * field_gradient_group_dir_comp[1][0]
  field_y_y_dy = field_group_comp[1] * field_gradient_group_dir_comp[1][1]
  field_z_y_dz = field_group_comp[2] * field_gradient_group_dir_comp[1][2]
  ## z
  field_x_z_dx = field_group_comp[0] * field_gradient_group_dir_comp[2][0]
  field_y_z_dy = field_group_comp[1] * field_gradient_group_dir_comp[2][1]
  field_z_z_dz = field_group_comp[2] * field_gradient_group_dir_comp[2][2]
  kappa_comp = np.array([
    field_x_x_dx + field_y_x_dy + field_z_x_dz,
    field_x_y_dx + field_y_y_dy + field_z_y_dz,
    field_x_z_dx + field_y_z_dy + field_z_z_dz
  ])
  kappa = np.sqrt(
    kappa_comp[0]**2 + kappa_comp[1]**2 + kappa_comp[2]**2
  )
  ## diagnostic plot
  ## curvature
  PlotFuncs.plotScalarField(
    field_slice       = np.log(kappa[:,:,0]),
    fig               = fig,
    ax                = axs[0,1],
    bool_add_colorbar = True,
    cmap_name         = "Reds",
    NormType          = colors.Normalize,
    cbar_title        = r"$\kappa$",
    bool_label_axis   = True
  )
  ## x
  PlotFuncs.plotScalarField(
    field_slice       = field_x_x_dx[:,:,0],
    fig               = fig,
    ax                = axs[1,0],
    bool_add_colorbar = True,
    bool_log_center_cbar = True,
    cmap_name         = "cmr.waterlily",
    NormType          = colors.Normalize,
    cbar_title        = r"$\kappa$",
    bool_label_axis   = True
  )
  PlotFuncs.plotScalarField(
    field_slice       = field_y_x_dy[:,:,0],
    fig               = fig,
    ax                = axs[1,1],
    bool_add_colorbar = True,
    bool_log_center_cbar = True,
    cmap_name         = "cmr.waterlily",
    NormType          = colors.Normalize,
    cbar_title        = r"$\kappa$",
    bool_label_axis   = True
  )
  PlotFuncs.plotScalarField(
    field_slice       = field_z_x_dz[:,0,:],
    fig               = fig,
    ax                = axs[1,2],
    bool_add_colorbar = True,
    bool_log_center_cbar = True,
    cmap_name         = "cmr.waterlily",
    NormType          = colors.Normalize,
    cbar_title        = r"$\kappa$",
    bool_label_axis   = True
  )
  ## y
  PlotFuncs.plotScalarField(
    field_slice       = field_x_y_dx[:,:,0],
    fig               = fig,
    ax                = axs[2,0],
    bool_add_colorbar = True,
    bool_log_center_cbar = True,
    cmap_name         = "cmr.waterlily",
    NormType          = colors.Normalize,
    cbar_title        = r"$\kappa$",
    bool_label_axis   = True
  )
  PlotFuncs.plotScalarField(
    field_slice       = field_y_y_dy[:,:,0],
    fig               = fig,
    ax                = axs[2,1],
    bool_add_colorbar = True,
    bool_log_center_cbar = True,
    cmap_name         = "cmr.waterlily",
    NormType          = colors.Normalize,
    cbar_title        = r"$\kappa$",
    bool_label_axis   = True
  )
  PlotFuncs.plotScalarField(
    field_slice       = field_z_y_dz[:,0,:],
    fig               = fig,
    ax                = axs[2,2],
    bool_add_colorbar = True,
    bool_log_center_cbar = True,
    cmap_name         = "cmr.waterlily",
    NormType          = colors.Normalize,
    cbar_title        = r"$\kappa$",
    bool_label_axis   = True
  )
  ## z
  PlotFuncs.plotScalarField(
    field_slice       = field_x_z_dx[:,:,0],
    fig               = fig,
    ax                = axs[3,0],
    bool_add_colorbar = True,
    bool_log_center_cbar = True,
    cmap_name         = "cmr.waterlily",
    NormType          = colors.Normalize,
    cbar_title        = r"$\kappa$",
    bool_label_axis   = True
  )
  PlotFuncs.plotScalarField(
    field_slice       = field_y_z_dy[:,:,0],
    fig               = fig,
    ax                = axs[3,1],
    bool_add_colorbar = True,
    bool_log_center_cbar = True,
    cmap_name         = "cmr.waterlily",
    NormType          = colors.Normalize,
    cbar_title        = r"$\kappa$",
    bool_label_axis   = True
  )
  PlotFuncs.plotScalarField(
    field_slice       = field_z_z_dz[:,0,:],
    fig               = fig,
    ax                = axs[3,2],
    bool_add_colorbar = True,
    bool_log_center_cbar = True,
    cmap_name         = "cmr.waterlily",
    NormType          = colors.Normalize,
    cbar_title        = r"$\kappa$",
    bool_label_axis   = True
  )


## ###############################################################
## OPPERATOR HANDLING PLOT CALLS
## ###############################################################
def plotSimData(filepath_sim_res, bool_verbose=True):
  ncols = 6
  nrows = 4
  fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*10, nrows*10))
  ## load magnetic field cube-data
  print("Loading data...")
  domain = np.linspace(-5, 5, 50)
  X, Y, Z = np.meshgrid(domain, -domain, domain, indexing="xy")
  field_x = X**2
  field_y = Y**2
  field_z = 0 * Z**2
  field_magn = np.sqrt(field_x**2 + field_y**2 + field_z**2)
  # field_magn = 1.0
  field_x /= field_magn
  field_y /= field_magn
  field_z /= field_magn
  # print("normalised field")
  # ## x
  # field_x_dx = (X**3 + 2*X*(Y**2 + Z**2)) / (X**2 + Y**2 + Z**2)**(3/2)
  # field_x_dy = (Y*X**2) / (X**2 + Y**2 + Z**2)**(3/2)
  # field_x_dz = (Z*X**2) / (X**2 + Y**2 + Z**2)**(3/2)
  # ## y
  # field_y_dx = (X*Y**2) / (X**2 + Y**2 + Z**2)**(3/2)
  # field_y_dy = (Y**3 + 2*Y*(X**2 + Z**2)) / (X**2 + Y**2 + Z**2)**(3/2)
  # field_y_dz = (Z*Y**2) / (X**2 + Y**2 + Z**2)**(3/2)
  # ## z
  # field_z_dx = 0 * (X*Z**2) / (X**2 + Y**2 + Z**2)**(3/2)
  # field_z_dy = 0 * (Y*Z**2) / (X**2 + Y**2 + Z**2)**(3/2)
  # field_z_dz = 0 * (Z**3 + 2*Z*(Y**2 + X**2)) / (X**2 + Y**2 + Z**2)**(3/2)
  print("un-normalised field")
  ## x
  field_x_dx = 2*X
  field_x_dy = 0*X
  field_x_dz = 0*X
  ## y
  field_y_dx = 0*Y
  field_y_dy = 2*Y
  field_y_dz = 0*Y
  ## z
  field_z_dx = 0*Z
  field_z_dy = 0*Z
  field_z_dz = 0*Z
  ## i d(fx)/di
  field_x_x_dx = field_x * field_x_dx
  field_y_x_dy = field_y * field_x_dy
  field_z_x_dz = field_z * field_x_dz
  ## i d(fy)/di
  field_x_y_dx = field_x * field_y_dx
  field_y_y_dy = field_y * field_y_dy
  field_z_y_dz = field_z * field_y_dz
  ## i d(fz)/di
  field_x_z_dx = field_x * field_z_dx
  field_y_z_dy = field_y * field_z_dy
  field_z_z_dz = field_z * field_z_dz
  ## expected curvature
  kappa_comp = np.array([
    field_x_x_dx + field_y_x_dy + field_z_x_dz,
    field_x_y_dx + field_y_y_dy + field_z_y_dz,
    field_x_z_dx + field_y_z_dy + field_z_z_dz
  ])
  kappa = np.sqrt(
    kappa_comp[0]**2 + kappa_comp[1]**2 + kappa_comp[2]**2
  )
  ## plot magnetic field components
  print("Plotting data...")
  PlotFuncs.plotVectorField(
    field_slice_x1        = field_x[:,:,0],
    field_slice_x2        = field_y[:,:,0],
    fig                   = fig,
    ax                    = axs[0,0],
    bool_plot_streamlines = True,
    streamline_linestyle  = "-",
    bool_plot_magnitude   = True,
    bool_add_colorbar     = True,
    cmap_name             = "cmr.iceburn",
    cbar_title            = r"$\ln\big(b^2 / b_{\rm rms}\big)$",
    bool_label_axis       = True
  )
  ## diagnostic plots
  ## expected curvature
  PlotFuncs.plotScalarField(
    field_slice       = np.log(kappa[:,:,0]),
    fig               = fig,
    ax                = axs[0,2],
    bool_add_colorbar = True,
    cmap_name         = "Reds",
    NormType          = colors.Normalize,
    cbar_title        = r"$\kappa$",
    bool_label_axis   = True
  )
  ## x
  PlotFuncs.plotScalarField(
    field_slice          = field_x_x_dx[:,:,0],
    fig                  = fig,
    ax                   = axs[1,3],
    bool_add_colorbar    = True,
    bool_log_center_cbar = True,
    cmap_name            = "cmr.waterlily",
    NormType             = colors.Normalize,
    cbar_title           = r"$\kappa$",
    bool_label_axis      = True
  )
  PlotFuncs.plotScalarField(
    field_slice          = field_y_x_dy[:,:,0],
    fig                  = fig,
    ax                   = axs[1,4],
    bool_add_colorbar    = True,
    bool_log_center_cbar = True,
    cmap_name            = "cmr.waterlily",
    NormType             = colors.Normalize,
    cbar_title           = r"$\kappa$",
    bool_label_axis      = True
  )
  PlotFuncs.plotScalarField(
    field_slice          = field_z_x_dz[:,:,0],
    fig                  = fig,
    ax                   = axs[1,5],
    bool_add_colorbar    = True,
    bool_log_center_cbar = True,
    cmap_name            = "cmr.waterlily",
    NormType             = colors.Normalize,
    cbar_title           = r"$\kappa$",
    bool_label_axis      = True
  )
  ## y
  PlotFuncs.plotScalarField(
    field_slice          = field_x_y_dx[:,:,0],
    fig                  = fig,
    ax                   = axs[2,3],
    bool_add_colorbar    = True,
    bool_log_center_cbar = True,
    cmap_name            = "cmr.waterlily",
    NormType             = colors.Normalize,
    cbar_title           = r"$\kappa$",
    bool_label_axis      = True
  )
  PlotFuncs.plotScalarField(
    field_slice          = field_y_y_dy[:,:,0],
    fig                  = fig,
    ax                   = axs[2,4],
    bool_add_colorbar    = True,
    bool_log_center_cbar = True,
    cmap_name            = "cmr.waterlily",
    NormType             = colors.Normalize,
    cbar_title           = r"$\kappa$",
    bool_label_axis      = True
  )
  PlotFuncs.plotScalarField(
    field_slice          = field_z_y_dz[:,:,0],
    fig                  = fig,
    ax                   = axs[2,5],
    bool_add_colorbar    = True,
    bool_log_center_cbar = True,
    cmap_name            = "cmr.waterlily",
    NormType             = colors.Normalize,
    cbar_title           = r"$\kappa$",
    bool_label_axis      = True
  )
  ## z
  PlotFuncs.plotScalarField(
    field_slice          = field_x_z_dx[:,:,0],
    fig                  = fig,
    ax                   = axs[3,3],
    bool_add_colorbar    = True,
    bool_log_center_cbar = True,
    cmap_name            = "cmr.waterlily",
    NormType             = colors.Normalize,
    cbar_title           = r"$\kappa$",
    bool_label_axis      = True
  )
  PlotFuncs.plotScalarField(
    field_slice          = field_y_z_dy[:,:,0],
    fig                  = fig,
    ax                   = axs[3,4],
    bool_add_colorbar    = True,
    bool_log_center_cbar = True,
    cmap_name            = "cmr.waterlily",
    NormType             = colors.Normalize,
    cbar_title           = r"$\kappa$",
    bool_label_axis      = True
  )
  PlotFuncs.plotScalarField(
    field_slice          = field_z_z_dz[:,0,:],
    fig                  = fig,
    ax                   = axs[3,5],
    bool_add_colorbar    = True,
    bool_log_center_cbar = True,
    cmap_name            = "cmr.waterlily",
    NormType             = colors.Normalize,
    cbar_title           = r"$\kappa$",
    bool_label_axis      = True
  )
  ## measure field curvature
  print("Computing field curvature...")
  fieldCurvature(fig, axs, [ field_x, field_y, field_z ])
  ## save figure
  print("Saving figure...")
  PlotFuncs.saveFigure(fig, f"field_curvature.png", bool_draft=True)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  plotSimData("/scratch/ek9/nk7952/Rm3000/Mach5/Pm1/144")


## ###############################################################
## PROGRAM PARAMTERS
## ###############################################################
BASEPATH = "/scratch/ek9/nk7952/"


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
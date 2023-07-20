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
def gradient_1ofd(field, cell_width, axis):
  F = -1 # shift forwards
  return (
    np.roll(field, F, axis=axis) - field
  ) / cell_width

def gradient_2ocd(field, cell_width, axis):
  F = -1 # shift forwards
  B = +1 # shift backwards
  return (
    np.roll(field, F, axis=axis) - np.roll(field, B, axis=axis)
  ) / (2*cell_width)

def gradient_4ocd(field, cell_width, axis):
  F = -1 # shift forwards
  B = +1 # shift backwards
  return (
    -   np.roll(field, 2*F, axis=axis)
    + 8*np.roll(field, F,   axis=axis)
    - 8*np.roll(field, B,   axis=axis)
    +   np.roll(field, 2*B, axis=axis)
  ) / (12*cell_width)

def fieldCurvature(field, funcGradient, box_width=1.0):
  ## format: (component, x, y, z)
  field = np.array(field)
  num_cells = field[0].shape[0]
  cell_width = box_width / num_cells
  ## df_j/dx_i: (component-j, gradient-direction-i, x, y, z)
  gradient_tensor = np.array([
    [
      funcGradient(field_component, cell_width, axis=gradient_dir)
      for gradient_dir in [0, 1, 2]
    ] for field_component in field
  ])
  ## compute field curvature: sum_i (f_i * df_j/dx_i)
  field_curvature = np.einsum("ixyz,jixyz->jxyz", field, gradient_tensor)
  return gradient_tensor, field_curvature

def plotField(fig, axs, field, label, index_row, index_col, dx=1):
  PlotFuncs.plotVectorField(
    field_slice_x1        = field[0][dx:-(dx+1), dx:-(dx+1), 0], # x
    field_slice_x2        = field[1][dx:-(dx+1), dx:-(dx+1), 0], # y
    fig                   = fig,
    ax                    = axs[index_row, index_col],
    # bool_plot_streamlines = True,
    bool_plot_magnitude   = True,
    bool_add_colorbar     = True,
    cmap_name             = "cmr.iceburn",
    NormType              = colors.Normalize,
    cbar_title            = r"$|" + label + "|$",
  )
  axs[index_row, index_col].set_xlabel(r"$" + label + "_x$")
  axs[index_row, index_col].set_ylabel(r"$" + label + "_y$")
  PlotFuncs.plotVectorField(
    field_slice_x1        = field[0][dx:-(dx+1), 0, dx:-(dx+1)], # x
    field_slice_x2        = field[2][dx:-(dx+1), 0, dx:-(dx+1)], # z
    fig                   = fig,
    ax                    = axs[index_row, index_col+1],
    # bool_plot_streamlines = True,
    bool_plot_magnitude   = True,
    bool_add_colorbar     = True,
    cmap_name             = "cmr.iceburn",
    NormType              = colors.Normalize,
    cbar_title            = r"$|" + label + "|$",
  )
  axs[index_row, index_col+1].set_xlabel(r"$" + label + "_x$")
  axs[index_row, index_col+1].set_ylabel(r"$" + label + "_z$")
  PlotFuncs.plotVectorField(
    field_slice_x1        = field[1][0, dx:-(dx+1), dx:-(dx+1)], # y
    field_slice_x2        = field[2][0, dx:-(dx+1), dx:-(dx+1)], # z
    fig                   = fig,
    ax                    = axs[index_row, index_col+2],
    # bool_plot_streamlines = True,
    bool_plot_magnitude   = True,
    bool_add_colorbar     = True,
    cmap_name             = "cmr.iceburn",
    NormType              = colors.Normalize,
    cbar_title            = r"$|" + label + "|$",
  )
  axs[index_row, index_col+2].set_xlabel(r"$" + label + "_y$")
  axs[index_row, index_col+2].set_ylabel(r"$" + label + "_z$")

def defineField(num_cells):
  domain = np.linspace(1, 2, num_cells, dtype=np.float64)
  Y, X, Z = np.meshgrid(domain, domain, domain, indexing="xy")
  ## field
  field_fx = X**2 + Y**2 + Z**2
  field_fy = X**2 + Y**2 + Z**2
  field_fz = X**2 + Y**2 + Z**2
  field = np.array([
    field_fx,
    field_fy,
    field_fz
  ])
  ## df_x/dx_i
  field_txdx = 2.0*X
  field_txdy = 2.0*Y
  field_txdz = 2.0*Z
  ## df_y/dx_i
  field_tydx = 2.0*X
  field_tydy = 2.0*Y
  field_tydz = 2.0*Z
  ## df_z/dx_i
  field_tzdx = 2.0*X
  field_tzdy = 2.0*Y
  field_tzdz = 2.0*Z
  ## gradient tensor
  gradient_tensor = np.array([
    [
      field_txdx,
      field_txdy,
      field_txdz
    ],
    [
      field_tydx,
      field_tydy,
      field_tydz
    ],
    [
      field_tzdx,
      field_tzdy,
      field_tzdz
    ],
  ])
  ## f_i d(f_x)/dx_i
  field_tx_txdx = field_fx * field_txdx
  field_ty_txdy = field_fy * field_txdy
  field_tz_txdz = field_fz * field_txdz
  ## f_i d(f_y)/dx_i
  field_tx_tydx = field_fx * field_tydx
  field_ty_tydy = field_fy * field_tydy
  field_tz_tydz = field_fz * field_tydz
  ## f_i d(f_z)/dx_i
  field_tx_tzdx = field_fx * field_tzdx
  field_ty_tzdy = field_fy * field_tzdy
  field_tz_tzdz = field_fz * field_tzdz
  ## field curvature: sum_i (f_i * df_j/dx_i)
  field_curvature = np.array([
    field_tx_txdx + field_ty_txdy + field_tz_txdz,
    field_tx_tydx + field_ty_tydy + field_tz_tydz,
    field_tx_tzdx + field_ty_tzdy + field_tz_tzdz
  ])
  return field, gradient_tensor, field_curvature


## ###############################################################
## OPPERATOR HANDLING PLOT CALLS
## ###############################################################
def plotTNB():
  print("Defining field...")
  list_num_cells = [ 10, 50, 100, 200 ]
  ncols = 3
  nrows = 4
  fig, axs = plt.subplots(
    ncols   = ncols,
    nrows   = nrows,
    figsize = (ncols*5, nrows*5)
  )
  list_colors = [
    "orange",
    "red",
    "cornflowerblue",
    "forestgreen"
  ]
  list_dx = [
    1,
    1,
    1,
    2
  ]
  list_funcGradient = [
    gradient_1ofd,
    gradient_2ocd,
    np.gradient,
    gradient_4ocd,
  ]
  for method_index in range(len(list_funcGradient)):
    for num_cells in list_num_cells:
      field, gradient_tensor_analytic, field_curvature_analytic = defineField(num_cells)
      print("Computing field curvature...")
      gradient_tensor_numeric, field_curvature_numeric = fieldCurvature(field, list_funcGradient[method_index])
      ## compute relative error
      dx = list_dx[method_index]
      err = [
        [
          np.abs(
            component_numeric[gradient_dir][dx:-(dx+1), dx:-(dx+1), dx:-(dx+1)]
            - component_analytic[gradient_dir][dx:-(dx+1), dx:-(dx+1), dx:-(dx+1)]
          )
          for gradient_dir in [0, 1, 2]
        ]
        for component_numeric, component_analytic in zip(
          gradient_tensor_numeric,
          gradient_tensor_analytic
        )
      ]
      ## diagnostic plots
      print("Plotting data...")
      ## field slices
      num_cells = field[0].shape[0]
      for component_index in range(3):
        for gradient_index in range(3):
          err_mean = np.mean(err[component_index][gradient_index])
          err_std  = np.std(err[component_index][gradient_index])
          axs[component_index+1,gradient_index].errorbar(
            num_cells, err_mean,
            yerr   = err_std,
            fmt    = "o",
            mfc    = list_colors[method_index],
            ecolor = list_colors[method_index],
            mec="black", elinewidth=1.5, markersize=9, capsize=7.5,
            linestyle="None", zorder=10
          )
  plotField(fig, axs, field, "f", index_row=0, index_col=0)
  for component_index in range(3):
    for gradient_index in range(3):
      axs[component_index+1,gradient_index].set_xscale("log")
      axs[component_index+1,gradient_index].set_yscale("log")
  ## save figure
  print("Saving figure...")
  PlotFuncs.saveFigure(fig, f"field_curvature.png", bool_draft=False)


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  plotTNB()
  sys.exit()


## END OF PROGRAM
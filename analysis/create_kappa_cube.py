#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import os, sys, h5py
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
  mach_regime = "Mach10"
  filepath_sim_res = f"/scratch/ek9/nk7952/Rm3000/{mach_regime}/Pm5/288/"
  dict_sim_inputs  = SimParams.readSimInputs(filepath_sim_res, True)
  mag_field = LoadData.loadFlashDataCube(
    filepath_file = f"{filepath_sim_res}/plt/Turb_hdf5_plt_cnt_0250",
    num_blocks    = dict_sim_inputs["num_blocks"],
    num_procs     = dict_sim_inputs["num_procs"],
    field_name    = "mag"
  )
  vel_field = LoadData.loadFlashDataCube(
    filepath_file = f"{filepath_sim_res}/plt/Turb_hdf5_plt_cnt_0250",
    num_blocks    = dict_sim_inputs["num_blocks"],
    num_procs     = dict_sim_inputs["num_procs"],
    field_name    = "vel"
  )
  ## compute tnb-basis
  print("Computing tnb-basis...")
  basis_t, basis_n, basis_b, kappa = computeTNBBasis(mag_field)
  ## create dataset
  h5 = h5py.File(f"cube_{mach_regime}.h5", "w")
  h5.create_dataset("magx", data=mag_field[0])
  h5.create_dataset("magy", data=mag_field[1])
  h5.create_dataset("magz", data=mag_field[2])
  h5.create_dataset("kappa", data=kappa)
  h5.create_dataset("velx", data=vel_field[0])
  h5.create_dataset("vely", data=vel_field[1])
  h5.create_dataset("velz", data=vel_field[2])
  h5.close()


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  plotTNB()
  sys.exit()


## END OF PROGRAM
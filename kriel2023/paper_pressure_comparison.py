#!/bin/env python3


## ###############################################################
## MODULES
## ###############################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
## HELPER FUNCTIONS
## ###############################################################
def getSimLabel(dict_sim_inputs):
  Mach = dict_sim_inputs["desired_Mach"]
  if Mach < 1: Mach_str = f"{Mach:.1f}"
  else: Mach_str = f"{Mach:.0f}"
  Re = dict_sim_inputs["Re"]
  Pm = dict_sim_inputs["Pm"]
  return "$\mathcal{M}$" + Mach_str + f"Re{Re:.0f}Pm{Pm:.0f}"


## ###############################################################
## OPERATOR CLASS
## ###############################################################
def plotSimData(filepath_sim_res):
  filepath_vis       = f"{PATH_PLOT}/preassure/"
  WWFnF.createFolder(filepath_vis, bool_verbose=False)
  filepath_plt_files = f"{filepath_sim_res}/plt"
  filename = "Turb_hdf5_plt_cnt_0100"
  dict_sim_inputs = SimParams.readSimInputs(filepath_sim_res)
  fig, ax = plt.subplots(figsize=(6, 6))
  ## ------------- VELOCITY FIELD
  print("Loading velocity field data...")
  vel_field = LoadData.loadFlashDataCube(
    filepath_file = f"{filepath_plt_files}/{filename}",
    num_blocks    = dict_sim_inputs["num_blocks"],
    num_procs     = dict_sim_inputs["num_procs"],
    field_name    = "vel"
  )
  ## ------------- MAGNETIC FIELD
  print("Loading magnetic field data...")
  mag_field = LoadData.loadFlashDataCube(
    filepath_file = f"{filepath_plt_files}/{filename}",
    num_blocks    = dict_sim_inputs["num_blocks"],
    num_procs     = dict_sim_inputs["num_procs"],
    field_name    = "mag"
  )
  ## ------------- DENSITY FIELD
  print("Loading density field data...")
  rho = LoadData.loadFlashDataCube(
    filepath_file = f"{filepath_plt_files}/{filename}",
    num_blocks    = dict_sim_inputs["num_blocks"],
    num_procs     = dict_sim_inputs["num_procs"],
    field_name    = "dens"
  )
  ## ------------- PLOT
  vel_magn = WWFields.fieldMagnitude(vel_field)
  mag_magn = WWFields.fieldMagnitude(mag_field)
  kin = np.array(rho) * np.array(vel_magn)
  print("Plotting...")
  cbar = PlotFuncs.plotScatter(
    fig, ax,
    list_x            = kin[:,:,0].flatten(),
    list_y            = mag_magn[:,:,0].flatten(),
    color             = np.log10(rho[:,:,0].flatten()),
    ms                = 1,
    cbar_title        = "density",
    cbar_orientation  = "horizontal",
    bool_add_colorbar = True
  )
  cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(PlotFuncs.labelLogFormatter))
  x = np.linspace(10**(-5), 10**(5), 10**4)
  PlotFuncs.plotData_noAutoAxisScale(ax, x, x, ls="-")
  ## label figure
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlim([ 10**(-2), 10**(2) ])
  ax.set_ylim([ 10**(-7), 1 ])
  ax.set_xlabel("kinetic energy")
  ax.set_ylabel("magnetic energy")
  ax.text(
    0.95, 0.935,
    getSimLabel(dict_sim_inputs),
    va        = "top",
    ha        = "right",
    transform = ax.transAxes,
    color     = "black",
    fontsize  = 20,
    zorder    = 10
  )
  ## save figure
  print("Saving figure...")
  sim_name     = SimParams.getSimName(dict_sim_inputs)
  fig_name     = f"{sim_name}_preassure_comparison.png"
  filepath_fig = f"{filepath_vis}/{fig_name}"
  fig.savefig(filepath_fig, dpi=100)
  plt.close(fig)
  print("Saved figure:", filepath_fig)
  print(" ")


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  # plotSimData(f"{PATH_SCRATCH_EK9}/Rm3000/Mach0.3/Pm125/288/")
  # plotSimData(f"{PATH_SCRATCH_EK9}/Rm3000/Mach0.3/Pm10/288/")
  # plotSimData(f"{PATH_SCRATCH_EK9}/Rm3000/Mach0.3/Pm5/288/")
  # plotSimData(f"{PATH_SCRATCH_EK9}/Rm3000/Mach0.3/Pm1/288/")
  # plotSimData(f"{PATH_SCRATCH_EK9}/Rm3000/Mach5/Pm125/288/")
  # plotSimData(f"{PATH_SCRATCH_EK9}/Rm3000/Mach5/Pm50/288/")
  # plotSimData(f"{PATH_SCRATCH_EK9}/Rm3000/Mach5/Pm25/288/")
  # plotSimData(f"{PATH_SCRATCH_EK9}/Rm3000/Mach5/Pm10/288/")
  plotSimData(f"{PATH_SCRATCH_EK9}/Rm3000/Mach5/Pm5/288/")
  # plotSimData(f"{PATH_SCRATCH_EK9}/Rm3000/Mach5/Pm1/288/")


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
#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import numpy as np
import matplotlib.pyplot as plt

from TheFlashModule import SimParams
from TheFlashModule import LoadFlashData
from ThePlottingModule import PlotFuncs

class Derivatives():
  def chained(state, cell_index, step_x1, step_x2, Nres):
    cell_width = 1 / Nres
    pos = [
      cell_index[0] + step_x1[0] + step_x2[0],
      cell_index[1] + step_x1[1] + step_x2[1],
      cell_index[2] + step_x1[2] + step_x2[2]
    ]
    return 1 / (4 * cell_width**2) * (
          state(pos[0], pos[1], pos[2])
        - state(pos[0], pos[1], pos[2])
        + state(pos[0], pos[1], pos[2])
        - state(pos[0], pos[1], pos[2])
    )

  def double(state, index, step, Nres):
    cell_width = 1 / Nres
    return 1 / (2 * cell_width**2) * (
              state(index[0] + step[0], index[1] + step[1], index[2] + step[2])
      - 2.0 * state(index[0]          , index[1]          , index[2]          )
            + state(index[0] - step[0], index[1] - step[1], index[2] - step[2])
    )


class MomentumDissipation():
  def __init__(self):
    self.a = 10
  
  def compute(self):
    term_x = 1
    term_y = 1
    term_z = 1

def main():
  X, Y
  field_x = 
  ## plot data
  print("Plotting data...")
  fig, ax = plt.subplots(figsize=(6,6))
  ## plot field slice
  PlotFuncs.plot2DField(
    fig                 = fig,
    ax                  = ax,
    field_slice_x1      = data_vel_x[0,:,:],
    field_slice_x2      = data_vel_y[0,:,:],
    bool_plot_magnitude = True,
    bool_plot_quiver    = True,
    bool_add_colorbar   = False,
    bool_label_axis     = False
  )
  ## save figure
  fig.savefig("flash_slice.png")
  plt.close(fig)
  print("Saved figure.")


# def main():
#   filepath_sim_res = f"/scratch/ek9/nk7952/Rm3000/Mach5/Pm10/144"
#   filepath_plt = f"{filepath_sim_res}/plt"
#   filename  = "Turb_hdf5_plt_cnt_0100"
#   ## read simulation input parameters
#   dict_sim_inputs = SimParams.readSimInputs(filepath_sim_res, True)
#   ## load velocity field
#   data_vel_x, data_vel_y, _ = LoadFlashData.loadFlashDataCube(
#     filepath_file = f"{filepath_plt}/{filename}",
#     num_blocks    = dict_sim_inputs["num_blocks"],
#     num_procs     = dict_sim_inputs["num_procs"],
#     field_name    = "vel"
#   )
#   ## plot data
#   print("Plotting data...")
#   fig, ax = plt.subplots(figsize=(6,6))
#   ## plot field slice
#   PlotFuncs.plot2DField(
#     fig                 = fig,
#     ax                  = ax,
#     field_slice_x1      = data_vel_x[0,:,:],
#     field_slice_x2      = data_vel_y[0,:,:],
#     bool_plot_magnitude = True,
#     bool_plot_quiver    = True,
#     bool_add_colorbar   = False,
#     bool_label_axis     = False
#   )
#   ## save figure
#   fig.savefig("flash_slice.png")
#   plt.close(fig)
#   print("Saved figure.")


## PROGRAM ENTRY POINT
if __name__ == "__main__":
  main()


## END OF PROGRAM
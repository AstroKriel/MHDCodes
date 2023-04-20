#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from TheFlashModule import SimParams
from TheFlashModule import LoadData
from ThePlottingModule import PlotFuncs

class FirstDerivatives():
  ''' index: [row, col] '''

  def fd1o(field, cell_index, unit_step, Nres, inv_step, cell_width):
    ''' d(field)/d(step): first order (1o), forward difference (fd)
      inv_step = 1 / cell_width where cell_width = box_width / Nres
    '''
    a = inv_step
    a = 1 / cell_width
    return a * (
        field[(cell_index[0] + unit_step[0]) % Nres,
              (cell_index[1] + unit_step[1]) % Nres]
      - field[ cell_index[0], cell_index[1]]
    )

  def bd1o(field, cell_index, unit_step, Nres, inv_step, cell_width):
    ''' d(field)/d(step): first order (1o), backward difference (bd)
      inv_step = 1 / cell_width where cell_width = box_width / Nres
    '''
    a = inv_step
    a = 1 / cell_width
    return a * (
        field[ cell_index[0], cell_index[1]]
      - field[(cell_index[0] - unit_step[0]) % Nres,
              (cell_index[1] - unit_step[1]) % Nres]
    )

  def cd2o(field, cell_index, unit_step, Nres, inv_2step, cell_width):
    ''' d(field)/d(step): second order (2o), centered difference (cd)
      inv_2step = 1 / (2 * cell_width) where cell_width = box_width / Nres
    '''
    a = inv_2step
    a = 1 / (2 * cell_width)
    return a * (
        field[(cell_index[0] + unit_step[0]) % Nres,
              (cell_index[1] + unit_step[1]) % Nres]
      - field[(cell_index[0] - unit_step[0]) % Nres,
              (cell_index[1] - unit_step[1]) % Nres]
    )

  def cd4o(field, cell_index, unit_step, Nres, inv_12step, cell_width):
    ''' d(field)/d(step): second order (2o), centered difference (cd)
      inv_2step = 1 / (2 * cell_width) where cell_width = box_width / Nres
    '''
    a = inv_12step
    a = 1 / (12 * cell_width)
    return a * (
      -   field[(cell_index[0] + 2*unit_step[0]) % Nres,
                (cell_index[1] + 2*unit_step[1]) % Nres]
      + 8*field[(cell_index[0] +   unit_step[0]) % Nres,
                (cell_index[1] +   unit_step[1]) % Nres]
      - 8*field[(cell_index[0] -   unit_step[0]) % Nres,
                (cell_index[1] -   unit_step[1]) % Nres]
      +   field[(cell_index[0] - 2*unit_step[0]) % Nres,
                (cell_index[1] - 2*unit_step[1]) % Nres]
    )


class SecondDerivatives():
  ''' index: [row, col] '''

  def cd2o(field, cell_index, unit_step, Nres, inv_step_sq, cell_width):
    ''' d^2(field)/d(step)^2: second order (2o), centered difference (cd)
      inv_2step = 1 / (2 * cell_width**2) where cell_width = box_width / Nres
    '''
    a = inv_step_sq
    a = 1 / (2 * cell_width**2)
    return a * (
          field[(cell_index[0] + unit_step[0]) % Nres,
                (cell_index[1] + unit_step[1]) % Nres]
      - 2*field[cell_index[0], cell_index[1]]
      +   field[(cell_index[0] - unit_step[0]) % Nres,
                (cell_index[1] - unit_step[1]) % Nres]
    )

  def cd4o(field, cell_index, unit_step, Nres, inv_12step_sq, cell_width):
    ''' d^2(field)/d(step)^2: fouth order (4o), centered difference (cd)
      inv_2step = 1 / (2 * cell_width**2) where cell_width = box_width / Nres
    '''
    a = inv_12step_sq
    # a = 1 / (12 * cell_width**2)
    return a * (
      -    field[ (cell_index[0] + 2*unit_step[0]) % Nres,
                  (cell_index[1] + 2*unit_step[1]) % Nres]
      + 16*field[ (cell_index[0] +   unit_step[0]) % Nres,
                  (cell_index[1] +   unit_step[1]) % Nres]
      - 30*field[cell_index[0], cell_index[1]]
      + 16*field[ (cell_index[0] -   unit_step[0]) % Nres,
                  (cell_index[1] -   unit_step[1]) % Nres]
      -    field[ (cell_index[0] - 2*unit_step[0]) % Nres,
                  (cell_index[1] - 2*unit_step[1]) % Nres]
    )


def main():
  filepath_sim_res = f"/scratch/ek9/nk7952/Rm3000/Mach5/Pm10/144"
  filepath_plt = f"{filepath_sim_res}/plt"
  filename  = "Turb_hdf5_plt_cnt_0100"
  ## read simulation input parameters
  dict_sim_inputs = SimParams.readSimInputs(filepath_sim_res, True)
  ## load velocity field
  data_vel_x, data_vel_y, _ = LoadData.loadFlashDataCube(
    filepath_file = f"{filepath_plt}/{filename}",
    num_blocks    = dict_sim_inputs["num_blocks"],
    num_procs     = dict_sim_inputs["num_procs"],
    field_name    = "vel"
  )
  ## plot data
  print("Plotting data...")
  fig, ax = plt.subplots(figsize=(6,6))
  ## plot field slice
  PlotFuncs.plotVectorField(
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


## PROGRAM ENTRY POINT
if __name__ == "__main__":
  main()


## END OF PROGRAM
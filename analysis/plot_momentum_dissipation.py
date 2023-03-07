import os
import numpy as np
import matplotlib.pyplot as plt

from numba import njit

from TheSimModule import SimParams
from TheLoadingModule import LoadFlashData
from ThePlottingModule import PlotFuncs

os.system("clear")

class Derivatives():
  def chained(state, index, dx1, dx2, Nres):
    dx = 1 / Nres
    pos = [
      index[0] + dx1[0] + dx2[0],
      index[1] + dx1[1] + dx2[1],
      index[2] + dx1[2] + dx2[2]
    ]
    return 1 / (4 * dx**2) * (
          state(pos[0], pos[1], pos[2])
        - state(pos[0], pos[1], pos[2])
        + state(pos[0], pos[1], pos[2])
        - state(pos[0], pos[1], pos[2])
    )
  
  def double(state, index, dx, Nres):
    dx = 1 / Nres
    return 1 / (2 * dx**2) * (
        1.0 * state(index[0] + dx[0], index[1] + dx[1], index[2] + dx[2])
      - 2.0 * state(index[0]        , index[1]        , index[2]        )
      + 1.0 * state(index[0] - dx[0], index[1] - dx[1], index[2] - dx[2])
    )


class MomentumDissipation():
  def __init__(self):
    self.a = 10
  
  def compute(self):
    term_x = 1
    term_y = 1
    term_z = 1


def main():
  filepath_sim = f"/scratch/ek9/nk7952/Rm3000/super_sonic/Pm1/288"
  filepath_plt = f"{filepath_sim}/plt"
  filename  = "Turb_hdf5_plt_cnt_0100"
  num_procs = [ 8, 8, 6 ]
  ## load magnitude of velocity field slice
  print("Loading velocity field data...")
  data_vel_scalar = LoadFlashData.loadPltData_slice_magnitude(
    filepath_file = f"{filepath_plt}/{filename}",
    num_blocks    = [ 36, 36, 48 ],
    num_procs     = num_procs,
    str_field     = "vel",
    bool_norm     = False
  )
  ## plot data
  print("Plotting data...")
  fig, ax = plt.subplots(figsize=(6,6))
  data_scalar = data_vel_scalar
  PlotFuncs.plot2DField(
    fig           = fig,
    ax            = ax,
    data          = data_scalar,
    bool_colorbar = False,
    bool_label    = False
  )
  step = 2*3
  data_vecs_x = data_vel_vecs[0][::step, ::step]
  data_vecs_y = data_vel_vecs[1][::step, ::step]
  x = np.linspace(-1.0, 1.0, len(data_vecs_x[0,:]))
  y = np.linspace(-1.0, 1.0, len(data_vecs_x[:,0]))
  X, Y = np.meshgrid(x, -y)
  norm = np.sqrt(data_vecs_x**2 + data_vecs_y**2)
  ax.quiver(X, Y, data_vecs_x / norm , data_vecs_y / norm, width=5e-3, color="red")
  ## save figure
  fig.savefig("lic_flash_data.png")
  plt.close(fig)
  print("Saved figure.")


## PROGRAM ENTRY POINT
if __name__ == "__main__":
  main()


## END OF PROGRAM
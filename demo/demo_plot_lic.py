import os
import numpy as np
import matplotlib.pyplot as plt

from numba import njit

from TheFlashModule import LoadFlashData
from ThePlottingModule import PlotFuncs


@njit
def get_noise(vector_field):
  vector_field = np.asarray(vector_field)
  num_rows, num_cols, _ = vector_field.shape
  return np.random.rand(num_rows, num_cols)

@njit
def lic_flow(vector_field, t=0, len_pix=5, scalar_field=None):
  vector_field = np.asarray(vector_field)
  num_rows, num_cols, num_dims = vector_field.shape
  ## check that the vector field is 2D
  if num_dims != 2:
    raise ValueError("Incorrect dimension of input vector_field.")
  ## generate a noisy background canvas if none is provided
  if scalar_field is None:
    scalar_field = get_noise(vector_field)
  ## initialise the output 2D LIC
  result = np.zeros((num_rows, num_cols))
  for i in range(num_rows):
    for j in range(num_cols):
      y = i
      x = j
      forward_sum = 0
      forward_total = 0
      ## advect forwards
      for k in range(len_pix):
        dx = vector_field[int(y), int(x), 0]
        dy = vector_field[int(y), int(x), 1]
        dt = dt_x = dt_y = 0
        if   dy > 0: dt_y = (-y + (np.floor(y) + 1)) /  dy
        elif dy < 0: dt_y = ( y - (np.ceil(y)  - 1)) / -dy
        if   dx > 0: dt_x = (-x + (np.floor(x) + 1)) /  dx
        elif dx < 0: dt_x = ( x - (np.ceil(x)  - 1)) / -dx
        if (dx != 0) or (dy != 0):
          dt = min(dt_x, dt_y)
        x = min(max(x + dx * dt, 0), num_cols - 1)
        y = min(max(y + dy * dt, 0), num_rows - 1)
        weight = pow(np.cos(t + 0.46 * k), 2)
        forward_sum += scalar_field[int(y), int(x)] * weight
        forward_total += weight
      y = i
      x = j
      backward_sum = 0
      backward_total = 0
      ## advect backwards
      for k in range(1, len_pix):
        dx = -vector_field[int(y), int(x), 0]
        dy = -vector_field[int(y), int(x), 1]
        dt_x = dt_y = 0
        if   dy > 0: dt_y = (-y + (np.floor(y) + 1)) /  dy
        elif dy < 0: dt_y = ( y - (np.ceil(y)  - 1)) / -dy
        if   dx > 0: dt_x = (-x + (np.floor(x) + 1)) /  dx
        elif dx < 0: dt_x = ( x - (np.ceil(x)  - 1)) / -dx
        if (dx != 0) or (dy != 0):
          dt = min(dt_x, dt_y)
        x = min(max(x + dx * dt, 0), num_cols - 1)
        y = min(max(y + dy * dt, 0), num_rows - 1)
        weight = pow(np.cos(t - 0.46 * k), 2)
        backward_sum += scalar_field[int(y), int(x)] * weight
        backward_total += weight
      numer = forward_sum + backward_sum
      denom = forward_total + backward_total
      result[i, j] = numer / denom
  return result

def test_genData():
  ## generate data
  x, y = np.meshgrid(
    np.linspace(-10, 10, 1000),
    np.linspace(-10, 10, 1000)
  )
  dx = -(y+1) / np.sqrt((x+1)**2 + (y+1)**2)
  dy =  (x+1) / np.sqrt((x+1)**2 + (y+1)**2)
  vector_field = np.stack((dx, dy), axis=2)
  lic_data = lic_flow(vector_field, len_pix=100)
  ## plot data
  fig, ax = plt.subplots()
  ax.imshow(lic_data)
  fig.savefig("lic_data.png")
  plt.close(fig)
  print("Saved figure.")

def test_loadData():
  nres = 288
  bool_lic = 1
  filepath_sim = f"/scratch/ek9/nk7952/Rm3000/super_sonic/Pm1/{nres}"
  filepath_plt = f"{filepath_sim}/plt"
  if nres == 576:
    filename  = "Turb_hdf5_plt_cnt_0100"
    num_procs = [ 16, 16, 12 ]
  elif nres == 288:
    filename  = "Turb_hdf5_plt_cnt_0100"
    num_procs = [ 8, 8, 6 ]
  ## load magnitude of velocity field slice
  print("Loading velocity field data...")
  data_vel_scalar = LoadFlashData.loadPltData_slice_magnitude(
    filepath_file = f"{filepath_plt}/{filename}",
    num_blocks    = [ 36, 36, 48 ],
    num_procs     = num_procs,
    str_field     = "dens",
    bool_norm     = True
  )
  ## load magnetic field slice
  print("Loading magnetic field data...")
  data_mag_vecs = LoadFlashData.loadPltData_slice_field(
    filepath_file = f"{filepath_plt}/{filename}",
    num_blocks    = [ 36, 36, 48 ],
    num_procs     = num_procs,
    str_field     = "mag"
  )
  ## calculating LIC
  if bool_lic:
    len_pix = 3
    print("Calculating LIC...")
    vector_field = np.stack((data_mag_vecs[0], data_mag_vecs[1]), axis=2)
    data_lic = lic_flow(
      vector_field = vector_field,
      scalar_field = data_vel_scalar,
      len_pix = len_pix
    )
    # for _ in range(2):
    #   data_lic = lic_flow(
    #     vector_field = vector_field,
    #     scalar_field = data_lic,
    #     len_pix = len_pix
    #   )
  ## plot data
  print("Plotting data...")
  fig, ax = plt.subplots(figsize=(6,6))
  ## plot LIC/home/586/nk7952/MHDCodes/demo/lic_flash_data.png
  if bool_lic:
    PlotFuncs.plot2DField(
      fig   = fig,
      ax    = ax,
      data  = data_lic
    )
  ## plot scalar field
  else:
    # data_scalar_tmp = np.concatenate((data_vel_scalar, data_vel_scalar, data_vel_scalar), axis=1)
    # data_scalar     = np.concatenate((data_scalar_tmp, data_scalar_tmp), axis=0)
    data_scalar = data_vel_scalar
    PlotFuncs.plot2DField(
      fig           = fig,
      ax            = ax,
      data          = data_scalar,
      bool_colorbar = False,
      bool_label    = False
    )
    ## plot quivers
    # data_vecs_x_tmp = np.concatenate((data_mag_vecs[0], data_mag_vecs[0], data_mag_vecs[0]), axis=1)
    # data_vecs_y_tmp = np.concatenate((data_mag_vecs[1], data_mag_vecs[1], data_mag_vecs[1]), axis=1)
    # data_vecs_x     = np.concatenate((data_vecs_x_tmp,  data_vecs_x_tmp),  axis=0)
    # data_vecs_y     = np.concatenate((data_vecs_y_tmp,  data_vecs_y_tmp),  axis=0)
  # step = 2*3
  # data_vecs_x = data_mag_vecs[0][::step, ::step]
  # data_vecs_y = data_mag_vecs[1][::step, ::step]
  # x = np.linspace(-1.0, 1.0, len(data_vecs_x[0,:]))
  # y = np.linspace(-1.0, 1.0, len(data_vecs_x[:,0]))
  # X, Y = np.meshgrid(x, -y)
  # norm = np.sqrt(data_vecs_x**2 + data_vecs_y**2)
  # # norm = 1.0
  # ax.quiver(X, Y, data_vecs_x / norm , data_vecs_y / norm, width=5e-3, color="red")
  # # ax.quiver(X, Y, data_vecs_x, data_vecs_y, width=2.5e-4, color="red", alpha=0.5)
  ## save figure
  fig.savefig("lic_flash_data.png")
  plt.close(fig)
  print("Saved figure.")


## PROGRAM ENTRY POINT
if __name__ == "__main__":
  # test_genData()
  test_loadData()


## END OF PROGRAM
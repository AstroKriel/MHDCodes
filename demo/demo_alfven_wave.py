import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

os.system("clear")

def getVMag(v_array):
  return np.sqrt(np.sum([
    v**2 for v in list(v_array)]
  ))

def cross_3D(v_1, v_2):
  ## cross_3D product of two vectors
  return np.array([
    v_1[1]*v_2[2] - v_1[2]*v_2[1],
    v_1[2]*v_2[0] - v_1[0]*v_2[2],
    v_1[0]*v_2[1] - v_1[1]*v_2[0]
  ])

def getState(amplitude, omega, time, v_k, v_pos):
  ## work with real component of the solution
  return amplitude * np.real(np.exp(
    1j * ( omega*time - np.dot(v_k, v_pos) )
  ))

def main():
  print("Initialising parameters...")
  ## ratio of specific heats
  gamma         = -5/3
  ## background state
  rho_0         = 1.0 # choose
  press_0       = 1.0 # choose
  v_mag_dir_hat = np.array([1, 0, 0]) # choose
  mag_strength  = 1.0 # choose
  v_vel_0       = np.array([0, 0, 0])
  v_mag_0       = mag_strength * v_mag_dir_hat
  ## perturbed state
  v_k           = np.array([0, 0, 1]) # choose: k_z =/= 0
  alpha_1       = 1 # choose
  alpha_2       = gamma * alpha_1
  rho_1         = rho_0 * (1 + alpha_1)
  press_1       = press_0 * (1 + alpha_2)
  alfven_speed  = getVMag(v_mag_0)**2 / (4*np.pi * rho_0) # v_A^2
  theta_mag_k   = np.dot(v_mag_0, v_k) / (getVMag(v_mag_0) * getVMag(v_k))
  omega         = np.sqrt( alfven_speed**2 * getVMag(v_k) * np.cos(theta_mag_k)**2 )
  v_vel_1       = np.array([0, 0, omega * alpha_1 / v_k[2]])
  v_alpha_3     = 1 / omega * cross_3D(v_k, cross_3D(v_mag_dir_hat, v_vel_1))
  v_mag_1       = mag_strength * (v_mag_dir_hat + v_alpha_3)
  ## initialise figure
  print("Initialising figure...")
  num_rows, num_cols = 4, 3
  fig = plt.figure(figsize=( num_cols*5.0, num_rows*3.0 ))
  fig_grid = GridSpec(num_rows, num_cols, figure=fig)
  ax_rho   = fig.add_subplot(fig_grid[0,0])
  ax_press = fig.add_subplot(fig_grid[1,0])
  ax_vel_x = fig.add_subplot(fig_grid[2,0])
  ax_vel_y = fig.add_subplot(fig_grid[2,1])
  ax_vel_z = fig.add_subplot(fig_grid[2,2])
  ax_mag_x = fig.add_subplot(fig_grid[3,0])
  ax_mag_y = fig.add_subplot(fig_grid[3,1])
  ax_mag_z = fig.add_subplot(fig_grid[3,2])
  ## initialise solutions
  num_x, num_y = 5, 5
  data_pos_x = np.linspace(0, 10, num_x)
  data_pos_y = np.linspace(0, 10, num_y)
  data_rho   = np.zeros((num_x, num_y))
  data_press = np.zeros((num_x, num_y))
  data_vel_x = np.zeros((num_x, num_y))
  data_vel_y = np.zeros((num_x, num_y))
  data_vel_z = np.zeros((num_x, num_y))
  data_mag_x = np.zeros((num_x, num_y))
  data_mag_y = np.zeros((num_x, num_y))
  data_mag_z = np.zeros((num_x, num_y))
  ## create solution datasets
  print("Computing solutions...")
  time = 0.0
  for index_x in range(num_x):
    for index_y in range(num_y):
      v_pos = np.array([data_pos_x[index_x], data_pos_y[index_y], 0])
      dict_params = { "omega" : omega, "time" : time, "v_k" : v_k, "v_pos" : v_pos }
      data_rho[index_x, index_y]   = getState(rho_1,      **dict_params)
      data_press[index_x, index_y] = getState(press_1,    **dict_params)
      data_vel_x[index_x, index_y] = getState(v_vel_1[0], **dict_params)
      data_vel_y[index_x, index_y] = getState(v_vel_1[1], **dict_params)
      data_vel_z[index_x, index_y] = getState(v_vel_1[2], **dict_params)
      data_mag_x[index_x, index_y] = getState(v_mag_1[0], **dict_params)
      data_mag_y[index_x, index_y] = getState(v_mag_1[1], **dict_params)
      data_mag_z[index_x, index_y] = getState(v_mag_1[2], **dict_params)
  ## plot data
  print("Plotting solutions...")
  ax_rho.imshow(data_rho)
  ax_press.imshow(data_press)
  ax_vel_x.imshow(data_vel_x)
  ax_vel_y.imshow(data_vel_y)
  ax_vel_z.imshow(data_vel_z)
  ax_mag_x.imshow(data_mag_x)
  ax_mag_y.imshow(data_mag_y)
  ax_mag_z.imshow(data_mag_z)
  ## save figure
  print("Saving figure...")
  filepath_fig = "fig_alfven_wave.png"
  fig.set_tight_layout(True)
  fig.savefig(filepath_fig)
  plt.close(fig)
  print("Saved figure:", filepath_fig)

if __name__ == "__main__":
  main()
  sys.exit()

## END OF DEMO PROGRAM
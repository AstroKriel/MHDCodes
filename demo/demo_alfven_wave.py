#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os, sys
import numpy as np
import matplotlib.pyplot as plt

## load user defined modules
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear")


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def cross_sym(v_1, v_2):
  print(f"{v_1} x {v_2}")
  result = [
    f"[ ({v_1[1]} {v_2[2]}) - ({v_1[2]} {v_2[1]}) ]",
    f"[ ({v_1[2]} {v_2[0]}) - ({v_1[0]} {v_2[2]}) ]",
    f"[ ({v_1[0]} {v_2[1]}) - ({v_1[1]} {v_2[0]}) ]"
  ]
  print("=\t", *result, sep="\n\t")
  print(" ")
  return result

def cross(v_1, v_2):
  return np.array([
    v_1[1]*v_2[2] - v_1[2]*v_2[1],
    v_1[2]*v_2[0] - v_1[0]*v_2[2],
    v_1[0]*v_2[1] - v_1[1]*v_2[0]
  ])

def getVMag(vec):
  return np.sqrt(np.sum([
    elem**2 for elem in list(vec)]
  ))

def getVNorm(vec):
  return np.array(vec) / getVMag(vec)

def getCosAngle(v_1, v_2):
  return np.dot(v_1, v_2) / (getVMag(v_1) * getVMag(v_2))

def getState(amplitude, omega, time, v_k, v_pos):
  ## real component of the solution
  return amplitude * np.cos(
    omega*time - np.dot(v_k, v_pos)
  )

def plotData(ax, data, label):
  im = ax.imshow(data)
  PlotFuncs.addColorbar_fromMappble(im, cbar_title=label)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  print("Initialising parameters...")
  ## ratio of specific heats
  gamma         = 5/3
  ## free parameters
  rho_0         = 1.0
  press_0       = 3/5
  alpha_2       = 0.0
  vel_1         = 1.0
  v_k           = 0.25 * np.array([ 3.0, 2.0, 1.0 ])
  v_mag_0       = np.array([ 0.5, 1.0, 1.5 ])
  ## compute helper variables
  k             = getVMag(v_k)
  cos_theta     = getCosAngle(v_mag_0, v_k)
  theta         = 180/np.pi * np.arccos(cos_theta)
  print(f"angle between vec(k) and u-vec(n): {round(theta, 2)}")
  ## background state
  ## inputs: rho_0, press_0, vel_1, v_mag_0
  v_vel_0       = np.zeros(3)
  v_perp        = cross(v_k, v_mag_0)
  if np.sum(abs(v_perp)) == 0: v_perp = np.array([ 0, 0, 1 ])
  v_vel_1       = vel_1 * getVNorm(v_perp)
  v_mag_0_hat   = getVNorm(v_mag_0)
  mag_0         = getVMag(v_mag_0)
  ## perturbed state
  alpha_1       = alpha_2 / gamma # from continuity eqn
  alfven_speed  = mag_0**2 / (4*np.pi * rho_0) # v_A^2
  omega         = np.sqrt(alfven_speed) * k * cos_theta # dispersion relation
  # alpha_3_z     = -k / omega * cos_theta * v_vel_1[2]
  # v_alpha_3     = np.array([ 0, 0, alpha_3_z ])
  check         = -np.dot(v_k, v_mag_0_hat) / omega * v_vel_1
  # if any(v_alpha_3 - check): raise Exception("Error: Something went wrong!")
  v_alpha_3     = check
  ## initialise figure
  print("Initialising figure...")
  fig, axs = plt.subplots(
    nrows              = 4,
    ncols              = 3,
    figsize            = (7*3, 4*4),
    sharex             = True,
    constrained_layout = True
  )
  ## initialise solutions
  num_x = 100
  num_y = 100
  data_pos_x = np.linspace(-5, 5, num_x)
  data_pos_y = np.linspace(-5, 5, num_y)
  data_rho   = np.zeros(( num_x, num_y ))
  data_press = np.zeros(( num_x, num_y ))
  data_vel_x = np.zeros(( num_x, num_y ))
  data_vel_y = np.zeros(( num_x, num_y ))
  data_vel_z = np.zeros(( num_x, num_y ))
  data_mag_x = np.zeros(( num_x, num_y ))
  data_mag_y = np.zeros(( num_x, num_y ))
  data_mag_z = np.zeros(( num_x, num_y ))
  ## compute solutions
  print("Computing solutions...")
  time = 0.3
  for index_x in range(num_x):
    for index_y in range(num_y):
      v_pos = np.array([
        data_pos_x[index_x],
        data_pos_y[index_y],
        0
      ])
      dict_params = { "omega" : omega, "time" : time, "v_k" : v_k, "v_pos" : v_pos }
      data_rho[index_x, index_y]   = rho_0   * (1.0            + getState(alpha_1,      **dict_params))
      data_press[index_x, index_y] = press_0 * (1.0            + getState(alpha_2,      **dict_params))
      data_vel_x[index_x, index_y] =           (v_vel_0[0]     + getState(v_vel_1[0],   **dict_params))
      data_vel_y[index_x, index_y] =           (v_vel_0[1]     + getState(v_vel_1[1],   **dict_params))
      data_vel_z[index_x, index_y] =           (v_vel_0[2]     + getState(v_vel_1[2],   **dict_params))
      data_mag_x[index_x, index_y] = mag_0   * (v_mag_0_hat[0] + getState(v_alpha_3[0], **dict_params))
      data_mag_y[index_x, index_y] = mag_0   * (v_mag_0_hat[1] + getState(v_alpha_3[1], **dict_params))
      data_mag_z[index_x, index_y] = mag_0   * (v_mag_0_hat[2] + getState(v_alpha_3[2], **dict_params))
  ## plot solutions
  print("Plotting solutions...")
  ## velocity field
  data_vel_z_slice = data_vel_z[:, num_x//2]
  axs[0,0].plot(data_vel_z_slice, "k-")
  plotData(axs[1,0], data_vel_x, "vel-x")
  plotData(axs[2,0], data_vel_y, "vel-y")
  plotData(axs[3,0], data_vel_z, "vel-z")
  axs[0,0].set_ylabel("vel-z (x-slice)")
  axs[1,0].set_ylabel("y")
  axs[2,0].set_ylabel("y")
  axs[3,0].set_ylabel("y")
  ## magnetic field
  data_mag_z_slice = data_mag_z[:, num_x//2]
  axs[0,1].plot(data_mag_z_slice, "k-")
  plotData(axs[1,1], data_mag_x, "mag-x")
  plotData(axs[2,1], data_mag_y, "mag-y")
  plotData(axs[3,1], data_mag_z, "mag-z")
  axs[0,1].set_ylabel("mag-z (x-slice)")
  axs[3,0].set_xlabel("x")
  axs[3,1].set_xlabel("x")
  axs[3,2].set_xlabel("x")
  ## density + pressure
  plotData(axs[2,2], data_rho,   "density")
  plotData(axs[3,2], data_press, "pressure")
  ## remove axis
  axs[0,2].axis("off")
  axs[1,2].axis("off")
  ## save figure
  PlotFuncs.saveFigure(fig, "fig_alfven_wave.png")


## ###############################################################
## RUN DEMO PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF DEMO PROGRAM
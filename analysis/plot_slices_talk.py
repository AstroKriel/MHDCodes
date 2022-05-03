#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import numpy as np
import cmasher as cmr # https://cmasher.readthedocs.io/user/introduction.html#colormap-overview
import matplotlib as mpl
import matplotlib.cm as cm

## load old user defined modules
from the_matplotlib_styler import *
from OldModules.the_useful_library import *
from OldModules.the_loading_library import *
from OldModules.the_fitting_library import *
from OldModules.the_plotting_library import *


## ###############################################################
## PREPARE WORKSPACE
##################################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend
plt.style.use("dark_background")


## ###############################################################
## FUNCTIONS
## ###############################################################
def funcAddColorbar(fig, axs, im, title, bool_right=True):
  ## get multi-axis bounds
  ax0_info = axs[0].get_position()
  ax1_info = axs[-1].get_position()
  ## get colorbar position
  ax_x0     = ax0_info.x0
  ax_x1     = ax1_info.x0
  ax_y0     = ax1_info.y0
  ax_width  = ax1_info.width
  ax_height = ax0_info.y0 - ax1_info.y0 + ax1_info.height
  ## add colorbar axis
  cbar_width = 0.035
  ax_cbar = fig.add_axes([
    (ax_x0 + ax_width + 0.01) if bool_right else (ax_x0 - cbar_width - 0.01),
    ax_y0,
    cbar_width,
    ax_height
  ])
  ## create colormap
  cbar = fig.colorbar(im, cax=ax_cbar, orientation="vertical")
  if not bool_right:
    ax_cbar.yaxis.set_ticks_position("left")

def funcPlotSlices(filepath_data, filepath_plot):
  filepath_0 = createFilepath([ filepath_data, "Re500",  "288", "sub_sonic", "Pm2", "plt" ]) # 
  filepath_1 = createFilepath([ filepath_data, "Rm3000", "288", "sub_sonic", "Pm2", "plt" ]) # 
  num_blocks = [36, 36, 48]
  num_procs  = list(np.array([4, 4, 3]) * 2)
  file_end_index = 150
  ## ################
  ## LOAD RE 500 DATA
  ## ################
  print("Loading velocity slices...")
  list_kin_field_0, _ = loadListFLASHFieldSlice(
    filepath_0,
    str_field = "vel",
    num_blocks = num_blocks,
    num_procs  = num_procs,
    file_end_index = file_end_index
  )
  print("Loading magnetic slices...")
  list_mag_field_0, _ = loadListFLASHFieldSlice(
    filepath_0,
    str_field = "mag",
    num_blocks = num_blocks,
    num_procs  = num_procs,
    file_end_index = file_end_index
  )
  print(" ")
  ## #################
  ## LOAD RM 3000 DATA
  ## #################
  print("Loading velocity slices...")
  list_kin_field_1, _ = loadListFLASHFieldSlice(
    filepath_1,
    str_field = "vel",
    num_blocks = num_blocks,
    num_procs  = num_procs,
    file_end_index = file_end_index
  )
  print("Loading magnetic slices...")
  list_mag_field_1, _ = loadListFLASHFieldSlice(
    filepath_1,
    str_field = "mag",
    num_blocks = num_blocks,
    num_procs  = num_procs,
    file_end_index = file_end_index
  )
  print(" ")
  ## ###################
  ## GET COLORBAR LIMITS
  ## ###################
  ## get colorbar limits
  kin_lims = [ 10**(-2), 2 ]
  mag_lims = [ 10**(-5), 10 ]
  ## #################
  ## PLOT FIELD SLICES
  ## #################
  ## initialise figure
  print("Saving field slices...")
  fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
  fig.subplots_adjust(hspace=0.025, wspace=0.035)
  ## loop over each slice
  for _, time_index in loopListWithUpdates(list_kin_field_0):
    ## Re 500 velocity field
    im_obj_0 = axs[0,0].imshow(
      list_kin_field_0[time_index],
      extent = [-1,1,-1,1],
      cmap = plt.get_cmap("cmr.freeze"),
      norm = mpl.colors.LogNorm(vmin=kin_lims[0], vmax=kin_lims[1])
    )
    ## Rm 3000 velocity field
    im_obj_0 = axs[1,0].imshow(
      list_kin_field_1[time_index],
      extent = [-1,1,-1,1],
      cmap = plt.get_cmap("cmr.freeze"),
      norm = mpl.colors.LogNorm(vmin=kin_lims[0], vmax=kin_lims[1])
    )
    ## Re 500 magnetic field
    im_obj_1 = axs[0,1].imshow(
      list_mag_field_0[time_index],
      extent = [-1,1,-1,1],
      cmap = plt.get_cmap("cmr.fall"),
      norm = mpl.colors.LogNorm(vmin=mag_lims[0], vmax=mag_lims[1])
    )
    ## Rm 3000 magnetic field
    im_obj_1 = axs[1,1].imshow(
      list_mag_field_1[time_index],
      extent = [-1,1,-1,1],
      cmap = plt.get_cmap("cmr.fall"),
      norm = mpl.colors.LogNorm(vmin=mag_lims[0], vmax=mag_lims[1])
    )
    ## add colorbar
    if time_index == 0:
      funcAddColorbar(fig, list(axs[:,0]), im_obj_0, r"$\bm{u}^2$", bool_right=False)
      funcAddColorbar(fig, list(axs[:,1]), im_obj_1, r"$\bm{b}^2$", bool_right=True)
    ## remove axis labels
    axs[1,0].axes.xaxis.set_visible(False)
    axs[1,0].axes.yaxis.set_visible(False)
    axs[1,1].axes.xaxis.set_visible(False)
    axs[1,1].axes.yaxis.set_visible(False)
    axs[0,0].axes.xaxis.set_visible(False)
    axs[0,0].axes.yaxis.set_visible(False)
    axs[0,1].axes.xaxis.set_visible(False)
    axs[0,1].axes.yaxis.set_visible(False)
    ## save plot
    fig_name = "energy_slice={:04d}.png".format(int(time_index))
    fig_filepath = createFilepath([filepath_plot, fig_name])
    plt.savefig(fig_filepath)
    ## clear figure and axis
    fig.artists.clear()
    axs[0,0].clear()
    axs[0,1].clear()
    axs[1,0].clear()
    axs[1,1].clear()


## ###############################################################
## DEFINE MAIN PROGRAM
## ###############################################################
def main():
  ## ####################
  ## PLOT SIMULATION DATA
  ## ####################
  filepath_data = "/scratch/ek9/nk7952" # "/Users/dukekriel/Documents/Projects/TurbulentDynamo/data" # 
  filepath_plot = "vis_folder/slices" # "/Users/dukekriel/Documents/Projects/TurbulentDynamo/figures/slices" # 
  filepath_slices = createFilepath([filepath_plot, "frames"])
  createFolder(filepath_slices)
  ## plot field slices
  funcPlotSlices(filepath_data, filepath_slices)
  ## animate field slices
  aniEvolution(
    filepath_slices,
    filepath_plot,
    input_name  = "energy_slice=%*.png",
    output_name = "energy_slice.mp4"
  )


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
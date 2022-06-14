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


## ###############################################################
## PREPARE WORKSPACE
##################################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## FUNCTIONS
## ###############################################################
def funcPlotSlices(filepath_data_base, filepath_plot_base):
  str_suite = "Re120Pm25"
  str_sim   = "k4.5"
  str_label_1 = r"$t/t_\mathrm{turb} = 10$"
  str_label_2 = r"$t/t_\mathrm{turb} = 40$"
  filepath_data_0 = createFilepath([filepath_data_base, "test_k_driv",  str_suite, str_sim, "Turb_hdf5_plt_cnt_0100"])
  filepath_data_1 = createFilepath([filepath_data_base, "test_k_driv",  str_suite, str_sim, "Turb_hdf5_plt_cnt_0400"])
  filepath_plot   = createFilepath([filepath_plot_base, "test_k_driv",  str_suite])
  num_blocks = [36, 36, 48] # [18, 18, 24] or [36, 36, 48]
  num_procs  = [4, 4, 3] # [4, 4, 3] or [8, 8, 6] or [16, 16, 12]
  print("Loading data...")
  ## ################
  ## LOAD RE 500 DATA
  ## ################
  kin_field_0 = loadFLASHFieldSlice(
    filepath_data_0,
    str_field = "vel",
    num_blocks = num_blocks,
    num_procs  = num_procs,
    bool_rms_norm = True
  )
  mag_field_0 = loadFLASHFieldSlice(
    filepath_data_0,
    str_field = "mag",
    num_blocks = num_blocks,
    num_procs  = num_procs,
    bool_rms_norm = True
  )
  ## #################
  ## LOAD RM 3000 DATA
  ## #################
  kin_field_1 = loadFLASHFieldSlice(
    filepath_data_1,
    str_field = "vel",
    num_blocks = num_blocks,
    num_procs  = num_procs,
    bool_rms_norm = True
  )
  mag_field_1 = loadFLASHFieldSlice(
    filepath_data_1,
    str_field = "mag",
    num_blocks = num_blocks,
    num_procs  = num_procs,
    bool_rms_norm = True
  )
  ## ###################
  ## GET COLORBAR LIMITS
  ## ###################
  cbar_lims_vel = [ 0.75*10**(-2), 1.15*10 ]
  cbar_lims_mag = [ 0.75*10**(-2), 50 ]
  ## #################
  ## PLOT FIELD SLICES
  ## #################
  print("Plotting data...")
  ## initialise figure
  fig, axs = plt.subplots(2, 2, figsize=(8*0.9, 8*0.9), sharex=True, sharey=True)
  fig.subplots_adjust(hspace=0.025/2, wspace=0.025)
  ## add simulation labels
  axs[0,0].text(
    0.05, 0.95, str_label_1,
    ha="left", va="top", transform=axs[0,0].transAxes, fontsize=16,
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", alpha=0.7)
  )
  axs[1,0].text(
    0.05, 0.95, str_label_1,
    ha="left", va="top", transform=axs[1,0].transAxes, fontsize=16,
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", alpha=0.7)
  )
  axs[0,1].text(
    0.05, 0.95, str_label_2,
    ha="left", va="top", transform=axs[0,1].transAxes, fontsize=16,
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", alpha=0.7)
  )
  axs[1,1].text(
    0.05, 0.95, str_label_2,
    ha="left", va="top", transform=axs[1,1].transAxes, fontsize=16,
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", alpha=0.7)
  )
  ## first dataset velocity field
  im_obj_0 = axs[0,0].imshow(
    kin_field_0,
    extent = [-1,1,-1,1],
    cmap = plt.get_cmap("cmr.freeze"),
    norm = mpl.colors.LogNorm(vmin=cbar_lims_vel[0], vmax=cbar_lims_vel[1])
  )
  ## second dataset velocity field
  im_obj_0 = axs[0,1].imshow(
    kin_field_1,
    extent = [-1,1,-1,1],
    cmap = plt.get_cmap("cmr.freeze"),
    norm = mpl.colors.LogNorm(vmin=cbar_lims_vel[0], vmax=cbar_lims_vel[1])
  )
  ## first dataset magnetic field
  im_obj_1 = axs[1,0].imshow(
    mag_field_0,
    extent = [-1,1,-1,1],
    cmap = plt.get_cmap("cmr.fall"),
    norm = mpl.colors.LogNorm(vmin=cbar_lims_mag[0], vmax=cbar_lims_mag[1])
  )
  ## second dataset magnetic field
  im_obj_1 = axs[1,1].imshow(
    mag_field_1,
    extent = [-1,1,-1,1],
    cmap = plt.get_cmap("cmr.fall"),
    norm = mpl.colors.LogNorm(vmin=cbar_lims_mag[0], vmax=cbar_lims_mag[1])
  )
  # ## add reference circles
  # axs[1,0].text(
  #     0.8-1, 0.075-1,
  #     r"$\ell/L = 1/3$",
  #     ha="left", va="bottom", fontsize=16,
  #     bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", alpha=0.7)
  # )
  # axs[1,1].text(
  #     0.325-1, 0.075-1,
  #     r"$\ell/L = 1/10$",
  #     ha="left", va="bottom", fontsize=16,
  #     bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", alpha=0.7)
  # )
  # axs[1,0].add_artist(
  #     plt.Circle(
  #         (0.375-1, 0.375-1), 1/3,
  #         color="white", fill=False, linewidth=2
  #     )
  # )
  # axs[1,1].add_artist(
  #     plt.Circle(
  #         (0.142-1, 0.142-1), 1/10,
  #         color="white", fill=False, linewidth=2
  #     )
  # )
  ## add colorbar
  funcAddColorbar(fig, list(axs[0,:]), im_obj_0, r"$u^2 / u^2_{\rm{rms}}$", bool_top=True)
  funcAddColorbar(fig, list(axs[1,:]), im_obj_1, r"$B^2 / B^2_{\rm{rms}}$", bool_top=False)
  ## remove axis labels
  axs[0,0].axes.xaxis.set_visible(False)
  axs[0,0].axes.yaxis.set_visible(False)
  axs[0,1].axes.xaxis.set_visible(False)
  axs[0,1].axes.yaxis.set_visible(False)
  axs[1,0].axes.xaxis.set_visible(False)
  axs[1,0].axes.yaxis.set_visible(False)
  axs[1,1].axes.xaxis.set_visible(False)
  axs[1,1].axes.yaxis.set_visible(False)
  ## save plot
  print("Saving figure...")
  fig_name = "fig_energy_slice_{}.pdf".format(str_sim)
  fig_filepath = filepath_plot + "/" + fig_name
  fig.savefig(fig_filepath)
  print("\t> Figure saved: " + fig_name)

def funcAddColorbar(fig, axs, im, title, bool_top=True):
  ## get multi-axis bounds
  ax0_info = axs[0].get_position()
  ax1_info = axs[-1].get_position()
  ## get colorbar position
  ax_x0     = ax0_info.x0
  ax_x1     = ax1_info.x0
  ax_y0     = ax1_info.y0
  ax_width  = ax1_info.width
  ax_height = ax0_info.y0 + ax1_info.height
  ## add colorbar axis
  cbar_height = 0.035
  ax_cbar = fig.add_axes([
    ax_x0,
    (ax_height + 0.01) if bool_top else (ax_y0 - cbar_height - 0.01),
    (ax_x1 + ax_width) - ax_x0,
    cbar_height
  ])
  ## create colormap
  _ = fig.colorbar(im, cax=ax_cbar, orientation="horizontal")
  if bool_top:
    ax_cbar.xaxis.set_ticks_position("top")
    ax_cbar.text(0.5, 2, title, ha="center", va="bottom", transform=ax_cbar.transAxes, fontsize=20)
  else: ax_cbar.text(0.5, -1, title, ha="center", va="top", transform=ax_cbar.transAxes, fontsize=20)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## define working + plotting directories
  filepath_base = "/Users/dukekriel/Documents/Studies/TurbulentDynamo/"
  filepath_data_base = filepath_base + "/data/super_sonic"
  filepath_plot_base = filepath_base + "/figures/super_sonic"
  print("Saving figures in: " + filepath_plot_base)
  ## plot field slices
  funcPlotSlices(filepath_data_base, filepath_plot_base)


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
#!/usr/bin/env python3

##################################################################
## MODULES
##################################################################
import os
import argparse
import numpy as np
import cmasher as cmr # https://cmasher.readthedocs.io/user/diverging.html

from tqdm import tqdm

## user defined libraries
from the_useful_library import *
from the_plotting_library import *
from the_matplotlib_styler import *


#################################################################
## PREPARE TERMINAL/WORKSPACE/CODE
#################################################################
os.system("clear")  # clear terminal window
plt.close("all")    # close all pre-existing plots
## work in a non-interactive mode
mpl.use("Agg")
plt.ioff()


##################################################################
## COMMAND LINE ARGUMENT INPUT
##################################################################
ap = argparse.ArgumentParser(description="A bunch of input arguments")
## ------------------- DEFINE OPTIONAL ARGUMENTS
ap.add_argument("-vis_folder",     type=str,   default="vis_folder", required=False)
ap.add_argument("-dat_folder",     type=str,   default="",           required=False)
ap.add_argument("-start_time",     type=int,   default=[1],          required=False, nargs="+")
ap.add_argument("-end_time",       type=int,   default=[np.inf],     required=False, nargs="+")
ap.add_argument("-num_blocks",     type=int,   default=[36, 36, 48], required=False, nargs="+")
ap.add_argument("-num_procs",      type=int,   default=[8, 8, 6],    required=False, nargs="+")
ap.add_argument("-plots_per_eddy", type=float, default=10, required=False)
ap.add_argument("-plot_every",     type=int,   default=1,  required=False)
## ------------------- DEFINE REQUIRED ARGUMENTS
ap.add_argument("-base_path",   type=str, required=True)
ap.add_argument("-sim_folders", type=str, required=True, nargs="+")
ap.add_argument("-sim_names",   type=str, required=True, nargs="+")
## ---------------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ---------------------------- SAVE PARAMETERS
start_time     = args["start_time"] # starting processing frame
end_time       = args["end_time"]   # the last file to process
plots_per_eddy = args["plots_per_eddy"] # number of plot files per eddy turnover time
plot_every     = args["plot_every"] # number of plot files per eddy turnover time
num_blocks     = args["num_blocks"]
num_procs      = args["num_procs"]
## ---------------------------- SAVE FILEPATH PARAMETERS
filepath_base = args["base_path"]   # home directory
folders_sims  = args["sim_folders"] # list of subfolders where each simulation"s data is stored
folders_data  = args["dat_folder"]  # where spectras are stored in simulation folder
folder_vis    = args["vis_folder"]  # subfolder where animation and plots will be saved
sim_names     = args["sim_names"]   # name of figures


##################################################################
## INITIALISING VARIABLES
##################################################################
## ---------------------------- START CODE
if len(sim_names) < len(folders_sims):
    raise Exception("You need to give a label to every simulation.")
## if a time-range isn't specified for one of the simulations, then use the default time-range
if len(start_time) < len(folders_sims): start_time.extend( [1] * (len(folders_sims) - len(start_time)) )
if len(end_time) < len(folders_sims): end_time.extend( [np.inf] * (len(folders_sims) - len(end_time)) )
## folders where spectra data is
filepaths_data = []
for folder_sim in folders_sims:
    filepaths_data.append(createFilepath([filepath_base, folder_sim, folders_data]))
## folder where visualisations will be saved
filepath_vis = createFilepath([filepath_base, folder_vis])
createFolder(filepath_vis)
## folder where spectra plots will be saved
filepath_frames = createFilepath([filepath_vis, "slices"])
createFolder(filepath_frames)
## print filepath information to the console
printInfo("Base filepath:", filepath_base)
printInfo("Figure folder:", filepath_vis)
for sim_index in range(len(filepaths_data)):
    printInfo("\t> Sim name:", sim_names[sim_index])
    printInfo("\t> Sim directory:", filepaths_data[sim_index])
    print(" ")


##################################################################
## RUNNING CODE
##################################################################
for filepath_data, sim_index in zip(filepaths_data, range(len(filepaths_data))):
    ## find min and max colorbar limits, save field slices and simulation times
    print("Loading data...")
    col_min_val, col_max_val, list_flash_fields, list_sim_times = loadFLASHFieldDataList(
        filepath_data,
        start_time[sim_index],
        end_time[sim_index],
        str_field  = "mag",
        num_blocks = num_blocks,
        num_procs  = num_procs,
        plots_per_eddy   = plots_per_eddy,
        plot_every_index = plot_every
    )
    ## plot simulation frames
    print("Plotting data slices...")
    for _, time_index in loopListWithUpdates(list_sim_times):
        plot2DField(
            field = list_flash_fields[time_index],
            filepath_fig = "{}/{}_{}.png".format(
                filepath_frames, sim_names[sim_index], str(int(time_index)).zfill(4)
            ),
            cmap_str   = "cmr.ocean",
            cbar_label = r"$\log_{10}(B^2)$",
            cbar_lims  = [col_min_val, col_max_val]
        )
    # ## animate frames
    # if len(list_sim_times) > 3:
    #     aniEvolution(
    #         filepath_frames,
    #         filepath_vis,
    #         createName([ sim_names[sim_index], "slice=%*.png" ]),
    #         createName([ sim_names[sim_index], "ani_slice.mp4" ])
    #     )


## END OF PROGRAM
#!/usr/bin/env python3

##################################################################
## MODULES
##################################################################
import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

## load old user defined modules
from the_dynamo_library import *
from OldModules.the_useful_library import *
from the_matplotlib_styler import *


##################################################################
## PREPARE TERMINAL/WORKSPACE/CODE
#################################################################
os.system("clear") # clear terminal window
plt.close("all")   # close all pre-existing plots
## work in a non-interactive mode
mpl.use("Agg")
plt.ioff()


##################################################################
## INPUT COMMAND LINE ARGUMENTS
##################################################################
ap = argparse.ArgumentParser(description="A bunch of input arguments")
## ------------------- DEFINE OPTIONAL ARGUMENTS
ap.add_argument("-vis_folder", type=str, default="vis_folder", required=False)
ap.add_argument("-dat_folder", type=str, default="plt",         required=False)
ap.add_argument("-start_time", type=int, default=[1],        required=False, nargs="+")
ap.add_argument("-end_time",   type=int, default=[np.inf],   required=False, nargs="+")
ap.add_argument("-num_blocks", type=int, default=[36, 36, 48], required=False, nargs="+")
ap.add_argument("-num_procs",  type=int, default=[8, 8, 6],  required=False, nargs="+")
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
    raise Exception("You need to give a name to every simulation.")
## if a time-range isn't specified for one of the simulations, then use the default time-range
if len(start_time) < len(folders_sims): start_time.extend( [1] * (len(folders_sims) - len(start_time)) )
if len(end_time) < len(folders_sims): end_time.extend( [np.inf] * (len(folders_sims) - len(end_time)) )
## folders where spectra data is
filepaths_data = []
for folder_sim in folders_sims:
    filepaths_data.append(createFilePath([filepath_base, folder_sim, folders_data]))
## folder where visualisations will be saved
filepath_vis = createFilePath([filepath_base, folder_vis])
createFolder(filepath_vis)
## folder where spectra plots will be saved
filepath_frames = createFilePath([filepath_vis, "slices"])
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
filepath_data = createFilePath([filepaths_data[0], "Turb_hdf5_plt_cnt_0300"])

# vector field magnitude
print("Calculate field magnitude...")
## load magnitude of field
field_data = loadFLASHFieldSlice(
    filepath_data   = filepath_data,
    num_blocks      = num_blocks,
    num_procs       = num_procs,
    str_field_type  = "dens",
    bool_print_info = False
)
## plot
plot2DField(
    np.log(field_data[:,:,0]),
    createFilePath([filepath_vis, "test_field"]),
    cbar_label = r"$\ln(\rho / \rho_0)$",
    cmap_str   = "cmr.wildfire",
    bool_mid_norm = True,
    mid_norm      = 0
)

## diveregence of vector field
print("Calculate field divergence...")
## load data
data_sorted_x, data_sorted_y, data_sorted_z = loadFLASH3DField(
    filepath_data   = filepath_data,
    num_blocks      = num_blocks,
    num_procs       = num_procs,
    str_field_type  = "vel",
    bool_print_info = False
)
## calculate divergence of the field
dx = 1/(num_blocks[0] * num_procs[0])
grad_x = np.gradient(data_sorted_x, dx, axis=1)
grad_y = np.gradient(data_sorted_y, dx, axis=2)
grad_z = np.gradient(data_sorted_z, dx, axis=0)
div_data = np.sum(
    [
        grad_x[:,:,0],
        grad_y[:,:,0],
        grad_z[:,:,0]
    ],
    axis = 0
)
## plot
plot2DField(
    div_data,
    createFilePath([filepath_vis, "test_div"]),
    cbar_label = r"$\nabla \cdot (\bm{u} / \bm{u}_0)$",
    cmap_str   = "cmr.iceburn",
    bool_mid_norm = True,
    mid_norm      = 0
)

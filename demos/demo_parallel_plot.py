#!/usr/bin/env python3

##################################################################
## MODULES
##################################################################
import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

## for parallel plotting and progress bar
from tqdm import tqdm
from concurrent.tasks import ProcessPoolExecutor, as_completed

from math import sin, cos, floor

##################################################################
## PREPARE TERMINAL/WORKSPACE/CODE
#################################################################
os.system('clear')  # clear terminal window
plt.close('all')    # close all pre-existing plots
mpl.use('Agg')      # run in non-interactive mode: plots faster (can only save, not show figures) 
                        # http://matplotlib.org/faq/usage_faq.html#what-is-a-backend

##################################################################
## FUNCTIONS
##################################################################
def parallelProcess(array, function, num_proc=8, use_kwargs=False, serial_num=3):
    """
        A parallel version of the map function with a progress bar. 
        FROM: http://danshiebler.com/2016-09-14-parallel-progress-bar/
        Arguments:
            > array (array-like): An array to iterate over.
            > function (function): A python function to apply to the elements of array
            > num_proc (int, default=8): The number of cores to use
            > use_kwargs (boolean, default=False): Whether to consider input arguments to function
            > serial_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    ## we run the first few iterations serially to catch bugs
    if serial_num > 0: front = [function(**a) if use_kwargs else function(a) for a in array[:serial_num]]
    ## if we set num_proc to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if num_proc==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[serial_num:])]
    ## assemble the workers
    with ProcessPoolExecutor(max_workers=num_proc) as pool:
        ## pass the elements from the input array into the function
        if use_kwargs: tasks = [pool.submit(function, **a) for a in array[serial_num:]]
        else: tasks = [pool.submit(function, a) for a in array[serial_num:]]
        kwargs = {
            'total': len(tasks),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        ## print out the progress as tasks complete
        for f in tqdm(as_completed(tasks), **kwargs): pass
    out = []
    ## get the results from the tasks 
    for index, future in tqdm(enumerate(tasks)):
        try: out.append(future.result())
        except Exception as e: out.append(e)
    return front + out

def calcCliffordPoints(a, b, c, d, NUM_POINTS):
    ## initialise the points arrays
    x = np.zeros(NUM_POINTS)
    y = np.zeros(NUM_POINTS)
    ## calculate the clifford points
    for index in range(NUM_POINTS):
        x[index] = sin(a * y[index-1]) + c * cos(a * x[index-1])
        y[index] = sin(b * x[index-1]) + d * cos(b * y[index-1])
    ## return the points
    return x, y

def calcDensityMap(x_points, y_points, NUM_CELLS):
    ## initialise the density map
    density_map = np.zeros([NUM_CELLS, NUM_CELLS])
    ## find the bounds of the points
    x_max = max(x_points)
    x_min = min(x_points)
    y_max = max(y_points)
    y_min = min(y_points)
    ## calculate the cell widths
    cell_width  = (x_max - x_min)/NUM_CELLS
    cell_height = (y_max - y_min)/NUM_CELLS
    ## loop over each of the points and increment the cell they fall in
    for x, y in zip(x_points, y_points):
        ## initialise the cell index
        tmp_col = -1
        tmp_row = -1
        ## find the cell in which the point falls
        tmp_col = floor(min((x - x_min)/cell_width,  NUM_CELLS-1))
        tmp_row = floor(min((y - y_min)/cell_height, NUM_CELLS-1))
        ## increment the grid cell count
        if (tmp_col >= 0) and (tmp_row >= 0): 
            density_map[tmp_row, tmp_col] += 1
    ## return density map
    return density_map

def plotDensityMap(density_map, plot_num):
    ## dimensions of the figure
    x_dim = 6.0
    y_dim = 5
    ## create figure
    fig = plt.figure(frameon=False)
    fig.set_size_inches(x_dim, y_dim)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ## plot and save image
    plt.imshow(density_map,
        vmin=np.min(density_map), vmax=0.9*np.max(density_map),
        origin='lower', extent=[0, x_dim, 0, y_dim], cmap='viridis')
    plt.savefig('density_'+str(plot_num).zfill(3)+'.png', dpi=300)
    ## close figure
    plt.close()

def plotCliffordAttractor(a, b, c, d, NUM_POINTS, NUM_CELLS, plot_num):
    ## calculate points
    x_points, y_points = calcCliffordPoints(a, b, c, d, num_points)
    ## calculate density map
    density_map = calcDensityMap(x_points, y_points, NUM_CELLS)
    ## plot and save figure
    plotDensityMap(density_map, plot_num)
    ## return completion status
    return True

##################################################################
## INPUT COMMAND LINE ARGUMENTS
##################################################################
ap = argparse.ArgumentParser(description='A bunch of input arguments')
## ------------------- DEFINE OPTIONAL ARGUMENTS
ap.add_argument('-num_proc',   type=int, required=False, default=8)
ap.add_argument('-num_points', type=int, required=False, default=10**6)
ap.add_argument('-num_cells',  type=int, required=False, default=10**3)
## ---------------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ---------------------------- SAVE PARAMETERS
num_proc   = args['num_proc']
num_points = args['num_points']
num_cells  = args['num_cells']

##################################################################
## PLOT CLIFFORD ATTRACTOR
##################################################################
## how many plots (i.e. different number of clifford attractors)
num_plots = 100
## define the set of parameters
a_range = 1.7 * np.ones(num_plots)
b_range = 1.7 * np.ones(num_plots)
c_range = 0.6 * np.ones(num_plots)
d_range = np.linspace(1, 2, num_plots)
## create the input array to the parallel function
input_array = [{"a":a,
                "b":b,
                "c":c,
                "d":d,
                "NUM_POINTS":num_points,
                "NUM_CELLS":num_cells,
                "plot_num":plot_num} 
                for a, b, c, d, plot_num 
                in zip(a_range, b_range, c_range, d_range, range(num_plots))]
## plot clifford attractors
print('Plotting clifford attractors:')
list_output = parallelProcess(input_array, plotCliffordAttractor, use_kwargs=True, num_proc=num_proc)
print(' ')

## animate plots
#############################
ffmpeg_input = ('ffmpeg -start_number 0 -i density_%3d.png' + 
                ' -vb 40M -framerate 40 -vf scale=1440:-1 -vcodec mpeg4 clifford_denisty.mp4')
print('Animating plots...')
os.system(ffmpeg_input)
print(' ')

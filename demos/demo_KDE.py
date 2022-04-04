## https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
## https://scikit-learn.org/stable/modules/density.html


#################################################################
## MODULES
#################################################################
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
import cmasher as cmr # https://cmasher.readthedocs.io/user/diverging.html

from matplotlib.gridspec import GridSpec

from the_fitting_library import FitFromKDE

#################################################################
## PREPARE TERMINAL/WORKSPACE/CODE
#################################################################
os.system("clear")  # clear terminal window
plt.close("all")    # close all pre-existing plots
## work in a non-interactive mode
mpl.use("Agg")
plt.ioff()


def firstTest(my_path):
    ## initialise figure
    fig_scales = plt.figure(figsize=(12,4), constrained_layout=True)
    fig_grids = GridSpec(ncols=3, nrows=1, figure=fig_scales)
    ax0 = fig_scales.add_subplot(fig_grids[0])
    ax1 = fig_scales.add_subplot(fig_grids[1])
    ax2 = fig_scales.add_subplot(fig_grids[2])
    ## LEFT AXIS
    ############
    ## generate 2D distribution of points
    data_x = np.random.normal(0, 0.1, 1000)
    data_y = np.random.normal(100, 2, 1000)
    ## plot distribution of points
    ax0.plot(data_x, data_y, "b.")
    ax0.set_aspect(1./ax0.get_data_ratio())
    ## MIDDLE AXIS
    ##############
    ## calculate bounds of distribution
    xmin = min(data_x)
    xmax = max(data_x)
    ymin = min(data_y)
    ymax = max(data_y)
    ## create meshgrid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    ## fit gaussian KDE
    positions = np.vstack([xx.ravel(), yy.ravel()])
    kernel = stats.gaussian_kde(
        np.vstack([data_x, data_y])
    )
    f = np.reshape(kernel(positions).T, xx.shape)
    ## plot KDE overlayed by data
    ax1.imshow(np.rot90(f), cmap=plt.get_cmap("cmr.arctic"), extent=[xmin, xmax, ymin, ymax])
    ax1.plot(data_x, data_y, "b.")
    ax1.set_aspect(1./ax1.get_data_ratio())
    ## RIGHT AXIS
    #############
    ## resample KDE
    resamp_stack = kernel.resample(size=200)
    ## plot actual data and KDE which is being sampled for reference
    ax2.imshow(np.rot90(f), cmap=plt.get_cmap("cmr.arctic"), extent=[xmin, xmax, ymin, ymax])
    ax2.plot(data_x, data_y, "b.")
    ## plot resampled data
    ax2.plot(resamp_stack[0], resamp_stack[1], 'r.')
    ax2.set_aspect(1./ax2.get_data_ratio())
    ##### SAVE PLOT
    plt.savefig(my_path + "scipy_KDE.png")
    plt.close()

def secondTest(my_path):
    ## initialise figure
    fig_scales = plt.figure(figsize=(8,4), constrained_layout=True)
    fig_grids = GridSpec(ncols=2, nrows=1, figure=fig_scales)
    ax0 = fig_scales.add_subplot(fig_grids[0])
    ax1 = fig_scales.add_subplot(fig_grids[1])
    ## LEFT AXIS
    ############
    ## generate a list of 2D distributions
    list_data_x = []
    list_data_y = []
    list_mean = [5, 10]
    for list_index in range(len(list_mean)):
        list_data_x.append( list(np.random.normal(list_mean[list_index], 0.5, 15)) )
        list_data_y.append( list(np.random.normal(list_mean[list_index], 3, 15)) )
    ## plot distribution of points
    ax0.plot(list_data_x, list_data_y, "b.")
    ax0.set_aspect(1./ax0.get_data_ratio())
    ## RIGHT AXIS
    #############
    ## plot orginal data
    ax1.plot(list_data_x, list_data_y, "b.")
    ## resample KDE
    KDE = FitFromKDE(list_data_x, list_data_y)
    KDE.resampleFrom1DKDE(bool_x=True, num_resamp=20)
    KDE.resampleFrom1DKDE(bool_y=True, num_resamp=20)
    resampled_x = KDE.resampled_x
    resampled_y = KDE.resampled_y
    ## plot resampled data
    ax1.plot(resampled_x, resampled_y, 'r.')
    ## fix axis
    ax1.set_aspect(1./ax1.get_data_ratio())
    ##### SAVE PLOT
    plt.savefig(my_path + "scipy_KDE.png")
    plt.close()


#################################################################
## INITIALISING DATA
#################################################################
my_path = "/Users/dukekriel/Documents/Projects/Turbulent-Dynamo/data/"

secondTest(my_path)

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
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit

## load old user defined modules
from ThePlottingModule import TheMatplotlibStyler, PlotFuncs
from TheUsefulModule import WWLists, WWFnF
from TheFittingModule import UserModels
from TheLoadingModule import LoadFlashData


## ###############################################################
## PREPARE WORKSPACE
##################################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## FUNCTIONS
## ###############################################################
def funcFindDomainIndices(data, start_point, end_point):
  index_start = WWLists.getIndexClosestValue(data, start_point)
  index_end   = WWLists.getIndexClosestValue(data, end_point)
  return [ index_start, index_end ]


def funcPlotExpFit(
    ax,
    data_x, data_y,
    index_start_fit, index_end_fit
  ):
  ## define fit domain
  data_fit_domain = np.linspace(
    data_x[index_start_fit],
    data_x[index_end_fit],
    10**2
  )
  ## interpolate the non-uniform data
  interp_spline = make_interp_spline(
    data_x[index_start_fit : index_end_fit],
    data_y[index_start_fit : index_end_fit]
  )
  ## uniformly sample interpolated data
  data_sampled_y = interp_spline(data_fit_domain)
  ## fit exponential function to sampled data (in log-linear domain)
  fit_params_log, _ = curve_fit(
    UserModels.ListOfModels.exp_loge,
    data_fit_domain,
    np.log(data_sampled_y)
  )
  ## undo log transformation
  fit_params_linear = [
    np.exp(fit_params_log[0] + 2),
    fit_params_log[1]
  ]
  ## ##################
  ## PLOTTING DATA
  ## ########
  ## initialise the plot domain
  data_plot_domain = np.linspace(2, 100, 10**3)
  ## evaluate exponential
  data_E_exp = UserModels.ListOfModels.exp_linear(
    data_plot_domain,
    *fit_params_linear
  )
  ## find where exponential enters / exists fit range
  index_E_start = WWLists.getIndexClosestValue(data_E_exp, data_y[index_start_fit])
  index_E_end   = WWLists.getIndexClosestValue(data_E_exp, data_y[index_end_fit])
  ## plot fit
  ax.plot(
    data_plot_domain[ index_E_start : index_E_end ],
    data_E_exp[ index_E_start : index_E_end ],
    color="black", ls="--", lw=2, zorder=5
  )


def funcPlotSat(
    ax,
    data_x, data_y,
    index_start_fit, index_end_fit
  ):
  ## subset data in saturated regime
  sub_data_x = data_x[ index_start_fit : index_end_fit ]
  sub_data_y = data_y[ index_start_fit : index_end_fit ]
  ## fit saturated range
  ax.plot(
    sub_data_x,
    [ np.mean(sub_data_y) ] * len(sub_data_x),
    color="black", ls=":", lw=2, zorder=5
  )


class PlotEnergyRatio():
  def __init__(
      self,
      filepath_data, filepath_figure,
      time_start, time_end, time_sat
    ):
    ## store filepaths
    self.filepath_data   = filepath_data
    self.filepath_figure = filepath_figure
    ## store data time ranges
    self.time_start = time_start
    self.time_end   = time_end
    self.time_sat   = time_sat
    ## initialise data
    self.list_time         = None
    self.list_E_ratio      = None
    self.list_index_E_grow = None
    self.list_index_E_sat  = None
    ## run method functions
    self.loadData()
    self.plotData()
  def loadData(self):
    ## load magnetic energy
    data_time, data_E_B = LoadFlashData.loadTurbData(
      filepath_data = self.filepath_data,
      var_y      = 29,
      t_turb     = 4,
      time_start = self.time_start,
      time_end   = self.time_end
    )
    ## load kinetic energy
    data_time, data_E_K = LoadFlashData.loadTurbData(
      filepath_data = self.filepath_data,
      var_y      = 6,
      t_turb     = 4,
      time_start = self.time_start,
      time_end   = self.time_end
    )
    ## compute energy ratio
    data_E_ratio = [
      E_B / E_K
      for E_B, E_K in zip(data_E_B, data_E_K)
    ]
    ## store data
    self.data_time    = data_time
    self.data_E_ratio = data_E_ratio
    self.index_range_E_grow = funcFindDomainIndices(data_E_ratio, 1e-7, 1e-2)
    self.index_range_E_sat  = funcFindDomainIndices(data_time, self.time_sat, np.inf)
  def plotData(self):
    ## initialise figure
    _, ax = plt.subplots()
    ## plot time evolving data
    ax.plot(
      self.data_time,
      self.data_E_ratio,
      color="black", ls="-", lw=2, zorder=3
    )
    ## fit exponential phase
    funcPlotExpFit(
      ax = ax,
      data_x = self.data_time,
      data_y = self.data_E_ratio,
      index_start_fit = self.index_range_E_grow[0],
      index_end_fit   = self.index_range_E_grow[1]
    )
    ## fit saturated phase
    funcPlotSat(
      ax = ax,
      data_x = self.data_time,
      data_y = self.data_E_ratio,
      index_start_fit = self.index_range_E_sat[0],
      index_end_fit   = self.index_range_E_sat[1]
    )
    ## add axis labels
    ax.set_xlabel(r"$t / t_\mathrm{turb}$")
    ax.set_ylabel(r"$E_{\rm{mag}} / E_{\rm{kin}}$")
    ## scale y-axis
    ax.set_yscale("log")
    ## adjust domain range
    x = np.linspace(0, 100)
    y = np.logspace(-9, 0, 100)
    ax.set_xlim([ x[0], x[-1] ])
    ax.set_ylim([ y[0], y[-1] ])
    ## colour domain regions
    ax.fill_betweenx(y, 0, 1,    facecolor="red",    alpha=0.5, zorder=1)
    ax.fill_betweenx(y, 1, 10,   facecolor="yellow", alpha=0.5, zorder=1)
    ax.fill_betweenx(y, 10, 30,  facecolor="green",  alpha=0.5, zorder=1)
    ax.fill_betweenx(y, 30, 100, facecolor="blue",   alpha=0.5, zorder=1)
    ## add log axis-ticks
    PlotFuncs.addLogAxisTicks(ax)
    ## save figure
    fig_filepath = WWFnF.createFilepath([self.filepath_figure, "fig_dynamo_regimes.pdf"])
    plt.savefig(fig_filepath)
    plt.close()
    print("Figure saved:", fig_filepath)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  PlotEnergyRatio(
    filepath_data   = "/Users/dukekriel/Documents/Studies/TurbulentDynamo/data/sub_sonic/Rm3000/288/Pm2",
    filepath_figure = "/Users/dukekriel/Documents/Studies/TurbulentDynamo/figures/sub_sonic",
    time_start = 0.01,
    time_end   = 100,
    time_sat   = 30
  )


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
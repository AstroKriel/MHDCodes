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

from matplotlib.collections import LineCollection
from scipy.interpolate import make_interp_spline

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
# plt.style.use('dark_background')


## ###############################################################
## FUNCTIONS
## ###############################################################
def funcFindDomain(data, start_point, end_point):
  ## find indices when to start fitting
  index_start = getIndexClosestValue(data, start_point)
  ## find indices when to stop fitting
  index_end = getIndexClosestValue(data, end_point)
  ## check that there are a suffient number of points to fit to
  num_points = len(data[index_start : index_end])
  if num_points < 3:
    ## if no sufficient domain to fit to, then don't define a fit range
    index_start = None
    index_end = None
    # print("\t> Insufficient number of points ('{:d}') to fit to.".format(num_points))
  ## return the list of indices
  return [index_start, index_end]

def funcLoadData(
    ## where data is
    filepath_data,
    ## time range to load data for
    time_start, time_end, time_sat,
    ## output: simulation data
    list_time, list_Mach,
    list_E_kin_group, list_E_mag_group, list_E_ratio,
    ## output: indices associated with kinematic and saturated phases
    list_index_E_grow, list_index_E_sat, list_index_Mach
  ):
  ## load 'Mach' data
  data_time, data_Mach = loadTurbData(
    filepath_data = filepath_data,
    var_y      = 8,
    t_turb     = 4,
    time_start = time_start,
    time_end   = time_end
  )
  ## load 'E_B' data
  data_time, data_E_B = loadTurbData(
    filepath_data = filepath_data,
    var_y      = 29,
    t_turb     = 4,
    time_start = time_start,
    time_end   = time_end
  )
  ## load 'E_K' data
  data_time, data_E_K = loadTurbData(
    filepath_data = filepath_data,
    var_y      = 6,
    t_turb     = 4,
    time_start = time_start,
    time_end   = time_end
  )
  ## calculate ratio: 'E_B / E_K'
  data_E_ratio = [
    E_B / E_K
    for E_B, E_K in zip(data_E_B, data_E_K)
  ]
  ## append data
  list_time.append(data_time)
  list_Mach.append(data_Mach)
  list_E_kin_group.append(data_E_K)
  list_E_mag_group.append(data_E_B)
  list_E_ratio.append(data_E_ratio)
  ## append indices of bounds
  list_index_E_grow.append(
    funcFindDomain(data_E_ratio, 1e-6, 1e-2)
  )
  list_index_E_sat.append(
    funcFindDomain(data_time, time_sat, np.inf)
  )
  list_index_Mach.append(
    funcFindDomain(data_time, 10, 25)
  )

def funcPlotExpFit(
    ax,
    data_x, data_y,
    index_start_fit, index_end_fit,
    color = "black"
  ):
  ## ##################
  ## SAMPLING DATA
  ## ########
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
  ## ##################
  ## FITTING DATA
  ## #######
  ## fit exponential function to sampled data (in log-linear domain)
  fit_params_log, fit_cov = curve_fit(
    ListOfModels.exp_loge,
    data_fit_domain,
    np.log(data_sampled_y)
  )
  ## undo log transformation
  fit_params_linear = [
    np.exp(fit_params_log[0]),
    fit_params_log[1]
  ]
  ## ##################
  ## PLOTTING DATA
  ## ########
  ## initialise the plot domain
  data_plot_domain = np.linspace(0, 100, 10**3)
  ## evaluate exponential
  data_E_exp = ListOfModels.exp_linear(
    data_plot_domain,
    *fit_params_linear
  )
  ## find where exponential enters / exists fit range
  index_E_start = getIndexClosestValue(data_E_exp, data_y[index_start_fit])
  index_E_end = getIndexClosestValue(data_E_exp, data_y[index_end_fit])
  ## plot fit
  ax.plot(
    data_plot_domain[index_E_start : index_E_end],
    data_E_exp[index_E_start : index_E_end],
    color="black", dashes=(5, 1.5), linewidth=2, zorder=5
  )

def funcMeasureExpFit(
    data_x, data_y,
    index_start_fit, index_end_fit,
    bool_return_str = True
  ):
  ## check there is a domain to fit to
  if (index_start_fit is None) or (index_end_fit is None):
    if bool_return_str:
      return "decaying"
    else: return None
  ## ##################
  ## SAMPLING DATA
  ## ########
  ## define list of subset domain
  d_index = (index_end_fit - index_start_fit) // 5
  list_index_start = [
    index_start_fit,
    index_start_fit,
    index_start_fit + d_index,
    index_start_fit + 2*d_index,
    index_start_fit + 3*d_index,
    index_start_fit + 4*d_index
  ]
  list_index_end = [
    index_end_fit,
    index_end_fit - 4*d_index,
    index_end_fit - 3*d_index,
    index_end_fit - 2*d_index,
    index_end_fit - d_index,
    index_end_fit
  ]
  list_fit_growth = []
  list_fit_std = []
  for fit_index in range(len(list_index_start)):
    ## define fit domain
    data_fit_domain = np.linspace(
      data_x[list_index_start[fit_index]],
      data_x[list_index_end[fit_index]],
      10**2
    )
    ## interpolate the non-uniform data
    interp_spline = make_interp_spline(
      data_x[list_index_start[fit_index] : list_index_end[fit_index]],
      data_y[list_index_start[fit_index] : list_index_end[fit_index]]
    )
    ## uniformly sample interpolated data
    data_sampled_y = interp_spline(data_fit_domain)
    ## ##################
    ## FITTING DATA
    ## #######
    ## fit exponential function to sampled data (in log-linear domain)
    fit_params_log, fit_cov = curve_fit(
      ListOfModels.exp_loge,
      data_fit_domain,
      np.log(data_sampled_y)
    )
    ## undo log transformation
    fit_params_linear = [
      np.exp(fit_params_log[0]),
      fit_params_log[1]
    ]
    ## get growth rate
    list_fit_growth.append(-1 * fit_params_linear[1])
    ## get error
    list_fit_std.append(np.sqrt(np.diag(fit_cov))[1])
  growth_rate = list_fit_growth[0]
  growth_std = max(list_fit_std)
  ## return growth rate string
  if bool_return_str:
    str_median = ("{0:.2g}").format(growth_rate)
    if "." not in str_median:
      str_median = ("{0:.1f}").format(growth_rate)
    if "." in str_median:
      if ("0" == str_median[0]) and (len(str_median.split(".")[1]) == 1):
        str_median = ("{0:.2f}").format(growth_rate)
      num_decimals = len(str_median.split(".")[1])
    str_std = ("{0:."+str(num_decimals)+"f}").format(growth_std)
    return "${} \pm {}$".format(str_median, str_std)
  ## return growth rate value
  else: return growth_rate

def funcPlotMach(
    ## plot parameters
    ax, list_colors, list_labels,
    ## simulation data
    list_time_group, list_Mach_group,
    ## indices of saturated phase
    list_index_E_grow
  ):
  ## #################
  ## CREATE INSET AXIS
  ## #########
  ## create inset axes (it should fill the bounding box allocated to it)
  ax_inset = ax.inset_axes([0.2, 0.11, 0.75, 0.235])
  ## #########################
  ## LOOP OVER EACH SIMULATION
  ## #########
  ## for each simulation
  for sim_index in range(len(list_labels)):
    ## plot full time evolving data
    ax.plot(
      list_time_group[sim_index],
      np.array(list_Mach_group[sim_index]),
      color     = list_colors[sim_index],
      linestyle = "-",
      linewidth = 2,
      zorder    = 3
    )
    ## get indices of end of tarnsient phase (t/t_turb approx 5)
    index_start, index_end = funcFindDomain(list_time_group[sim_index], 0, 5)
    ## plot zoomed in data
    ax_inset.plot(
      list_time_group[sim_index][index_start : index_end],
      np.array(list_Mach_group[sim_index][index_start : index_end]),
      color     = list_colors[sim_index],
      linestyle = "-",
      linewidth = 2,
      zorder    = 3
    )
  ## add legend
  addLegend(
    ax = ax,
    loc  = "upper right",
    bbox = (0.95, 1.0),
    artists = [ "-", "-" ],
    colors  = [ "green", "orange" ],
    legend_labels = [ r"Re470Pm2", r"Re1700Pm2" ],
    rspacing = 0.5,
    cspacing = 0.25,
    ncol = 1,
    fontsize = 15,
    labelcolor = "white"
  )
  ax.text(
    0.675, 0.93,
    r"Re470Pm2", color="black",
    va="top", ha="left", transform=ax.transAxes, fontsize=15, zorder=10
  )
  ax.text(
    0.675, 0.81,
    r"Re1700Pm2", color="black",
    va="top", ha="left", transform=ax.transAxes, fontsize=15, zorder=10
  )
  ## tune inset axis
  ax_inset.set_xlim(0, 5)
  ax_inset.set_ylim(0, 0.5)
  ax.indicate_inset_zoom(ax_inset, edgecolor="black")
  ## tune main axis
  ax.set_ylim(-0.02, 0.52)
  ## add axis labels
  ax.set_ylabel(r"$\mathcal{M}$", fontsize=22)
  y_major = mpl.ticker.FixedLocator([0, 0.1, 0.2, 0.3, 0.4, 0.5])
  ax.yaxis.set_major_locator(y_major)

def funcPlotKinetic(
    ## plot parameters
    ax, list_colors, list_labels,
    ## simulation data
    list_time_group, list_E_kin_group, list_index_E_sat
  ):
  ## #################
  ## CREATE INSET AXIS
  ## #########
  ## create inset axes (it should fill the bounding box allocated to it)
  ax_inset = ax.inset_axes([0.2, 0.11, 0.75, 0.55])
  ## #########################
  ## LOOP OVER EACH SIMULATION
  ## #########
  ## for each simulation
  for sim_index in range(len(list_labels)):
    ## plot time evolving data
    ax.plot(
      list_time_group[sim_index],
      np.array(list_E_kin_group[sim_index]) / np.mean(list_E_kin_group[sim_index][
        list_index_E_sat[sim_index][0] : list_index_E_sat[sim_index][1]
      ]),
      color     = list_colors[sim_index],
      linestyle = "-",
      linewidth = 2,
      zorder    = 3
    )
    ## get indices of end of tarnsient phase (t/t_turb approx 5)
    index_start, index_end = funcFindDomain(list_time_group[sim_index], 0, 5)
    ## plot zoomed in data
    ax_inset.plot(
      list_time_group[sim_index][index_start : index_end],
      np.array(list_E_kin_group[sim_index][index_start : index_end]),
      color     = list_colors[sim_index],
      linestyle = "-",
      linewidth = 2,
      zorder    = 3
    )
  ## tune inset axis
  ax_inset.set_xlim(0, 5)
  ax_inset.set_ylim(10**(-5), 10**(1))
  ax_inset.set_yscale("log")
  ax.indicate_inset_zoom(ax_inset, edgecolor="black")
  ## add axis labels
  ax.set_ylabel(r"$E_{\rm{kin}} / \langle E_{\rm{kin}}\rangle$", fontsize=22)
  ## scale y-axis
  ax.set_yscale("log")
  ax.set_ylim([0.5*10**(-5), 3*10**(1)])
  locmin = mpl.ticker.LogLocator(
    base=10.0,
    subs=np.arange(2, 10) * 0.1,
    numticks=100
  )
  ax.yaxis.set_minor_locator(locmin)
  ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
  y_major = mpl.ticker.LogLocator(base=10.0, numticks=10)
  ax.yaxis.set_major_locator(y_major)

def funcPlotMagnetic(
    ## plot parameters
    ax, list_colors, list_labels,
    ## simulation data
    list_time_group, list_E_mag_group
  ):
  ## #########################
  ## LOOP OVER EACH SIMULATION
  ## #########
  ## for each simulation
  for sim_index in range(len(list_labels)):
    ## plot time evolving data
    ax.plot(
      list_time_group[sim_index],
      np.array(list_E_mag_group[sim_index]) / list_E_mag_group[sim_index][0],
      color     = list_colors[sim_index],
      linestyle = "-",
      linewidth = 2,
      zorder    = 3
    )
  ## add axis labels
  ax.set_ylabel(r"$E_{\rm{mag}} / E_{\rm{mag}, 0}$", fontsize=22)
  ## scale y-axis
  ax.set_xlim(0, 100)
  ax.set_yscale("log")
  ax.set_ylim([10**(-1), 3*10**(9)])
  locmin = mpl.ticker.LogLocator(
    base=10.0,
    subs=np.arange(2, 10) * 0.1,
    numticks=100
  )
  ax.yaxis.set_minor_locator(locmin)
  ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
  y_major = mpl.ticker.LogLocator(base=10.0, numticks=6)
  ax.yaxis.set_major_locator(y_major)

def funcPlotEnergyRatio(
    ## plot parameters
    ax, list_colors, list_labels,
    ## max of plot domain
    max_time,
    ## simulation data
    list_time_group, list_E_ratio_group,
    ## indices of kinematic and saturated phases
    list_index_E_grow, list_index_E_sat
  ):
  ## #########################
  ## LOOP OVER EACH SIMULATION
  ## #########
  ## for each simulation
  for sim_index in range(len(list_labels)):
    ## plot time evolving data
    ax.plot(
      list_time_group[sim_index],
      list_E_ratio_group[sim_index],
      color     = list_colors[sim_index],
      linestyle = "-",
      linewidth = 1.5,
      zorder    = 3
    )
    ## fit kinematic phase and save growth rate
    funcPlotExpFit(
      ax = ax,
      data_x = list_time_group[sim_index],
      data_y = list_E_ratio_group[sim_index],
      index_start_fit = list_index_E_grow[sim_index][0],
      index_end_fit   = list_index_E_grow[sim_index][1],
      color = list_colors[sim_index]
    )
    ## subset data in saturated regime
    sub_time_sat = list_time_group[sim_index][
      list_index_E_sat[sim_index][0] : list_index_E_sat[sim_index][1]
    ]
    sub_E_ratio_sat = list_E_ratio_group[sim_index][
      list_index_E_sat[sim_index][0] : list_index_E_sat[sim_index][1]
    ]
    ## fit saturated range
    ax.plot(
      sub_time_sat,
      [ np.mean(sub_E_ratio_sat) ] * len(sub_time_sat),
      color="black", ls=(0, (2, 1.5)), linewidth=2, zorder=5
    )
  ## ###############################
  ## KINETMATIC PHASE BOUNDING RANGE
  ## #########
  ## lower limit
  ax.axhline(
    y=list_E_ratio_group[0][list_index_E_grow[0][0]],
    ls=(0, (7.5, 5)), color="black", lw=1, alpha=0.5, zorder=3
  )
  ## upper limit
  ax.axhline(
    y=list_E_ratio_group[0][list_index_E_grow[0][1]],
    ls=(0, (7.5, 5)), color="black", lw=1, alpha=0.5, zorder=3
  )
  ## colour fitting region
  ax.axhspan(
    list_E_ratio_group[0][list_index_E_grow[0][0]],
    list_E_ratio_group[0][list_index_E_grow[0][1]],
    color  = "black",
    alpha  = 0.05,
    zorder = 0.25
  )
  ## add legend
  addLegend(
    ax = ax,
    loc  = "lower right",
    bbox = (0.95, 0.0),
    artists = [ (0, (5, 1.5)), (0, (2, 1.5)) ],
    colors  = [ "black", "black" ],
    legend_labels = [
      r"Kinematic phase",
      r"Saturated phase"
    ],
    rspacing = 0.5,
    cspacing = 0.25,
    ncol = 1,
    fontsize = 15,
    labelcolor = "white",
    lw = 1.75
  )
  ax.text(
    0.6, 0.265,
    r"Kinematic phase", color="black",
    va="top", ha="left", transform=ax.transAxes, fontsize=15, zorder=10
  )
  ax.text(
    0.6, 0.145,
    r"Saturated phase", color="black",
    va="top", ha="left", transform=ax.transAxes, fontsize=15, zorder=10
  )
  ## add axis labels
  ax.set_xlabel(r"$t / t_\mathrm{turb}$", fontsize=22)
  ax.set_ylabel(r"$E_{\rm{mag}} / E_{\rm{kin}}$", fontsize=22)
  ## scale y-axis
  ax.set_yscale("log")
  ## adjust domain range
  ax.set_xlim([0, max_time])
  ax.set_ylim([10**(-9), 10**(0)])
  # y_major = mpl.ticker.LogLocator(base=10.0, numticks=15)
  y_minor = mpl.ticker.LogLocator(
    base=10.0,
    subs=np.arange(2, 10) * 0.1,
    numticks=100
  )
  # ax.yaxis.set_major_locator(y_major)
  ax.yaxis.set_minor_locator(y_minor)
  ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
  y_major = mpl.ticker.LogLocator(base=10.0, numticks=6)
  ax.yaxis.set_major_locator(y_major)

def funcPlotTurb(filepath_data, filepath_plot):
  ## ####################
  ## INITIALISE VARIABLES
  ## ####################
  ## list of filepaths
  list_filepaths_data = [
    createFilepath([ filepath_data, "Re500",  "288", "Pm2" ]),
    createFilepath([ filepath_data, "Rm3000", "288", "Pm2" ])
  ]
  ## define simulation labels
  list_labels = [ r"Re470Pm2", r"Re1700Pm2" ]
  ## define simulation colors
  list_colors_kin = [ "green", "orange" ]
  list_colors_mag = [ "green", "orange" ]
  ## ####################
  ## LOAD SIMULATION DATA
  ## ####################
  ## define plotting domain for each simulation
  list_time_end = [ 100, 100 ]
  list_time_sat = [ 50, 30 ]
  ## initialise list of data
  list_time_group    = []
  list_Mach_group    = []
  list_E_kin_group   = []
  list_E_mag_group   = []
  list_E_ratio_group = []
  list_index_E_grow  = []
  list_index_E_sat   = []
  ## load data for each simulation
  print("Loading simulation data...")
  for sim_index in range(len(list_filepaths_data)):
    print("\t> " + list_filepaths_data[sim_index])
    funcLoadData(
      list_filepaths_data[sim_index],
      0.01, list_time_end[sim_index], list_time_sat[sim_index],
      list_time_group, list_Mach_group,
      list_E_kin_group, list_E_mag_group, list_E_ratio_group,
      list_index_E_grow, list_index_E_sat, []
    )
  ## #####################
  ## PLOT ENERGY EVOLUTION
  ## #####################
  fig, axs = plt.subplots(3, 1, figsize=(6, 3*3.5), sharex=True)
  fig.subplots_adjust(hspace=0.05)
  # ## plot kinetic energy
  # funcPlotKinetic(
  #     axs[0,0], list_colors_mag, list_labels,
  #     list_time_group, list_E_kin_group, list_index_E_sat
  # )
  ## plot Mach evolution
  funcPlotMach(
    axs[0], list_colors_kin, list_labels,
    list_time_group, list_Mach_group,
    list_index_E_grow
  )
  ## plot magnetic energy
  funcPlotMagnetic(
    axs[1], list_colors_mag, list_labels,
    list_time_group, list_E_mag_group
  )
  ## plot energy ratio evolution
  funcPlotEnergyRatio(
    axs[2],
    list_colors_mag, list_labels,
    max(list_time_end),
    list_time_group, list_E_ratio_group,
    list_index_E_grow, list_index_E_sat
  )
  ## save plot
  fig_name = "fig_energy_time.pdf"
  fig_filepath = createFilepath([filepath_plot, fig_name])
  plt.savefig(fig_filepath)
  print("\t> Figure saved: " + fig_name)

class funcPlotStuff():
  def __init__(
      self,
      filepath_data, filepath_plot
    ):
    ## store filepaths
    self.filepath_data = filepath_data
    self.filepath_plot = filepath_plot
    ## initialised lists
    self.list_Re = []
    self.list_Rm = []
    self.list_Pm = []
    self.list_E_growth = []
    self.list_E_sat    = []
    self.list_markers  = []
    ## load simulation data
    self.loadData()
    ## flatten lists
    self.list_Re = flattenList(self.list_Re)
    self.list_Rm = flattenList(self.list_Rm)
    self.list_Pm = flattenList(self.list_Pm)
    self.list_markers = flattenList(self.list_markers)
    ## create list of colors
    self.list_color = [
      "cornflowerblue" if Re < 100
      else "orangered"
      for Re in self.list_Re
    ]
    ## plot data
    self.plotGrowth()
    self.plotSat()
  def loadData(self):
    ## load Re = 10
    self.list_Re.append([ 10, 10, 10, 10 ])
    self.list_Rm.append([ 270, 540, 1300, 2500 ])
    self.list_Pm.append([ 27, 54, 130, 250 ])
    self.list_markers.append([ "s", "s", "s", "s" ])
    self.loadData_sim(
      sim_name = "Re10",
      list_sim_folders = [ "Pm25", "Pm50", "Pm125", "Pm250" ],
      list_time_sat = [ 30, 60, 30, 30 ]
    )
    ## load Re = 500
    self.list_Re.append([ 433, 466, 466 ])
    self.list_Rm.append([ 433, 933, 1866 ])
    self.list_Pm.append([ 1, 2, 4 ])
    self.list_markers.append([ "D", "D", "D" ])
    self.loadData_sim(
      sim_name = "Re500",
      list_sim_folders = [ "Pm1", "Pm2", "Pm4" ],
      list_time_sat = [ 50, 50, 50 ]
    )
    ## load Rm = 3000
    self.list_Re.append([ 3601, 1676, 599, 288, 139, 64, 27, 12 ])
    self.list_Rm.append([ 3601, 3361, 3001, 2881, 3481, 3241, 3481, 3121 ])
    self.list_Pm.append([ 1, 2, 5, 10, 25, 50, 128, 260 ])
    self.list_markers.append([ "o", "o", "o", "o", "o", "o", "o", "o" ])
    self.loadData_sim(
      sim_name = "Rm3000",
      list_sim_folders = [ "Pm1", "Pm2", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ],
      list_time_sat = [ 30, 30, 30, 30, 30, 30, 30, 30 ]
    )
    ## load keta = 127
    self.list_Re.append([ 73, 48, 27, 16 ])
    self.list_Rm.append([ 1851, 2452, 3465, 4000 ])
    self.list_Pm.append([ 25, 51, 128, 250 ])
    self.list_markers.append([ "v", "v", "v", "v" ])
    self.loadData_sim(
      sim_name = "keta",
      list_sim_folders = [ "Pm25", "Pm50", "Pm125", "Pm250" ],
      list_time_sat = [ 30, 30, 30, 30 ]
    )
  def loadData_sim(
      self,
      sim_name, list_sim_folders, list_time_sat
    ):
    for sim_index in range(len(list_sim_folders)):
      ## initialise list of data
      list_time = []
      list_E_ratio = []
      list_index_E_grow = []
      list_index_E_sat  = []
      ## load data
      funcLoadData(
        createFilepath([self.filepath_data, sim_name, "288", list_sim_folders[sim_index]]),
        0.01, np.inf, list_time_sat[sim_index],
        list_time, [],
        [], [], list_E_ratio,
        list_index_E_grow, list_index_E_sat, []
      )
      ## measure growth rate
      self.list_E_growth.append(
        funcMeasureExpFit(
          data_x = list_time[0],
          data_y = list_E_ratio[0],
          index_start_fit = list_index_E_grow[0][0],
          index_end_fit   = list_index_E_grow[0][1],
          bool_return_str = False
        )
      )
      ## measure saturation ratio
      self.list_E_sat.append(
        np.mean(
          list_E_ratio[0][
            list_index_E_sat[0][0] : list_index_E_sat[0][1]
          ]
        )
      )
  def plotGrowth(self):
    ## create figure
    fig, axs = plt.subplots(3, 1, figsize=(6*1.1, 3.5*3*1.1))
    fig.subplots_adjust(hspace=0.25)
    ## plot simulation data points
    for sim_index in range(len(self.list_Pm)):
      ## check megnetic energy grows
      if self.list_E_growth[sim_index] is None: continue
      ## plot data
      axs[0].plot(
        self.list_Re[sim_index],
        self.list_E_growth[sim_index],
        marker = self.list_markers[sim_index],
        color = self.list_color[sim_index],
        markeredgecolor = "black"
      )
      axs[1].plot(
        self.list_Rm[sim_index],
        self.list_E_growth[sim_index],
        marker = self.list_markers[sim_index],
        color = self.list_color[sim_index],
        markeredgecolor = "black"
      )
      axs[2].plot(
        self.list_Pm[sim_index],
        self.list_E_growth[sim_index],
        marker = self.list_markers[sim_index],
        color = self.list_color[sim_index],
        markeredgecolor = "black"
      )
    ## add legend: simulation marker
    addLegend(
      ax = axs[0],
      artists = [ "s", "D", "o", "v" ],
      colors  = [ "black" ] * 4,
      legend_labels  = [ r"Re $= 10$", r"Re $\approx 450$", r"Rm $\approx 3300$", r"$k_{\eta, \rm{theory}} \approx 125$" ],
      place_pos = 0,
      ms = 6,
      spacing = 0.75,
      ncol = 1
    )
    axs[0].text(
      0.05, 0.9,
      r"Re $< 100$", color="blue",
      va="top", ha="left", transform=axs[0].transAxes, fontsize=14
    )
    axs[0].text(
      0.05, 0.8,
      r"Re $> 100$", color="red",
      va="top", ha="left", transform=axs[0].transAxes, fontsize=14
    )
    ## label axis
    axs[0].set_xlabel(r"Re")
    axs[1].set_xlabel(r"Rm")
    axs[2].set_xlabel(r"Pm")
    axs[0].set_ylabel(r"$\Gamma_k(1/t_{\rm{eddy}})$")
    axs[1].set_ylabel(r"$\Gamma_k(1/t_{\rm{eddy}})$")
    axs[2].set_ylabel(r"$\Gamma_k(1/t_{\rm{eddy}})$")
    ## adjust axis scale
    axs[0].set_xscale("log")
    axs[1].set_xscale("log")
    axs[2].set_xscale("log")
    ## use integers for tick labels
    axs[0].xaxis.set_major_formatter(ScalarFormatter())
    axs[1].xaxis.set_major_formatter(ScalarFormatter())
    axs[1].xaxis.set_minor_formatter(NullFormatter())
    axs[2].xaxis.set_major_formatter(ScalarFormatter())
    ## save plot
    fig_name = "fig_energy_growth.pdf"
    fig_filepath = createFilepath([self.filepath_plot, fig_name])
    plt.savefig(fig_filepath)
    print("\t> Figure saved: " + fig_name)
  def plotSat(self):
    ## create figure
    fig, axs = plt.subplots(3, 1, figsize=(6*1.1, 3.5*3*1.1))
    fig.subplots_adjust(hspace=0.25)
    ## annotate sub-axis
    ## plot simulation data points
    for sim_index in range(len(self.list_Pm)):
      ## check megnetic energy grows
      if isinstance(self.list_E_growth[sim_index], str): continue
      ## plot data
      axs[0].plot(
        self.list_Re[sim_index],
        self.list_E_sat[sim_index],
        marker = self.list_markers[sim_index],
        color = self.list_color[sim_index],
        markeredgecolor = "black"
      )
      axs[1].plot(
        self.list_Rm[sim_index],
        self.list_E_sat[sim_index],
        marker = self.list_markers[sim_index],
        color = self.list_color[sim_index],
        markeredgecolor = "black"
      )
      axs[2].plot(
        self.list_Pm[sim_index],
        self.list_E_sat[sim_index],
        marker = self.list_markers[sim_index],
        color = self.list_color[sim_index],
        markeredgecolor = "black"
      )
    ## add legend: simulation marker
    addLegend(
      ax = axs[0],
      artists = [ "s", "D", "o", "v" ],
      colors  = [ "black" ] * 4,
      legend_labels  = [ r"Re $= 10$", r"Re $= 500$", r"Rm $= 3000$", r"$k_{\eta, \rm{theory}} \approx 127$" ],
      place_pos = 0,
      ms = 6,
      spacing = 0.75,
      ncol = 1
    )
    axs[0].text(
      0.95, 0.9,
      r"Re $< 100$", color="blue",
      va="top", ha="right", transform=axs[0].transAxes, fontsize=14
    )
    axs[0].text(
      0.95, 0.8,
      r"Re $> 100$", color="red",
      va="top", ha="right", transform=axs[0].transAxes, fontsize=14
    )
    ## label axis
    axs[0].set_xlabel(r"Re")
    axs[1].set_xlabel(r"Rm")
    axs[2].set_xlabel(r"Pm")
    axs[0].set_ylabel(r"$(E_{\rm{mag}} / E_{\rm{kin}})_{\rm{sat}}$")
    axs[1].set_ylabel(r"$(E_{\rm{mag}} / E_{\rm{kin}})_{\rm{sat}}$")
    axs[2].set_ylabel(r"$(E_{\rm{mag}} / E_{\rm{kin}})_{\rm{sat}}$")
    ## adjust axis scale
    axs[0].set_xscale("log")
    axs[1].set_xscale("log")
    axs[2].set_xscale("log")
    ## use integers for tick labels
    axs[0].xaxis.set_major_formatter(ScalarFormatter())
    axs[1].xaxis.set_major_formatter(ScalarFormatter())
    axs[1].xaxis.set_minor_formatter(NullFormatter())
    axs[2].xaxis.set_major_formatter(ScalarFormatter())
    axs[0].yaxis.set_major_formatter(ScalarFormatter())
    axs[1].yaxis.set_major_formatter(ScalarFormatter())
    axs[2].yaxis.set_major_formatter(ScalarFormatter())
    ## save plot
    fig_name = "fig_energy_sat.pdf"
    fig_filepath = createFilepath([self.filepath_plot, fig_name])
    plt.savefig(fig_filepath)
    print("\t> Figure saved: " + fig_name)

def funcPrintForm(
    distribution,
    num_digits = 2
  ):
  str_median = ("{0:.2g}").format(np.percentile(distribution, 50))
  # if "." not in str_median:
  #     # print(np.percentile(distribution, 50))
  #     str_median = ("{0:.1f}").format(np.percentile(distribution, 50))
  if "." in str_median:
    if ("0" == str_median[0]) and (len(str_median.split(".")[1]) == 1):
      str_median = ("{0:.2f}").format(np.percentile(distribution, 50))
    num_decimals = len(str_median.split(".")[1])
  else: num_decimals = 2
  str_low  = ("-{0:."+str(num_decimals)+"f}").format(np.percentile(distribution, 50) - np.percentile(distribution, 16))
  str_high = ("+{0:."+str(num_decimals)+"f}").format(np.percentile(distribution, 84) - np.percentile(distribution, 50))
  return r"${}_{}^{}$".format(
    str_median,
    "{" + str_low  + "}",
    "{" + str_high + "}"
  )

def funcPrintSimStats(
    filepath_data,
    time_start, time_end, time_sat
  ):
  ## initialise list of data
  list_time = []
  list_Mach = []
  list_E_ratio = []
  list_index_E_grow = []
  list_index_E_sat  = []
  list_index_Mach   = []
  ## load data
  funcLoadData(
    filepath_data,
    time_start, time_end, time_sat,
    list_time, list_Mach,
    [], [], list_E_ratio,
    list_index_E_grow, list_index_E_sat, list_index_Mach
  )
  ## collect statistics
  mach = funcPrintForm(
    list_Mach[0][list_index_Mach[0][0] : list_index_Mach[0][1]]
  )
  growth_rate = funcMeasureExpFit(
    data_x = list_time[0],
    data_y = list_E_ratio[0],
    index_start_fit = list_index_E_grow[0][0],
    index_end_fit   = list_index_E_grow[0][1],
    bool_return_str = True
  )
  if "\pm" in growth_rate:
    E_ratio_sat = funcPrintForm(
      list_E_ratio[0][list_index_E_sat[0][0] : list_index_E_sat[0][1]]
    )
  else: E_ratio_sat = "--"
  ## print statistics
  print("& " + mach + " & " + growth_rate  + " & " + E_ratio_sat)

def funcPrintStats(filepath_data):
  ## Re = 10
  sim_folders_Re10 = [ "Pm25", "Pm50", "Pm125", "Pm250" ]
  list_time_sat = [ 30, 60, 30, 30 ]
  print("Statistics for Re10...")
  for sim_index in range(len(sim_folders_Re10)):
    funcPrintSimStats(
      createFilepath([filepath_data, "Re10", "288", sim_folders_Re10[sim_index]]),
      0.01, np.inf, list_time_sat[sim_index]
    )
  ## Re = 500
  sim_folders_Re500 = [ "Pm1", "Pm2", "Pm4" ]
  list_time_sat = [ 50, 50, 50 ]
  print("Statistics for Re500")
  for sim_index in range(len(sim_folders_Re500)):
    funcPrintSimStats(
      createFilepath([filepath_data, "Re500", "288", sim_folders_Re500[sim_index]]),
      0.01, np.inf, list_time_sat[sim_index]
    )
  ## Rm = 3000
  sim_folders_Rm3000 = [ "Pm1", "Pm2", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
  list_time_sat = [ 30, 30, 30, 30, 30, 30, 30, 30 ]
  print("Statistics for Rm3000")
  for sim_index in range(len(sim_folders_Rm3000)):
    funcPrintSimStats(
      createFilepath([filepath_data, "Rm3000", "288", sim_folders_Rm3000[sim_index]]),
      0.01, np.inf, list_time_sat[sim_index]
    )
  ## keta
  sim_folders_keta = [ "Pm25", "Pm50", "Pm125", "Pm250" ]
  list_time_sat = [ 30, 30, 30, 30 ]
  print("Statistics for keta..")
  for sim_index in range(len(sim_folders_keta)):
    funcPrintSimStats(
      createFilepath([filepath_data, "keta", "288", sim_folders_keta[sim_index]]),
      0.01, np.inf, list_time_sat[sim_index]
    )


## ###############################################################
## DEFINE MAIN PROGRAM
## ###############################################################
def main():
  ## ####################
  ## INITIALISE VARIABLES
  ## ####################
  ## filepath to data
  filepath_data = "/Users/dukekriel/Documents/Studies/TurbulentDynamo/data/sub_sonic"
  ## filepath to figures
  filepath_plot = "/Users/dukekriel/Documents/Studies/TurbulentDynamo/figures/sub_sonic"
  print("Saving figures in: " + filepath_plot)

  ## #####################
  ## PLOT ENERGY EVOLUTION
  ## #####################
  funcPlotTurb(filepath_data, filepath_plot)

  # ## ##################
  # ## MEASURE STATISTICS
  # ## ##################
  # print(" ")
  # funcPlotStuff(filepath_data, filepath_plot)

  # ## ##################
  # ## MEASURE STATISTICS
  # ## ##################
  # print(" ")
  # funcPrintStats(filepath_data)


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
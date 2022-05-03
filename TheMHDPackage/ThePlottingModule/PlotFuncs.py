## START OF MODULE


## ###############################################################
## DEPENDENCIES: REQUIRED MODULES
## ###############################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmasher as cmr
  ## https://cmasher.readthedocs.io/user/introduction.html#colormap-overview
  ## cmr sequential maps: tropical, ocean, arctic, bubblegum, lavender
  ## cmr diverging maps: iceburn, wildfire, fusion

from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, NullFormatter
from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

from TheUsefulModule import WWFnF
from TheFittingModule import FitDistro


## ###############################################################
## ANIMATIONS
## ###############################################################
def aniEvolution(
    filepath_frames,
    filepath_ani_movie,
    input_name,
    output_name,
    bool_hide_updates = False
  ):
  ''' Animate the plot frames.
  '''
  if not(bool_hide_updates): print("Animating plots...")
  ## create filepath to where plots are saved
  filepath_input = WWFnF.createFilepath([
    filepath_frames,
    input_name
  ])
  ## create filepath to where animation should be saved
  filepath_output = WWFnF.createFilepath([
    filepath_ani_movie,
    output_name
  ])
  ## animate the plot-frames
  os.system(
    "ffmpeg -y -start_number 0 -i " + filepath_input +
    ("" if not(bool_hide_updates) else " -loglevel quiet") +
    " -vb 40M -framerate 40 -vf scale=1440:-1 -vcodec mpeg4 " + filepath_output
  )
  if not(bool_hide_updates):
    print("Animation finished: " + filepath_output)
    print(" ")


## ###############################################################
## PLOTTING HELPER FUNCTIONS
## ###############################################################
def plotColorbar(mappable, cbar_label=None):
  ''' from: https://joseph-long.com/writing/colorbars/
  '''
  last_axes = plt.gca()
  ax = mappable.axes
  fig = ax.figure
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.1)
  cbar = fig.colorbar(mappable, cax=cax)
  cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")
  plt.sca(last_axes)
  return cbar

class MidpointNormalize(colors.Normalize):
  def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
    self.midpoint = midpoint
    colors.Normalize.__init__(self, vmin, vmax, clip)
  def __call__(self, value, clip=None):
    x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
    return np.ma.masked_array(np.interp(value, x, y))

class MidPointLogNorm(colors.LogNorm):
  ## https://stackoverflow.com/questions/48625475/python-shifted-logarithmic-colorbar-white-color-offset-to-center
  def __init__(
      self,
      midpoint = None,
      vmin = None,
      vmax = None,
      clip = False
    ):
    colors.LogNorm.__init__(
      self,
      vmin = vmin,
      vmax = vmax,
      clip = clip
    )
    self.midpoint = midpoint
  def __call__(
      self,
      value,
      clip = None
    ):
    x = [
      np.log(self.vmin),
      np.log(self.midpoint),
      np.log(self.vmax)
    ]
    y = [0, 0.5, 1]
    return np.ma.masked_array(np.interp(np.log(value), x, y))

class FixLogAxis():
  ''' Makes sure that the axis' domain range spans at least an order of magnitude.
  '''
  def __init__(
      self,
      ax,
      bool_fix_x_axis = False,
      bool_fix_y_axis = False
    ):
    ## save figure axis
    self.ax = ax
    ## adjust the x-axis
    if bool_fix_x_axis:
      tmp_min, tmp_max = ax.get_xlim()
      self.fix(
        tmp_axis    = ax.xaxis,
        tmp_min     = tmp_min,
        tmp_max     = tmp_max,
        bool_x_axis = True
      )
    ## adjust the y-axis
    if bool_fix_y_axis:
      tmp_min, tmp_max = ax.get_ylim()
      self.fix(
        tmp_axis    = ax.yaxis,
        tmp_min     = tmp_min,
        tmp_max     = tmp_max,
        bool_y_axis = True
      )
  def fix(
      self,
      tmp_axis, tmp_min, tmp_max,
      bool_x_axis=False, bool_y_axis=False
    ):
    ## ######################
    ## CHANGE THE AXIS DOMAIN
    ## #######
    if abs(np.log10(tmp_min) - np.log10(tmp_max)) < 0.5:
      ## centre the points and make sure the range wider
      new_min = 10**( (np.log10(tmp_min) + np.log10(tmp_max))/2 - 0.25 )
      new_max = 10**( (np.log10(tmp_min) + np.log10(tmp_max))/2 + 0.25 )
      if bool_x_axis: self.ax.set_xlim([new_min, new_max])
      elif bool_y_axis: self.ax.set_ylim([new_min, new_max])
      tmp_min = new_min
      tmp_max = new_max
    # elif 1.1 > abs(np.log10(tmp_min) - np.log10(tmp_max)) > 1:
    #     new_min = 10**( np.log10(tmp_min) - 0.1 )
    #     new_max = 10**( np.log10(tmp_max) + 0.1 )
    #     if bool_x_axis: self.ax.set_xlim([new_min, new_max])
    #     elif bool_y_axis: self.ax.set_ylim([new_min, new_max])
    #     tmp_min = new_min
    #     tmp_max = new_max
    ## #####################
    ## CHANGE THE AXIS TICKS
    ## #######
    # print(abs(np.log10(tmp_min) - np.log10(tmp_max))) # TODO: different crit thresh for x and y-axis
    if abs(np.log10(tmp_min) - np.log10(tmp_max)) < 0.75:
      ## show major and minor axis tick labels
      for axis in [tmp_axis]:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(ScalarFormatter())
    else:
      ## only show major axis tick labels
      for axis in [tmp_axis]:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())


## ###############################################################
## ADD TO PLOTS
## ###############################################################
def plotLabelBox(
    fig, ax,
    list_fig_labels = [],
    xpos            = 0.05,
    ypos            = 0.95,
    box_alignment   = (0.0, 1.0), # align: left, upper
    alpha           = 0.5,
    fontsize        = 16
  ):
  if len(list_fig_labels) == 0: return
  list_labels_str = [
    TextArea(tmp_label)
    for tmp_label in list_fig_labels
  ]
  texts_vbox = VPacker(
    children = list_labels_str,
    pad = 2.5,
    sep = 5.0,
  )
  ann = AnnotationBbox(
    texts_vbox,
    fontsize      = fontsize,
    xy            = (xpos, ypos),
    xycoords      = ax.transAxes,
    box_alignment = box_alignment,
    bboxprops     = dict(color="grey", facecolor="white", boxstyle="round", alpha=alpha)
  )
  ann.set_figure(fig)
  fig.artists.append(ann)

def showDataOutsideAxis(ax, data_x, data_y, range_y):
    ## loop over data points
    for x, y in zip(data_x, data_y):
      ## if the data point lies outside the figure bounds 
      if (y < range_y[0]) or (y > range_y[1]):
        ## indicate point x-value
        ax.axvline(x=x, ls="-", color="black", alpha=0.1)

def insetPlot(
    ax, data_x, data_y,
    ax_inset_bounds = [
      0.0, 1.0, 1.0, 0.5
    ],
    label_x    = None,
    label_y    = None,       
    range_y    = None,
    bool_log_y = False,
    fs = 20
  ):
  ## create inset axis
  ax_inset = ax.inset_axes(ax_inset_bounds)
  ax_inset.xaxis.tick_top()
  ## plot data
  ax_inset.plot(data_x, data_y, "k.")
  ## add axis label
  ax_inset.set_xlabel(label_x, fontsize=fs)
  ax_inset.set_ylabel(label_y, fontsize=fs)
  ax_inset.xaxis.set_label_position("top") 
  ## set y-axis range
  if range_y is not None:
    ax_inset.set_ylim(range_y)
    ## check if any points are outside of the figure window
    showDataOutsideAxis(ax_inset, data_x, data_y, range_y)
  ## transform y-axis range
  if bool_log_y:
    ax_inset.set_yscale("log")
    FixLogAxis(ax_inset, bool_fix_y_axis=True)
  ## return inset axis
  return ax_inset

def insetPDF(
    ax, data,
    ax_inset_bounds = [
      0.0, 1.0, 1.0, 0.5
    ],
    label_x    = None,
    label_y    = None,       
    range_y    = None,
    bool_log_y = False
  ):
  ## create inset axis
  ax_inset = ax.inset_axes(ax_inset_bounds)
  ax_inset.xaxis.tick_top()
  ## plot data
  plotPDF(ax_inset, data)
  ## add axis label
  ax_inset.set_xlabel(label_x)
  ax_inset.set_ylabel(label_y)
  ax_inset.xaxis.set_label_position("top")
  ## return inset axis
  return ax_inset

def addLegend(
    ax, artists, legend_labels,
    # place_pos = 1,
    colors   = [ "k" ],
    loc      = "upper right",
    bbox     = (1.0, 1.0),
    ms       = 8,
    lw       = 2,
    fontsize = None,
    title    = None,
    ncol     = None,
    tpad     = None,
    rspacing = None,
    cspacing = None,
    labelcolor = None
  ):
  ## check that the inputs are the correct length
  if len(artists) < len(legend_labels): artists.extend( artists[0] * len(legend_labels) )
  if len(colors)  < len(legend_labels): colors.extend( colors[0] * len(legend_labels) )
  ## useful lists
  list_markers = [ ".", "o", "s", "D", "^", "v" ] # list of marker styles
  list_lines   = [
    "-", "--", "-.", ":",
    (0, (5, 1.5)),
    (0, (2, 1.5)),
    (0, (7, 3, 4, 3, 4, 3)),
    (0, (6, 3, 1, 3, 1, 3, 1, 3))
  ] # list of line styles
  ## iniialise list of artists for legend
  legend_artists = []
  ## create legend artists
  for artist, color in  zip(artists, colors):
    ## if the artist is a marker
    if artist in list_markers:
      legend_artists.append( 
        Line2D(
          [0], [0],
          marker = artist,
          color  = color,
          linewidth = 0,
          markeredgecolor = "white",
          markersize = ms
        )
      )
    ## if the artist is a line
    elif artist in list_lines:
      legend_artists.append(
        Line2D(
          [0], [0],
          linestyle = artist,
          color     = color,
          linewidth = lw
        )
      )
    ## otherwise throw an error
    else: raise Exception("Artist '{}' is not valid.".format(artist))
  ## draw the legend
  legend = ax.legend(
    legend_artists, legend_labels, title=title,
    loc = loc,
    bbox_to_anchor = bbox,
    handletextpad  = 0.5 if tpad is None else tpad,
    labelspacing   = 0.5 if rspacing is None else rspacing,
    columnspacing  = 0.5 if cspacing is None else cspacing,
    fontsize = 16 if fontsize is None else fontsize,
    ncol = 1 if ncol is None else ncol,
    labelcolor = "black" if labelcolor is None else labelcolor
  )
  ## add legend
  ax.add_artist(legend)


## ###############################################################
## PLOT 2D FIELDS
## ###############################################################
def plot2DField(
    ## 2D field data
    field, 
    ## filepath where figure will be saved
    filepath_fig,
    ## pass a figure object (optional)
    fig = None,
    ax  = None,
    ## colormap details (optional)
    bool_colorbar = True,
    cmap_str      = "cmr.arctic",
    cbar_label    = None,
    cbar_lims     = None,
    bool_mid_norm = False,
    mid_norm      = 0,
    ## label figure (optional)
    list_fig_labels  = None,
    bool_hide_labels = False
  ):
  ## check that a figure object has been passed
  if (fig is None) or (ax is None):
    fig, ax = fig, ax = plt.subplots(constrained_layout=True)
  ## plot slice
  im_obj = ax.imshow(
    field,
    extent = [-1,1,-1,1],
    cmap   = plt.get_cmap(cmap_str),
    clim   = cbar_lims if (cbar_lims is not None) else None,
    norm   = MidpointNormalize(
      midpoint = mid_norm,
      vmin = cbar_lims[0] if (cbar_lims is not None) else None,
      vmax = cbar_lims[1] if (cbar_lims is not None) else None
    ) if bool_mid_norm else None
  )
  ## add colorbar
  if bool_colorbar:
    plotColorbar(im_obj, cbar_label)
  ## add labels
  if list_fig_labels is not None:
    plotLabelBox(fig, ax, list_fig_labels)
  ## add axis labels
  if not bool_hide_labels:
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"])
    ax.set_yticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"])
  ## save figure
  plt.savefig(filepath_fig)
  ## clear figure and axis
  fig.artists.clear()
  ax.clear()


## ###############################################################
## PLOT DISTRIBUTION STATISTICS
## ###############################################################
def plotPDF(
    ax, vals,
    num_bins = 10,
    label    = "",
    orientation = "vertical",
    fill_color  = None,
    cmap_str    = "cmr.tropical",
    num_cols    = 1,
    col_index   = 0,
    bool_log    = False,
    bool_set_axis_lim = False
  ):
  if len(vals) > 3:
    ## calculate density of data
    dens, bin_edges = np.histogram(vals, bins=num_bins, density=True)
    ## normalise density
    dens_norm = np.append(0, dens / dens.sum())
    ## extract colours from Cmasher's colormap
    cmasher_colormap = plt.get_cmap(cmap_str, num_cols)
    my_colormap = cmasher_colormap(np.linspace(0, 1, num_cols))
    ## log bins
    if bool_log:
      ## create bin locations
      _, bins = np.histogram(vals, bins=num_bins)
      bin_edges = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
      if "v" in orientation: ax.set_xscale("log")
      else: ax.set_yscale("log")
    ## fill PDF with colour
    if "v" in orientation:
      ## fill PDF
      ax.fill_between(
        bin_edges,
        dens_norm, 
        step  = "pre",
        alpha = 0.2,
        color = my_colormap[col_index] if fill_color is None else fill_color
      )
      ## plot PDF lines
      ax.plot(
        bin_edges,
        dens_norm, 
        drawstyle = "steps",
        label = label,
        color = my_colormap[col_index] if fill_color is None else fill_color
      )
      ## set x-lim range
      if bool_set_axis_lim:
        bin_step = bin_edges[1] - bin_edges[0]
        ax.set_xlim([
          min(bin_edges[1:]) - bin_step/2,
          max(bin_edges[1:]) + bin_step/2
        ])
    else:
      ## fill PDF
      ax.fill_betweenx(
        bin_edges,
        dens_norm, 
        step  = "post",
        alpha = 0.2,
        color = my_colormap[col_index] if fill_color is None else fill_color
      )
      ## plot PDF lines
      ax.plot(
        dens_norm, 
        bin_edges,
        drawstyle = "steps",
        label = label,
        color = my_colormap[col_index] if fill_color is None else fill_color
      )
      ## set y-lim range
      if bool_set_axis_lim:
        bin_step = bin_edges[1] - bin_edges[0]
        ax.set_ylim([
          min(bin_edges[1:]) - bin_step/2,
          max(bin_edges[1:]) + bin_step/2
        ])
      ## put y-axis tights on the right
      ax.yaxis.tick_right()

def plotHistogram(
    ax, vals,
    num_bins   = 10,
    sim_label  = "",
    fill_color = None,
    cmap_str   = "cmr.tropical",
    num_cols   = 1,
    col_index  = 0,
    alpha      = 0.4,
    bool_log   = False
  ):
  if len(vals) > 3:
    ## extract colours from Cmasher's colormap
    cmasher_colormap = plt.get_cmap(cmap_str, num_cols)
    my_colormap = cmasher_colormap(np.linspace(0, 1, num_cols))
    if bool_log:
      ## create bin locations
      _, bins = np.histogram(vals, bins=num_bins)
      logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
      ## plot bins
      ax.hist(
        vals,
        histtype = "step",
        bins  = logbins,
        label = sim_label,
        color = my_colormap[col_index] if fill_color is None else fill_color,
        fill  = True,
        alpha = alpha
      )
      ax.hist(
        vals,
        histtype = "step", 
        bins  = logbins,
        color = my_colormap[col_index] if fill_color is None else fill_color,
        fill  = False
      )
      ## log axis
      ax.set_xscale("log")
    else:
      ## stacked histogram
      ax.hist(
        vals,
        histtype = "step",
        bins  = num_bins,
        label = sim_label,
        color = my_colormap[col_index] if fill_color is None else fill_color,
        fill  = True,
        alpha = alpha
      )
      ax.hist(
        vals,
        histtype = "step", 
        bins  = num_bins,
        color = my_colormap[col_index] if fill_color is None else fill_color,
        fill  = False
      )

def plotErrorBar(
    ax, data_x, data_y,
    ## color inputs
    color     = None,
    num_cols  = 1,
    col_index = 0,
    cmap_str  = "cmr.tropical",
    ## marker info
    alpha  = 1,
    marker = "o",
    ms     = 8,
    label  = None
  ):
  ## if a color has not been defined
  if color is None:
    ## extract colours from Cmasher's colormap
    cmasher_colormap = plt.get_cmap(cmap_str, num_cols)
    my_colormap = cmasher_colormap(np.linspace(0, 1, num_cols))
    color = my_colormap[col_index]
  ## calculate quantiles: x-data
  bool_plot_x_error = False
  x_median = data_x
  if isinstance(data_x, (list, np.ndarray)):
    if len(data_x) > 1:
      bool_plot_x_error = True
      x_median = np.percentile(data_x, 50)
      x_p16    = np.percentile(data_x, 16)
      x_p84    = np.percentile(data_x, 84)
  ## calculate quantiles: y-data
  bool_plot_y_error = False
  y_median = data_y
  if isinstance(data_y, (list, np.ndarray)):
    if len(data_y) > 1:
      bool_plot_y_error = True
      y_median = np.percentile(data_y, 50)
      y_p16    = np.percentile(data_y, 16)
      y_p84    = np.percentile(data_y, 84)
  ## plot errorbar
  return ax.errorbar(
    x_median,
    y_median,
    xerr = np.vstack([
      x_median - x_p16,
      x_p84 - x_median
    ]) if bool_plot_x_error else None,
    yerr = np.vstack([
      y_median - y_p16,
      y_p84 - y_median
    ]) if bool_plot_y_error else None,
    color  = color,
    alpha  = alpha,
    fmt    = marker,
    label  = label,
    markersize=ms, elinewidth=2, linestyle="None", markeredgecolor="black", capsize=7.5, zorder=10
  )

def plotList1DDistributions(
    ax, list_x, list_y_group,
    color = "black",
    alpha = 1.0
  ):
  for group_index in range(len(list_x)):
    plot_y = list_y_group[group_index]
    plot_x = [ list_x[group_index] ] * len(plot_y)
    ax.plot(plot_x, plot_y, ".", color=color, alpha=alpha, zorder=7)

def plotList2DDistributions(
    ax, list_x_group, list_y_group,
    color = "black",
    alpha = 1.0
  ):
  for group_index in range(len(list_x_group)):
    plot_x = list_x_group[group_index]
    plot_y = list_y_group[group_index]
    ax.plot(plot_x, plot_y, ".", color=color, alpha=alpha, zorder=7)


## ###############################################################
## FITTING LINE TO MANY DISTRIBUTIONS
## ###############################################################
class CreateDistributionStatsLabel():
  def coef(param_list, num_digits=3):
    str_median = ("{0:."+str(num_digits)+"g}").format(np.percentile(param_list, 50))
    num_decimals = 2
    if "." in str_median:
      num_decimals = len(str_median.split(".")[1])
    str_low  = ("-{0:."+str(num_decimals)+"f}").format(np.percentile(param_list, 50) - np.percentile(param_list, 16))
    str_high = ("+{0:."+str(num_decimals)+"f}").format(np.percentile(param_list, 84) - np.percentile(param_list, 50))
    return r"${}_{}^{}$\;".format(
      str_median,
      "{" + str_low  + "}",
      "{" + str_high + "}"
    )
  def exp(param_list, num_digits=3):
    str_median = ("{0:."+str(num_digits)+"g}").format(np.percentile(param_list, 50))
    num_decimals = 2
    if "." in str_median:
      num_decimals = len(str_median.split(".")[1])
      if num_decimals < 2:
        num_decimals = 2
        str_median += "0"
    else:
      str_median += ".00"
    str_low  = ("-{0:."+str(num_decimals)+"f}").format(np.percentile(param_list, 50) - np.percentile(param_list, 16))
    str_high = ("+{0:."+str(num_decimals)+"f}").format(np.percentile(param_list, 84) - np.percentile(param_list, 50))
    return r"$^{}$".format(
      "{" + str_median +
        "_{" + str_low  + "}" +
        "^{" + str_high + "}" +
      "}"
    )
  def offset(param_list, num_digits=3):
    str_median = ("{0:."+str(num_digits)+"g}").format(np.percentile(param_list, 50))
    num_decimals = 2
    if "." in str_median:
      num_decimals = len(str_median.split(".")[1])
    str_low  = ("-{0:."+str(num_decimals)+"f}").format(np.percentile(param_list, 50) - np.percentile(param_list, 16))
    str_high = ("+{0:."+str(num_decimals)+"f}").format(np.percentile(param_list, 84) - np.percentile(param_list, 50))
    return r"${}_{}^{}$".format(
      str_median,
      "{" + str_low  + "}",
      "{" + str_high + "}"
    )

class CreateFunctionLabel():
  def __init__(
      self,
      list_params,
      func_label,
      var_str,
      num_digits = 3,
      bool_hide_coef = False
    ):
    self.list_params = list_params
    self.num_digits = num_digits
    self.var_str = var_str
    self.label = ""
    self.bool_hide_coef = bool_hide_coef
    if "PowerLawOffset".lower() in func_label.lower(): self.labelPowerLawOffset()
    elif "PowerLaw".lower() in func_label.lower(): self.labelPowerLaw()
    elif "LinearOffset".lower() in func_label.lower(): self.labelLinearOffset()
    elif "Linear".lower() in func_label.lower(): self.labelLinear()
    else: raise Exception("Undefined KDE function: can't create label for '{}'".format(func_label))
  def labelPowerLawOffset(self):
    str_coef = CreateDistributionStatsLabel.coef(self.list_params[0], self.num_digits)
    str_exp  = CreateDistributionStatsLabel.exp(self.list_params[1], self.num_digits)
    str_off  = CreateDistributionStatsLabel.offset(self.list_params[2], self.num_digits)
    if np.percentile(self.list_params[2], 50) > 0: str_sum_sep = r" $+$ "
    else: str_sum_sep = r" $-$ "
    ## save output string
    if self.bool_hide_coef: self.label = self.var_str + str_exp + str_sum_sep + str_off
    else: self.label = str_coef + self.var_str + str_exp + str_sum_sep + str_off
  def labelPowerLaw(self):
    str_coef = CreateDistributionStatsLabel.coef(self.list_params[0], self.num_digits)
    str_exp  = CreateDistributionStatsLabel.exp(self.list_params[1], self.num_digits)
    ## save output string
    if self.bool_hide_coef: self.label = self.var_str + str_exp
    else: self.label = str_coef + self.var_str + str_exp
  def labelLinearOffset(self):
    str_coef = CreateDistributionStatsLabel.coef(self.list_params[0], self.num_digits)
    str_off  = CreateDistributionStatsLabel.offset(self.list_params[1], self.num_digits)
    if np.percentile(self.list_params[1], 50) > 0: str_sum_sep = r" $+$ "
    else: str_sum_sep = r" $-$ "
    ## save output string
    if self.bool_hide_coef: self.label = self.var_str + str_sum_sep + str_off
    else: self.label = str_coef + self.var_str + str_sum_sep + str_off
  def labelLinear(self):
    str_coef = CreateDistributionStatsLabel.coef(self.list_params[0], self.num_digits)
    ## save output string
    if self.bool_hide_coef: self.label = self.var_str
    else: self.label = str_coef + self.var_str

def plotDistributionFit(
    ax, var_str, func_label,
    ## fitting parameters
    input_x, input_y,
    func_fit, func_plot,
    bool_resample = False,
    bool_log_fit  = False,
    list_func_indices_unlog = [],
    errors     = None,
    p0         = None,
    bounds     = None,
    num_resamp = 10**3,
    maxfev     = 10**3,
    ## label parameters
    bool_show_label = True,
    pre_label  = r"",
    num_digits = 3,
    bool_hide_coef = False,
    plot_domain = None,
    plot_args   = {
      "x":0.05,
      "y":0.95,
      "va":"top",
      "ha":"left", 
      "color":"black",
      "ls":"--",
      "bool_box":True
    },
    ## debug
    bool_debug = False
  ):
  ## ###########
  ## LOG X-INPUT
  ## ###
  ## if resampling from input distribution(s)
  if bool_resample:
    input_x = FitDistro.resampleFrom1DKDE(input_x, num_resamp=num_resamp)
  ## if fitting to a number of distributions
  if isinstance(input_x[0], (list, np.ndarray)) and len(list(input_x[0])) > 1:
    ## define fitting domain
    if plot_domain is None:
      plot_domain = np.linspace(
        min([ np.percentile(sub_list, 16) for sub_list in input_x ]), # smallest x-value
        max([ np.percentile(sub_list, 84) for sub_list in input_x ]), # largest x-value
        100
      )
    ## log transform
    if bool_log_fit:
      fit_x = [[
          np.log(x)
          if x > 0 else np.log(np.median(sub_list)) # make sure not to log 0
          for x in sub_list
        ] for sub_list in input_x
      ]
    ## no log transoform
    else: fit_x = input_x
  else:
    ## define fitting domain
    if plot_domain is None:
      plot_domain = np.linspace(min(input_x), max(input_x), 100)
    ## log transform
    if bool_log_fit:
      fit_x = [
        np.log(x)
        if x > 0 else np.log(np.median(input_x))
        for x in input_x
      ]
    ## no log transoform
    else: fit_x = input_x
  ## ###########
  ## LOG Y-INPUT
  ## ###
  ## if resampling from input distribution(s)
  if bool_resample:
    input_y = FitDistro.resampleFrom1DKDE(input_y, num_resamp=num_resamp)
  ## log transform
  if bool_log_fit:
    ## if fitting to a number of distributions
    if isinstance(input_y[0], (list, np.ndarray)) and len(list(input_y[0])) > 1:
      fit_y = [[
          np.log(y)
          if y > 0
          else np.log(np.median(sub_list)) # make sure not to log 0
          for y in sub_list
        ] for sub_list in input_y
      ]
    else:
      fit_y = [
        np.log(y)
        if y > 0 else np.log(np.median(input_y)) # make sure not to log 0
        for y in input_y
      ]
  ## no log transform
  else: fit_y = input_y
  ## #######################
  ## FIT TO RESAMPLED POINTS
  ## #########
  list_fit_params, _ = FitDistro.fitToDistributions(
    input_x = fit_x,
    input_y = fit_y,
    func    = func_fit,
    p0      = p0,
    bounds  = bounds,
    errors  = errors,
    maxfev  = maxfev
  )
  ## undo log-transform
  if bool_log_fit:
    for index_unlog in list_func_indices_unlog:
      list_fit_params[index_unlog] = np.exp(list_fit_params[index_unlog])
  ## initialise plot variables
  num_params = len(list_fit_params)
  ## calculate the median of the fitted parameters
  median_params = [
    np.median(list_fit_params[index_param])
    for index_param in range(num_params)
  ]
  ## ##########################
  ## PLOT FITS TO DISTRIBUTIONS
  ## #########
  ## plot all fitted lines
  if bool_debug:
    list_lines = [] # plot of each fit
    ## loop over each fit
    for fit_index in range(len(list_fit_params[0])):
      ## get the fitted parameters
      fit_params = [
        list_param[fit_index]
        for list_param in list_fit_params
      ]
      ## create plot artist
      list_lines.append(
        np.column_stack((
          plot_domain,
          func_plot(plot_domain, *fit_params)
        ))
      )
    ## plot all artists
    ax.add_collection(
      LineCollection(list_lines, linestyle="-", color=plot_args["color"], alpha=0.01),
      autolim = False # ignore these points when setting the axis limits
    )
  ## plot line of best fit
  ax.add_collection(
    LineCollection(
      [
        np.column_stack((
          plot_domain,
          func_plot(plot_domain, *median_params)
        ))
      ],
      linestyle=plot_args["ls"], lw=2, color=plot_args["color"], zorder=7
    ),
    autolim = False # ignore these  points when setting the axis limits
  )
  ## check the distribution of resampled points being fitted to
  if bool_debug:
    if isinstance(fit_x[0], list):
      plotList2DDistributions(ax, fit_x, fit_y, color="red", alpha=0.05)
    else: plotList1DDistributions(ax, fit_x, fit_y, color="red", alpha=0.05)
  ## #############
  ## ADD FIT LABEL
  ## #######
  if bool_show_label:
    ax.text(
      **FitDistro.returnDicWithoutKeys(plot_args, ["ls", "bool_box"]),
      s = pre_label + CreateFunctionLabel(
        list_params = list_fit_params,
        func_label  = func_label,
        var_str     = var_str,
        num_digits  = num_digits,
        bool_hide_coef = bool_hide_coef
      ).label,
      fontsize=16, transform=ax.transAxes,
      bbox = dict(
        facecolor = "white",
        edgecolor = "gray",
        boxstyle  = "round",
        alpha     = 0.85
      ) if plot_args["bool_box"] else None
    )
  ## return the fitted parameters
  return list_fit_params


## END OF MODULE
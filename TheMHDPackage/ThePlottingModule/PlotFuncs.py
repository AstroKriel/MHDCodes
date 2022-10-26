## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmasher as cmr
  ## https://cmasher.readthedocs.io/user/introduction.html#colormap-overview
  ## cmr sequential maps: tropical, ocean, arctic, bubblegum, lavender
  ## cmr diverging maps: iceburn, wildfire, fusion

from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox
from mpl_toolkits.axes_grid1 import make_axes_locatable

from TheUsefulModule import WWFnF
from ThePlottingModule import TheMatplotlibStyler


## ###############################################################
## ANIMATIONS
## ###############################################################
def aniEvolution(
    filepath_frames, filepath_movie,
    input_name, output_name
  ):
  ''' Animate plot frames into .mp4 video.
  '''
  ## create filepath to where plots are saved
  filepath_input = WWFnF.createFilepath([
    filepath_frames,
    input_name
  ])
  ## create filepath to where animation should be saved
  filepath_output = WWFnF.createFilepath([
    filepath_movie,
    output_name
  ])
  ## animate the plot-frames
  os.system(f"ffmpeg -y -start_number 0 -i {filepath_input} -loglevel quiet -vb 40M -framerate 40 -vf scale=1440:-1 -vcodec mpeg4 {filepath_output}")
  print("Saved animation:", filepath_output)
  print(" ")

def initFigureGrid(fig_aspect_ratio, num_rows, num_cols, fig_scale=1.0):
  fig = plt.figure(
    constrained_layout = True,
    figsize            = (
      fig_scale * fig_aspect_ratio[1] * num_cols,
      fig_scale * fig_aspect_ratio[0] * num_rows
  ))
  fig_grid = GridSpec(num_rows, num_cols, figure=fig)
  return fig, fig_grid


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


## ###############################################################
## ADD TO PLOTS
## ###############################################################
def addLinearAxisTicks(
    ax,
    bool_minor_ticks = False,
    bool_major_ticks = False,
    num_minor_ticks  = 10,
    num_major_ticks  = 10
  ):
  ## add minor axis ticks
  if bool_minor_ticks:
    y_minor = mpl.ticker.LinearLocator(numticks=num_minor_ticks)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
  ## add major axis ticks
  if bool_major_ticks:
    y_major = mpl.ticker.LinearLocator(numticks=num_major_ticks)
    ax.yaxis.set_major_locator(y_major)

def addLogAxisTicks(
    ax,
    bool_minor_ticks = False,
    bool_major_ticks = False,
    num_minor_ticks  = 100,
    num_major_ticks  = 6
  ):
  ## add minor axis ticks
  if bool_minor_ticks:
    y_minor = mpl.ticker.LogLocator(
      base     = 10.0,
      subs     = np.arange(2, 10) * 0.1,
      numticks = num_minor_ticks
    )
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
  ## add major axis ticks
  if bool_major_ticks:
    y_major = mpl.ticker.LogLocator(
      base     = 10.0,
      numticks = num_major_ticks
    )
    ax.yaxis.set_major_locator(y_major)

def plotBoxOfLabels(
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
    TextArea(tmp_label, textprops={"fontsize":fontsize})
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
    bboxprops     = dict(color="grey", facecolor="white", boxstyle="round", alpha=alpha, zorder=10)
  )
  ann.set_figure(fig)
  fig.artists.append(ann)

def insetPlot(
    ax, data_x, data_y,
    ax_inset_bounds = [ 0.0, 1.0, 1.0, 0.5 ],
    label_x    = None,
    label_y    = None,       
    range_y    = None,
    bool_log_y = False,
    fontsize   = 20
  ):
  ## create inset axis
  ax_inset = ax.inset_axes(ax_inset_bounds)
  ax_inset.xaxis.tick_top()
  ## plot data
  ax_inset.plot(data_x, data_y, "k.")
  ## add axis label
  ax_inset.set_xlabel(label_x, fontsize=fontsize)
  ax_inset.set_ylabel(label_y, fontsize=fontsize)
  ax_inset.xaxis.set_label_position("top")
  ## return inset axis
  return ax_inset

def insetPDF(
    ax, data,
    ax_inset_bounds = [ 0.0, 1.0, 1.0, 0.5 ],
    label_x = None,
    label_y = None
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
    ax, list_artists, list_legend_labels,
    list_marker_colors = [ "k" ],
    label_color = "black",
    loc         = "upper right",
    bbox        = (1.0, 1.0),
    ms          = 8,
    lw          = 2,
    title       = None,
    ncol        = 1,
    rspacing    = 0.5,
    cspacing    = 0.5,
    fontsize    = 16,
  ):
  ## helper function
  def checkListLength(list_input, list_ref):
    if len(list_input) < len(list_ref):
      list_input.extend( list_input[0] * (len(list_ref)-len(list_input)) )
  ## check that the inputs are the correct length
  checkListLength(list_artists, list_legend_labels)
  checkListLength(list_marker_colors, list_legend_labels)
  ## lists of artists (marker and line styles) the user can choose from
  list_markers = [ ".", "o", "s", "D", "^", "v" ]
  list_lines   = [
    "-", "--", "-.", ":",
    (0, (5, 1.5)),
    (0, (2, 1.5)),
    (0, (7, 3, 4, 3, 4, 3)),
    (0, (6, 3, 1, 3, 1, 3, 1, 3))
  ]
  ## iniialise list of artists for legend
  list_legend_artists = []
  for artist, marker_color in zip(list_artists, list_marker_colors):
    ## if the artist is a marker
    if artist in list_markers:
      list_legend_artists.append( 
        Line2D(
          [0], [0],
          marker = artist,
          color  = marker_color,
          linewidth= 0, markeredgecolor= "white", markersize= ms
        )
      )
    ## if the artist is a line
    elif artist in list_lines:
      list_legend_artists.append(
        Line2D(
          [0], [0],
          linestyle = artist,
          color     = marker_color,
          linewidth = lw
        )
      )
    ## otherwise throw an error
    else: Exception(f"Artist '{artist}' is not valid.")
  ## draw the legend
  legend = ax.legend(
    list_legend_artists,
    list_legend_labels,
    frameon=False, facecolor=None,
    title          = title,
    loc            = loc,
    bbox_to_anchor = bbox,
    ncol           = ncol,
    borderpad      = 0.8,
    handletextpad  = 0.5,
    labelspacing   = rspacing,
    columnspacing  = cspacing,
    fontsize       = fontsize,
    labelcolor     = label_color
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
    plotBoxOfLabels(fig, ax, list_fig_labels)
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
        bins     = logbins,
        color    = my_colormap[col_index] if fill_color is None else fill_color,
        label    = sim_label,
        alpha    = alpha,
        fill     = True
      )
      ax.hist(
        vals,
        histtype = "step", 
        bins     = logbins,
        color    = my_colormap[col_index] if fill_color is None else fill_color,
        fill     = False
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
      list_params, func_label, var_str,
      num_digits     = 3,
      bool_hide_coef = False
    ):
    self.list_params    = list_params
    self.num_digits     = num_digits
    self.var_str        = var_str
    self.label          = ""
    self.bool_hide_coef = bool_hide_coef
    if "PowerLawOffset".lower() in func_label.lower(): self.labelPowerLawOffset()
    elif "PowerLaw".lower() in func_label.lower(): self.labelPowerLaw()
    elif "LinearOffset".lower() in func_label.lower(): self.labelLinearOffset()
    elif "Linear".lower() in func_label.lower(): self.labelLinear()
    else: Exception(f"Undefined KDE function: can't create label for '{func_label}'")

  def labelPowerLawOffset(self):
    str_coef = CreateDistributionStatsLabel.coef(self.list_params[0], self.num_digits)
    str_exp  = CreateDistributionStatsLabel.exp(self.list_params[1], self.num_digits)
    str_off  = CreateDistributionStatsLabel.offset(self.list_params[2], self.num_digits)
    if np.percentile(self.list_params[2], 50) > 0: str_sum_sep = r" $+$ "
    else: str_sum_sep = r" $-$ "
    if self.bool_hide_coef: self.label = self.var_str + str_exp + str_sum_sep + str_off
    else: self.label = str_coef + self.var_str + str_exp + str_sum_sep + str_off

  def labelPowerLaw(self):
    str_coef = CreateDistributionStatsLabel.coef(self.list_params[0], self.num_digits)
    str_exp  = CreateDistributionStatsLabel.exp(self.list_params[1], self.num_digits)
    if self.bool_hide_coef: self.label = self.var_str + str_exp
    else: self.label = str_coef + self.var_str + str_exp

  def labelLinearOffset(self):
    str_coef = CreateDistributionStatsLabel.coef(self.list_params[0], self.num_digits)
    str_off  = CreateDistributionStatsLabel.offset(self.list_params[1], self.num_digits)
    if np.percentile(self.list_params[1], 50) > 0: str_sum_sep = r" $+$ "
    else: str_sum_sep = r" $-$ "
    if self.bool_hide_coef: self.label = self.var_str + str_sum_sep + str_off
    else: self.label = str_coef + self.var_str + str_sum_sep + str_off

  def labelLinear(self):
    str_coef = CreateDistributionStatsLabel.coef(self.list_params[0], self.num_digits)
    if self.bool_hide_coef: self.label = self.var_str
    else: self.label = str_coef + self.var_str


## END OF LIBRARY
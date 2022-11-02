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

from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox
from mpl_toolkits.axes_grid1 import make_axes_locatable

## load user defined modules
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
  filepath_input  = f"{filepath_frames}/{input_name}"
  filepath_output = f"{filepath_movie}/{output_name}"
  ## animate the plot-frames
  os.system(f"ffmpeg -y -start_number 0 -i {filepath_input} -loglevel quiet -vb 40M -framerate 40 -vf scale=1440:-1 -vcodec mpeg4 {filepath_output}")
  print("Saved animation:", filepath_output)
  print(" ")


## ###############################################################
## WORKING WITH FIGURES AND COLORBARS
## ###############################################################
def createFigure_grid(
    num_rows         = 1,
    num_cols         = 1,
    fig_scale        = 1.0,
    fig_aspect_ratio = (4,6)
  ):
  fig = plt.figure(
    constrained_layout = True,
    figsize            = (
      fig_scale * fig_aspect_ratio[1] * num_cols,
      fig_scale * fig_aspect_ratio[0] * num_rows
  ))
  fig_grid = GridSpec(num_rows, num_cols, figure=fig)
  return fig, fig_grid

def saveFigure(fig, filepath_fig):
  print("Saving figure...")
  fig.set_tight_layout(True)
  fig.savefig(filepath_fig)
  plt.close(fig)
  print("Saved figure:", filepath_fig)

def createCmap(cmap_name, vmin=0.0, vmax=1.0):
  ## cmr cmaps span [0.0, 1.0], so pass (vmin, vmax) to subset the cmap
  return cmr.get_sub_cmap(cmap_name, vmin, vmax)


## ###############################################################
## ADD TO PLOTS
## ###############################################################
def addColorbar_fromCmap(
    fig, ax, cmap,
    vmin=0.0, vmax=1.0, label=None, fontsize=16, orientation="horizontal"
  ):
  norm = colors.Normalize(vmin=vmin, vmax=vmax)
  smap = ScalarMappable(cmap=cmap, norm=norm)
  div  = make_axes_locatable(ax)
  if "h" in orientation:   cax = div.new_vertical(size="5%", pad=0.1)
  elif "v" in orientation: cax = div.new_horizontal(size="5%", pad=0.1)
  else: raise Exception(f"ERROR: '{orientation}' is not a supported orientation!")
  fig.add_axes(cax)
  cbar = fig.colorbar(mappable=smap, cax=cax, orientation=orientation)
  if "h" in orientation:
    cax.set_title(label, fontsize=fontsize)
    cax.xaxis.set_ticks_position("top")
  else: cbar.ax.set_ylabel(label, rotation=-90, va="bottom", fontsize=fontsize)

def addColorbar_fromMappble(mappable, cbar_title=None):
  ''' from: https://joseph-long.com/writing/colorbars/
  '''
  ax_old  = plt.gca()
  ax_new  = mappable.axes
  fig     = ax_new.figure
  div     = make_axes_locatable(ax_new)
  ax_cbar = div.append_axes("right", size="5%", pad=0.1)
  cbar    = fig.colorbar(mappable, cax=ax_cbar)
  cbar.ax.set_ylabel(cbar_title, rotation=-90, va="bottom")
  plt.sca(ax_old)
  return cbar

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

def addBoxOfLabels(
    fig, ax, list_labels,
    ## default: (left, upper)
    xpos          = 0.05,
    ypos          = 0.95,
    box_alignment = (0.0, 1.0),
    alpha         = 0.5,
    fontsize      = 16
  ):
  if len(list_labels) == 0: return
  list_text_areas = [
    TextArea(label, textprops={"fontsize" : fontsize})
    for label in list_labels
  ]
  texts_vbox = VPacker(
    children = list_text_areas,
    pad      = 2.5,
    sep      = 5.0,
  )
  abox = AnnotationBbox(
    texts_vbox,
    fontsize      = fontsize,
    xy            = (xpos, ypos),
    xycoords      = ax.transAxes,
    box_alignment = box_alignment,
    bboxprops     = dict(color="grey", facecolor="white", boxstyle="round", alpha=alpha, zorder=10)
  )
  abox.set_figure(fig)
  fig.artists.append(abox)

def addInsetAxis(
    ax, 
    ax_inset_bounds = [ 0.0, 1.0, 1.0, 0.5 ],
    label_x         = None,
    label_y         = None,
    fontsize        = 20
  ):
  ## create inset axis
  ax_inset = ax.inset_axes(ax_inset_bounds)
  ax_inset.xaxis.tick_top()
  ## add axis label
  ax_inset.set_xlabel(label_x, fontsize=fontsize)
  ax_inset.set_ylabel(label_y, fontsize=fontsize)
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
  checkListLength(list_artists,       list_legend_labels)
  checkListLength(list_marker_colors, list_legend_labels)
  ## iniialise list of artists to draw
  list_legend_artists = []
  ## lists of artists the user can choose from
  list_markers = [ ".", "o", "s", "D", "^", "v" ]
  list_lines   = [
    "-", "--", "-.", ":",
    (0, (5, 1.5)),
    (0, (2, 1.5)),
    (0, (7, 3, 4, 3, 4, 3)),
    (0, (6, 3, 1, 3, 1, 3, 1, 3))
  ]
  for artist, marker_color in zip(list_artists, list_marker_colors):
    ## if the artist is a marker
    if artist in list_markers:
      list_legend_artists.append( 
        Line2D([0], [0], marker=artist, color=marker_color, linewidth=0, markeredgecolor="white", markersize=ms)
      )
    ## if the artist is a line
    elif artist in list_lines:
      list_legend_artists.append(
        Line2D([0], [0], linestyle=artist, color=marker_color, linewidth=lw)
      )
    ## otherwise throw an error
    else: raise Exception(f"ERROR: '{artist}' is not a valid valid.")
  ## create legend
  legend = ax.legend(
    list_legend_artists,
    list_legend_labels,
    frameon        = False,
    facecolor      = None,
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
  ## draw legend
  ax.add_artist(legend)


## ###############################################################
## PLOT 2D FIELDS
## ###############################################################
class MidpointNormalize(colors.Normalize):
  def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
    self.midpoint = midpoint
    colors.Normalize.__init__(self, vmin, vmax, clip)
  def __call__(self, value, clip=None):
    x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
    return np.ma.masked_array(np.interp(value, x, y))

def plot2DField(
    field, filepath_fig,
    fig              = None,
    ax               = None,
    bool_save        = False,
    bool_colorbar    = True,
    cmap_str         = "cmr.arctic",
    cbar_title       = None,
    cbar_lims        = None,
    bool_mid_norm    = False,
    mid_norm         = 0,
    list_labels      = None,
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
    addColorbar_fromMappble(im_obj, cbar_title)
  ## add labels
  if list_labels is not None:
    addBoxOfLabels(fig, ax, list_labels)
  ## add axis labels
  if not bool_hide_labels:
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"])
    ax.set_yticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"])
  ## save figure
  if bool_save:
    plt.savefig(filepath_fig)
    ## clear figure and axis
    fig.artists.clear()
    ax.clear()


## ###############################################################
## CREATE LABELS FOR STATISTICAL MODELS (TODO: update)
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
    else: raise Exception(f"Undefined KDE function: can't create label for '{func_label}'")

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
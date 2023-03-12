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
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

## load user defined modules
from ThePlottingModule.TheMatplotlibStyler import *
from TheUsefulModule import WWFnF, WWLists


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
    fig_aspect_ratio = (4, 6)
  ):
  fig = plt.figure(
    constrained_layout = True,
    figsize            = (
      fig_scale * fig_aspect_ratio[1] * num_cols,
      fig_scale * fig_aspect_ratio[0] * num_rows
  ))
  fig_grid = GridSpec(num_rows, num_cols, figure=fig)
  return fig, fig_grid

def saveFigure(fig, filepath_fig, bool_verbose=True):
  if not fig.get_constrained_layout():
    fig.set_tight_layout(True)
  fig.savefig(filepath_fig)
  plt.close(fig)
  if bool_verbose: print("Saved figure:", filepath_fig)

def createNorm(vmin=0.0, vmax=1.0):
  return colors.Normalize(vmin=vmin, vmax=vmax)

def createCmap(
    cmap_name,
    cmin=0.0, cmax=1.0,
    vmin=0.0, vmax=1.0
  ):
  ## cmaps span cmin=0.0 to cmax=1.0, so pass (cmin, cmax) to subset a cmap color-range
  cmap = cmr.get_sub_cmap(cmap_name, cmin, cmax)
  ## define value range of colorbar: [vmin, vmax]
  norm = createNorm(vmin, vmax)
  return cmap, norm


## ###############################################################
## PLOT DATA
## ###############################################################
def plotData_noAutoAxisScale(
    ax, x, y,
    color="k", ls=":", lw=1, label=None, alpha=1.0, zorder=1
  ):
  col = LineCollection(
    [ np.column_stack((x, y)) ],
    colors = color,
    ls     = ls,
    lw     = lw,
    label  = label,
    alpha  = alpha,
    zorder = zorder
  )
  ax.add_collection(col, autolim=False)

def plotErrorBar_1D(
    ax, x, array_y,
    color="k", marker="o", label=None
  ):
  array_y = [ elem for elem in array_y if elem is not None ]
  if len(array_y) < 5: return
  y_p16  = np.nanpercentile(array_y, 16)
  y_p50  = np.nanpercentile(array_y, 50)
  y_p84  = np.nanpercentile(array_y, 84)
  y_1sig = np.vstack([
    y_p50 - y_p16,
    y_p84 - y_p50
  ])
  ax.errorbar(
    x, y_p50,
    yerr  = y_1sig,
    color = color,
    fmt   = marker,
    label = label,
    markersize=7, markeredgecolor="black", capsize=7.5, elinewidth=2,
    linestyle="None", zorder=10
  )

def plotPDF(ax, list_data, color):
  list_dens, list_bin_edges = np.histogram(list_data, bins=10, density=True)
  list_dens_norm = np.append(0, list_dens / list_dens.sum())
  ax.fill_between(list_bin_edges, list_dens_norm, step="pre", alpha=0.2, color=color)
  ax.plot(list_bin_edges, list_dens_norm, drawstyle="steps", color=color)


## ###############################################################
## ADD TO PLOTS
## ###############################################################
def addSubplot_secondAxis(fig, grid_elem, shared_axis):
  ax0 = fig.add_subplot(grid_elem)
  if "x" in shared_axis.lower():
    ax1 = ax0.twinx()
  elif "y" in shared_axis.lower():
    ax1 = ax0.twiny()
  else: raise Exception("Error: user has not indicated which axis (x or y) should be shared:", shared_axis.lower())
  ax0.set_zorder(1) # default zorder is 0 for ax0 and ax1
  ax0.set_frame_on(False) # prevents ax0 from hiding ax1
  return [ ax0, ax1 ]

def addLegend(
    ax,
    loc      = "upper right",
    bbox     = (0.99, 0.99),
    ncol     = 1,
    fontsize = 20,
    alpha    = 0.85,
    zorder   = 10
  ):
  ax.legend(
    ncol=ncol, loc=loc, bbox_to_anchor=bbox, fontsize=fontsize,
    frameon=True, facecolor="white", edgecolor="grey", framealpha=alpha
  ).set_zorder(zorder)

def addLegend_joinedAxis(
    axs,
    loc      = "upper right",
    bbox     = (0.99, 0.99),
    ncol     = 1,
    fontsize = 20,
    alpha    = 0.85,
    zorder   = 10
  ):
  list_lines_ax0, list_labels_ax0 = axs[0].get_legend_handles_labels()
  list_lines_ax1, list_labels_ax1 = axs[1].get_legend_handles_labels()
  list_lines  = list_lines_ax0  + list_lines_ax1
  list_labels = list_labels_ax0 + list_labels_ax1
  axs[0].legend(
    list_lines,
    list_labels,
    ncol=ncol, loc=loc, bbox_to_anchor=bbox, fontsize=fontsize,
    frameon=True, facecolor="white", edgecolor="grey", framealpha=alpha
  ).set_zorder(zorder)

def addLegend_withBox(
    ax,
    loc      = "upper right",
    bbox     = (1.0, 1.0),
    ncol     = 1,
    alpha    = 0.85,
    fontsize = 20,
    zorder   = 10
  ):
  ax.legend(
    loc=loc, bbox_to_anchor=bbox, ncol=ncol, framealpha=alpha, fontsize=fontsize,
    frameon=True, facecolor="white", edgecolor="grey"
  ).set_zorder(zorder)

def addColorbar_fromCmap(
    fig, ax, cmap,
    norm=None, vmin=0.0, vmax=1.0,
    label=None, fontsize=16, orientation="horizontal", size=10
  ):
  if norm is None: norm = createNorm(vmin, vmax)
  smap   = ScalarMappable(cmap=cmap, norm=norm)
  ax_div = make_axes_locatable(ax)
  if   "h" in orientation: cax = ax_div.append_axes(position="top",   size=f"{size:.1f}%", pad="2%")
  elif "v" in orientation: cax = ax_div.append_axes(position="right", size=f"{size:.1f}%", pad="2%")
  else: raise Exception(f"Error: '{orientation}' is not a supported orientation!")
  # fig.add_axes(cax)
  cbar = fig.colorbar(mappable=smap, cax=cax, orientation=orientation)
  if "h" in orientation:
    cax.set_title(label, fontsize=fontsize)
    cax.xaxis.set_ticks_position("top")
  else: cbar.ax.set_ylabel(label, rotation=-90, va="bottom", fontsize=fontsize)

def addColorbar_fromMappble(mappable, cbar_title=None, size=7.5):
  ''' from: https://joseph-long.com/writing/colorbars/
  '''
  ax_old  = plt.gca()
  ax_new  = mappable.axes
  fig     = ax_new.figure
  div     = make_axes_locatable(ax_new)
  ax_cbar = div.append_axes("right", size=f"{size:.1f}%", pad="2%")
  cbar    = fig.colorbar(mappable, cax=ax_cbar)
  cbar.ax.set_ylabel(cbar_title, rotation=-90, va="bottom")
  plt.sca(ax_old)
  return cbar

def addAxisTicks_linear(
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

def addAxisTicks_log10(
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
    list_colors = [ "k" ],
    xpos        = 0.05,
    ypos        = 0.95,
    bbox        = (0.0, 1.0),
    alpha       = 0.5,
    fontsize    = 16,
  ):
  if len(list_labels) == 0: return
  WWLists.ensureListLength(list_colors, list_labels)
  list_text_areas = [
    TextArea(label, textprops={
      "fontsize" : fontsize,
      "color"    : list_colors[index_label]
    })
    for index_label, label in enumerate(list_labels)
    if label is not None
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
    box_alignment = bbox,
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
  ## check that the inputs are the correct length
  WWLists.ensureListLength(list_artists,       list_legend_labels)
  WWLists.ensureListLength(list_marker_colors, list_legend_labels)
  ## iniialise list of artists to draw
  list_legend_artists = []
  ## lists of artists the user can choose from
  list_markers = [ ".", "o", "s", "D", "^", "v" ]
  list_lines   = [ "-", "--", "-.", ":" ]
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
    else: raise Exception(f"Error: '{artist}' is not a valid valid.")
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

def labelDualAxis_sharedX(
    axs,
    label_left  = r"",
    label_right = r"",
    color_left  = "black",
    color_right = "black"
  ):
  axs[0].set_ylabel(label_left,  color=color_left)
  axs[1].set_ylabel(label_right, color=color_right, rotation=-90, labelpad=40)
  ## colour left/right axis-splines
  axs[0].tick_params(axis="y", colors=color_left)
  axs[1].tick_params(axis="y", colors=color_right)
  axs[1].spines["left" ].set_color(color_left)
  axs[1].spines["right"].set_color(color_right)

def labelDualAxis_sharedY(
    axs,
    label_bottom = r"",
    label_top    = r"",
    color_bottom = "black",
    color_top    = "black"
  ):
  axs[0].set_xlabel(label_bottom, color=color_bottom)
  axs[1].set_xlabel(label_top,    color=color_top, labelpad=20)
  ## colour bottom/top axis-splines
  axs[0].tick_params(axis="x", colors=color_bottom)
  axs[1].tick_params(axis="x", colors=color_top)
  axs[1].spines["bottom"].set_color(color_bottom)
  axs[1].spines["top"   ].set_color(color_top)


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
    data,
    fig              = None,
    ax               = None,
    filepath_fig     = None,
    cmap_str         = "cmr.arctic",
    cbar_title       = None,
    cbar_bounds      = None,
    bool_colorbar    = True,
    bool_label       = True
  ):
  ## check that a figure object has been passed
  if (fig is None) or (ax is None):
    fig, ax = fig, ax = plt.subplots(constrained_layout=True)
  ## plot slice
  im_obj = ax.imshow(
    data,
    extent = [-1.0, 1.0, -1.0, 1.0],
    cmap   = plt.get_cmap(cmap_str),
    norm   = colors.LogNorm(
      vmin = 0.9*np.min(data) if cbar_bounds is None else cbar_bounds[0],
      vmax = 1.1*np.max(data) if cbar_bounds is None else cbar_bounds[1]
    )
  )
  ## add colorbar
  if bool_colorbar:
    addColorbar_fromMappble(im_obj, cbar_title)
  ## add axis labels
  if bool_label:
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"])
    ax.set_yticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"])
  else: ax.set_axis_off()
  ## save figure
  if filepath_fig is not None:
    plt.savefig(filepath_fig)
    ## clear figure and axis
    fig.artists.clear()
    ax.clear()


## END OF LIBRARY
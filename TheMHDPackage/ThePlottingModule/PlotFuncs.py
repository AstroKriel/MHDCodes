## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import os, functools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import cmasher as cmr
  ## https://cmasher.readthedocs.io/user/introduction.html#colormap-overview
  ## cmr sequential maps: tropical, ocean, arctic, bubblegum, lavender
  ## cmr diverging maps: iceburn, wildfire, fusion

from scipy.stats import gaussian_kde
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

## load user defined modules
from ThePlottingModule.TheMatplotlibStyler import *
from TheUsefulModule import WWLists


## ###############################################################
## ANIMATING FRAMES
## ###############################################################
def aniEvolution(filepath_frames, filepath_movie, input_name, output_name):
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
    num_rows, num_cols,
    fig_scale        = 1.0,
    fig_aspect_ratio = (4, 6)
  ):
  fig = plt.figure(
    constrained_layout = True,
    figsize            = (
      fig_scale * fig_aspect_ratio[1] * num_cols,
      fig_scale * fig_aspect_ratio[0] * num_rows
  ))
  fig_grid = GridSpec(
    num_rows, num_cols,
    figure = fig,
    wspace = -1,
    hspace = -1
  )
  return fig, fig_grid

def saveFigure(fig, filepath_fig, bool_tight=True, bool_draft=False, bool_verbose=True):
  # if bool_tight and not(fig.get_constrained_layout()):
  fig.set_tight_layout(True)
  if bool_draft: dpi = 50
  else: dpi = 200
  fig.savefig(filepath_fig, dpi=dpi)
  plt.close(fig)
  if bool_verbose: print("Saved figure:", filepath_fig)

def createNorm(vmin=0.0, vmax=1.0, NormType=colors.Normalize):
  return NormType(vmin=vmin, vmax=vmax)

def createCmap(
    cmap_name,
    cmin=0.0, cmax=1.0,
    vmin=0.0, vmid=None, vmax=1.0,
    NormType = colors.Normalize
  ):
  if vmid is not None:
    NormType = functools.partial(MidpointNormalize, vmid=vmid)
  ## cmaps span cmin=0.0 to cmax=1.0, so pass (cmin, cmax) to subset a cmap color-range
  cmap = cmr.get_sub_cmap(cmap_name, cmin, cmax)
  ## define value range of colorbar: [vmin, vmax]
  norm = createNorm(vmin, vmax, NormType)
  return cmap, norm

class MidpointNormalize(colors.Normalize):
  def __init__(self, vmin=None, vmid=None, vmax=None, clip=False):
    self.vmid = vmid
    colors.Normalize.__init__(self, vmin, vmax, clip)

  def __call__(self, value, clip=None):
    return np.ma.masked_array(np.interp(
      value,
      [ self.vmin, self.vmid, self.vmax ],
      [ 0, 0.5, 1 ]
    ))


## ###############################################################
## PLOT 1D DATA
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
    label   = None,
    color   = "k",
    marker  = "o",
    capsize = 7.5,
    alpha   = 1.0,
    zorder  = 5
  ):
  array_y = [
    y
    for y in array_y
    if y is not None
  ]
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
    yerr    = y_1sig,
    color   = color,
    fmt     = marker,
    label   = label,
    alpha   = alpha,
    capsize = capsize,
    zorder  = zorder,
    markersize=7, markeredgecolor="black",
    elinewidth=1.5, linestyle="None"
  )

def plotPDF(
    ax, list_data,
    num_bins     = 10,
    weights      = None,
    color        = "black",
    bool_flip_ax = False
  ):
  list_dens, list_bin_edges = np.histogram(list_data, bins=num_bins, weights=weights)
  list_dens_norm = np.append(0, list_dens / list_dens.sum())
  if bool_flip_ax:
    ax.plot(list_dens_norm[::-1], list_bin_edges[::-1], drawstyle="steps", color=color)
    ax.fill_between(list_dens_norm[::-1], list_bin_edges[::-1], step="pre", alpha=0.2, color=color)
  else:
    ax.plot(list_bin_edges, list_dens_norm, drawstyle="steps", color=color)
    ax.fill_between(list_bin_edges, list_dens_norm, step="pre", alpha=0.2, color=color)
  return list_bin_edges, list_dens_norm


## ###############################################################
## PLOT 2D DATA
## ###############################################################
def plotScatter(
    fig, ax, list_x, list_y,
    vmin              = None,
    vmax              = None,
    ms                = 1,
    color             = None,
    fontsize          = 20,
    cbar_title        = None,
    cbar_orientation  = "horizontal",
    bool_add_colorbar = False
  ):
  if color is None:
    ## color by the point density
    xy_stack = np.vstack([ list_x, list_y ])
    color = gaussian_kde(xy_stack)(xy_stack)
  plot_obj = ax.scatter(list_x, list_y, c=color, s=ms)
  plot_obj.set_clim(vmin=vmin, vmax=vmax)
  if bool_add_colorbar:
    cbar = addColorbar_fromMappble(
      fig, ax, plot_obj,
      cbar_title  = cbar_title,
      orientation = cbar_orientation,
      fontsize    = fontsize
    )
  else: cbar = None
  return plot_obj, cbar

def plotScalarField(
    field_slice,
    fig               = None,
    ax                = None,
    bool_add_colorbar = False,
    bool_center_cbar  = False,
    cbar_orientation  = "horizontal",
    cmap_name         = "cmr.arctic",
    NormType          = colors.LogNorm,
    cbar_bounds       = None,
    cbar_title        = None,
    bool_label_axis   = False
  ):
  ## check that a figure object has been passed
  if (fig is None) or (ax is None):
    fig, ax = fig, ax = plt.subplots(constrained_layout=True)
  ## plot scalar field
  if bool_center_cbar: NormType = functools.partial(MidpointNormalize, vmid=0)
  im_obj = ax.imshow(
    field_slice,
    extent = [-1, 1, -1, 1],
    cmap   = plt.get_cmap(cmap_name),
    norm   = NormType(
      vmin = 0.9*np.min(field_slice) if cbar_bounds is None else cbar_bounds[0],
      vmax = 1.1*np.max(field_slice) if cbar_bounds is None else cbar_bounds[1]
    )
  )
  ## add colorbar
  if bool_add_colorbar:
    addColorbar_fromMappble(
      fig, ax, im_obj,
      cbar_title  = cbar_title,
      orientation = cbar_orientation
    )
  ## add axis labels
  if bool_label_axis:
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"])
    ax.set_yticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"])
  else:
    # ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])

def plotVectorField(
    field_slice_x1,
    field_slice_x2,
    fig                   = None,
    ax                    = None,
    bool_plot_magnitude   = True,
    bool_add_colorbar     = False,
    bool_log_center_cbar  = False,
    cbar_orientation      = "horizontal",
    cmap_name             = "cmr.arctic",
    NormType              = colors.LogNorm,
    cbar_bounds           = None,
    cbar_title            = None,
    bool_plot_quiver      = False,
    quiver_step           = 1,
    quiver_width          = 5e-3,
    field_color           = "white",
    bool_plot_streamlines = False,
    streamline_width      = 1.0,
    streamline_linestyle  = "->",
    bool_label_axis       = False
  ):
  ## check that a figure object has been passed
  if (fig is None) or (ax is None):
    fig, ax = fig, ax = plt.subplots(constrained_layout=True)
  ## plot magnitude of vector field
  if bool_plot_magnitude:
    field_magnitude = np.sqrt(field_slice_x1**2 + field_slice_x2**2)
    if bool_log_center_cbar:
      field_magnitude = np.log(field_magnitude)
      NormType = functools.partial(MidpointNormalize, vmid=0)
    im_obj = ax.imshow(
      field_magnitude,
      extent = [-1.0, 1.0, -1.0, 1.0],
      cmap   = plt.get_cmap(cmap_name),
      norm   = NormType(
        vmin = 0.9*np.min(field_magnitude) if cbar_bounds is None else cbar_bounds[0],
        vmax = 1.1*np.max(field_magnitude) if cbar_bounds is None else cbar_bounds[1]
      )
    )
    ## add colorbar
    if bool_add_colorbar:
      addColorbar_fromMappble(
        fig, ax, im_obj,
        cbar_title  = cbar_title,
        orientation = cbar_orientation
      )
  ## overlay vector field
  if bool_plot_quiver:
    field_vecs_x1 = field_slice_x1[::quiver_step, ::quiver_step]
    field_vecs_x2 = field_slice_x2[::quiver_step, ::quiver_step]
    x = np.linspace(-1.0, 1.0, len(field_vecs_x1[0,:]))
    y = np.linspace(-1.0, 1.0, len(field_vecs_x2[:,0]))
    X, Y = np.meshgrid(x, -y)
    norm = np.sqrt(field_vecs_x1**2 + field_vecs_x2**2)
    norm[norm == 0.0] = 1.0
    ax.quiver(
      X, Y,
      field_vecs_x1 / norm,
      field_vecs_x2 / norm,
      width = quiver_width,
      color = field_color
    )
  if bool_plot_streamlines:
    x = np.linspace(-1.0, 1.0, len(field_slice_x1[0,:]))
    y = np.linspace(-1.0, 1.0, len(field_slice_x2[:,0]))
    X, Y = np.meshgrid(x, -y)
    ax.streamplot(
      X, Y,
      field_slice_x1,
      field_slice_x2,
      color      = field_color,
      linewidth  = streamline_width,
      density    = 1,
      arrowstyle = streamline_linestyle,
      arrowsize  = 1
    )
  ## add axis labels
  if bool_label_axis:
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"])
    ax.set_yticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"])
  else:
    a = 10
    # # ax.set_axis_off()
    # ax.set_xticks([])
    # ax.set_yticks([])


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
    lw       = 2.0,
    fontsize = 20,
    alpha    = 0.85,
    zorder   = 10
  ):
  obj_legend = ax.legend(
    ncol=ncol, loc=loc, bbox_to_anchor=bbox, fontsize=fontsize, framealpha=alpha,
    frameon=True, facecolor="white", edgecolor="grey"
  )
  obj_legend.set_zorder(zorder)
  for line in obj_legend.get_lines():
    line.set_linewidth(lw)
  return obj_legend

def addLegend_fromArtists(
    ax, list_artists, list_legend_labels,
    list_marker_colors = [ "k" ],
    bool_frame         = False,
    label_color        = "black",
    loc                = "upper right",
    bbox               = (1.0, 1.0),
    ms                 = 8,
    lw                 = 2,
    title              = None,
    ncol               = 1,
    handletextpad      = 0.5,
    rspacing           = 0.5,
    cspacing           = 0.5,
    fontsize           = 16,
    alpha              = 0.6
  ):
  if len(list_artists) + len(list_legend_labels) == 0: return
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
        Line2D([0], [0], marker=artist, color=marker_color, linewidth=0, markeredgecolor="black", markersize=ms)
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
    frameon        = bool_frame,
    title          = title,
    loc            = loc,
    bbox_to_anchor = bbox,
    ncol           = ncol,
    borderpad      = 0.45,
    handletextpad  = handletextpad,
    labelspacing   = rspacing,
    columnspacing  = cspacing,
    fontsize       = fontsize,
    labelcolor     = label_color,
    framealpha     = alpha,
    facecolor      = "white",
    edgecolor      = "grey"
  )
  ## draw legend
  ax.add_artist(legend)

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
    ncol=ncol, loc=loc, bbox_to_anchor=bbox, fontsize=fontsize, framealpha=alpha,
    frameon=True, facecolor="white", edgecolor="grey"
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

def labelLogFormatter(x, pos):
  if x % 1 == 0:
    return r"$10^{" + f"{x:.0f}" + "}$"
  else: return r"$10^{" + f"{x:.1f}" + "}$"

def addColorbar_fromCmap(
    fig, ax, cmap,
    norm=None, vmin=0.0, vmax=1.0,
    orientation    = "horizontal",
    bool_log_ticks = False,
    cbar_title     = None,
    cbar_title_pad = 10,
    fontsize       = 16,
    size           = 10
  ):
  if norm is None: norm = createNorm(vmin, vmax)
  mappable = ScalarMappable(cmap=cmap, norm=norm)
  ax_div = make_axes_locatable(ax)
  if   "h" in orientation: ax_cbar = ax_div.append_axes(position="top",   size=f"{size:.1f}%", pad="2%")
  elif "v" in orientation: ax_cbar = ax_div.append_axes(position="right", size=f"{size:.1f}%", pad="2%")
  else: raise Exception(f"Error: '{orientation}' is not a supported orientation!")
  # fig.add_axes(ax_cbar)
  cbar = fig.colorbar(mappable=mappable, cax=ax_cbar, orientation=orientation)
  if "h" in orientation:
    ax_cbar.set_title(cbar_title, fontsize=fontsize, pad=cbar_title_pad)
    ax_cbar.xaxis.set_ticks_position("top")
    if bool_log_ticks:
      cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(labelLogFormatter))
  else:
    cbar.ax.set_ylabel(cbar_title, fontsize=fontsize, rotation=-90, va="bottom")
    if bool_log_ticks:
      cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(labelLogFormatter))
  return cbar

def addColorbar_fromMappble(
    mappable,
    fig         = None,
    ax          = None,
    orientation = "vertical",
    cbar_title  = None,
    size        = 7.5,
    title_pad   = 12.5,
    fontsize    = 20
  ):
  ''' from: https://joseph-long.com/writing/colorbars/
  '''
  if (fig is None) or (ax is None):
    ax  = mappable.axes
    fig = ax.figure
  ax_div = make_axes_locatable(ax)
  if   "h" in orientation: ax_cbar = ax_div.append_axes(position="top",   size=f"{size:.1f}%", pad="2%")
  elif "v" in orientation: ax_cbar = ax_div.append_axes(position="right", size=f"{size:.1f}%", pad="2%")
  cbar = fig.colorbar(mappable=mappable, cax=ax_cbar, orientation=orientation)
  if "h" in orientation:
    ax_cbar.set_title(cbar_title, fontsize=fontsize, pad=title_pad)
    ax_cbar.xaxis.set_ticks_position("top")
  else: cbar.ax.set_ylabel(cbar_title, fontsize=fontsize, rotation=-90, va="bottom")
  plt.sca(ax)
  return cbar

def addAxisTicks_linear(
    ax,
    bool_y_axis      = True,
    bool_minor_ticks = False,
    bool_major_ticks = False,
    num_minor_ticks  = None,
    num_major_ticks  = None
  ):
  if bool_y_axis: ax_axis = ax.yaxis
  else:           ax_axis = ax.xaxis
  ## add minor axis ticks
  if bool_minor_ticks:
    loc_minor = mpl.ticker.LinearLocator(numticks=num_minor_ticks)
    ax_axis.set_minor_locator(loc_minor)
    ax_axis.set_minor_formatter(mpl.ticker.NullFormatter())
  ## add major axis ticks
  if bool_major_ticks:
    y_major = mpl.ticker.LinearLocator(numticks=num_major_ticks)
    ax_axis.set_major_locator(y_major)

def addAxisTicks_log10(
    ax,
    bool_y_axis      = True,
    bool_minor_ticks = False,
    bool_major_ticks = False,
    num_minor_ticks  = None,
    num_major_ticks  = None
  ):
  if bool_y_axis: ax_axis = ax.yaxis
  else:           ax_axis = ax.xaxis
  ## add minor axis ticks
  if bool_minor_ticks:
    loc_minor = mpl.ticker.LogLocator(
      base     = 10.0,
      subs     = np.arange(2, 10) * 0.1,
      numticks = num_minor_ticks
    )
    ax_axis.set_minor_locator(loc_minor)
    ax_axis.set_minor_formatter(mpl.ticker.NullFormatter())
  ## add major axis ticks
  if bool_major_ticks:
    loc_major = mpl.ticker.LogLocator(
      base     = 10.0,
      numticks = num_major_ticks
    )
    ax_axis.set_major_locator(loc_major)

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
    if (label is not None) and len(label) > 0
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
  ax_inset.tick_params(top=True, bottom=True, labeltop=True, labelbottom=False)
  ## add axis label
  ax_inset.set_xlabel(label_x, fontsize=fontsize)
  ax_inset.set_ylabel(label_y, fontsize=fontsize)
  ax_inset.xaxis.set_label_position("top")
  ## return inset axis
  return ax_inset


## ###############################################################
## LABELLING PLOTS
## ###############################################################
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


## END OF LIBRARY
#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

## load user defined modules
from TheUsefulModule import WWFnF, WWObjs
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def getShockWidth(Mach, Re):
  return Mach**2 / (Re * (Mach - 1)**2)

def addLegend_Re(ax):
  args = { "va":"bottom", "ha":"right", "transform":ax.transAxes, "fontsize":15 }
  ax.text(0.925, 0.225, r"Re $< 100$", color="cornflowerblue", **args)
  ax.text(0.925, 0.1,   r"Re $> 100$", color="orangered",  **args)

def plotScale(ax, x_median, y_median, color=None, ecolor=None, marker="o", x_1sig=None, y_1sig=None, ms=8, zorder=5):
  ax.errorbar(
    x_median, y_median,
    xerr   = x_1sig,
    yerr   = y_1sig,
    fmt    = marker,
    mfc    = "whitesmoke" if color is None else color,
    ecolor = "black" if ecolor is None else ecolor, 
    zorder = zorder,
    markersize = ms,
    mec="black", elinewidth=1.5, capsize=7.5, linestyle="None"
  )

def addText(ax, pos, text, rotation=0):
  ax.text(
    pos[0], pos[1],
    text,
    transform = ax.transAxes,
    va        = "center",
    ha        = "left",
    rotation  = rotation,
    rotation_mode = "anchor",
    color     = "black",
    fontsize  = 17,
    zorder    = 10
  )


## ###############################################################
## LOAD + PLOT MHD SCALES
## ###############################################################
class PlotSimScales():
  def __init__(self):
    ## load simulationd data
    self._loadAllSimulationData()
    ## check the number of different Mach numbers
    self.Mach_group_sim = [
      self._getMachString(sim_name)
      for sim_name in self.dict_scales_group_sim
      if self._meetsSimCondition(sim_name)
    ]
    ## define name of figures
    self.bool_Mach_varied = len(set(self.Mach_group_sim)) > 1
    if self.bool_Mach_varied:
      self.plot_name = "Mach_varied"
    else: self.plot_name = self.Mach_group_sim[0]
    self.filepath_vis = f"{PATH_PAPER}/{self.plot_name}/"
    WWFnF.createFolder(self.filepath_vis, bool_verbose=False)
    print("Saving figures in:", self.filepath_vis)
    print(" ")

  def _getShockWidth(self, sim_name):
    Mach = self.dict_scales_group_sim[sim_name]["Mach"]["val"]
    Re = self.dict_scales_group_sim[sim_name]["Re"]
    # if Mach < 1.1: return np.nan
    return getShockWidth(Mach, Re)

  def _getMachString(self, sim_name):
    Mach = self.dict_scales_group_sim[sim_name]["Mach"]["val"]
    if Mach < 0.7:
      return f"Mach{float(Mach):.1f}"
    else: return f"Mach{int(round(Mach)):d}"

  def _meetsSimCondition(self, sim_name):
    Mach = self.dict_scales_group_sim[sim_name]["Mach"]["val"]
    Mach_string = self._getMachString(sim_name)
    Rm_string = str(int(self.dict_scales_group_sim[sim_name]["Rm"]))
    # return (Mach_string == "Mach5") and (Rm_string == "3000")
    return True

  def _addLabel(self, ax):
    ## add Mach annotation
    if not(self.bool_Mach_varied):
      Mach_string = self.Mach_group_sim[0].replace("Mach", "")
      ax.text(
        0.05, 0.9,
        r"$\mathcal{M} =$ " + Mach_string,
        transform = ax.transAxes, va="top", ha="left", fontsize=16
      )

  def _loadAllSimulationData(self):
    print("Loading datasets...")
    self.dict_scales_group_sim = WWObjs.readJsonFile2Dict(
      filepath     = PATH_PAPER,
      filename     = "dataset.json",
      bool_verbose = False
    )
    ## get list of simulation names
    self.sim_name_group_sim = list(self.dict_scales_group_sim.keys())
    ## define simulation markers based on Mach number
    self.marker_group_sim = [
      "o" if self.dict_scales_group_sim[sim_name]["Mach"]["val"] < 1.1 else "D"
      for sim_name in self.dict_scales_group_sim
    ]
    ## color simulations by Mach number
    self.color_by_Mach_group_sim = [
      "cornflowerblue"
      if self.dict_scales_group_sim[sim_name]["Mach"]["val"] < 0.7 else
      "orangered"
      for sim_name in self.dict_scales_group_sim
    ]
    ## color simulations by shockwidth
    self.shockwidth_group_sim = [
      self._getShockWidth(sim_name)
      for sim_name in self.dict_scales_group_sim
    ]
    self.cmap, self.norm = PlotFuncs.createCmap(
      cmap_name = "cmr.rainforest_r",
      cmin = 0.075,
      cmax = 0.55,
      vmin = 0.5,
      vmid = 3,
      vmax = 3.75,
    )
    self.color_by_shockwidth_group_sim = [
      self.cmap(self.norm(np.log10(1/shockwidth)))
      if not np.isnan(shockwidth) else None
      for shockwidth in self.shockwidth_group_sim
    ]
    ## create empty space
    print(" ")

  def _plotDependance_knu_Re(self):
    ## plot data
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    list_colors = ["#C85DEF", "white", "#FFAB1A", "black"]
    list_zorder = [5, 9, 3, 7]
    cmap = colors.ListedColormap(list_colors)
    for sim_index, sim_name in enumerate(self.sim_name_group_sim):
      if not(self._meetsSimCondition(sim_name)): continue
      Mach = self.dict_scales_group_sim[sim_name]["Mach"]["val"]
      if Mach < 0.7: order_index = 0 # Mach = 0.3
      elif Mach > 7: order_index = 3 # Mach = 10
      elif Mach > 2: order_index = 2 # Mach = 5
      else:          order_index = 1 # Mach = 1
      plotScale(
        ax       = ax,
        x_median = self.dict_scales_group_sim[sim_name]["Re"],
        y_median = self.dict_scales_group_sim[sim_name]["k_nu_kin"]["inf"]["val"] / 2,
        y_1sig   = self.dict_scales_group_sim[sim_name]["k_nu_kin"]["inf"]["std"],
        # marker   = self.marker_group_sim[sim_index],
        # color    = self.color_by_Mach_group_sim[sim_index],
        marker   = "o",
        color    = list_colors[order_index],
        zorder   = list_zorder[order_index]
      )
    ## plot reference lines
    x = np.linspace(10**(-1), 10**(5), 10**4)
    PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.325*x**(2/3), ls="-",  zorder=0, lw=1.5)
    PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.1*x**(3/4),   ls="--", zorder=0, lw=1.5)
    PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.625*x**(3/8), ls=":",  zorder=0, lw=1.5)
    ## label figure
    PlotFuncs.addLegend_fromArtists(
      ax,
      list_artists       = [ "-", "--", ":" ],
      list_legend_labels = [
        r"$\propto \mathrm{Re}^{2/3}$",
        r"$\propto \mathrm{Re}^{3/4}$",
        r"$\propto \mathrm{Re}^{3/8}$",
      ],
      list_marker_colors = [ "k" ],
      label_color        = "black",
      loc                = "lower right",
      bbox               = (1.0, 0.0),
      lw                 = 1.5
    )
    cbar = PlotFuncs.addColorbar_fromCmap(
      fig        = ax.get_figure(),
      ax         = ax,
      cmap       = cmap,
      cbar_title = r"$\mathcal{M}$",
      cbar_title_pad=7, orientation="horizontal", size=8, fontsize=20
    )
    tick_placement = [0.3, 1, 5, 10]
    cbar.set_ticks(np.arange(0.125, 1, 0.25))
    cbar.set_ticklabels([ f"${val:.1f}$" for val in tick_placement ])
    ## adjust axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([ 7, 5000 ])
    ax.set_ylim([ 1, 100 ])
    ax.set_xlabel(r"$\mathrm{Re}$", fontsize=20)
    ax.set_ylabel(r"$k_\nu / k_\mathrm{turb}$", fontsize=20)
    ## save plot
    name = f"scaling_knu_kin.pdf"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{name}")
    print(" ")

  def _plotDependance_knu_comparison(self):
    ## plot data
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for sim_index, sim_name in enumerate(self.sim_name_group_sim):
      if not(self._meetsSimCondition(sim_name)): continue
      plotScale(
        ax       = ax,
        x_median = self.dict_scales_group_sim[sim_name]["k_nu_kin"]["inf"]["val"],
        x_1sig   = self.dict_scales_group_sim[sim_name]["k_nu_kin"]["inf"]["std"],
        y_median = self.dict_scales_group_sim[sim_name]["k_nu_vel"]["inf"]["val"],
        y_1sig   = self.dict_scales_group_sim[sim_name]["k_nu_vel"]["inf"]["std"],
        marker   = "o",
        color    = "black"
      )
    ## plot reference lines
    x = np.linspace(10**(-1), 10**(5), 10**4)
    PlotFuncs.plotData_noAutoAxisScale(ax, x, x, ls="-", zorder=0, lw=1.5)
    PlotFuncs.addLegend_fromArtists(
      ax,
      list_artists       = [ "-" ],
      list_legend_labels = [ r"$1:1$" ],
      list_marker_colors = [ "k" ],
      label_color        = "black",
      loc                = "lower right",
      bbox               = (1.0, -0.01),
      lw                 = 1.5,
      fontsize           = 18
    )
    ## adjust axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k_\nu \mathrm{\,with\,} \bm{\psi} = \bm{u} \sqrt{\rho/2}$", fontsize=20)
    ax.set_ylabel(r"$k_\nu \mathrm{\,with\,} \bm{\psi} = \bm{u} / \sqrt{2}$", fontsize=20)
    ## save plot
    name = f"scaling_knu_comparison.pdf"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{name}")
    print(" ")

  def _plotDependance_keta_Pm(self, domain_name):
    keta_type = domain_name.split("_")[-1]
    ## plot data
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    list_colors = ["#C85DEF", "white", "#FFAB1A", "black"]
    list_zorder = [5, 9, 3, 7]
    cmap = colors.ListedColormap(list_colors)
    for sim_index, sim_name in enumerate(self.sim_name_group_sim):
      if not(self._meetsSimCondition(sim_name)): continue
      Mach = self.dict_scales_group_sim[sim_name]["Mach"]["val"]
      Re = self.dict_scales_group_sim[sim_name]["Re"]
      if Mach < 0.7: order_index = 0 # Mach = 0.3
      elif Mach > 7: order_index = 3 # Mach = 10
      elif Mach > 2: order_index = 2 # Mach = 5
      else:          order_index = 1 # Mach = 1
      v1 = self.dict_scales_group_sim[sim_name][domain_name]["inf"]["val"]
      d1 = self.dict_scales_group_sim[sim_name][domain_name]["inf"]["std"]
      v2 = self.dict_scales_group_sim[sim_name]["k_nu_kin"]["inf"]["val"]
      d2 = self.dict_scales_group_sim[sim_name]["k_nu_kin"]["inf"]["std"]
      plotScale(
        ax       = ax,
        x_median = self.dict_scales_group_sim[sim_name]["Pm"],
        y_median = (v1 / v2),
        y_1sig   = (v1 / v2) * np.sqrt((d1 / v1)**2 + (d2 / v2)**2),
        # marker   = self.marker_group_sim[sim_index],
        # color    = self.color_by_Mach_group_sim[sim_index],
        marker   = "D" if Re > 100 else "o",
        color    = list_colors[order_index],
        zorder   = list_zorder[order_index]
      )
    ## plot reference lines
    x = np.linspace(10**(-1), 10**(5), 10**4)
    if "cur" in keta_type.lower():
      PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.385*x**(1/2), ls="-", lw=1.5)
      list_artists = [ "-" ]
      list_labels = [ r"$\propto \mathrm{Pm}^{1/2}$" ]
    elif "mag" in keta_type.lower():
      PlotFuncs.plotData_noAutoAxisScale(ax, x, x**(1/3), ls="-", lw=1.5)
      PlotFuncs.plotData_noAutoAxisScale(ax, x, x**(1/7), ls="--", lw=1.5)
      list_artists = [ "-", "--" ]
      list_labels = [
        r"$\mathrm{Pm}^{1/3}$",
        r"$\mathrm{Pm}^{1/7}$",
      ]
    ## label figure
    if self.bool_Mach_varied:
      PlotFuncs.addLegend_fromArtists(
        ax,
        list_artists       = [ "o", "D" ],
        list_marker_colors = [ "black" ],
        list_legend_labels = [
          r"$\mathrm{Re} < 100$",
          r"$\mathrm{Re} > 100$"
        ],
        label_color   = "black",
        loc           = "upper left",
        bbox          = (-0.025, 1.0),
        lw            = 1.5,
        handletextpad = 0.01
      )
    else: self._addLabel(ax)
    PlotFuncs.addLegend_fromArtists(
      ax,
      list_artists       = list_artists,
      list_legend_labels = list_labels,
      list_marker_colors = [ "k" ],
      label_color        = "black",
      loc                = "lower right",
      bbox               = (1.0, 0.0),
      lw                 = 1.5
    )
    cbar = PlotFuncs.addColorbar_fromCmap(
      fig        = ax.get_figure(),
      ax         = ax,
      cmap       = cmap,
      cbar_title = r"$\mathcal{M}$",
      cbar_title_pad=7, orientation="horizontal", size=8, fontsize=20
    )
    tick_placement = [0.3, 1, 5, 10]
    cbar.set_ticks(np.arange(0.125, 1, 0.25))
    cbar.set_ticklabels([ f"${val:.1f}$" for val in tick_placement ])
    ## adjust axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([ 0.7, 400 ])
    if   "cur" in keta_type.lower(): ax.set_ylim([ 0.1, 10 ])
    elif "mag" in keta_type.lower(): ax.set_ylim([ 0.7, 10 ])
    ax.set_xlabel(r"$\mathrm{Pm}$", fontsize=20)
    if   "cur" in keta_type.lower(): ax.set_ylabel(r"$k_\eta / k_\nu$", fontsize=20)
    elif "mag" in keta_type.lower(): ax.set_ylabel(r"$k_\mathrm{Rm} / k_\nu$", fontsize=20)
    ## save plot
    name = f"scaling_keta_{keta_type}.pdf"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{name}")
    print(" ")

  def _plotDependance_kp_keta(self):
    ## plot data
    factor = 1.45
    fig, ax = plt.subplots(1, 1, figsize=(6*factor, 4.75*factor))
    ax_inset = ax.inset_axes([ 0.025, 0.575, 0.46, 0.3875 ])
    ax_inset.tick_params(left=True, right=True, labelleft=False, labelright=True)
    ax_inset.yaxis.set_label_position("right")
    for sim_index, sim_name in enumerate(self.sim_name_group_sim):
      if not(self._meetsSimCondition(sim_name)): continue
      color = self.color_by_shockwidth_group_sim[sim_index]
      Mach = self.dict_scales_group_sim[sim_name]["Mach"]["val"]
      if Mach < 1.1: color = None
      plotScale(
        ax       = ax,
        x_median = self.dict_scales_group_sim[sim_name]["k_eta_cur"]["inf"]["val"],
        x_1sig   = self.dict_scales_group_sim[sim_name]["k_eta_cur"]["inf"]["std"],
        y_median = self.dict_scales_group_sim[sim_name]["k_p"]["inf"]["val"],
        y_1sig   = self.dict_scales_group_sim[sim_name]["k_p"]["inf"]["std"],
        marker   = self.marker_group_sim[sim_index],
        color    = color,
        ms       = 8
      )
      Mach_string = self._getMachString(sim_name)
      Rm_string = str(int(self.dict_scales_group_sim[sim_name]["Rm"]))
      if (Mach_string == "Mach5") and (Rm_string == "3000"):
        plotScale(
          ax       = ax_inset,
          x_median = self.dict_scales_group_sim[sim_name]["k_eta_cur"]["inf"]["val"],
          x_1sig   = self.dict_scales_group_sim[sim_name]["k_eta_cur"]["inf"]["std"],
          y_median = self.dict_scales_group_sim[sim_name]["k_p"]["inf"]["val"],
          y_1sig   = self.dict_scales_group_sim[sim_name]["k_p"]["inf"]["std"],
          marker   = self.marker_group_sim[sim_index],
          color    = color,
          ms       = 8
        )
    ## plot reference lines
    x = np.linspace(10**(-1), 10**(5), 10**4)
    PlotFuncs.plotData_noAutoAxisScale(ax, x, 0.39*x, ls="-", lw=1.5)
    PlotFuncs.plotData_noAutoAxisScale(ax_inset, x, 0.4*x, ls="-", lw=1.5)
    ## label figure
    if self.bool_Mach_varied:
      PlotFuncs.addLegend_fromArtists(
        ax,
        list_artists       = [ "o", "D" ],
        list_marker_colors = [
          "whitesmoke",
          "black"
        ],
        list_legend_labels = [
          r"$\mathcal{M} \leq 1$",
          r"$\mathcal{M} > 1$"
        ],
        label_color   = "black",
        loc           = "lower right",
        bbox          = (1.0, -0.01),
        lw            = 1.5,
        handletextpad = 0.01,
        fontsize      = 18
      )
    else: self._addLabel(ax)
    cbar = PlotFuncs.addColorbar_fromCmap(
      fig        = ax.get_figure(),
      ax         = ax,
      cmap       = self.cmap,
      norm       = self.norm,
      cbar_title = r"$\log_{10}\big( k_\mathrm{shock} / k_\mathrm{turb} \big)$",
      cbar_title_pad=12, orientation="horizontal", size=5.5, fontsize=18
    )
    cbar.ax.tick_params(labelsize=16)
    PlotFuncs.addLegend_fromArtists(
      ax,
      list_artists       = [ "-" ],
      list_legend_labels = [ r"$(0.40 \pm 0.06)\, k_\eta$" ],
      list_marker_colors = [ "k" ],
      label_color        = "black",
      loc                = "lower left",
      bbox               = (0.0, -0.01),
      lw                 = 1.5,
      fontsize           = 18
    )
    list_cbar_ticks = []
    for x in range(10):
      y = 3.5 - 0.5*x
      if y > 0.25: list_cbar_ticks.append(y)
    cbar.set_ticks(list_cbar_ticks)
    ## adjust main axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([ 5, 130 ])
    ax.set_ylim([ 0.9, 80 ])
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlabel(r"$k_\eta$", fontsize=20)
    ax.set_ylabel(r"$k_\mathrm{p}$", fontsize=20)
    ## adjust inset axis
    ax_inset.set_xlim([ 12, 55 ])
    ax_inset.set_ylim([ -1, 11 ])
    ax_inset.tick_params(axis="x", labelsize=16)
    ax_inset.tick_params(axis="y", labelsize=16)
    ax_inset.set_xticks([ 20, 40 ])
    ax_inset.text(
      0.665, 0.85,
      r"$\mathrm{increasing}\, \nu$",
      va        = "top",
      ha        = "center",
      transform = ax_inset.transAxes,
      color     = "black",
      fontsize  = 16,
      zorder    = 10
    )
    ax_inset.arrow(
      x  = 52.5,
      y  = 9.75,
      dx = -24,
      dy = 0,
      color = "black",
      head_width = 1.5
    )
    ax_inset.text(
      0.5, 0.0825,
      r"$\mathrm{Rm} = 3000$, $\mathcal{M} = 5$",
      va        = "bottom",
      ha        = "center",
      transform = ax_inset.transAxes,
      color     = "black",
      fontsize  = 16,
      zorder    = 10
    )
    ## save plot
    name = f"scaling_kp.pdf"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{name}")
    print(" ")
  
  def _plotDependance_kp_kshock(self):
    ## plot data
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    list_vals = []
    for sim_index, sim_name in enumerate(self.sim_name_group_sim):
      shockwidth = self.shockwidth_group_sim[sim_index]
      # Mach_string = self._getMachString(sim_name)
      # Rm_string = str(int(self.dict_scales_group_sim[sim_name]["Rm"]))
      # condition = (Mach_string == "Mach5") and (Rm_string == "3000")
      # if not condition: continue
      v1 = self.dict_scales_group_sim[sim_name]["k_p"]["inf"]["val"]
      d1 = self.dict_scales_group_sim[sim_name]["k_p"]["inf"]["std"]
      v2 = self.dict_scales_group_sim[sim_name]["k_eta_cur"]["inf"]["val"]
      d2 = self.dict_scales_group_sim[sim_name]["k_eta_cur"]["inf"]["std"]
      Mach = self.dict_scales_group_sim[sim_name]["Mach"]["val"]
      if Mach < 1.1:
        color  = "whitesmoke"
        marker = "o"
        zorder = 3
        list_vals.append(v1 / v2)
      else:
        color  = "forestgreen" # self.color_by_shockwidth_group_sim[sim_index]
        marker = "D"
        zorder = 5
      plotScale(
        ax       = ax,
        x_median = 1/shockwidth,
        y_median = (v1 / v2),
        y_1sig   = (v1 / v2) * np.sqrt((d1 / v1)**2 + (d2 / v2)**2),
        color    = color,
        # ecolor   = color,
        marker   = marker,
        ms       = 10,
        zorder   = zorder
      )
    ## plot reference lines
    ax.axvline(x=10**(2), ls=":", color="black", zorder=0, lw=1.5)
    x = np.logspace(-5, 5, 10**4)
    PlotFuncs.plotData_noAutoAxisScale(ax, x, x**(-1/3), ls="-", zorder=0, lw=1.5)
    list_labels = [
      r"$(k_\mathrm{shock} / k_\mathrm{turb})^{-1/3}$"
    ]
    list_artists = [ "-" ]
    if len(list_vals) > 3:
      ave_val = np.mean(list_vals)
      std_val = np.std(list_vals)
      ax.axhline(y=ave_val, ls="--", zorder=1, lw=1.5, color="k")
      list_labels.append(f"${ave_val:.2f} \pm {std_val:.2f}$")
      list_artists.append("--")
    ## add text labels
    PlotFuncs.addLegend_fromArtists(
      ax,
      list_artists       = list_artists,
      list_legend_labels = list_labels,
      list_marker_colors = [ "k" ],
      label_color        = "black",
      loc                = "lower left",
      bbox               = (-0.015, -0.02),
      lw                 = 1.5,
      fontsize           = 18
    )
    PlotFuncs.addLegend_fromArtists(
      ax,
      list_artists       = [ "o", "D" ],
      list_marker_colors = [
        "whitesmoke",
        "forestgreen"
      ],
      list_legend_labels = [
        r"$\mathcal{M} \leq 1$",
        r"$\mathcal{M} > 1$"
      ],
      label_color   = "black",
      loc           = "upper right",
      bbox          = (0.99, 0.99),
      ms            = 10,
      lw            = 1.5,
      handletextpad = 0.01,
      fontsize      = 18
    )
    ## adjust axis
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([ 1e-3, 1e5 ])
    ax.set_ylim([ 4e-2, 2 ])
    ax.set_xticks([
      10**(x)
      for x in range(-3, 6)
    ])
    ax.set_xticklabels([
      f"$10^{{{x:.0f}}}$"
      if (x % 2 == 1) else
      ""
      for x in range(-3, 6)
    ])
    # add x and y axis labels
    ax.set_xlabel(r"$k_\mathrm{shock} / k_\mathrm{turb}$", fontsize=20)
    ax.set_ylabel(r"$k_\mathrm{p} / k_\eta$", fontsize=20)
    ## save plot
    name = f"scaling_kshock.pdf"
    PlotFuncs.saveFigure(fig, f"{self.filepath_vis}/{name}")
    print(" ")

  def plotRoutines(self):
    # self._plotDependance_knu_Re()
    # self._plotDependance_knu_comparison()
    # self._plotDependance_keta_Pm("k_eta_cur")
    # self._plotDependance_keta_Pm("k_eta_mag")
    # self._plotDependance_kp_keta()
    self._plotDependance_kp_kshock()


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  obj = PlotSimScales()
  obj.plotRoutines()


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
PATH_PAPER = "/home/586/nk7952/MHDCodes/kriel2023/"


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
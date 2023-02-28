## ########################################################
## LOAD MODULES
## ########################################################
import os, sys
import numpy as np
import matplotlib as mpl

from ThePlottingModule import PlotFuncs


## ########################################################
## PREPARE WORKSPACE
## ########################################################
os.system("clear")


## ########################################################
## PLOTTING QUOKKA'S WEAK SCALING
## ########################################################
class PlotWeakScaling():
  def __init__(self) -> None:
    pass

  def runRoutines(self):
    self._initData()
    self._initFigure()
    self._plotData()
    self._labelPlot()
    self._savePlot()

  def _initData(self):
    self.list_nodes = [ 0.25, 1, 8, 64 ]
    self.list_gpus  = [ 1, 4, 32, 256 ]
    ## hydro only
    self.list_eff_vs_1gpu_hydro  = [ 1.0, 0.62, 0.51, 0.52 ]
    self.list_eff_vs_1node_hydro = [ np.nan, 1.0, 0.82, 0.83 ]
    self.list_speed_hydro        = [ 113.34, 70.68, 57.75, 58.50 ]
    ## hydro + radiation
    self.list_eff_vs_1gpu_radhydro  = [ 1.0, 0.46, 0.35, 0.35 ]
    self.list_eff_vs_1node_radhydro = [ np.nan, 1.0, 0.77, 0.76 ]
    self.list_speed_radhydro        = [ 22.55, 10.32, 7.92, 7.87 ]
  
  def _initFigure(self):
    self.fig, fig_grid = PlotFuncs.createFigure_grid(
      fig_scale        = 1.0,
      fig_aspect_ratio = (4.0, 7.0),
      num_rows         = 3,
      num_cols         = 1
    )
    self.ax_vs_1node = self.fig.add_subplot(fig_grid[0])
    self.ax_vs_1gpu  = self.fig.add_subplot(fig_grid[1])
    self.ax_speed    = self.fig.add_subplot(fig_grid[2])
    self.axs = [
      self.ax_vs_1node,
      self.ax_vs_1gpu,
      self.ax_speed
    ]

  def _plotData(self):
    ## plot efficiency vs 1 node
    self.axs[0].plot(
      self.list_nodes,
      self.list_eff_vs_1node_hydro,
      color="blue", label=r"hydro only",
      ls="-", lw=2, marker="o", ms=10
    )
    self.axs[0].plot(
      self.list_nodes,
      self.list_eff_vs_1node_radhydro,
      color="orange", label=r"radiation $+$ hydro",
      ls="--", lw=2, marker="o", ms=10
    )
    ## plot efficiency vs 1 gpu
    self.axs[1].plot(
      self.list_nodes,
      self.list_eff_vs_1gpu_hydro,
      color="blue", label=r"hydro only",
      ls="-", lw=2, marker="o", ms=10
    )
    self.axs[1].plot(
      self.list_nodes,
      self.list_eff_vs_1gpu_radhydro,
      color="orange", label=r"radiation $+$ hydro",
      ls="--", lw=2, marker="o", ms=10
    )
    ## plot update speed
    self.axs[2].plot(
      self.list_nodes,
      self.list_speed_hydro,
      color="blue", label=r"hydro only",
      ls="-", lw=2, marker="o", ms=10
    )
    self.axs[2].plot(
      self.list_nodes,
      self.list_speed_radhydro,
      color="orange", label=r"radiation $+$ hydro",
      ls="--", lw=2, marker="o", ms=10
    )

  def _labelPlot(self):
    def scaleNodeAxis(ax):
      node_bounds = [ 0.1, 100 ]
      ax.set_xlim(node_bounds)
      ax.set_xscale("log")
    def addTopAxis(ax):
      ax_top = ax.twiny()
      ax_top.set_xscale("log")
      ax_top.set_xlim(ax.get_xlim())
      ax_top.set_xticks(self.list_nodes)
      ax_top.minorticks_off()
      ax_top.set_xticklabels([
        f"{x:d}"
        for x in self.list_gpus
      ])
      return ax_top
    scaleNodeAxis(self.axs[0])
    scaleNodeAxis(self.axs[1])
    scaleNodeAxis(self.axs[2])
    axs_0_top = addTopAxis(self.axs[0])
    axs_1_top = addTopAxis(self.axs[1])
    axs_2_top = addTopAxis(self.axs[2])
    self.axs[0].set_xticklabels([])
    self.axs[1].set_xticklabels([])
    # axs_1_top.set_xticklabels([])
    axs_2_top.set_xticklabels([])
    axs_1_top.set_xlabel(r"Number of GPUs")
    self.axs[0].set_ylabel(r"Parallel efficiency" + "\n" + r"(vs 1 Node)")
    self.axs[1].set_ylabel(r"Parallel efficiency" + "\n" + r"(vs 1 GPU)")
    self.axs[2].set_ylabel(r"Update speed" + "\n" + r"(Mzones GPU$^{-1}$ s$^{-1}$)")
    self.axs[1].set_xlabel(r"Number of Nodes")
    self.axs[1].legend(loc="right", fontsize=20)
    self.axs[1].text(
      0.95, 0.95, "weak scaling",
      transform=self.axs[1].transAxes, va="top", ha="right", fontsize=25
    )

  def _savePlot(self):
    PlotFuncs.saveFigure(self.fig, "quokka_weak_scaling.png")


## ########################################################
## PLOTTING QUOKKA'S STRONG SCALING
## ########################################################
class PlotStrongScaling():
  def __init__(self) -> None:
    pass

  def runRoutines(self):
    self._initData()
    self._initFigure()
    self._plotData()
    self._labelPlot()
    self._savePlot()

  def _initData(self):
    self.list_nodes = [ 1, 2, 4, 8 ]
    self.list_gpus  = [ 4, 8, 16, 32 ]
    self.list_speed = [ 4.95, 4.30, 3.26, 2.61 ]
    self.list_eff   = [ 1.0, 0.87, 0.66, 0.53 ]
  
  def _initFigure(self):
    self.fig, fig_grid = PlotFuncs.createFigure_grid(
      fig_scale        = 1.25,
      fig_aspect_ratio = (4.0, 6.0),
      num_rows         = 1,
      num_cols         = 1
    )
    self.ax   = self.fig.add_subplot(fig_grid[0])

  def _plotData(self):
    ## plot efficiency
    self.ax.plot(
      self.list_nodes,
      self.list_eff,
      color="orange", label=r"radiation $+$ hydro",
      ls="--", lw=2, marker="o", ms=10
    )

  def _labelPlot(self):
    def addTopAxis(ax):
      ax_top = ax.twiny()
      ax_top.set_xscale("log")
      ax_top.set_xlim(ax.get_xlim())
      ax_top.set_xticks(self.list_nodes)
      ax_top.minorticks_off()
      ax_top.set_xticklabels([
        f"{x:d}"
        for x in self.list_gpus
      ])
      return ax_top
    def addRightAxis(ax):
      ax_right = ax.twinx()
      ax_right.set_ylim(ax.get_ylim())
      ax_right.set_yticks(self.list_eff)
      ax_right.minorticks_off()
      ax_right.yaxis.set_major_locator(
        mpl.ticker.FixedLocator(self.list_eff)
      )
      return ax_right
    self.ax.set_xscale("log")
    self.ax.set_xlim([ 0.9, 11 ])
    self.ax.set_ylim([ 0.46, 1.05 ])
    ax_top   = addTopAxis(self.ax)
    ax_right = addRightAxis(self.ax)
    self.ax.xaxis.set_minor_locator(
      mpl.ticker.LogLocator(
        base     = 10.0,
        subs     = np.arange(0.1, 1, 0.1),
        numticks = 12
      )
    )
    self.ax.xaxis.set_minor_formatter(
      mpl.ticker.NullFormatter()
    )
    ax_top.set_xlabel(r"Number of GPUs")
    self.ax.set_ylabel(r"Parallel efficiency")
    ax_right.set_ylabel(r"Update speed" + "\n" + r"(Mzones GPU$^{-1}$ s$^{-1}$)")
    self.ax.set_xlabel(r"Number of Nodes")
    self.ax.legend(loc="lower left", fontsize=20)
    self.ax.text(
      0.95, 0.95, "strong scaling",
      transform=self.ax.transAxes, va="top", ha="right", fontsize=25
    )

  def _savePlot(self):
    PlotFuncs.saveFigure(self.fig, "quokka_strong_scaling.png")


## ########################################################
## ESTIMATE COST OF SIMULATIONS
## ########################################################
def estSimCost(t_adv, t_run, u_sig, eps, phi, cfl, dx, L_max=None, NL=None):
  if L_max is not None:
    list_L  = list(range(0, L_max))
  if NL is not None:
    num_cells = NL**2
  else:
    list_NL = [ 256**3, 206**3, 267**3 ]
    list_NL.extend([
      int(267**3 * 8**( (L-2) / 2 ))
      for L in list_L[3:]
    ])
    num_cells = sum([
      2**L * NL
      for L, NL in zip(list_L, list_NL)
    ])
  cost_step = t_adv / (eps * phi)
  num_steps = t_run * u_sig / (cfl * dx)
  sim_cost  = cost_step * num_steps * num_cells
  print(f"{5*sim_cost:.0f}")
  return sim_cost


## ########################################################
## PROGRAM MAIN
## ########################################################
BOOL_PLOT_WEAK   = 0
BOOL_PLOT_STRONG = 0

def main():
  ## estimate cost of simulation
  ## check Kumholz proposal
  estSimCost(
    t_run = 20 * np.sqrt(10**3),
    u_sig = 2,
    t_adv = 0.2 * 1e-6,
    eps   = 1,
    phi   = 100,
    cfl   = 0.2,
    dx    = 40/256,
    L_max = 3
  )
  ## plot weak scaling
  if BOOL_PLOT_WEAK:
    obj_weak_scaling = PlotWeakScaling()
    obj_weak_scaling.runRoutines()
  ## print empty space
  if BOOL_PLOT_WEAK and BOOL_PLOT_STRONG: print(" ")
  ## plot strong scaling
  if BOOL_PLOT_STRONG:
    obj_strong_scaling = PlotStrongScaling()
    obj_strong_scaling.runRoutines()


## ########################################################
## PROGRAM ENTRY POINT
## ########################################################
if __name__ == "__main__":
  main()
  sys.exit(0)


## END OF PROGRAM
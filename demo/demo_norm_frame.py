import os, sys
import numpy as np
from ThePlottingModule import PlotFuncs

os.system("clear")

def tuneAxis(ax):
  ax.legend(loc="lower left", fontsize=15)
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlim([ 10**(-3),  10**(+5) ])
  ax.set_ylim([ 10**(-15), 10**(-1) ])

def plotModel(axs, data_x, list_dscale, pstyle, zorder, list_sub_label):
  data_y = data_x**(3/2) * np.exp(-data_x / list_dscale[0])
  data_y_norm = data_y / sum(data_y)
  axs[0].plot(
    data_x,
    data_y_norm,
    pstyle,
    lw     = 2,
    label  = r"$x_\eta = $"+str(list_dscale[0]),
    zorder = zorder
  )
  for index in range(len(list_dscale)-1):
    index_ax = index + 1
    axs[index_ax].plot(
      data_x / list_dscale[index_ax],
      data_y_norm,
      pstyle,
      lw     = 2,
      label  = r"$x_{\eta" + list_sub_label[index] + r"} =$ " + str(round(list_dscale[index_ax], 3)),
      zorder = zorder
    )

def main():
  ## initialise figure
  fig, fig_grid = PlotFuncs.createFigure_grid(num_rows=2, num_cols=2)
  ax0 = fig.add_subplot(fig_grid[0, 0])
  ax1 = fig.add_subplot(fig_grid[0, 1])
  ax2 = fig.add_subplot(fig_grid[1, 0])
  ax3 = fig.add_subplot(fig_grid[1, 1])
  axs = [ ax0, ax1, ax2, ax3 ]
  ## plot data
  data_x_1 = np.logspace(0, 4, 1000)
  data_x_2 = np.logspace(0, 2, 1000)
  plotModel(axs, data_x_2, [5,   5,   1,  1/60], "r-", 4, ["", ",1", ",2"])
  plotModel(axs, data_x_2, [25,  25,  5,  1/12], "b-", 3, ["", ",1", ",2"])
  plotModel(axs, data_x_1, [100, 100, 20, 1/3],  "g-", 2, ["", ",1", ",2"])
  plotModel(axs, data_x_1, [300, 300, 60, 1],    "k-", 1, ["", ",1", ",2"])
  ## label top, left axis
  axs[0].set_xlabel(r"$x$")
  axs[0].set_ylabel(r"$y = x^{3/2} \exp(-x/x_\eta) /$ total")
  tuneAxis(axs[0])
  ## label top, right axis
  axs[1].set_xlabel(r"$x / x_\eta$")
  tuneAxis(axs[1])
  ## label bottom, left axis
  axs[2].set_xlabel(r"$x / x_{\eta,1}$")
  axs[2].set_ylabel(r"$y$")
  tuneAxis(axs[2])
  ## label bottom, right axis
  axs[3].set_xlabel(r"$x / x_{\eta,2}$")
  tuneAxis(axs[3])
  ## save figure
  PlotFuncs.saveFigure(fig, "demo_norm_frame.png")

if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
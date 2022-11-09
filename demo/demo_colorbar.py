## Demo file by Neco for Marcus

import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt

from ThePlottingModule import PlotFuncs


def main():
  ## INITIALISE STUFF
  ## ----------------
  fig, ax = plt.subplots()
  ## plot random data
  data_x = np.linspace(1, 10, 100)
  data_y = data_x**2
  ax.plot(data_x, data_y, "k-")
  ## THIS IS THE STUFF YOU SHOULD CARE ABOUT
  ## ---------------------------------------
  ## trucated bounds in [0.0, 1.0]
  vmin, vmax = 0.0, 0.4
  ## create the cmap object
  cmap = PlotFuncs.createCmap(
    cmap_name   = "cmr.neutral_r",
    vmin        = vmin,
    vmax        = vmax
  )
  ## add colorbar to axis
  label = "this is a title"
  cmap_args = {
    "fig"   : fig,
    "ax"    : ax,
    "cmap"  : cmap,
    "vmin"  : vmin,
    "vmax"  : vmax,
    "label" : label
  }
  # PlotFuncs.addColorbar_fromCmap(orientation="vertical", **cmap_args) # right
  PlotFuncs.addColorbar_fromCmap(orientation="horizontal", **cmap_args) # top
  ## SAVE FIGURE
  ## -----------
  filepath_fig = f"demo_colorbar.png"
  PlotFuncs.saveFigure(fig, filepath_fig)

if __name__ == "__main__":
  main()

## END OF DEMO PROGRAM
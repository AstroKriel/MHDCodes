## ###############################################################
## IMPORT MODULES
## ###############################################################
import os, sys
import multiprocessing as mproc
import concurrent.futures as cfut
import numpy as np

from ThePlottingModule.TheMatplotlibStyler import *

## 'tmpfile' needs to be loaded before any 'matplotlib' libraries,
## so matplotlib stores its cache in a temporary directory.
## (necessary when plotting in parallel)
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()

import matplotlib.pyplot as plt


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## PLOTTING FUNCTION
## ###############################################################
def plotFunc(num_points, lock=None):
  print("looking at func with param:", num_points)
  fig, ax = plt.subplots(nrows=1, ncols=1)
  x = np.linspace(1, 100, num_points)
  y = x**3 - x**2
  ax.plot(x, y, ls="-", lw=2, marker="o", ms=10)
  ax.set_xlabel(r"this is the x axis")
  ax.set_ylabel(r"this is the y axis")
  if lock is not None: lock.acquire()
  fig.savefig(f"plot_{num_points:d}.png")
  plt.close(fig)
  if lock is not None: lock.release()


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  list_num_points = [ 2, 5, 10, 20, 100 ]
  if BOOL_MPROC:
    with cfut.ProcessPoolExecutor() as executor:
      manager = mproc.Manager()
      lock = manager.Lock()
      ## loop over all simulation folders
      futures = [
        executor.submit(
          plotFunc,
          num_points, lock
        ) for num_points in list_num_points
      ]
      ## wait to ensure that all scheduled and running tasks have completed
      cfut.wait(futures)
      ## check if any tasks failed
      for future in cfut.as_completed(futures):
        future.result()
  else: [
    plotFunc(num_points)
    for num_points in list_num_points
  ]


## ###############################################################
## PROGRAM ENTRY POINT
## ###############################################################
BOOL_MPROC = 1

if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
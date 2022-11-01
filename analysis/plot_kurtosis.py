#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import csv
import matplotlib.pyplot as plt

## load old user defined modules
from ThePlottingModule import *
from TheUsefulModule import *
from ThePlottingModule import PlotFuncs


## ###############################################################
## PREPARE WORKSPACE
## ###############################################################
os.system("clear") # clear terminal window
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def loadData():
  list_sim_data     = []
  abs_filepath_data = WWFnF.createFilepath([ ABS_FILEPATH_BASE, REL_FILEPATH_DATA, "kurtosis_data_gradv.csv" ])
  with open(abs_filepath_data, "r") as dataset_obj:
    for sim_data in csv.reader(dataset_obj):
      list_sim_data.append(sim_data)
  return list_sim_data

def plotData(list_sim_data):
  ## create figure
  factor = 1.45
  _, ax = plt.subplots(figsize=(8.5/factor, 5/factor))
  ## plot data
  for sim_index in range(1, len(list_sim_data)):
    Re  = float(list_sim_data[sim_index][1])
    val = float(list_sim_data[sim_index][3])
    err = float(list_sim_data[sim_index][4])
    print("\t", Re, val, err)
    ax.errorbar(
      x     = Re,
      y     = val,
      yerr  = err,
      color = "cornflowerblue" if Re < 100 else "darkgray" if Re > 2000 else "orangered",
      marker="o", markeredgecolor="black", markersize=9, elinewidth=2, capsize=7.5, zorder=10
    )
  ## undject y-axis range
  ax.set_ylim([-1.0, 2.0])
  ## log-scale x-axis
  ax.set_xscale("log")
  ## add vertical lines
  ax.axvline(x=100, ls="--", color="black")
  ax.text(
    110, 1.9,
    r"Re $= 100$", color="black",
    ha="left", va="top", fontsize=17,
    rotation = 90
  )
  ## add horizontal line
  ax.axhline(y=0, ls="--", color="black")
  ax.text(
    3850, 0.05,
    r"Gaussian", color="black",
    ha="right", va="bottom", fontsize=17
  )
  ## add legends
  ax.text(
    0.05, 0.925,
    r"Rm $\approx 3300$", color="black",
    ha="left", va="top", transform=ax.transAxes, fontsize=17
  )
  PlotFuncs.addLegend(
    ax = ax,
    loc  = "upper left",
    bbox = (-0.0125, 0.9),
    artists = [ "o", "s", "D" ],
    colors  = [ "k" ] * 3,
    legend_labels = [
      r"$\partial_x u_x$",
      r"$\partial_y u_y$",
      r"$\partial_z u_z$"
    ],
    rspacing = 0.25,
    cspacing = 0.25,
    ncol = 1,
    fontsize = 17,
    labelcolor = "black",
    lw = 1.5
  )
  ## add axis labels
  ax.set_xlabel(r"Re", fontsize=22)
  ax.set_ylabel(r"$\mathcal{K} - 3$", fontsize=22)
  ## save figure
  plot_name = "fig_kurtosis.pdf"
  abs_filepath_plot = WWFnF.createFilepath([
    ABS_FILEPATH_BASE,
    REL_FILEPATH_PLOT,
    plot_name
  ])
  plt.savefig(abs_filepath_plot)
  print("Saved figure:", abs_filepath_plot)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  ## LOAD KURTOSIS DATASET
  ## ---------------------
  print("Loading data...")
  list_sim_data = loadData()
  ## PLOT KURTOSIS DATA
  ## ------------------
  print("Plotting data...")
  plotData(list_sim_data)


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
ABS_FILEPATH_BASE = "/Users/dukekriel/Documents/Studies/TurbulentDynamo/"
REL_FILEPATH_DATA = "data/sub_sonic/"
REL_FILEPATH_PLOT = "figures/sub_sonic/"


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit()


## END OF PROGRAM
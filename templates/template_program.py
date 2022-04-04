#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import argparse
import numpy as np

## needs to be loaded before matplotlib
## so matplotlib cache is stored in a temporary location when plotting in parallel
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()

## plotting stuff
import matplotlib.pyplot as plt
import copy # for making a seperate instance of object

## user defined libraries
from the_useful_library import *
from the_loading_library import *
from the_fitting_library import *
from the_plotting_library import *


## ###############################################################
## PREPARE WORKSPACE
##################################################################
os.system("clear") # clear terminal window
plt.ioff()
plt.switch_backend("agg") # use a non-interactive plotting backend


## ###############################################################
## DEFINE MAIN PROGRAM
## ###############################################################
def main():
    ## something
    doStuff = None


## ###############################################################
## RUN PROGRAM
## ###############################################################
if __name__ == "__main__":
    main()
    sys.exit()


## END OF PROGRAM
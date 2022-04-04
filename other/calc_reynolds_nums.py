#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import sys
import numpy as np

from the_useful_library import *
from the_loading_library import *
from the_fitting_library import *


def funcUpdateReynolds(filepath_data, Re, Rm):
    spectra_obj = loadPickleObject(filepath_data, "spectra_obj_fixed.pkl")
    updateAttr(spectra_obj, "Re", Re)
    updateAttr(spectra_obj, "Rm", Rm)
    savePickleObject(spectra_obj, filepath_data, "spectra_obj_fixed.pkl")


filepath_base = "/Users/dukekriel/Documents/Projects/TurbulentDynamo/data"


print("Updating Re10...")
list_Re10_folders = [ "Pm25", "Pm50", "Pm125", "Pm250" ]
list_Re10_Mach = [ 0.27, 0.27, 0.26, 0.25 ]
list_Re10_nu   = [ 2.50e-02 ] * 4
list_Re10_eta  = [ 
    1.00e-03,
    5.00e-04,
    2.00e-04,
    1.00e-04
]
list_Re10_Re = [
    Mach / nu
    for Mach, nu in zip( list_Re10_Mach, list_Re10_nu )
]
list_Re10_Rm = [
    Mach / eta
    for Mach, eta in zip( list_Re10_Mach, list_Re10_eta )
]
for sim_index in range(len(list_Re10_folders)):
    funcUpdateReynolds(
        createFilepath([ filepath_base, "Re10", "288", list_Re10_folders[sim_index] ]),
        list_Re10_Re[sim_index],
        list_Re10_Rm[sim_index]
    )


print("Updating Re500...")
list_Re500_folders = [ "Pm1", "Pm2", "Pm4" ]
list_Re500_Mach = [ 0.26, 0.28, 0.28 ]
list_Re500_nu   = [ 6.00e-04 ] * 3
list_Re500_eta  = [ 
    6.00e-04,
    3.00e-04, 
    1.50e-04 
]
list_Re500_Re = [
    Mach / nu
    for Mach, nu in zip( list_Re500_Mach, list_Re500_nu )
]
list_Re500_Rm = [
    Mach / eta
    for Mach, eta in zip( list_Re500_Mach, list_Re500_eta )
]
for sim_index in range(len(list_Re500_folders)):
    funcUpdateReynolds(
        createFilepath([ filepath_base, "Re500", "288", list_Re500_folders[sim_index] ]),
        list_Re500_Re[sim_index],
        list_Re500_Rm[sim_index]
    )


print("Updating Rm3000...")
list_Rm3000_folders = [ "Pm1", "Pm2", "Pm5", "Pm10", "Pm25", "Pm50", "Pm125", "Pm250" ]
list_Rm3000_Mach = [ 0.30, 0.28, 0.25, 0.24, 0.29, 0.27, 0.29, 0.26 ]
list_Rm3000_nu   = [ 
    8.33e-05,
    1.67e-04,
    4.17e-04,
    8.33e-04,
    2.08e-03,
    4.17e-03,
    1.04e-02,
    2.08e-02
]
list_Rm3000_eta  = [ 
    8.33e-05
] * 8
list_Rm3000_Re = [
    Mach / nu
    for Mach, nu in zip( list_Rm3000_Mach, list_Rm3000_nu )
]
list_Rm3000_Rm = [
    Mach / eta
    for Mach, eta in zip( list_Rm3000_Mach, list_Rm3000_eta )
]
for sim_index in range(len(list_Rm3000_folders)):
    funcUpdateReynolds(
        createFilepath([ filepath_base, "Rm3000", "288", list_Rm3000_folders[sim_index] ]),
        list_Rm3000_Re[sim_index],
        list_Rm3000_Rm[sim_index]
    )


print("Updating keta...")
list_keta_folders = [ "Pm25", "Pm50", "Pm125", "Pm250" ]
list_keta_Mach = [ 0.25, 0.26, 0.27, 0.25 ]
list_keta_nu   = [ 
    3.38e-03,
    5.32e-03,
    9.74e-03,
    1.56e-02 
]
list_keta_eta  = [ 
    1.35e-04,
    1.06e-04,
    7.79e-05,
    6.25e-05 
]
list_keta_Re = [
    Mach / nu
    for Mach, nu in zip( list_keta_Mach, list_keta_nu )
]
list_keta_Rm = [
    Mach / eta
    for Mach, eta in zip( list_keta_Mach, list_keta_eta )
]
for sim_index in range(len(list_keta_folders)):
    funcUpdateReynolds(
        createFilepath([ filepath_base, "keta", "288", list_keta_folders[sim_index] ]),
        list_keta_Re[sim_index],
        list_keta_Rm[sim_index]
    )


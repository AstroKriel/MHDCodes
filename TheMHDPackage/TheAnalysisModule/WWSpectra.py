## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import numpy as np


## ###############################################################
## FUNCTIONS
## ###############################################################
def AveNormSpectraData(list_power_group_t):
  ## normalise spectra data
  list_power_norm_group_t = [
    np.array(list_power) / np.sum(list_power)
    for list_power in list_power_group_t
  ]
  ## return average spectra
  return np.median(list_power_norm_group_t, axis=0)


## END OF LIBRARY
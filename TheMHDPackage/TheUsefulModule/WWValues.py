## START OF MODULE


## ###############################################################
## DEPENDENCIES: REQUIRED MODULES
## ###############################################################
import numpy as np


## ###############################################################
## WORKING WITH NUMBERS
## ###############################################################
def roundDecimalsDown(number:float, decimals:int=1):
  factor = 10 ** decimals
  return np.floor(number * factor) / factor

def roundDecimalsUp(number:float, decimals:int=1):
  factor = 10 ** decimals
  return np.ceil(number * factor) / factor


## END OF MODULE
## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import numpy as np


## ###############################################################
## WORKING WITH ARRAYS
## ###############################################################
def normaliseData(vals):
  ''' normaliseData
  PURPOSE:
    Normalise values by translating and scaling the distribution of elements that lie on [a,b] 
    to one with the same shape but instead lies on [0,1]. All elements become scaled by: 
      (x-a)/(b-a) for all x in vals.
    (This is different to normalising all elements in the set by the magnitude of the largest element).
  '''
  if not(type(vals) == np.ndarray): vals = np.array(vals)
  vals = vals - vals.min()
  if (vals.max() == 0): return vals
  return vals / vals.max()


## END OF LIBRARY
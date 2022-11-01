## START OF LIBRARY


## ###############################################################
## MODULES
## ###############################################################
import numpy as np
from tqdm.auto import tqdm


## ###############################################################
## WORKING WITH LISTS
## ###############################################################
def getCommonElements(list1, list2):
  return sorted( list( set(list1).intersection(set(list2)) ) )

def getUnionElements(list1, list2):
  return sorted( list( set(list1 + list2) ) )

def getIndexClosestValue(input_vals, target_value):
  array_vals = np.asarray(input_vals)
  ## check there are sufficient points
  if array_vals.shape[0] < 3:
    raise Exception(f"ERROR: There are insuffient elements:", array_vals.shape)
  return np.argmin(np.abs(array_vals - target_value))

def flattenList(list_elems):
  return list(np.concatenate(list_elems).flat)

def loopListWithUpdates(list_elems, bool_hide_updates=False):
  lst_len = len(list_elems)
  return zip(
    list_elems,
    tqdm(
      range(lst_len),
      total   = lst_len - 1,
      disable = (lst_len < 3) or bool_hide_updates
    )
  )


## END OF LIBRARY
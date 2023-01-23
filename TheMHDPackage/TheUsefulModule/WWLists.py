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
    raise Exception(f"Error: There is an insuffient number of elements in:", input_vals)
  if target_value ==  np.inf: return np.nanargmax(array_vals)
  if target_value == -np.inf: return np.nanargmin(array_vals)
  return np.nanargmin(np.abs(array_vals - target_value))

def ensureListLength(list_input, list_ref):
  if len(list_input) < len(list_ref):
    list_input.extend(
      [ list_input[0] ] * int( len(list_ref) - len(list_input) )
    )

def flattenList(list_elems):
  return list(np.concatenate(list_elems).flat)

def loopListWithUpdates(list_elems, bool_verbose=True):
  lst_len = len(list_elems)
  return zip(
    list_elems,
    tqdm(
      range(lst_len),
      total   = lst_len - 1,
      disable = (lst_len < 3) or not(bool_verbose)
    )
  )

def getElemFromLoL(list_of_list, index_elem):
  return [
    list_elems[index_elem]
    for list_elems in list_of_list
  ]


## END OF LIBRARY
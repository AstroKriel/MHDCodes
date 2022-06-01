## START OF MODULE


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

def getIndexValueExceeded(input_vals, target_value):
  ## work with arrays
  input_vals = np.asarray(input_vals)
  ## check there are sufficient points in the array
  if input_vals.shape[0] < 3:
    raise Exception("Insuffient points. Array has shape '{}'.".format( input_vals.shape ))
  ## check that the conversion worked
  if isinstance(input_vals, np.ndarray):
    if target_value < min(input_vals): return np.argmin(input_vals)
    if target_value > max(input_vals): return np.argmax(input_vals)
    return np.argmax(input_vals > target_value) # gets first instance of where target_value is exceeded
  ## otherwise throw an error
  else: raise Exception("Values stored as '{:s}' instead of 'numpy array'.".format( type(input_vals) ))

def getIndexClosestValue(input_vals, target_value):
  ## work with arrays
  input_vals = np.asarray(input_vals)
  ## check there are sufficient points in the array
  if input_vals.shape[0] < 3:
    raise Exception("Insuffient points. Array has shape '{}'.".format( input_vals.shape ))
  ## check that the conversion worked
  if isinstance(input_vals, np.ndarray):
    if target_value < min(input_vals): return np.argmin(input_vals)
    if target_value > max(input_vals): return np.argmax(input_vals)
    return np.argmin(np.abs(input_vals - target_value)) # gets the index of the value closest to target_index
  ## otherwise throw an error
  else: raise Exception("Values stored as '{:s}' instead of 'numpy array'.".format( type(input_vals) ))

def getIndexListMin(list_elems):
  ## returns: min_value, min_index
  return min((val, idx) for (idx, val) in enumerate(list_elems))[1]

def getIndexListMax(list_elems):
  ## returns: max_value, max_index
  return max((val, idx) for (idx, val) in enumerate(list_elems))[1]

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

def extendInputList(list_elems, list_str, des_len, des_val=None):
  ## if (the input list is not empty) and (desired value is None / not specified)
  if (len(list_elems) > 0) and (des_val is None):
    ## then extend the list with the list's first entry
    des_val = list_elems[0]
  ## check that the list is shorter than desired
  if (len(list_elems) < des_len) or (list_elems[0] is None):
    ## if (the input list is not defined), but (desired input is a list of correct length)
    if ( 
        (list_elems[0] is None) and 
        ( isinstance(des_val, list) and (len(des_val) == des_len) )
      ):
      print("\t> Set contents of '{:s}' to '{:}'".format(
        list_str,
        len(list_elems),
        len(des_val),
        des_val
      ))
      ## set the contents of the input list as the desired list's contents
      list_elems[:] = des_val
    else:
      print("\t> Extended '{:s}' from length '{:d}' to length '{:d}' with '{:}'".format(
        list_str,
        len(list_elems),
        des_len,
        des_val
      ))
      ## extend the list with desired entry
      list_elems.extend( [des_val] * (des_len - len(list_elems)) )

def getIndexListInBounds(list_elems, bounds):
  return [ 
    elem_index 
    for elem_index, elem_val in enumerate(list_elems)
    if (bounds[0] < elem_val) and (elem_val < bounds[1])
  ]

def subsetListByBounds(list_elems, bounds):
  return [
    elem_val
    for elem_index, elem_val in enumerate(list_elems)
    if (bounds[0] < elem_val) and (elem_val < bounds[1])
  ]

def subsetListByIndices(list_elems, list_indices):
  return [
    elem_val
    for elem_index, elem_val in enumerate(list_elems)
    if elem_index in list_indices
  ]

def removeRedudantOuterList(list_elems):
  return next(iter(list_elems))


## END OF MODULE
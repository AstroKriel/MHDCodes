#!/usr/bin/env python3

## ###############################################################
## MODULES
## ###############################################################
import os
import re
import sys
import argparse
import numpy as np

from tqdm.auto import tqdm
from scipy.stats import norm

## always import the c-version of pickle
try: import cPickle as pickle
except ModuleNotFoundError: import pickle


## ###############################################################
## WORKING WITH ARGUMENT INPUTS
## ###############################################################
def str2bool(v):
  """ str2bool
  BASED ON: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
  """
  if isinstance(v, bool): return v
  if v.lower() in ("yes", "true", "t", "y", "1"): return True
  elif v.lower() in ("no", "false", "f", "n", "0"): return False
  else: raise argparse.ArgumentTypeError("Boolean value expected.")

class MyParser(argparse.ArgumentParser):
  def error(self, message):
    self.print_help()
    sys.exit(1)


## ###############################################################
## WORKING WITH FILES
## ###############################################################
def makeFilter(
    str_contains       = None,
    str_not_contains   = None,
    str_startswith     = None,
    str_endswith       = None,
    file_index_placing = None,
    file_start_index   = 0,
    file_end_index     = np.inf,
    str_split_by       = "_"
  ):
  """ makeFilter
    PURPOSE: Create a filter condition for files that look a particular way.
  """
  def meetsCondition(element):
    ## if str_contains specified, then look for condition
    if str_contains is not None: bool_contains = element.__contains__(str_contains)
    else: bool_contains = True # don't consider condition
    ## if str_not_contains specified, then look for condition
    if str_not_contains is not None: bool_not_contains = not(element.__contains__(str_not_contains))
    else: bool_not_contains = True # don't consider condition
    ## if str_startswith specified, then look for condition
    if str_startswith is not None: bool_startswith = element.startswith(str_startswith)
    else: bool_startswith = True # don't consider condition
    ## if str_endswith specified, then look for condition
    if str_endswith is not None: bool_endswith = element.endswith(str_endswith)
    else: bool_endswith = True # don't consider condition
    ## make sure that the file has the right name structure (i.e. check all conditions have been met)
    if (
        bool_contains and 
        bool_not_contains and 
        bool_startswith and 
        bool_endswith
      ):
      ## if the index range also needs to be checked
      if file_index_placing is not None:
        ## check that the file index falls within the specified range
        if len(element.split(str_split_by)) > abs(file_index_placing):
          bool_time_after  = (
            int(element.split(str_split_by)[file_index_placing]) >= file_start_index
          )
          bool_time_before = (
            int(element.split(str_split_by)[file_index_placing]) <= file_end_index
          )
          ## if the file meets all the required conditions
          if (bool_time_after and bool_time_before): return True
      ## otherwise, all specified conditions have been met
      else: return True
    ## otherwise, don't look at the file
    else: return False
  return meetsCondition

def getFilesFromFolder(
    folder_directory, 
    str_contains       = None,
    str_startswith     = None,
    str_endswith       = None,
    str_not_contains   = None,
    file_index_placing = None,
    file_start_index   = 0,
    file_end_index     = np.inf
  ):
  ''' getFilesFromFolder
    PURPOSE: Return the names of files that meet the required conditions in the specified folder.
  '''
  myFilter = makeFilter(
    str_contains,
    str_not_contains,
    str_startswith,
    str_endswith,
    file_index_placing,
    file_start_index,
    file_end_index
  )
  return list(filter(myFilter, sorted(os.listdir(folder_directory))))


## ###############################################################
## WORKING WITH FOLDERS
## ###############################################################
def createFolder(folder_name, bool_hide_updates=False):
  """ createFolder
  PURPOSE: Create a folder if and only if it does not already exist.
  """
  if not(os.path.exists(folder_name)):
    os.makedirs(folder_name)
    if not(bool_hide_updates):
      print("SUCCESS: Folder created. \n\t" + folder_name + "\n")
  elif not(bool_hide_updates):
    print("WARNING: Folder already exists (folder not created). \n\t" + folder_name + "\n")

def createFilepath(folder_names):
  """ creatFilePath
  PURPOSE: Concatinate a list of folder names into a single string separated by '/'.
  """
  return re.sub( '/+', '/', "/".join([folder for folder in folder_names if folder != ""]) )

def createName(name_elems):
  """ creatFilePath
  PURPOSE: Concatinate a list of folder names into a single string separated by '_.
  """
  return re.sub( '_+', '_', "_".join([elems for elems in name_elems if elems != ""]) )


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
  ## if (the input list is not empty) and (desired value is not specified)
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
      print("\t> Extended '{:s}' from length '{:d}' to length '{:d}' with '{:}'...".format(
        list_str,
        len(list_elems),
        len(des_val),
        des_val
      ))
      ## set the contents of the input list as the desired list's contents
      list_elems[:] = des_val
    else:
      print("\t> Extended '{:s}' from length '{:d}' to length '{:d}' with '{:}'...".format(
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


## ###############################################################
## WORKING WITH NUMBERS
## ###############################################################
def roundDecimalsDown(number:float, decimals:int=1):
  factor = 10 ** decimals
  return np.floor(number * factor) / factor

def roundDecimalsUp(number:float, decimals:int=1):
  factor = 10 ** decimals
  return np.ceil(number * factor) / factor


## ###############################################################
## WORKING WITH DISTRIBUTIONS
## ###############################################################
def sampleGaussFromQuantiles(
    p1, p2,
    x1, x2,
    num_samples = 10**3
  ):
  ''' From: Cook (2010)
    'Determining distribution parameters from quantiles'
  '''
  ## calculate the inverse of the CFD
  cdf_inv_p1 = norm.ppf(p1)
  cdf_inv_p2 = norm.ppf(p2)
  ## calculate mean of the normal distribution
  norm_mean = (
      (x1 * cdf_inv_p2) - (x2 * cdf_inv_p1)
    ) / (cdf_inv_p2 - cdf_inv_p1)
  ## calculate standard deviation of the normal distribution
  norm_std = (x2 - x1) / (cdf_inv_p2 - cdf_inv_p1)
  ## return sampled points
  return norm(loc=norm_mean, scale=norm_std).rvs(size=num_samples)


## ###############################################################
## WORKING WITH OBJECTS
## ###############################################################
def savePickleObject(obj, filepath_folder, obj_filename):
  ## create filepath where object will be saved
  obj_filepath = createFilepath([filepath_folder, obj_filename])
  ## if the file exists, then delete it
  if os.path.isfile(obj_filepath):
    os.remove(obj_filepath)
  ## save new object
  with open(obj_filepath, "wb") as output:
    pickle.dump(obj, output, -1)
  ## print success to terminal
  print("\t> Object saved: " + obj_filepath)

def loadPickleObject(
    filepath_folder,
    obj_filename,
    bool_check = False,
    bool_hide_updates = False
  ):
  ## create filepath where object will be loaded from
  obj_filepath = createFilepath([filepath_folder, obj_filename])
  ## if the file exists, then read it in
  if os.path.isfile(obj_filepath):
    if not bool_hide_updates: print("\t> Loading: " + obj_filepath)
    with open(obj_filepath, "rb") as input:
      return pickle.load(input)
  else:
    if bool_check: return -1
    else: raise Exception("No object '{:s}' found in '{:s}'.".format(
      obj_filename,
      filepath_folder
    ))

def updateAttr(obj, attr, desired_val):
  ## check that the new attribute value is not None
  if desired_val is not None:
    ## check that the new value is not the same as the old value
    if not(getattr(obj, attr) == desired_val):
      ## change the attribute value
      setattr(obj, attr, desired_val)
      return True
  ## don't change the attribute value
  return False

def printObjAttrNames(obj):
  ## loop over all the attribute variable names in the object
  for attr in vars(obj):
    print(attr)


## ###############################################################
## WORKING WITH DICTIONARIES
## ###############################################################
def returnDicWithoutKeys(dic, keys):
  return {k: v for k, v in dic.items() if k not in keys}


## ###############################################################
## PRINTING TO THE TERMINAL
## ###############################################################
def printInfo(str_justified, input_info, num_char_spacing=15):
  ## make sure that the second input won't overlap with the first
  if len(str_justified) > num_char_spacing:
    num_char_spacing = len(str_justified) + 1
  ## if the input is a string, then print it
  if isinstance(input_info, str):
    print(str_justified.ljust(num_char_spacing+1) + input_info)
  ## if the input is a number (i.e. int, float), then print it
  if isinstance(input_info, (int, float)):
    print(str_justified.ljust(num_char_spacing+1) + str(input_info))
  ## otherwise if the input is a list, then print the list elements
  elif isinstance(input_info, (list, np.ndarray)):
    ## if the list is a list of strings
    if isinstance(input_info[0], str):
      print(str_justified.ljust(num_char_spacing), input_info)
    ## otherwise assume its a list of numbers (i.e. int, float)
    else: print(str_justified.ljust(num_char_spacing), [ str(elem) for elem in input_info ])


# END OF LIBRARY